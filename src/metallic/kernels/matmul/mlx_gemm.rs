use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::metallic::{
    Context, Dtype, MetalError,
    encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_bytes_slice, set_compute_pipeline_state},
    kernels::{GemmKernel, GemmTile, GemmTranspose, KernelFunction, gemm_kernel_symbol},
    resource_cache::ResourceCache,
    tensor::MpsMatrixBatchView,
};

use super::{MatMulBackend, MatMulBackendKind, MpsMatMulBackend};

const SWIZZLE_LOG: i32 = 0;

static MLX_GEMM_ENABLED: OnceLock<bool> = OnceLock::new();

pub fn mlx_gemm_enabled() -> bool {
    *MLX_GEMM_ENABLED.get_or_init(|| match std::env::var("METALLIC_GEMM_BACKEND") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            match value.as_str() {
                "mps" | "cpu" | "0" | "off" | "false" => false,
                "mlx" | "1" | "on" | "true" => true,
                _ => true,
            }
        }
        Err(_) => true,
    })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemmParams {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    tiles_n: i32,
    tiles_m: i32,
    batch_stride_a: u64,
    batch_stride_b: u64,
    batch_stride_d: u64,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemmAddmmParams {
    ldc: i32,
    fdc: i32,
    batch_stride_c: u64,
    alpha: f32,
    beta: f32,
}

pub struct MlxGemmBackend {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    ldc: i32,
    fdc: i32,
    tiles_n: i32,
    tiles_m: i32,
    gemm_k_iterations_aligned: i32,
    batch_size: usize,
    batch_ndim: i32,
    batch_stride_a: u64,
    batch_stride_b: u64,
    batch_stride_c: Option<u64>,
    batch_stride_d: u64,
    alpha: f32,
    beta: f32,
    use_out_source: bool,
    kernel: GemmKernel,
    dtype: Dtype,
}

impl MlxGemmBackend {
    pub fn try_new<T: crate::metallic::TensorElement>(
        ctx: &mut Context<T>,
        transpose_left: bool,
        transpose_right: bool,
        left_view: &MpsMatrixBatchView,
        right_view: &MpsMatrixBatchView,
        result_view: &MpsMatrixBatchView,
        left_dtype: Dtype,
        right_dtype: Dtype,
        result_dtype: Dtype,
        alpha_beta: Option<(f32, f32)>,
    ) -> Result<Option<Self>, MetalError> {
        if !mlx_gemm_enabled() {
            return Ok(None);
        }

        if left_dtype != right_dtype || left_dtype != result_dtype {
            return Ok(None);
        }

        if !matches!(left_dtype, Dtype::F32 | Dtype::F16) {
            return Ok(None);
        }

        let elem_size = left_dtype.size_bytes();

        let (m, k) = if transpose_left {
            (left_view.columns, left_view.rows)
        } else {
            (left_view.rows, left_view.columns)
        };
        let (k_right, n) = if transpose_right {
            (right_view.columns, right_view.rows)
        } else {
            (right_view.rows, right_view.columns)
        };

        if k != k_right {
            return Err(MetalError::InvalidOperation(format!(
                "Cannot multiply matrices (with transpose): {}x{} and {}x{}",
                m, k, k_right, n
            )));
        }

        let m_i32 = i32::try_from(m).map_err(|_| MetalError::OperationNotSupported("Matrix rows exceed supported limits".to_string()))?;
        let n_i32 =
            i32::try_from(n).map_err(|_| MetalError::OperationNotSupported("Matrix columns exceed supported limits".to_string()))?;
        let k_i32 = i32::try_from(k)
            .map_err(|_| MetalError::OperationNotSupported("Matrix interior dimension exceeds supported limits".to_string()))?;

        let lda = Self::stride_in_elements(left_view.row_bytes, elem_size)?;
        let ldb = Self::stride_in_elements(right_view.row_bytes, elem_size)?;
        let ldd = Self::stride_in_elements(result_view.row_bytes, elem_size)?;

        let batch_stride_a = Self::stride_in_elements_u64(left_view.matrix_bytes, elem_size)?;
        let batch_stride_b = Self::stride_in_elements_u64(right_view.matrix_bytes, elem_size)?;
        let batch_stride_d = Self::stride_in_elements_u64(result_view.matrix_bytes, elem_size)?;

        let (alpha, beta, use_out_source, do_axpby, batch_stride_c, ldc, fdc) = if let Some((alpha, beta)) = alpha_beta {
            let ldc = ldd;
            let fdc = 1;
            let batch_stride_c = batch_stride_d;
            (alpha, beta, true, true, Some(batch_stride_c), ldc, fdc)
        } else {
            (1.0f32, 0.0f32, false, false, None, 0, 0)
        };

        let batch_size = result_view.batch;
        let has_batch = batch_size > 1;
        let batch_ndim = if has_batch { 1 } else { 0 };

        let transpose = GemmTranspose::from_flags(transpose_left, transpose_right);
        let tile = Self::select_tile(
            &ctx.device,
            left_dtype,
            transpose_left,
            transpose_right,
            m_i32,
            n_i32,
            k_i32,
            batch_size,
        );
        let (bm, bn, bk, wm, wn) = tile.dimensions();

        let tiles_m = Self::ceil_div(m_i32, bm);
        let tiles_n = Self::ceil_div(n_i32, bn);
        let gemm_k_iterations_aligned = if bk > 0 { k_i32 / bk } else { 0 };

        let align_m = m_i32 % bm == 0;
        let align_n = n_i32 % bn == 0;
        let align_k = if bk > 0 { k_i32 % bk == 0 } else { false };
        let do_gather = false;

        let kernel = GemmKernel { transpose, tile };
        gemm_kernel_symbol(transpose, left_dtype, tile)?;
        let kernel_fn = KernelFunction::MlxGemm(kernel);

        let flags = [
            (10u16, has_batch),
            (100u16, use_out_source),
            (110u16, do_axpby),
            (200u16, align_m),
            (201u16, align_n),
            (202u16, align_k),
            (300u16, do_gather),
        ];

        let pipeline = ctx
            .kernel_manager
            .get_pipeline_with_constants(kernel_fn, left_dtype, &ctx.device, Some(&flags))?;

        Ok(Some(Self {
            pipeline,
            m: m_i32,
            n: n_i32,
            k: k_i32,
            lda,
            ldb,
            ldd,
            ldc,
            fdc,
            tiles_n,
            tiles_m,
            gemm_k_iterations_aligned,
            batch_size,
            batch_ndim,
            batch_stride_a,
            batch_stride_b,
            batch_stride_c,
            batch_stride_d,
            alpha,
            beta,
            use_out_source,
            kernel,
            dtype: left_dtype,
        }))
    }

    fn stride_in_elements(bytes: usize, elem_size: usize) -> Result<i32, MetalError> {
        let elems = Self::stride_in_elements_u64(bytes, elem_size)?;
        i32::try_from(elems).map_err(|_| MetalError::OperationNotSupported("Stride exceeds supported limits".to_string()))
    }

    fn stride_in_elements_u64(bytes: usize, elem_size: usize) -> Result<u64, MetalError> {
        if bytes % elem_size != 0 {
            return Err(MetalError::InvalidOperation(
                "Tensor strides are not aligned to element size".to_string(),
            ));
        }
        Ok((bytes / elem_size) as u64)
    }

    fn ceil_div(value: i32, divisor: i32) -> i32 {
        (value + divisor - 1) / divisor
    }

    pub fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        left_buf: &Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        left_offset: usize,
        right_buf: &Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        right_offset: usize,
        result_buf: &Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        result_offset: usize,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, left_buf, left_offset);
        set_buffer(&encoder, 1, right_buf, right_offset);
        if self.use_out_source {
            set_buffer(&encoder, 2, result_buf, result_offset);
        }
        set_buffer(&encoder, 3, result_buf, result_offset);

        let params = GemmParams {
            m: self.m,
            n: self.n,
            k: self.k,
            lda: self.lda,
            ldb: self.ldb,
            ldd: self.ldd,
            tiles_n: self.tiles_n,
            tiles_m: self.tiles_m,
            batch_stride_a: self.batch_stride_a,
            batch_stride_b: self.batch_stride_b,
            batch_stride_d: self.batch_stride_d,
            swizzle_log: SWIZZLE_LOG,
            gemm_k_iterations_aligned: self.gemm_k_iterations_aligned,
            batch_ndim: self.batch_ndim,
        };
        set_bytes(&encoder, 4, &params);

        if self.use_out_source {
            let addmm_params = GemmAddmmParams {
                ldc: self.ldc,
                fdc: self.fdc,
                batch_stride_c: self.batch_stride_c.unwrap_or(0),
                alpha: self.alpha,
                beta: self.beta,
            };
            set_bytes(&encoder, 5, &addmm_params);
        }

        let batch_shape = [self.batch_size as i32];
        set_bytes_slice(&encoder, 6, &batch_shape);

        let mut batch_strides = [self.batch_stride_a, self.batch_stride_b, 0u64];
        if let Some(stride_c) = self.batch_stride_c {
            batch_strides[2] = stride_c;
        }
        let stride_len = if self.use_out_source { 3 } else { 2 };
        set_bytes_slice(&encoder, 7, &batch_strides[..stride_len]);

        let (_, _, _, wm, wn) = self.kernel.tile.dimensions();
        let threads_per_tg = MTLSize {
            width: 32,
            height: wn as NSUInteger,
            depth: wm as NSUInteger,
        };
        let groups = MTLSize {
            width: self.tiles_n.max(1) as NSUInteger,
            height: self.tiles_m.max(1) as NSUInteger,
            depth: self.batch_size.max(1) as NSUInteger,
        };

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();

        Ok(())
    }

    pub(crate) fn kernel(&self) -> GemmKernel {
        self.kernel
    }

    pub(crate) fn kernel_symbol(&self) -> Result<&'static str, MetalError> {
        gemm_kernel_symbol(self.kernel.transpose, self.dtype, self.kernel.tile)
    }

    fn architecture_suffix(device: &Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>) -> Option<char> {
        unsafe {
            let architecture = device.architecture();
            let name = architecture.name();
            let suffix = name.to_string().chars().last()?;
            Some(suffix.to_ascii_lowercase())
        }
    }

    fn select_tile(
        device: &Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
        dtype: Dtype,
        transpose_left: bool,
        transpose_right: bool,
        m: i32,
        n: i32,
        k: i32,
        batch_size: usize,
    ) -> GemmTile {
        let mut bm = 64;
        let mut bn = 64;
        let mut bk = 16;
        let mut wm = 2;
        let mut wn = 2;

        let arch = Self::architecture_suffix(device);
        let is_float = matches!(dtype, Dtype::F32);
        let large_matmul = {
            let batch = batch_size as u128;
            let m = m as u128;
            let n = n as u128;
            batch.saturating_mul(m).saturating_mul(n) >= (1u128 << 20)
        };
        let reasonable_k = {
            let max_dim = m.max(n) as u128;
            let k_extent = k as u128;
            max_dim.saturating_mul(2) > k_extent
        };

        match arch {
            Some('g') | Some('p') => {
                if !transpose_left && transpose_right {
                    bm = 64;
                    bn = 32;
                    bk = 32;
                    wm = 2;
                    wn = 2;
                } else if !is_float {
                    bm = 64;
                    bn = 64;
                    bk = 16;
                    wm = 1;
                    wn = 2;
                }
            }
            Some('d') => {
                if large_matmul {
                    if !is_float {
                        if reasonable_k {
                            bm = 64;
                            bn = 64;
                            bk = 16;
                            wm = 1;
                            wn = 2;
                        } else if !transpose_left && transpose_right {
                            bm = 64;
                            bn = 32;
                            bk = 32;
                            wm = 2;
                            wn = 2;
                        } else {
                            bm = 32;
                            bn = 64;
                            bk = 16;
                            wm = 1;
                            wn = 2;
                        }
                    }
                } else if !is_float {
                    if !transpose_left && transpose_right {
                        bm = 64;
                        bn = 32;
                        bk = 32;
                        wm = 2;
                        wn = 2;
                    } else {
                        bm = 64;
                        bn = 64;
                        bk = 16;
                        wm = 1;
                        wn = 2;
                    }
                } else if !transpose_left && transpose_right {
                    bm = 32;
                    bn = 64;
                    bk = 16;
                    wm = 1;
                    wn = 2;
                } else {
                    bm = 64;
                    bn = 32;
                    bk = 32;
                    wm = 2;
                    wn = 2;
                }
            }
            _ => {
                // Medium devices keep the default tile configuration
            }
        }

        GemmTile::from_dims(bm, bn, bk, wm, wn).unwrap_or(GemmTile::Bm32Bn32Bk16Wm2Wn2)
    }
}

pub fn try_create_backend<T: crate::metallic::TensorElement>(
    ctx: &mut Context<T>,
    cache: Option<&mut ResourceCache>,
    transpose_left: bool,
    transpose_right: bool,
    left_view: &MpsMatrixBatchView,
    right_view: &MpsMatrixBatchView,
    result_view: &MpsMatrixBatchView,
    left_dtype: Dtype,
    right_dtype: Dtype,
    result_dtype: Dtype,
    alpha_beta: Option<(f32, f32)>,
) -> Result<MatMulBackend, MetalError> {
    if let Some(backend) = MlxGemmBackend::try_new(
        ctx,
        transpose_left,
        transpose_right,
        left_view,
        right_view,
        result_view,
        left_dtype,
        right_dtype,
        result_dtype,
        alpha_beta,
    )? {
        ctx.note_matmul_backend(MatMulBackendKind::Mlx);
        return Ok(MatMulBackend::Mlx(backend));
    }

    let cache = cache.ok_or_else(|| MetalError::InvalidOperation("Resource cache required for matmul".to_string()))?;

    let backend = MpsMatMulBackend::new(
        ctx,
        cache,
        transpose_left,
        transpose_right,
        left_view,
        right_view,
        result_view,
        alpha_beta.unwrap_or((1.0, 0.0)),
        left_dtype,
        right_dtype,
        result_dtype,
    )?;

    ctx.note_matmul_backend(MatMulBackendKind::Mps);

    Ok(MatMulBackend::Mps(backend))
}
