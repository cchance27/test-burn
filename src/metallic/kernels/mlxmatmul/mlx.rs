use crate::metallic::kernels::{KernelFunction, KernelInvocable};
use crate::metallic::{
    Context, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage,
    encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state},
    tensor::Dtype,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder as _, MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLSize,
};
use rustc_hash::FxHashMap;
use std::convert::{TryFrom, TryInto};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::metallic::kernels::matmul::MatMulBackend;
use crate::metallic::resource_cache::ResourceCache;

const BM: usize = 32;
const BN: usize = 32;
const BK: usize = 16;
const WN: usize = 2;
const WM: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GemmParams {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    tiles_n: i32,
    tiles_m: i32,
    batch_stride_a: isize,
    batch_stride_b: isize,
    batch_stride_d: isize,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GemmAddmmParams {
    ldc: i32,
    fdc: i32,
    batch_stride_c: usize,
    alpha: f32,
    beta: f32,
}

#[derive(Hash, PartialEq, Eq)]
struct PipelineKey {
    name: String,
    has_batch: bool,
    use_out_source: bool,
    do_axpby: bool,
    align_m: bool,
    align_n: bool,
    align_k: bool,
}

pub(crate) struct MlxKernelCache {
    library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
    pipelines: FxHashMap<PipelineKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl Default for MlxKernelCache {
    fn default() -> Self {
        Self {
            library: None,
            pipelines: FxHashMap::default(),
        }
    }
}

impl MlxKernelCache {
    fn library<'a>(
        &'a mut self,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
        if let Some(library) = &self.library {
            return Ok(library.clone());
        }

        let source = include_str!("mlx.metal");
        let source_ns = NSString::from_str(source);
        let library = device
            .newLibraryWithSource_options_error(&source_ns, None)
            .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;
        self.library = Some(library.clone());
        Ok(library)
    }

    fn pipeline(
        &mut self,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        name: &str,
        constants: &MlxConstants,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        let key = PipelineKey {
            name: name.to_string(),
            has_batch: constants.has_batch,
            use_out_source: constants.use_out_source,
            do_axpby: constants.do_axpby,
            align_m: constants.align_m,
            align_n: constants.align_n,
            align_k: constants.align_k,
        };

        if let Some(pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline.clone());
        }

        let library = self.library(device)?;
        let fn_name = NSString::from_str(&key.name);
        let constant_values = constants.to_function_constants();
        let function = library
            .newFunctionWithName_constantValues_error(&fn_name, &constant_values)
            .map_err(|err| MetalError::FunctionCreationFailed(err.to_string()))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|_err| MetalError::PipelineCreationFailed)?;
        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}

struct MlxConstants {
    has_batch: bool,
    use_out_source: bool,
    do_axpby: bool,
    align_m: bool,
    align_n: bool,
    align_k: bool,
}

impl MlxConstants {
    fn to_function_constants(&self) -> Retained<MTLFunctionConstantValues> {
        let constants = MTLFunctionConstantValues::new();

        set_bool_constant(&constants, 10, self.has_batch);
        set_bool_constant(&constants, 100, self.use_out_source);
        set_bool_constant(&constants, 110, self.do_axpby);
        set_bool_constant(&constants, 200, self.align_m);
        set_bool_constant(&constants, 201, self.align_n);
        set_bool_constant(&constants, 202, self.align_k);
        set_bool_constant(&constants, 300, false); // gather_bias

        constants
    }
}

fn set_bool_constant(constants: &MTLFunctionConstantValues, index: usize, value: bool) {
    let mut raw = value as u8;
    let ptr = NonNull::new(&mut raw as *mut u8).expect("stack pointer is never null");
    unsafe {
        constants.setConstantValue_type_atIndex(ptr.cast::<c_void>(), MTLDataType::Bool, index);
    }
}

pub struct MatMulMlxOp;

struct MatMulMlx<T: TensorElement> {
    left: Tensor<T>,
    right: Tensor<T>,
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    params: GemmParams,
    addmm: Option<GemmAddmmParams>,
    batch_size: usize,
    batch_strides: [usize; 3],
    threadgroups: MTLSize,
    threads_per_tg: MTLSize,
    use_out_source: bool,
}

impl KernelInvocable for MatMulMlxOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, Option<&'a Tensor<T>>, bool, bool, f32, f32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (left, right, existing_out, transpose_left, transpose_right, alpha, beta) = args;
        if beta != 0.0 && existing_out.is_none() {
            return Err(MetalError::InvalidOperation("beta requires an existing output tensor".to_string()));
        }

        // Avoid MPS-specific batch compaction: MLX GEMM supports arbitrary batch strides directly.
        let left_tensor = left.clone();
        let right_tensor = right.clone();
        let left_view = left.as_mps_matrix_batch_view()?;
        let right_view = right.as_mps_matrix_batch_view()?;

        let batch = left_view.batch;
        if right_view.batch != batch {
            return Err(MetalError::InvalidOperation(
                "Left and right operands must share the same batch".to_string(),
            ));
        }

        let (a_rows_base, a_cols_base) = (left_view.rows, left_view.columns);
        let (b_rows_base, b_cols_base) = (right_view.rows, right_view.columns);

        let (a_rows, a_cols) = if transpose_left {
            (a_cols_base, a_rows_base)
        } else {
            (a_rows_base, a_cols_base)
        };
        let (b_rows, b_cols) = if transpose_right {
            (b_cols_base, b_rows_base)
        } else {
            (b_rows_base, b_cols_base)
        };

        if a_cols != b_rows {
            return Err(MetalError::InvalidOperation(format!(
                "Cannot multiply matrices with shapes {}x{} and {}x{}",
                a_rows, a_cols, b_rows, b_cols
            )));
        }

        let m = a_rows;
        let n = b_cols;
        let k = a_cols;

        if m == 0 || n == 0 || k == 0 {
            return Err(MetalError::InvalidOperation("MatMul dimensions must be non-zero".to_string()));
        }

        let (lda, layout_a_transposed) = determine_layout(&left_tensor, a_rows_base, a_cols_base)?;
        let (ldb, layout_b_transposed) = determine_layout(&right_tensor, b_rows_base, b_cols_base)?;

        let effective_a_trans = layout_a_transposed ^ transpose_left;
        let effective_b_trans = layout_b_transposed ^ transpose_right;

        let dtype = T::DTYPE;
        let function_name = gemm_function_name(dtype, effective_a_trans, effective_b_trans)?;

        let has_batch = batch > 1;
        let align_m = m % BM == 0;
        let align_n = n % BN == 0;
        let align_k = k % BK == 0;
        let requires_epilogue = alpha != 1.0 || beta != 0.0;
        let use_out_source = requires_epilogue;
        let constants = MlxConstants {
            has_batch,
            use_out_source,
            do_axpby: requires_epilogue,
            align_m,
            align_n,
            align_k,
        };

        let pipeline = ctx.mlx_kernel_cache.pipeline(&ctx.device, &function_name, &constants)?;

        let elem_size = dtype.size_bytes();
        let batch_stride_a = isize::try_from(left_view.matrix_bytes / elem_size)
            .map_err(|_| MetalError::InvalidOperation("batch stride for A exceeds isize".to_string()))?;
        let batch_stride_b = isize::try_from(right_view.matrix_bytes / elem_size)
            .map_err(|_| MetalError::InvalidOperation("batch stride for B exceeds isize".to_string()))?;

        let out_dims = output_dims(batch, m, n);
        let out = if let Some(existing) = existing_out {
            existing.clone()
        } else if use_out_source {
            Tensor::zeros(out_dims, ctx, true)?
        } else {
            Tensor::new(out_dims, TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?
        };

        if let Some(existing) = existing_out {
            let out_view = existing.as_mps_matrix_batch_view()?;
            if out_view.batch != batch || out_view.rows != m || out_view.columns != n {
                return Err(MetalError::InvalidOperation(format!(
                    "Existing output shape mismatch: expected batch={}, rows={}, cols={}, got batch={}, rows={}, cols={}",
                    batch, m, n, out_view.batch, out_view.rows, out_view.columns
                )));
            }
        }

        if use_out_source {
            ctx.prepare_tensors_for_active_cmd(&[&left_tensor, &right_tensor, &out])?;
        } else {
            ctx.prepare_tensors_for_active_cmd(&[&left_tensor, &right_tensor])?;
        }

        let out_view = out.as_mps_matrix_batch_view()?;
        let ldd = determine_leading_dimension(&out, m, n)?;
        let fdc = determine_minor_stride(&out)?;
        let batch_stride_d = isize::try_from(out_view.matrix_bytes / elem_size)
            .map_err(|_| MetalError::InvalidOperation("batch stride for output exceeds isize".to_string()))?;

        let swizzle_log = 0;
        let tile = 1usize << swizzle_log;
        let tn = div_ceil(n, BN) * tile;
        let mut tm = div_ceil(m, BM);
        tm = div_ceil(tm, tile);

        let tiles_n = i32::try_from(tn).map_err(|_| MetalError::InvalidOperation("tiles_n exceeds i32".to_string()))?;
        let tiles_m = i32::try_from(tm).map_err(|_| MetalError::InvalidOperation("tiles_m exceeds i32".to_string()))?;

        let params = GemmParams {
            m: m.try_into()
                .map_err(|_| MetalError::InvalidOperation("m exceeds i32".to_string()))?,
            n: n.try_into()
                .map_err(|_| MetalError::InvalidOperation("n exceeds i32".to_string()))?,
            k: k.try_into()
                .map_err(|_| MetalError::InvalidOperation("k exceeds i32".to_string()))?,
            lda,
            ldb,
            ldd,
            tiles_n,
            tiles_m,
            batch_stride_a,
            batch_stride_b,
            batch_stride_d,
            swizzle_log,
            gemm_k_iterations_aligned: (k / BK) as i32,
            batch_ndim: if has_batch { 1 } else { 0 },
        };

        let addmm = if use_out_source {
            Some(GemmAddmmParams {
                ldc: ldd,
                fdc,
                batch_stride_c: out_view.matrix_bytes / elem_size,
                alpha,
                beta,
            })
        } else {
            None
        };

        let batch_strides = [
            if has_batch {
                usize::try_from(batch_stride_a)
                    .map_err(|_| MetalError::InvalidOperation("batch stride for A must be positive".to_string()))?
            } else {
                0
            },
            if has_batch {
                usize::try_from(batch_stride_b)
                    .map_err(|_| MetalError::InvalidOperation("batch stride for B must be positive".to_string()))?
            } else {
                0
            },
            if has_batch && use_out_source {
                out_view.matrix_bytes / elem_size
            } else {
                0
            },
        ];
        let threadgroups = MTLSize {
            width: tn,
            height: tm,
            depth: batch.max(1),
        };
        let threads_per_tg = MTLSize {
            width: 32,
            height: WN,
            depth: WM,
        };

        {
            let command_buffer = {
                let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
                command_buffer.clone()
            };
            ctx.register_matmul_dispatch(&command_buffer, MatMulBackend::Mlx);
        }

        let op = MatMulMlx {
            left: left_tensor,
            right: right_tensor,
            out: out.clone(),
            pipeline,
            params,
            addmm,
            batch_size: batch,
            batch_strides,
            threadgroups,
            threads_per_tg,
            use_out_source,
        };

        Ok((Box::new(op), out))
    }
}

impl<T: TensorElement> Operation for MatMulMlx<T> {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.left.buf, self.left.offset);
        set_buffer(&encoder, 1, &self.right.buf, self.right.offset);
        if self.use_out_source {
            set_buffer(&encoder, 2, &self.out.buf, self.out.offset);
        }
        set_buffer(&encoder, 3, &self.out.buf, self.out.offset);
        set_bytes(&encoder, 4, &self.params);
        if let Some(addmm) = &self.addmm {
            set_bytes(&encoder, 5, addmm);
        }
        let batch_shape: i32 = self
            .batch_size
            .try_into()
            .map_err(|_| MetalError::InvalidOperation("Batch size exceeds i32".to_string()))?;
        set_bytes(&encoder, 6, &batch_shape);
        set_bytes(&encoder, 7, &self.batch_strides);

        dispatch_threadgroups(&encoder, self.threadgroups, self.threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

fn determine_layout<T: TensorElement>(tensor: &Tensor<T>, rows: usize, cols: usize) -> Result<(i32, bool), MetalError> {
    if tensor.strides.len() < 2 {
        return Err(MetalError::InvalidShape(
            "Matrix tensor must have at least two dimensions".to_string(),
        ));
    }
    let stride_minor = tensor.strides[tensor.strides.len() - 1];
    let stride_major = tensor.strides[tensor.strides.len() - 2];

    let row_major = (cols == 1 || stride_minor == 1) && (rows <= 1 || stride_major >= cols);
    if row_major {
        let lda_usize = if rows <= 1 { cols } else { stride_major };
        let lda = i32::try_from(lda_usize)
            .map_err(|_| MetalError::InvalidOperation(format!("Leading dimension for row-major layout exceeds i32: {}", lda_usize)))?;
        return Ok((lda, false));
    }

    let column_major = (rows == 1 || stride_major == 1) && (cols <= 1 || stride_minor >= rows);
    if column_major {
        let lda_usize = if cols <= 1 { rows } else { stride_minor };
        let lda = i32::try_from(lda_usize)
            .map_err(|_| MetalError::InvalidOperation(format!("Leading dimension for column-major layout exceeds i32: {}", lda_usize)))?;
        return Ok((lda, true));
    } else {
        Err(MetalError::InvalidOperation(format!(
            "Matrix is not contiguous in a supported layout: strides={:?}",
            tensor.strides
        )))
    }
}

fn determine_leading_dimension<T: TensorElement>(tensor: &Tensor<T>, rows: usize, cols: usize) -> Result<i32, MetalError> {
    if tensor.strides.len() < 2 {
        return Err(MetalError::InvalidShape(
            "Output tensor must have at least two dimensions".to_string(),
        ));
    }
    let stride = tensor.strides[tensor.strides.len() - 2];
    if stride < cols && rows > 1 {
        return Err(MetalError::InvalidOperation(format!(
            "Output stride too small for dimensions: stride={}, rows={}, cols={}",
            stride, rows, cols
        )));
    }
    Ok(stride as i32)
}

fn determine_minor_stride<T: TensorElement>(tensor: &Tensor<T>) -> Result<i32, MetalError> {
    if tensor.strides.is_empty() {
        return Err(MetalError::InvalidShape(
            "Output tensor must have at least one dimension".to_string(),
        ));
    }
    Ok(tensor.strides[tensor.strides.len() - 1] as i32)
}

fn gemm_function_name(dtype: Dtype, a_trans: bool, b_trans: bool) -> Result<String, MetalError> {
    let dtype_str = match dtype {
        Dtype::F32 => "f32",
        Dtype::F16 => "f16",
    };

    let a_tag = if a_trans { "t" } else { "n" };
    let b_tag = if b_trans { "t" } else { "n" };

    Ok(format!(
        "gemm_{}{}_{dtype}_{dtype}_{BM}_{BN}_{BK}_{WM}_{WN}",
        a_tag,
        b_tag,
        dtype = dtype_str,
        BM = BM,
        BN = BN,
        BK = BK,
        WM = WM,
        WN = WN,
    ))
}

fn output_dims(batch: usize, rows: usize, cols: usize) -> Vec<usize> {
    if batch > 1 { vec![batch, rows, cols] } else { vec![rows, cols] }
}

fn div_ceil(value: usize, divisor: usize) -> usize {
    (value + divisor - 1) / divisor
}
