use super::*;
use crate::metallic::kernels::kernel_manager::MlxPipelineKey;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputePipelineState, MTLResource, MTLResourceUsage};
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixMultiplication};
#[cfg(test)]
use std::cell::Cell;

use super::{
    KernelFunction, KernelInvocable, MatMulBackend, MatMulBackendPreference, matrix_strides_from_view, mps_matrix_from_buffer, supports_mlx,
};
use crate::metallic::{
    Context, MetalError, Operation, Tensor, TensorElement,
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey},
    instrumentation::MatMulDispatchHandle,
    resource_cache::ResourceCache,
    tensor::MpsMatrixBatchView,
};

pub struct MatMulAlphaBetaOp;

enum MatMulAlphaBetaImplementation {
    Mps(MatMulAlphaBetaMps),
    Mlx(MatMulAlphaBetaMlx),
}

struct MatMulAlphaBeta {
    implementation: MatMulAlphaBetaImplementation,
}

struct MatMulAlphaBetaMps {
    left_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    left_offset: usize,
    right_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    right_offset: usize,
    result_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    result_offset: usize,
    left_desc: Retained<MPSMatrixDescriptor>,
    right_desc: Retained<MPSMatrixDescriptor>,
    result_desc: Retained<MPSMatrixDescriptor>,
    gemm: Retained<MPSMatrixMultiplication>,
    batch_size: usize,
    profiler: Option<MatMulDispatchHandle>,
}

#[cfg(test)]
thread_local! {
    static LAST_MLX_ALPHA_BETA_FLAGS: Cell<Option<(bool, bool)>> = Cell::new(None);
}

#[cfg(test)]
fn record_mlx_alpha_beta_flags(use_out_source: bool, scale_only: bool) {
    LAST_MLX_ALPHA_BETA_FLAGS.with(|cell| cell.set(Some((use_out_source, scale_only))));
}

#[cfg(test)]
pub(crate) fn take_last_mlx_alpha_beta_flags() -> Option<(bool, bool)> {
    LAST_MLX_ALPHA_BETA_FLAGS.with(|cell| {
        let value = cell.get();
        cell.set(None);
        value
    })
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

struct MatMulAlphaBetaMlx {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    left_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    left_offset: usize,
    right_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    right_offset: usize,
    result_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    result_offset: usize,
    params: super::GemmParams,
    addmm_params: GemmAddmmParams,
    batch_size: usize,
    batch_stride_a: usize,
    batch_stride_b: usize,
    tiles_n: usize,
    tiles_m: usize,
    use_out_source: bool,
    profiler: Option<MatMulDispatchHandle>,
}

impl KernelInvocable for MatMulAlphaBetaOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, bool, f32, f32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (left, right, result, transpose_left, transpose_right, alpha, beta) = args;

        let (left_tensor, left_view) = left.ensure_mps_contiguous_batch(ctx)?;
        let (right_tensor, right_view) = right.ensure_mps_contiguous_batch(ctx)?;
        let result_view = result.as_mps_matrix_batch_view()?;

        ctx.prepare_tensors_for_active_cmd(&[&left_tensor, &right_tensor, result])?;

        let (eff_left_rows, eff_left_cols) = if transpose_left {
            (left_view.columns, left_view.rows)
        } else {
            (left_view.rows, left_view.columns)
        };
        let (eff_right_rows, eff_right_cols) = if transpose_right {
            (right_view.columns, right_view.rows)
        } else {
            (right_view.rows, right_view.columns)
        };

        if eff_left_cols != eff_right_rows {
            return Err(MetalError::InvalidOperation(format!(
                "Cannot multiply matrices (with transpose): {}x{} and {}x{}",
                eff_left_rows, eff_left_cols, eff_right_rows, eff_right_cols
            )));
        }

        if left_view.batch != right_view.batch || left_view.batch != result_view.batch {
            return Err(MetalError::InvalidOperation(
                "Batched matmul requires consistent batch dimensions".to_string(),
            ));
        }

        if result_view.rows != eff_left_rows || result_view.columns != eff_right_cols {
            return Err(MetalError::InvalidOperation(format!(
                "Result tensor dimensions ({}, {}) do not match expected output dimensions ({}, {})",
                result_view.rows, result_view.columns, eff_left_rows, eff_right_cols
            )));
        }

        let preference = ctx.matmul_backend_preference();
        let mut selected_backend = MatMulBackend::Mps;
        let mut implementation: Option<MatMulAlphaBetaImplementation> = None;

        if preference != MatMulBackendPreference::ForceMps
            && supports_mlx(&left_tensor, &left_view, &right_tensor, &right_view)
            && super::mlx_operand_is_contiguous(result, &result_view)
        {
            match MatMulAlphaBetaMlx::new(
                ctx,
                &left_tensor,
                &right_tensor,
                result,
                left_view,
                right_view,
                result_view,
                eff_left_rows,
                eff_left_cols,
                eff_right_cols,
                transpose_left,
                transpose_right,
                alpha,
                beta,
            ) {
                Ok(mlx_op) => {
                    selected_backend = if transpose_left || transpose_right {
                        MatMulBackend::MlxTransposed
                    } else {
                        MatMulBackend::Mlx
                    };
                    implementation = Some(MatMulAlphaBetaImplementation::Mlx(mlx_op));
                }
                Err(err) => {
                    if preference == MatMulBackendPreference::ForceMlx {
                        return Err(err);
                    }
                }
            }
        } else if preference == MatMulBackendPreference::ForceMlx {
            return Err(MetalError::OperationNotSupported(
                "MatMul MLX backend requires contiguous inputs".to_string(),
            ));
        }

        let implementation = if let Some(impl_) = implementation {
            impl_
        } else {
            let cache = cache.ok_or_else(|| MetalError::InvalidOperation("Resource cache required for matmul".to_string()))?;
            let mps_op = MatMulAlphaBetaMps::new(
                ctx,
                cache,
                &left_tensor,
                &right_tensor,
                result,
                left_view,
                right_view,
                result_view,
                eff_left_rows,
                eff_left_cols,
                eff_right_cols,
                transpose_left,
                transpose_right,
                alpha,
                beta,
            )?;
            MatMulAlphaBetaImplementation::Mps(mps_op)
        };

        let command_buffer = {
            let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
            command_buffer.clone()
        };
        let profiler = ctx.register_matmul_dispatch(&command_buffer, selected_backend);

        let implementation = match implementation {
            MatMulAlphaBetaImplementation::Mps(op) => MatMulAlphaBetaImplementation::Mps(op.with_profiler(profiler)),
            MatMulAlphaBetaImplementation::Mlx(op) => MatMulAlphaBetaImplementation::Mlx(op.with_profiler(profiler)),
        };

        Ok((Box::new(MatMulAlphaBeta { implementation }), result.clone()))
    }
}

impl Operation for MatMulAlphaBeta {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        match &self.implementation {
            MatMulAlphaBetaImplementation::Mps(op) => op.encode(command_buffer),
            MatMulAlphaBetaImplementation::Mlx(op) => op.encode(command_buffer),
        }
    }
}

impl MatMulAlphaBetaMps {
    #[allow(clippy::too_many_arguments)]
    fn new<T: TensorElement>(
        ctx: &Context<T>,
        cache: &mut ResourceCache,
        left_tensor: &Tensor<T>,
        right_tensor: &Tensor<T>,
        result: &Tensor<T>,
        left_view: MpsMatrixBatchView,
        right_view: MpsMatrixBatchView,
        result_view: MpsMatrixBatchView,
        eff_left_rows: usize,
        eff_left_cols: usize,
        eff_right_cols: usize,
        transpose_left: bool,
        transpose_right: bool,
        alpha: f32,
        beta: f32,
    ) -> Result<Self, MetalError> {
        let left_dtype = left_tensor.dtype;
        let right_dtype = right_tensor.dtype;
        let result_dtype = result.dtype;

        let gemm_key = MpsGemmKey {
            transpose_left,
            transpose_right,
            result_rows: eff_left_rows,
            result_columns: eff_right_cols,
            interior_columns: eff_left_cols,
            batch_size: result_view.batch,
            alpha,
            beta,
        };

        let gemm = cache.get_or_create_gemm(gemm_key, &ctx.device)?;

        let left_desc_key = MpsMatrixDescriptorKey {
            rows: left_view.rows,
            columns: left_view.columns,
            row_bytes: left_view.row_bytes,
            matrices: left_view.batch,
            matrix_bytes: left_view.matrix_bytes,
            dtype: left_dtype,
        };
        let left_desc = cache.get_or_create_descriptor(left_desc_key, &ctx.device)?;

        let right_desc_key = MpsMatrixDescriptorKey {
            rows: right_view.rows,
            columns: right_view.columns,
            row_bytes: right_view.row_bytes,
            matrices: right_view.batch,
            matrix_bytes: right_view.matrix_bytes,
            dtype: right_dtype,
        };
        let right_desc = cache.get_or_create_descriptor(right_desc_key, &ctx.device)?;

        let result_desc_key = MpsMatrixDescriptorKey {
            rows: eff_left_rows,
            columns: eff_right_cols,
            row_bytes: result_view.row_bytes,
            matrices: result_view.batch,
            matrix_bytes: result_view.matrix_bytes,
            dtype: result_dtype,
        };
        let result_desc = cache.get_or_create_descriptor(result_desc_key, &ctx.device)?;

        Ok(Self {
            left_buf: left_tensor.buf.clone(),
            left_offset: left_tensor.offset,
            right_buf: right_tensor.buf.clone(),
            right_offset: right_tensor.offset,
            result_buf: result.buf.clone(),
            result_offset: result.offset,
            left_desc,
            right_desc,
            result_desc,
            gemm,
            batch_size: result_view.batch,
            profiler: None,
        })
    }

    fn with_profiler(mut self, profiler: MatMulDispatchHandle) -> Self {
        self.profiler = Some(profiler);
        self
    }

    fn encode(&self, command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Result<(), MetalError> {
        if let Some(profiler) = &self.profiler {
            profiler.sample_start_blit(command_buffer);
        }

        let left = mps_matrix_from_buffer(&self.left_buf, self.left_offset, &self.left_desc);
        let right = mps_matrix_from_buffer(&self.right_buf, self.right_offset, &self.right_desc);
        let result = mps_matrix_from_buffer(&self.result_buf, self.result_offset, &self.result_desc);

        unsafe {
            self.gemm.setBatchStart(0 as NSUInteger);
            self.gemm.setBatchSize(self.batch_size as NSUInteger);
        }
        super::encode_mps_matrix_multiplication(&self.gemm, command_buffer, &left, &right, &result);

        if let Some(profiler) = &self.profiler {
            profiler.sample_end_blit(command_buffer);
        }

        Ok(())
    }
}

impl MatMulAlphaBetaMlx {
    #[allow(clippy::too_many_arguments)]
    fn new<T: TensorElement>(
        ctx: &mut Context<T>,
        left_tensor: &Tensor<T>,
        right_tensor: &Tensor<T>,
        result: &Tensor<T>,
        left_view: MpsMatrixBatchView,
        right_view: MpsMatrixBatchView,
        result_view: MpsMatrixBatchView,
        eff_left_rows: usize,
        eff_left_cols: usize,
        eff_right_cols: usize,
        transpose_left: bool,
        transpose_right: bool,
        alpha: f32,
        beta: f32,
    ) -> Result<Self, MetalError> {
        let m = eff_left_rows;
        let n = eff_right_cols;
        let k = eff_left_cols;

        let has_batch = left_view.batch > 1;
        let align_m = m % 32 == 0;
        let align_n = n % 32 == 0;
        let align_k = k % 16 == 0;

        let scale_only = alpha != 1.0 && beta == 0.0;
        let use_out_source = if scale_only { false } else { alpha != 1.0 || beta != 0.0 };
        let do_axpby = use_out_source && (alpha != 1.0 || beta != 1.0);

        #[cfg(test)]
        record_mlx_alpha_beta_flags(use_out_source, scale_only);

        let pipeline_key = MlxPipelineKey {
            dtype: result.dtype,
            transpose_left,
            transpose_right,
            has_batch,
            align_m,
            align_n,
            align_k,
            use_out_source,
            do_axpby,
            scale_only,
        };

        let pipeline = ctx.kernel_manager.get_mlx_pipeline(pipeline_key, &ctx.device)?;

        let m_i32 = i32::try_from(m).map_err(|_| MetalError::InvalidOperation("matmul rows exceed i32 range".to_string()))?;
        let n_i32 = i32::try_from(n).map_err(|_| MetalError::InvalidOperation("matmul columns exceed i32 range".to_string()))?;
        let k_i32 =
            i32::try_from(k).map_err(|_| MetalError::InvalidOperation("matmul interior dimension exceeds i32 range".to_string()))?;

        let left_elem_size = left_tensor.dtype.size_bytes();
        let right_elem_size = right_tensor.dtype.size_bytes();
        let result_elem_size = result.dtype.size_bytes();

        let (left_row_stride, left_matrix_stride) = matrix_strides_from_view(&left_view, left_elem_size)
            .ok_or_else(|| MetalError::InvalidOperation("MatMul MLX backend requires contiguous left operand".to_string()))?;
        let (right_row_stride, right_matrix_stride) = matrix_strides_from_view(&right_view, right_elem_size)
            .ok_or_else(|| MetalError::InvalidOperation("MatMul MLX backend requires contiguous right operand".to_string()))?;
        let (result_row_stride, result_matrix_stride) = matrix_strides_from_view(&result_view, result_elem_size)
            .ok_or_else(|| MetalError::InvalidOperation("MatMul MLX backend requires contiguous result tensor".to_string()))?;

        let lda = i32::try_from(left_row_stride).map_err(|_| MetalError::InvalidOperation("lda exceeds i32 range".to_string()))?;
        let ldb = i32::try_from(right_row_stride).map_err(|_| MetalError::InvalidOperation("ldb exceeds i32 range".to_string()))?;
        let ldd = i32::try_from(result_row_stride).map_err(|_| MetalError::InvalidOperation("ldd exceeds i32 range".to_string()))?;

        let bm = 32usize;
        let bn = 32usize;
        let bk = 16usize;
        let swizzle_log = 0i32;
        let tile = 1usize << swizzle_log;
        let tiles_n = n.div_ceil(bn) * tile;
        let tiles_m = m.div_ceil(bm).div_ceil(tile);

        let batch_stride_a = left_matrix_stride;
        let batch_stride_b = right_matrix_stride;

        let params = super::GemmParams {
            m: m_i32,
            n: n_i32,
            k: k_i32,
            lda,
            ldb,
            ldd,
            tiles_n: i32::try_from(tiles_n).map_err(|_| MetalError::InvalidOperation("tiles_n exceeds i32 range".to_string()))?,
            tiles_m: i32::try_from(tiles_m).map_err(|_| MetalError::InvalidOperation("tiles_m exceeds i32 range".to_string()))?,
            batch_stride_a,
            batch_stride_b,
            batch_stride_d: result_matrix_stride,
            swizzle_log,
            gemm_k_iterations_aligned: i32::try_from(k / bk.max(1))
                .map_err(|_| MetalError::InvalidOperation("gemm k iterations exceed i32 range".to_string()))?,
            batch_ndim: if has_batch { 1 } else { 0 },
            alpha_scale: if scale_only { alpha } else { 1.0 },
        };

        let ldc = i32::try_from(result_row_stride).map_err(|_| MetalError::InvalidOperation("ldc exceeds i32 range".to_string()))?;
        let fdc = 1i32;

        let addmm_params = GemmAddmmParams {
            ldc,
            fdc,
            batch_stride_c: result_matrix_stride,
            alpha,
            beta,
        };

        Ok(Self {
            pipeline,
            left_buf: left_tensor.buf.clone(),
            left_offset: left_tensor.offset,
            right_buf: right_tensor.buf.clone(),
            right_offset: right_tensor.offset,
            result_buf: result.buf.clone(),
            result_offset: result.offset,
            params,
            addmm_params,
            batch_size: left_view.batch,
            batch_stride_a,
            batch_stride_b,
            tiles_n,
            tiles_m,
            use_out_source,
            profiler: None,
        })
    }

    fn with_profiler(mut self, profiler: MatMulDispatchHandle) -> Self {
        self.profiler = Some(profiler);
        self
    }

    fn encode(&self, command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        if let Some(profiler) = &self.profiler {
            profiler.sample_start_compute(&encoder);
        }

        super::set_compute_pipeline_state(&encoder, &self.pipeline);
        super::set_buffer(&encoder, 0, &self.left_buf, self.left_offset);
        super::set_buffer(&encoder, 1, &self.right_buf, self.right_offset);
        if self.use_out_source {
            super::set_buffer(&encoder, 2, &self.result_buf, self.result_offset);
        }
        super::set_buffer(&encoder, 3, &self.result_buf, self.result_offset);
        super::set_bytes(&encoder, 4, &self.params);
        if self.use_out_source {
            super::set_bytes(&encoder, 5, &self.addmm_params);
        }

        let batch_size_i32 =
            i32::try_from(self.batch_size).map_err(|_| MetalError::InvalidOperation("matmul batch size exceeds i32 range".to_string()))?;
        super::set_bytes(&encoder, 6, &batch_size_i32);

        let batch_strides = [self.batch_stride_a, self.batch_stride_b];
        super::set_bytes(&encoder, 7, &batch_strides);

        let left_resource = buffer_as_resource(&self.left_buf);
        let right_resource = buffer_as_resource(&self.right_buf);
        let result_resource = buffer_as_resource(&self.result_buf);

        encoder.useResource_usage(left_resource, MTLResourceUsage::Read);
        encoder.useResource_usage(right_resource, MTLResourceUsage::Read);
        let result_usage = if self.use_out_source {
            MTLResourceUsage::Read | MTLResourceUsage::Write
        } else {
            MTLResourceUsage::Write
        };
        encoder.useResource_usage(result_resource, result_usage);

        let grid = objc2_metal::MTLSize {
            width: self.tiles_n as NSUInteger,
            height: self.tiles_m as NSUInteger,
            depth: self.batch_size as NSUInteger,
        };
        let threads = objc2_metal::MTLSize {
            width: 32 as NSUInteger,
            height: 2 as NSUInteger,
            depth: 2 as NSUInteger,
        };
        super::dispatch_threadgroups(&encoder, grid, threads);

        if let Some(profiler) = &self.profiler {
            profiler.sample_end_compute(&encoder);
        }

        encoder.endEncoding();

        Ok(())
    }
}

fn buffer_as_resource(buffer: &Retained<ProtocolObject<dyn MTLBuffer>>) -> &ProtocolObject<dyn MTLResource> {
    ProtocolObject::<dyn MTLResource>::from_ref::<ProtocolObject<dyn MTLBuffer>>(&**buffer)
}
