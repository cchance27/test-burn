use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState, MTLResource, MTLResourceUsage,
    MTLSize,
};
use objc2_metal_performance_shaders::{MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};
use std::time::Duration;

use super::{KernelFunction, KernelInvocable, kernel_manager::MlxPipelineKey};
use crate::metallic::instrumentation::MatMulDispatchHandle;
use crate::metallic::tensor::MpsMatrixBatchView;
use crate::metallic::{
    Context, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage,
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey},
    encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state},
    resource_cache::ResourceCache,
};

#[cfg(test)]
mod matmul_test;

// Include additional mps kernels
mod matmul_alpha_beta;
#[cfg(test)]
mod matmul_alpha_beta_test;
pub use matmul_alpha_beta::MatMulAlphaBetaOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatMulBackend {
    Total,
    Mps,
    Mlx,
    MlxTransposed,
}

#[derive(Clone, Copy, Debug)]
pub struct MatMulSample {
    pub backend: MatMulBackend,
    pub duration: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatMulBackendPreference {
    Auto,
    ForceMps,
    ForceMlx,
}

// Public, user-facing, zero-sized struct for the matmul operation with transpose options.
pub struct MatMulOp;

// Internal struct that holds data for the regular `Operation` trait.
struct MatMulMps {
    pub left_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub left_offset: usize,
    pub right_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub right_offset: usize,
    pub result_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub result_offset: usize,
    pub left_desc: Retained<MPSMatrixDescriptor>,
    pub right_desc: Retained<MPSMatrixDescriptor>,
    pub result_desc: Retained<MPSMatrixDescriptor>,
    pub gemm: Retained<MPSMatrixMultiplication>,
    pub batch_size: usize,
    profiler: Option<MatMulDispatchHandle>,
}

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
    batch_stride_a: usize,
    batch_stride_b: usize,
    batch_stride_d: usize,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
}

impl MatMulMps {
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
    ) -> Result<Self, MetalError> {
        let left_dtype = left_tensor.dtype;
        let right_dtype = right_tensor.dtype;
        let result_dtype = result.dtype;

        debug_assert_eq!(left_dtype, right_dtype);
        debug_assert_eq!(left_dtype, result_dtype);

        let gemm_key = MpsGemmKey {
            transpose_left,
            transpose_right,
            result_rows: eff_left_rows,
            result_columns: eff_right_cols,
            interior_columns: eff_left_cols,
            batch_size: result_view.batch,
            alpha: 1.0,
            beta: 0.0,
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
}

impl MatMulMps {
    fn with_profiler(mut self, profiler: MatMulDispatchHandle) -> Self {
        self.profiler = Some(profiler);
        self
    }
}

struct MatMulMlx {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    left_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    left_offset: usize,
    right_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    right_offset: usize,
    result_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    result_offset: usize,
    params: GemmParams,
    batch_size: usize,
    batch_stride_a: usize,
    batch_stride_b: usize,
    tiles_n: usize,
    tiles_m: usize,
    profiler: Option<MatMulDispatchHandle>,
}

impl MatMulMlx {
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
    ) -> Result<Self, MetalError> {
        let m = eff_left_rows;
        let n = eff_right_cols;
        let k = eff_left_cols;

        debug_assert_eq!(left_view.batch, right_view.batch);
        debug_assert_eq!(left_view.batch, result_view.batch);

        let align_m = m % 32 == 0;
        let align_n = n % 32 == 0;
        let align_k = k % 16 == 0;
        let has_batch = left_view.batch > 1;

        let pipeline_key = MlxPipelineKey {
            dtype: result.dtype,
            transpose_left,
            transpose_right,
            has_batch,
            align_m,
            align_n,
            align_k,
        };

        let pipeline = ctx.kernel_manager.get_mlx_pipeline(pipeline_key, &ctx.device)?;

        let m_i32 = i32::try_from(m).map_err(|_| MetalError::InvalidOperation("matmul rows exceed i32 range".to_string()))?;
        let n_i32 = i32::try_from(n).map_err(|_| MetalError::InvalidOperation("matmul columns exceed i32 range".to_string()))?;
        let k_i32 =
            i32::try_from(k).map_err(|_| MetalError::InvalidOperation("matmul interior dimension exceeds i32 range".to_string()))?;

        let left_elem_size = left_tensor.dtype.size_bytes();
        let right_elem_size = right_tensor.dtype.size_bytes();
        let result_elem_size = result.dtype.size_bytes();

        debug_assert_eq!(left_elem_size, right_elem_size);
        debug_assert_eq!(left_elem_size, result_elem_size);

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
        let batch_stride_d = result_matrix_stride;

        let params = GemmParams {
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
            batch_stride_d,
            swizzle_log,
            gemm_k_iterations_aligned: i32::try_from(k / bk.max(1))
                .map_err(|_| MetalError::InvalidOperation("gemm k iterations exceed i32 range".to_string()))?,
            batch_ndim: if has_batch { 1 } else { 0 },
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
            batch_size: left_view.batch,
            batch_stride_a,
            batch_stride_b,
            tiles_n,
            tiles_m,
            profiler: None,
        })
    }
}

impl MatMulMlx {
    fn with_profiler(mut self, profiler: MatMulDispatchHandle) -> Self {
        self.profiler = Some(profiler);
        self
    }
}

enum MatMulImplementation {
    Mps(MatMulMps),
    Mlx(MatMulMlx),
}

struct MatMul {
    implementation: MatMulImplementation,
}

impl Operation for MatMul {
    fn encode(&self, command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>, cache: &mut ResourceCache) -> Result<(), MetalError> {
        match &self.implementation {
            MatMulImplementation::Mps(op) => op.encode(command_buffer, cache),
            MatMulImplementation::Mlx(op) => op.encode(command_buffer, cache),
        }
    }
}

// Implement `KernelInvocable` for the public struct.
impl KernelInvocable for MatMulOp {
    // Input arguments for the call - two input tensors + transpose options
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, bool, bool); // (left, right, transpose_left, transpose_right)
    // The output type

    // For MPS operations, return None since they don't use KernelFunction
    fn function_id() -> Option<KernelFunction> {
        None // MPS operations don't use kernel functions
    }

    // This `new` method is called by `ctx.call()`.
    // It creates the output tensor and the internal `Operation` struct.
    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>, // MPS doesn't use this
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (left, right, transpose_left, transpose_right) = args;

        let (left_tensor, left_view) = left.ensure_mps_contiguous_batch(ctx)?;
        let (right_tensor, right_view) = right.ensure_mps_contiguous_batch(ctx)?;

        ctx.prepare_tensors_for_active_cmd(&[&left_tensor, &right_tensor])?;

        // Calculate effective dimensions based on transpose
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

        if left_view.batch != right_view.batch {
            return Err(MetalError::InvalidOperation(
                "Batched matmul requires operands to share the same batch size".to_string(),
            ));
        }

        // Create the output tensor (eff_left_rows x eff_right_cols)
        let out_dims = if left_view.batch > 1 {
            vec![left_view.batch, eff_left_rows, eff_right_cols]
        } else {
            vec![eff_left_rows, eff_right_cols]
        };
        let out = Tensor::new(out_dims, TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        let result_view = out.as_mps_matrix_batch_view()?;
        let preference = ctx.matmul_backend_preference();
        let mut selected_backend = MatMulBackend::Mps;
        let mut implementation: Option<MatMulImplementation> = None;

        if preference != MatMulBackendPreference::ForceMps && supports_mlx(&left_tensor, &left_view, &right_tensor, &right_view) {
            match MatMulMlx::new(
                ctx,
                &left_tensor,
                &right_tensor,
                &out,
                left_view,
                right_view,
                result_view,
                eff_left_rows,
                eff_left_cols,
                eff_right_cols,
                transpose_left,
                transpose_right,
            ) {
                Ok(mlx_op) => {
                    selected_backend = if transpose_left || transpose_right {
                        MatMulBackend::MlxTransposed
                    } else {
                        MatMulBackend::Mlx
                    };
                    implementation = Some(MatMulImplementation::Mlx(mlx_op));
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
            let mps_op = MatMulMps::new(
                ctx,
                cache,
                &left_tensor,
                &right_tensor,
                &out,
                left_view,
                right_view,
                result_view,
                eff_left_rows,
                eff_left_cols,
                eff_right_cols,
                transpose_left,
                transpose_right,
            )?;
            MatMulImplementation::Mps(mps_op)
        };

        let command_buffer = {
            let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
            command_buffer.clone()
        };
        let profiler = ctx.register_matmul_dispatch(&command_buffer, selected_backend);

        let implementation = match implementation {
            MatMulImplementation::Mps(op) => MatMulImplementation::Mps(op.with_profiler(profiler)),
            MatMulImplementation::Mlx(op) => MatMulImplementation::Mlx(op.with_profiler(profiler)),
        };

        Ok((Box::new(MatMul { implementation }), out))
    }
}

// Implement `Operation` for the internal struct.
// This contains the low-level logic to encode the kernel onto the command buffer.
impl Operation for MatMulMps {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        if let Some(profiler) = &self.profiler {
            profiler.sample_start_blit(command_buffer);
        }

        // Wrap buffers into MPSMatrix views
        let left = mps_matrix_from_buffer(&self.left_buf, self.left_offset, &self.left_desc);
        let right = mps_matrix_from_buffer(&self.right_buf, self.right_offset, &self.right_desc);
        let result = mps_matrix_from_buffer(&self.result_buf, self.result_offset, &self.result_desc);

        // Encode the MPS matrix multiplication
        unsafe {
            self.gemm.setBatchStart(0 as NSUInteger);
            self.gemm.setBatchSize(self.batch_size as NSUInteger);
        }
        encode_mps_matrix_multiplication(&self.gemm, command_buffer, &left, &right, &result);

        if let Some(profiler) = &self.profiler {
            profiler.sample_end_blit(command_buffer);
        }

        Ok(())
    }
}

impl Operation for MatMulMlx {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        if let Some(profiler) = &self.profiler {
            profiler.sample_start_compute(&encoder);
        }

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.left_buf, self.left_offset);
        set_buffer(&encoder, 1, &self.right_buf, self.right_offset);
        set_buffer(&encoder, 3, &self.result_buf, self.result_offset);
        set_bytes(&encoder, 4, &self.params);

        let batch_size_i32 =
            i32::try_from(self.batch_size).map_err(|_| MetalError::InvalidOperation("matmul batch size exceeds i32 range".to_string()))?;
        set_bytes(&encoder, 6, &batch_size_i32);

        let batch_strides = [self.batch_stride_a, self.batch_stride_b];
        set_bytes(&encoder, 7, &batch_strides);

        let left_resource = buffer_as_resource(&self.left_buf);
        let right_resource = buffer_as_resource(&self.right_buf);
        let result_resource = buffer_as_resource(&self.result_buf);

        encoder.useResource_usage(left_resource, MTLResourceUsage::Read);
        encoder.useResource_usage(right_resource, MTLResourceUsage::Read);
        encoder.useResource_usage(result_resource, MTLResourceUsage::Write);

        let grid = MTLSize {
            width: self.tiles_n as NSUInteger,
            height: self.tiles_m as NSUInteger,
            depth: self.batch_size as NSUInteger,
        };
        let threads = MTLSize {
            width: 32 as NSUInteger,
            height: 2 as NSUInteger,
            depth: 2 as NSUInteger,
        };
        dispatch_threadgroups(&encoder, grid, threads);

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

/// Create an `MPSMatrix` view into an existing `MTLBuffer`.
///
/// # Arguments
///
/// * `buffer` - The retained Metal buffer containing f32 elements.
/// * `offset` - A byte offset into the buffer where the matrix data begins.
/// * `descriptor` - Describes the matrix layout (rows, columns, rowBytes).
pub fn mps_matrix_from_buffer(
    buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
    offset: usize,
    descriptor: &Retained<MPSMatrixDescriptor>,
) -> Retained<MPSMatrix> {
    let rows = unsafe { descriptor.rows() };
    let row_bytes = unsafe { descriptor.rowBytes() };
    let matrices = unsafe { descriptor.matrices() };
    let total_bytes = if matrices <= 1 {
        row_bytes * rows
    } else {
        let matrix_bytes = unsafe { descriptor.matrixBytes() };
        (matrices - 1) * matrix_bytes + rows * row_bytes
    };
    let size = total_bytes;
    debug_assert!(offset + size <= buffer.length(), "matrix dimensions exceed buffer length");
    unsafe { MPSMatrix::initWithBuffer_offset_descriptor(MPSMatrix::alloc(), buffer, offset, descriptor) }
}

/// Encodes a matrix multiplication operation to a command buffer.
///
/// # Arguments
///
/// * `op` - The `MPSMatrixMultiplication` operation to encode.
/// * `command_buffer` - The command buffer to encode the operation into.
/// * `left` - The left matrix operand.
/// * `right` - The right matrix operand.
/// * `result` - The result matrix.
pub fn encode_mps_matrix_multiplication(
    op: &Retained<MPSMatrixMultiplication>,
    command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    left: &Retained<MPSMatrix>,
    right: &Retained<MPSMatrix>,
    result: &Retained<MPSMatrix>,
) {
    unsafe { op.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(command_buffer, left, right, result) }
}

/// Returns true when the MLX kernel can service the requested matmul without
/// additional data movement.
fn supports_mlx<T: TensorElement>(
    left: &Tensor<T>,
    left_view: &MpsMatrixBatchView,
    right: &Tensor<T>,
    right_view: &MpsMatrixBatchView,
) -> bool {
    if left.dtype != right.dtype {
        return false;
    }

    matches!(
        left.dtype,
        crate::metallic::tensor::Dtype::F16 | crate::metallic::tensor::Dtype::F32
    ) && mlx_operand_is_contiguous(left, left_view)
        && mlx_operand_is_contiguous(right, right_view)
}

fn mlx_operand_is_contiguous<T: TensorElement>(tensor: &Tensor<T>, view: &MpsMatrixBatchView) -> bool {
    if tensor.dims().len() < 2 {
        return false;
    }

    let elem_size = tensor.dtype.size_bytes();
    let Some((row_stride, matrix_stride)) = matrix_strides_from_view(view, elem_size) else {
        return false;
    };

    if row_stride != view.columns {
        return false;
    }

    let Some(expected_matrix_stride) = view.rows.checked_mul(row_stride) else {
        return false;
    };
    if matrix_stride != expected_matrix_stride {
        return false;
    }

    matches!(tensor.strides.last(), Some(1))
}

fn matrix_strides_from_view(view: &MpsMatrixBatchView, elem_size: usize) -> Option<(usize, usize)> {
    if elem_size == 0 {
        return None;
    }

    if view.row_bytes % elem_size != 0 {
        return None;
    }

    let row_stride = view.row_bytes / elem_size;

    if view.matrix_bytes == 0 {
        let matrix_stride = view.rows.checked_mul(row_stride)?;
        Some((row_stride, matrix_stride))
    } else {
        if view.matrix_bytes % elem_size != 0 {
            return None;
        }
        Some((row_stride, view.matrix_bytes / elem_size))
    }
}
