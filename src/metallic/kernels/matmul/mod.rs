use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputePipelineState};
use objc2_metal_performance_shaders::{MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};

use super::{KernelFunction, KernelInvocable};
use crate::metallic::{
    Context, Dtype, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage,
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey},
    resource_cache::ResourceCache,
};

#[cfg(test)]
mod matmul_test;

// Include additional mps kernels
mod matmul_alpha_beta;
#[cfg(test)]
mod matmul_alpha_beta_test;
pub use matmul_alpha_beta::MatMulAlphaBetaOp;

// Public, user-facing, zero-sized struct for the matmul operation with transpose options.
pub struct MatMulOp;

// Internal struct that holds data for the regular `Operation` trait.
struct MatMul {
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
}

struct NoopOperation;

impl Operation for NoopOperation {
    fn encode(
        &self,
        _command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        Ok(())
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
        let mut out = Tensor::new(out_dims, TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        let result_view = out.as_mps_matrix_batch_view()?;
        let left_dtype = left_tensor.dtype;
        let right_dtype = right_tensor.dtype;
        let result_dtype = out.dtype;

        debug_assert_eq!(left_dtype, right_dtype);
        debug_assert_eq!(left_dtype, result_dtype);

        let requires_cpu = requires_cpu_matmul(&[left_dtype, right_dtype, result_dtype]);

        if requires_cpu {
            cpu_matmul_fallback(
                &left_tensor,
                &left_view,
                transpose_left,
                &right_tensor,
                &right_view,
                transpose_right,
                &mut out,
                &result_view,
                eff_left_rows,
                eff_left_cols,
                eff_right_cols,
                1.0,
                0.0,
                None,
            )?;

            out.flush_host_writes()?;

            return Ok((Box::new(NoopOperation), out));
        }

        // Get or create MPSMatrixMultiplication operation from cache
        let gemm_key = MpsGemmKey {
            transpose_left,
            transpose_right,
            result_rows: eff_left_rows,
            result_columns: eff_right_cols,
            interior_columns: eff_left_cols, // This is the "k" dimension after applying transpose
            batch_size: result_view.batch,
            alpha: 1.0,
            beta: 0.0,
        };

        let cache = cache.ok_or_else(|| MetalError::InvalidOperation("Resource cache required for matmul".to_string()))?;
        let gemm = cache.get_or_create_gemm(gemm_key, &ctx.device)?;

        // Create MPS matrix descriptors based on original dimensions (not transposed ones)
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

        // Create the internal operation struct.
        let op = MatMul {
            left_buf: left_tensor.buf.clone(),
            left_offset: left_tensor.offset,
            right_buf: right_tensor.buf.clone(),
            right_offset: right_tensor.offset,
            result_buf: out.buf.clone(),
            result_offset: out.offset,
            left_desc,
            right_desc,
            result_desc,
            gemm,
            batch_size: result_view.batch,
        };

        // Return the boxed operation and the output tensor.
        Ok((Box::new(op), out))
    }
}

// Implement `Operation` for the internal struct.
// This contains the low-level logic to encode the kernel onto the command buffer.
impl Operation for MatMul {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
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
        Ok(())
    }
}

fn cpu_matmul_fallback<T: TensorElement>(
    left: &Tensor<T>,
    left_view: &MpsMatrixBatchView,
    transpose_left: bool,
    right: &Tensor<T>,
    right_view: &MpsMatrixBatchView,
    transpose_right: bool,
    out: &mut Tensor<T>,
    result_view: &MpsMatrixBatchView,
    eff_left_rows: usize,
    eff_left_cols: usize,
    eff_right_cols: usize,
    alpha: f32,
    beta: f32,
    beta_input: Option<&[T::Scalar]>,
) -> Result<(), MetalError> {
    let elem_size = std::mem::size_of::<T::Scalar>();

    debug_assert_eq!(left_view.row_bytes % elem_size, 0);
    debug_assert_eq!(right_view.row_bytes % elem_size, 0);
    debug_assert_eq!(result_view.row_bytes % elem_size, 0);
    debug_assert_eq!(left_view.matrix_bytes % elem_size, 0);
    debug_assert_eq!(right_view.matrix_bytes % elem_size, 0);
    debug_assert_eq!(result_view.matrix_bytes % elem_size, 0);

    let left_row_stride = left_view.row_bytes / elem_size;
    let left_batch_stride = left_view.matrix_bytes / elem_size;
    let right_row_stride = right_view.row_bytes / elem_size;
    let right_batch_stride = right_view.matrix_bytes / elem_size;
    let result_row_stride = result_view.row_bytes / elem_size;
    let result_batch_stride = result_view.matrix_bytes / elem_size;

    let left_data = left.as_slice();
    let right_data = right.as_slice();
    let out_data = out.as_mut_slice();

    let beta_slice = beta_input.unwrap_or(&[]);
    let apply_beta = beta_input.is_some() && beta != 0.0;

    let batch = result_view.batch.max(1);

    for batch_idx in 0..batch {
        let left_base = batch_idx * left_batch_stride;
        let right_base = batch_idx * right_batch_stride;
        let result_base = batch_idx * result_batch_stride;

        for row in 0..eff_left_rows {
            for col in 0..eff_right_cols {
                let mut acc = 0.0f32;

                for k in 0..eff_left_cols {
                    let (left_row_idx, left_col_idx) = if transpose_left { (k, row) } else { (row, k) };
                    let left_index = left_base + left_row_idx * left_row_stride + left_col_idx;

                    let (right_row_idx, right_col_idx) = if transpose_right { (col, k) } else { (k, col) };
                    let right_index = right_base + right_row_idx * right_row_stride + right_col_idx;

                    let left_val = T::to_f32(left_data[left_index]);
                    let right_val = T::to_f32(right_data[right_index]);
                    acc += left_val * right_val;
                }

                let result_index = result_base + row * result_row_stride + col;
                let scaled = acc * alpha;
                let combined = if apply_beta {
                    let existing = T::to_f32(beta_slice[result_index]);
                    scaled + beta * existing
                } else {
                    scaled
                };
                out_data[result_index] = T::from_f32(combined);
            }
        }
    }

    Ok(())
}

fn requires_cpu_matmul(dtypes: &[Dtype]) -> bool {
    dtypes.iter().any(|dtype| !matches!(dtype, Dtype::F32 | Dtype::F16))
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
