use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputePipelineState};
use objc2_metal_performance_shaders::{MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};

use super::{KernelFunction, KernelInvocable};
use crate::metallic::{
    Context, MetalError, Operation, Tensor,
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey},
    resource_cache::ResourceCache,
};

mod matmul_test;

// Include additional mps kernels
mod matmul_alpha_beta;
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
}

// Implement `KernelInvocable` for the public struct.
impl KernelInvocable for MatMulOp {
    // Input arguments for the call - two input tensors + transpose options
    type Args = (Tensor, Tensor, bool, bool); // (left, right, transpose_left, transpose_right)
    // The output type

    // For MPS operations, return None since they don't use KernelFunction
    fn function_id() -> Option<KernelFunction> {
        None // MPS operations don't use kernel functions
    }

    // This `new` method is called by `ctx.call()`.
    // It creates the output tensor and the internal `Operation` struct.
    fn new(
        ctx: &mut Context,
        args: Self::Args,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>, // MPS doesn't use this
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        let (mut left, mut right, transpose_left, transpose_right) = args;

        // Validate dimensions for matrix multiplication
        if left.dims().len() != 2 || right.dims().len() != 2 {
            return Err(MetalError::InvalidOperation("matmul requires 2D tensors".to_string()));
        }

        ctx.prepare_tensors_for_active_cmd(&mut [&mut left, &mut right]);

        let left_rows = left.dims()[0];
        let left_cols = left.dims()[1];
        let right_rows = right.dims()[0];
        let right_cols = right.dims()[1];

        // Calculate effective dimensions based on transpose
        let (eff_left_rows, eff_left_cols) = if transpose_left {
            (left_cols, left_rows)
        } else {
            (left_rows, left_cols)
        };
        let (eff_right_rows, eff_right_cols) = if transpose_right {
            (right_cols, right_rows)
        } else {
            (right_rows, right_cols)
        };

        if eff_left_cols != eff_right_rows {
            return Err(MetalError::InvalidOperation(format!(
                "Cannot multiply matrices (with transpose): {}x{} and {}x{}",
                eff_left_rows, eff_left_cols, eff_right_rows, eff_right_cols
            )));
        }

        // Create the output tensor (eff_left_rows x eff_right_cols)
        let out_dims = vec![eff_left_rows, eff_right_cols];
        let out = Tensor::create_tensor_pooled(out_dims, ctx)?;

        // Get or create MPSMatrixMultiplication operation from cache
        let gemm_key = MpsGemmKey {
            transpose_left,
            transpose_right,
            result_rows: eff_left_rows,
            result_columns: eff_right_cols,
            interior_columns: eff_left_cols, // This is the "k" dimension after applying transpose
            alpha: 1.0,
            beta: 0.0,
        };

        let cache = cache.ok_or_else(|| MetalError::InvalidOperation("Resource cache required for matmul".to_string()))?;
        let gemm = cache.get_or_create_gemm(gemm_key, &ctx.device)?;

        // Create MPS matrix descriptors based on original dimensions (not transposed ones)
        let left_desc_key = MpsMatrixDescriptorKey {
            rows: left_rows,
            columns: left_cols,
            row_bytes: left_cols * std::mem::size_of::<f32>(),
        };
        let left_desc = cache.get_or_create_descriptor(left_desc_key, &ctx.device)?;

        let right_desc_key = MpsMatrixDescriptorKey {
            rows: right_rows,
            columns: right_cols,
            row_bytes: right_cols * std::mem::size_of::<f32>(),
        };
        let right_desc = cache.get_or_create_descriptor(right_desc_key, &ctx.device)?;

        let result_desc_key = MpsMatrixDescriptorKey {
            rows: eff_left_rows,
            columns: eff_right_cols,
            row_bytes: eff_right_cols * std::mem::size_of::<f32>(),
        };
        let result_desc = cache.get_or_create_descriptor(result_desc_key, &ctx.device)?;

        // Create the internal operation struct.
        let op = MatMul {
            left_buf: left.buf.clone(),
            left_offset: left.offset,
            right_buf: right.buf.clone(),
            right_offset: right.offset,
            result_buf: out.buf.clone(),
            result_offset: out.offset,
            left_desc,
            right_desc,
            result_desc,
            gemm,
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
        encode_mps_matrix_multiplication(&self.gemm, command_buffer, &left, &right, &result);
        Ok(())
    }
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
    let size = unsafe { descriptor.rowBytes() * descriptor.rows() };
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
