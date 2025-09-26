use super::*;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputePipelineState};
use objc2_metal_performance_shaders::{MPSDataType, MPSMatrixDescriptor, MPSMatrixMultiplication};

use super::{KernelFunction, KernelInvocable};
use crate::metallic::{Context, MetalError, Operation, Tensor, cache_keys::MpsGemmKey, resource_cache::ResourceCache};

// Public struct for matmul with alpha/beta scaling
pub struct MatMulAlphaBetaOp;

// Internal struct that holds data for the alpha/beta `Operation` trait.
struct MatMulAlphaBeta {
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
impl KernelInvocable for MatMulAlphaBetaOp {
    // Input arguments for the call - two input tensors + transpose options + alpha/beta
    type Args = (Tensor, Tensor, Tensor, bool, bool, f32, f32); // (left, right, result, transpose_left, transpose_right, alpha, beta)
    // The output type
    type Output = Tensor;

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
    ) -> Result<(Box<dyn Operation>, Self::Output), MetalError> {
        let (left, right, result, transpose_left, transpose_right, alpha, beta) = args;

        // Validate dimensions for matrix multiplication
        if left.dims().len() != 2 || right.dims().len() != 2 {
            return Err(MetalError::InvalidOperation("matmul requires 2D tensors".to_string()));
        }

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

        // Validate result tensor dimensions match expected output dimensions
        if result.dims()[0] != eff_left_rows || result.dims()[1] != eff_right_cols {
            return Err(MetalError::InvalidOperation(format!(
                "Result tensor dimensions ({}, {}) do not match expected output dimensions ({}, {})",
                result.dims()[0],
                result.dims()[1],
                eff_left_rows,
                eff_right_cols
            )));
        }

        // Get or create MPSMatrixMultiplication operation from cache
        let gemm_key = MpsGemmKey {
            transpose_left,
            transpose_right,
            result_rows: eff_left_rows,
            result_columns: eff_right_cols,
            interior_columns: eff_left_cols, // This is the "k" dimension after applying transpose
            alpha,
            beta,
        };

        let gemm = cache
            .ok_or_else(|| MetalError::InvalidOperation("Resource cache required for matmul".to_string()))?
            .get_or_create_gemm(gemm_key, &ctx.device)?;

        // Create MPS matrix descriptors based on original dimensions (not transposed ones)
        let left_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                left_rows,
                left_cols,
                left_cols * std::mem::size_of::<f32>(),
                MPSDataType::Float32,
            )
        };

        let right_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                right_rows,
                right_cols,
                right_cols * std::mem::size_of::<f32>(),
                MPSDataType::Float32,
            )
        };

        let result_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                eff_left_rows,
                eff_right_cols,
                eff_right_cols * std::mem::size_of::<f32>(),
                MPSDataType::Float32,
            )
        };

        // Create the internal operation struct.
        let op = MatMulAlphaBeta {
            left_buf: left.buf.clone(),
            left_offset: left.offset,
            right_buf: right.buf.clone(),
            right_offset: right.offset,
            result_buf: result.buf.clone(),
            result_offset: result.offset,
            left_desc,
            right_desc,
            result_desc,
            gemm,
        };

        // Return the boxed operation and the result tensor (already provided)
        Ok((Box::new(op), result))
    }
}

// Implement `Operation` for the internal struct.
// This contains the low-level logic to encode the kernel onto the command buffer.
impl Operation for MatMulAlphaBeta {
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
