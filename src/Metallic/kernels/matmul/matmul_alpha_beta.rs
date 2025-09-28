use super::*;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputePipelineState};
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixMultiplication};

use super::{KernelFunction, KernelInvocable};
use crate::metallic::{
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey},
    resource_cache::ResourceCache,
    Context, MetalError, Operation, Tensor,
};

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
    type Args<'a> = (&'a Tensor, &'a Tensor, &'a Tensor, bool, bool, f32, f32);
    // The output type

    // For MPS operations, return None since they don't use KernelFunction
    fn function_id() -> Option<KernelFunction> {
        None // MPS operations don't use kernel functions
    }

    // This `new` method is called by `ctx.call()`.
    // It creates the output tensor and the internal `Operation` struct.
    fn new<'a>(
        ctx: &mut Context,
        args: Self::Args<'a>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>, // MPS doesn't use this
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        let (left, right, result, transpose_left, transpose_right, alpha, beta) = args;

        // Validate dimensions for matrix multiplication
        if left.dims().len() != 2 || right.dims().len() != 2 {
            return Err(MetalError::InvalidOperation("matmul requires 2D tensors".to_string()));
        }

        ctx.prepare_tensors_for_active_cmd(&[left, right, result]);

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
        Ok((Box::new(op), result.clone()))
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
