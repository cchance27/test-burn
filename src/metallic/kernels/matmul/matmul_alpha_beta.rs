use super::*;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLComputePipelineState};
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixMultiplication};

use super::{KernelFunction, KernelInvocable, MatMulBackend};
use crate::metallic::{
    Context, MetalError, Operation, Tensor, TensorElement,
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey},
    instrumentation::{MatMulDispatchKind, MatMulDispatchTiming, MatmulDims},
    resource_cache::ResourceCache,
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
    pub batch_size: usize,
    pub dispatch_timing: Option<MatMulDispatchTiming>,
}

// Implement `KernelInvocable` for the public struct.
impl KernelInvocable for MatMulAlphaBetaOp {
    // Input arguments for the call - two input tensors + transpose options + alpha/beta
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, bool, f32, f32);
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
        let (left, right, result, transpose_left, transpose_right, alpha, beta) = args;

        let (left_tensor, left_view) = left.ensure_mps_contiguous_batch(ctx)?;
        let (right_tensor, right_view) = right.ensure_mps_contiguous_batch(ctx)?;
        let result_view = result.as_mps_matrix_batch_view()?;

        ctx.prepare_tensors_for_active_cmd(&[&left_tensor, &right_tensor, result])?;

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

        if left_view.batch != right_view.batch || left_view.batch != result_view.batch {
            return Err(MetalError::InvalidOperation(
                "Batched matmul requires consistent batch dimensions".to_string(),
            ));
        }

        // Validate result tensor dimensions match expected output dimensions
        if result_view.rows != eff_left_rows || result_view.columns != eff_right_cols {
            return Err(MetalError::InvalidOperation(format!(
                "Result tensor dimensions ({}, {}) do not match expected output dimensions ({}, {})",
                result_view.rows, result_view.columns, eff_left_rows, eff_right_cols
            )));
        }

        let left_dtype = left_tensor.dtype;
        let right_dtype = right_tensor.dtype;
        let result_dtype = result.dtype;

        let matmul_left_view = left_view;
        let matmul_right_view = right_view;
        let matmul_result_view = result_view;

        let matmul_left_buf = left_tensor.buf.clone();
        let matmul_left_offset = left_tensor.offset;
        let matmul_right_buf = right_tensor.buf.clone();
        let matmul_right_offset = right_tensor.offset;
        let matmul_result_buf = result.buf.clone();
        let matmul_result_offset = result.offset;

        let left_desc_dtype = left_dtype;
        let right_desc_dtype = right_dtype;
        let result_desc_dtype = result_dtype;

        // Get or create MPSMatrixMultiplication operation from cache
        let gemm_key = MpsGemmKey {
            transpose_left,
            transpose_right,
            result_rows: eff_left_rows,
            result_columns: eff_right_cols,
            interior_columns: eff_left_cols, // This is the "k" dimension after applying transpose
            batch_size: matmul_result_view.batch,
            alpha,
            beta,
        };

        let cache = cache.ok_or_else(|| MetalError::InvalidOperation("Resource cache required for matmul".to_string()))?;
        let gemm = cache.get_or_create_gemm(gemm_key, &ctx.device)?;

        // Create MPS matrix descriptors based on the buffers consumed by MPS
        let left_desc_key = MpsMatrixDescriptorKey {
            rows: matmul_left_view.rows,
            columns: matmul_left_view.columns,
            row_bytes: matmul_left_view.row_bytes,
            matrices: matmul_left_view.batch,
            matrix_bytes: matmul_left_view.matrix_bytes,
            dtype: left_desc_dtype,
        };
        let left_desc = cache.get_or_create_descriptor(left_desc_key, &ctx.device)?;

        let right_desc_key = MpsMatrixDescriptorKey {
            rows: matmul_right_view.rows,
            columns: matmul_right_view.columns,
            row_bytes: matmul_right_view.row_bytes,
            matrices: matmul_right_view.batch,
            matrix_bytes: matmul_right_view.matrix_bytes,
            dtype: right_desc_dtype,
        };
        let right_desc = cache.get_or_create_descriptor(right_desc_key, &ctx.device)?;

        let result_desc_key = MpsMatrixDescriptorKey {
            rows: eff_left_rows,
            columns: eff_right_cols,
            row_bytes: matmul_result_view.row_bytes,
            matrices: matmul_result_view.batch,
            matrix_bytes: matmul_result_view.matrix_bytes,
            dtype: result_desc_dtype,
        };
        let result_desc = cache.get_or_create_descriptor(result_desc_key, &ctx.device)?;

        let dims = MatmulDims {
            batch: matmul_result_view.batch,
            m: eff_left_rows,
            n: eff_right_cols,
            k: eff_left_cols,
        };

        let dispatch_timing = {
            let command_buffer = {
                let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
                command_buffer.clone()
            };
            ctx.register_matmul_dispatch(&command_buffer, MatMulBackend::Mps, Some(dims), MatMulDispatchKind::Blit)
                .timing()
                .cloned()
        };

        // Create the internal operation struct.
        let op = MatMulAlphaBeta {
            left_buf: matmul_left_buf,
            left_offset: matmul_left_offset,
            right_buf: matmul_right_buf,
            right_offset: matmul_right_offset,
            result_buf: matmul_result_buf,
            result_offset: matmul_result_offset,
            left_desc,
            right_desc,
            result_desc,
            gemm,
            batch_size: matmul_result_view.batch,
            dispatch_timing,
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

        if let Some(timing) = &self.dispatch_timing {
            if matches!(timing.kind(), MatMulDispatchKind::Blit) {
                if let Some(encoder) = command_buffer.blitCommandEncoder() {
                    unsafe {
                        let _: () = msg_send![
                            &*encoder,
                            sampleCountersInBuffer: timing.sample_buffer().as_ref()
                            atSampleIndex: timing.start_index()
                            withBarrier: true
                        ];
                        let _: () = msg_send![&*encoder, endEncoding];
                    }
                }
            }
        }

        // Encode the MPS matrix multiplication
        unsafe {
            self.gemm.setBatchStart(0 as NSUInteger);
            self.gemm.setBatchSize(self.batch_size as NSUInteger);
        }
        encode_mps_matrix_multiplication(&self.gemm, command_buffer, &left, &right, &result);

        if let Some(timing) = &self.dispatch_timing {
            if matches!(timing.kind(), MatMulDispatchKind::Blit) {
                if let Some(encoder) = command_buffer.blitCommandEncoder() {
                    unsafe {
                        let _: () = msg_send![
                            &*encoder,
                            sampleCountersInBuffer: timing.sample_buffer().as_ref()
                            atSampleIndex: timing.end_index()
                            withBarrier: false
                        ];
                        let _: () = msg_send![&*encoder, endEncoding];
                    }
                }
            }
        }

        Ok(())
    }
}
