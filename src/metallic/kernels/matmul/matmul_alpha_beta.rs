use crate::metallic::kernels::KernelFunction;

use super::{KernelInvocable, MatMulBackend, mlx_gemm};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputePipelineState};

use crate::metallic::{Context, MetalError, Operation, Tensor, TensorElement, resource_cache::ResourceCache};

// Public struct for matmul with alpha/beta scaling
pub struct MatMulAlphaBetaOp;

// Internal struct that holds data for the alpha/beta `Operation` trait.
struct MatMulAlphaBeta {
    left_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    left_offset: usize,
    right_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    right_offset: usize,
    result_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    result_offset: usize,
    backend: MatMulBackend,
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

        let backend = mlx_gemm::try_create_backend(
            ctx,
            cache,
            transpose_left,
            transpose_right,
            &left_view,
            &right_view,
            &result_view,
            left_dtype,
            right_dtype,
            result_dtype,
            Some((alpha, beta)),
        )?;

        let op = MatMulAlphaBeta {
            left_buf: left_tensor.buf.clone(),
            left_offset: left_tensor.offset,
            right_buf: right_tensor.buf.clone(),
            right_offset: right_tensor.offset,
            result_buf: result.buf.clone(),
            result_offset: result.offset,
            backend,
        };

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
        match &self.backend {
            MatMulBackend::Mps(backend) => backend.encode(
                command_buffer,
                &self.left_buf,
                self.left_offset,
                &self.right_buf,
                self.right_offset,
                &self.result_buf,
                self.result_offset,
            ),
            MatMulBackend::Mlx(backend) => backend.encode(
                command_buffer,
                &self.left_buf,
                self.left_offset,
                &self.right_buf,
                self.right_offset,
                &self.result_buf,
                self.result_offset,
                _cache,
            ),
        }
    }
}
