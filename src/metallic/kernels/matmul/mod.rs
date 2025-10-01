use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputePipelineState};
use objc2_metal_performance_shaders::{MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};
use std::sync::Arc;
use std::time::Duration;

use super::KernelInvocable;
use crate::metallic::context::MatMulInstrumentation;
use crate::metallic::{
    CommandBuffer, Context, Dtype, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage,
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey},
    resource_cache::ResourceCache,
    tensor::MpsMatrixBatchView,
};

#[cfg(test)]
mod matmul_test;

mod matmul_alpha_beta;
#[cfg(test)]
mod matmul_alpha_beta_test;
mod mlx_gemm;
pub use matmul_alpha_beta::MatMulAlphaBetaOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatMulBackendKind {
    Mps,
    Mlx,
}

#[derive(Clone, Copy, Debug)]
pub struct MatMulSample {
    pub backend: MatMulBackendKind,
    pub duration: Duration,
}

pub enum MatMulBackend {
    Mps(MpsMatMulBackend),
    Mlx(mlx_gemm::MlxGemmBackend),
}

impl MatMulBackend {
    fn kind(&self) -> MatMulBackendKind {
        match self {
            MatMulBackend::Mps(_) => MatMulBackendKind::Mps,
            MatMulBackend::Mlx(_) => MatMulBackendKind::Mlx,
        }
    }
}

pub struct MpsMatMulBackend {
    left_desc: Retained<MPSMatrixDescriptor>,
    right_desc: Retained<MPSMatrixDescriptor>,
    result_desc: Retained<MPSMatrixDescriptor>,
    gemm: Retained<MPSMatrixMultiplication>,
    batch_size: usize,
}

impl MpsMatMulBackend {
    pub fn new<T: TensorElement>(
        ctx: &mut Context<T>,
        cache: &mut ResourceCache,
        transpose_left: bool,
        transpose_right: bool,
        left_view: &MpsMatrixBatchView,
        right_view: &MpsMatrixBatchView,
        result_view: &MpsMatrixBatchView,
        alpha_beta: (f32, f32),
        left_dtype: Dtype,
        right_dtype: Dtype,
        result_dtype: Dtype,
    ) -> Result<Self, MetalError> {
        let (alpha, beta) = alpha_beta;

        let gemm_key = MpsGemmKey {
            transpose_left,
            transpose_right,
            result_rows: result_view.rows,
            result_columns: result_view.columns,
            interior_columns: if transpose_left { left_view.rows } else { left_view.columns },
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
            rows: result_view.rows,
            columns: result_view.columns,
            row_bytes: result_view.row_bytes,
            matrices: result_view.batch,
            matrix_bytes: result_view.matrix_bytes,
            dtype: result_dtype,
        };
        let result_desc = cache.get_or_create_descriptor(result_desc_key, &ctx.device)?;

        Ok(Self {
            left_desc,
            right_desc,
            result_desc,
            gemm,
            batch_size: result_view.batch,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        left_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
        left_offset: usize,
        right_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
        right_offset: usize,
        result_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
        result_offset: usize,
    ) -> Result<(), MetalError> {
        let left = mps_matrix_from_buffer(left_buf, left_offset, &self.left_desc);
        let right = mps_matrix_from_buffer(right_buf, right_offset, &self.right_desc);
        let result = mps_matrix_from_buffer(result_buf, result_offset, &self.result_desc);

        unsafe {
            self.gemm.setBatchStart(0 as NSUInteger);
            self.gemm.setBatchSize(self.batch_size as NSUInteger);
        }
        encode_mps_matrix_multiplication(&self.gemm, command_buffer, &left, &right, &result);

        Ok(())
    }
}

// Public, user-facing, zero-sized struct for the matmul operation with transpose options.
pub struct MatMulOp;

// Internal struct that holds data for the regular `Operation` trait.
struct MatMul {
    left_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    left_offset: usize,
    right_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    right_offset: usize,
    result_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    result_offset: usize,
    backend: MatMulBackend,
    backend_kind: MatMulBackendKind,
    instrumentation: Arc<MatMulInstrumentation>,
}

impl KernelInvocable for MatMulOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, bool, bool);

    fn function_id() -> Option<super::KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (left, right, transpose_left, transpose_right) = args;

        let (left_tensor, left_view) = left.ensure_mps_contiguous_batch(ctx)?;
        let (right_tensor, right_view) = right.ensure_mps_contiguous_batch(ctx)?;

        ctx.prepare_tensors_for_active_cmd(&[&left_tensor, &right_tensor])?;

        if left_view.batch != right_view.batch {
            return Err(MetalError::InvalidOperation(
                "Batched matmul requires operands to share the same batch size".to_string(),
            ));
        }

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

        let out_dims = if left_view.batch > 1 {
            vec![left_view.batch, eff_left_rows, eff_right_cols]
        } else {
            vec![eff_left_rows, eff_right_cols]
        };
        let out = Tensor::new(out_dims, TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        let result_view = out.as_mps_matrix_batch_view()?;

        let left_dtype = left_tensor.dtype;
        let right_dtype = right_tensor.dtype;
        let result_dtype = out.dtype;

        debug_assert_eq!(left_dtype, right_dtype);
        debug_assert_eq!(left_dtype, result_dtype);

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
            None,
        )?;

        let instrumentation = ctx.matmul_instrumentation_handle();
        let backend_kind = backend.kind();

        let op = MatMul {
            left_buf: left_tensor.buf.clone(),
            left_offset: left_tensor.offset,
            right_buf: right_tensor.buf.clone(),
            right_offset: right_tensor.offset,
            result_buf: out.buf.clone(),
            result_offset: out.offset,
            backend,
            backend_kind,
            instrumentation,
        };

        Ok((Box::new(op), out))
    }
}

impl Operation for MatMul {
    fn register_completion(&self, command_buffer: &CommandBuffer) -> Result<(), MetalError> {
        let instrumentation = Arc::clone(&self.instrumentation);
        let backend = self.backend_kind;
        command_buffer.observe_completion(move |duration| {
            instrumentation.record(backend, duration);
        });
        Ok(())
    }

    fn encode(&self, command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>, cache: &mut ResourceCache) -> Result<(), MetalError> {
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
                cache,
            ),
        }
    }
}

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

pub fn encode_mps_matrix_multiplication(
    op: &Retained<MPSMatrixMultiplication>,
    command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    left: &Retained<MPSMatrix>,
    right: &Retained<MPSMatrix>,
    result: &Retained<MPSMatrix>,
) {
    unsafe { op.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(command_buffer, left, right, result) }
}
