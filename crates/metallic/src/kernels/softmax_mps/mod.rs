use super::*;
use crate::cache_keys::MpsMatrixDescriptorKey;

use crate::kernels::matmul_mps::mps_matrix_from_buffer;
use crate::resource_cache::ResourceCache;
use crate::{Dtype, TensorElement};
use metallic_instrumentation::GpuProfiler;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::MTLCommandBuffer;
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixSoftMax};

// Public, user-facing, zero-sized struct for the MPS Softmax operation.
pub struct SoftmaxMpsOp;

// Internal operation that uses MPS for softmax computation
pub struct SoftmaxMpsOperation<T: TensorElement> {
    attn: Tensor<T>,
    descriptor: Retained<MPSMatrixDescriptor>,
    softmax: Retained<MPSMatrixSoftMax>,
    batch: usize,
}

impl<T: TensorElement> Operation for SoftmaxMpsOperation<T> {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        // MPS-backed op: ensure CPU-scope timing is used in latency mode for exact attribution
        GpuProfiler::mark_use_cpu_scope_for_cb(command_buffer);

        let attn_matrix = mps_matrix_from_buffer(&self.attn.buf, self.attn.offset, &self.descriptor);
        unsafe {
            self.softmax.setBatchStart(0 as NSUInteger);
            self.softmax.setBatchSize(self.batch as NSUInteger);
            self.softmax
                .encodeToCommandBuffer_inputMatrix_resultMatrix(command_buffer, &attn_matrix, &attn_matrix);
        }
        Ok(())
    }
}

impl KernelInvocable for SoftmaxMpsOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, usize, usize, usize, Dtype); // (attn, batch, rows, columns, dtype)

    fn function_id() -> Option<KernelFunction> {
        None // MPS operations don't have kernel functions
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (attn, batch, rows, columns, dtype) = args;

        let view = attn.as_mps_matrix_batch_view()?;

        if view.rows != rows || view.columns != columns {
            return Err(MetalError::InvalidShape(format!(
                "Attention matrix dimensions {:?} don't match expected {} x {}",
                attn.dims(),
                rows,
                columns
            )));
        }

        if view.batch != batch {
            return Err(MetalError::InvalidShape(format!(
                "Attention batch dimension {} does not match expected {}",
                view.batch, batch
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&[attn])?;

        let descriptor_key = MpsMatrixDescriptorKey {
            rows,
            columns,
            row_bytes: view.row_bytes,
            matrices: view.batch,
            matrix_bytes: view.matrix_bytes,
            dtype,
        };

        let device = &ctx.device;
        let cache = cache.ok_or(MetalError::ResourceCacheRequired)?;

        let descriptor = cache.get_or_create_descriptor(descriptor_key, device)?;
        let softmax = cache.get_or_create_softmax_full(rows, columns, dtype, false, device)?;

        let op = SoftmaxMpsOperation {
            attn: attn.clone(),
            descriptor,
            softmax,
            batch,
        };

        Ok((Box::new(op), attn.clone()))
    }
}

// Constructor for use directly from context and dispatcher
pub fn create_softmax_mps_operation_from_context<T: TensorElement>(
    attn: Tensor<T>,
    descriptor: Retained<MPSMatrixDescriptor>,
    softmax: Retained<MPSMatrixSoftMax>,
    batch: usize,
) -> SoftmaxMpsOperation<T> {
    SoftmaxMpsOperation {
        attn,
        descriptor,
        softmax,
        batch,
    }
}
