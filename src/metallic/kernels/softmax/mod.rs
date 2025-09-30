use super::*;
use crate::metallic::cache_keys::{MpsMatrixDescriptorKey, MpsSoftMaxKey};
use crate::metallic::kernels::matmul::mps_matrix_from_buffer;
use crate::metallic::resource_cache::ResourceCache;
use crate::metallic::tensor::MpsMatrixBatchView;
use crate::metallic::{Dtype, TensorElement};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::MTLCommandBuffer;
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixSoftMax};
use std::env;
use std::mem::size_of;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

pub const METALLIC_SOFTMAX_BACKEND_ENV: &str = "METALLIC_SOFTMAX_BACKEND";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftmaxBackendPreference {
    Auto,
    KernelOnly,
    MpsOnly,
}

impl SoftmaxBackendPreference {
    fn forces_kernel(self) -> bool {
        matches!(self, Self::KernelOnly)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftmaxBackend {
    Kernel,
    Mps,
}

#[derive(Clone, Copy, Debug)]
pub struct SoftmaxSample {
    pub backend: SoftmaxBackend,
    pub duration: Duration,
}

static BACKEND_PREFERENCE: OnceLock<SoftmaxBackendPreference> = OnceLock::new();

pub fn softmax_backend_preference() -> SoftmaxBackendPreference {
    *BACKEND_PREFERENCE.get_or_init(|| {
        let raw = env::var(METALLIC_SOFTMAX_BACKEND_ENV)
            .unwrap_or_else(|_| "auto".to_string())
            .to_lowercase();
        match raw.trim() {
            "kernel" | "compute" | "pipeline" => SoftmaxBackendPreference::KernelOnly,
            "mps" | "metal" => SoftmaxBackendPreference::MpsOnly,
            "auto" | "" | "default" => SoftmaxBackendPreference::Auto,
            other => {
                eprintln!("Unknown {METALLIC_SOFTMAX_BACKEND_ENV} value '{other}', falling back to auto");
                SoftmaxBackendPreference::Auto
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub fn apply_softmax<T: TensorElement>(
    ctx: &mut Context<T>,
    mut cache: Option<&mut ResourceCache>,
    attn: &Tensor<T>,
    batch: usize,
    rows: usize,
    columns: usize,
    causal: bool,
    query_offset: u32,
    allow_mps: bool,
) -> Result<Tensor<T>, MetalError> {
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

    let rows_total = batch * rows;
    let dtype = attn.dtype;

    let preference = softmax_backend_preference();
    let start = Instant::now();

    let supports_mps_dtype = matches!(dtype, Dtype::F32 | Dtype::F16);
    let can_use_mps = allow_mps && supports_mps_dtype && !causal && query_offset == 0 && !preference.forces_kernel();
    if can_use_mps && let Some(cache_slot) = cache.as_mut() {
        let cache_ref: &mut ResourceCache = cache_slot;
        try_apply_mps_softmax(ctx, cache_ref, attn, &view, batch, rows, columns, dtype)?;
        ctx.record_softmax_backend_sample(SoftmaxBackend::Mps, start.elapsed());
        return Ok(attn.clone());
    }

    let result = match cache {
        Some(cache_ref) => ctx.call_with_cache::<SoftmaxOp>(
            (attn, rows_total as u32, rows as u32, columns as u32, causal as u32, query_offset),
            cache_ref,
        )?,
        None => ctx.call::<SoftmaxOp>((attn, rows_total as u32, rows as u32, columns as u32, causal as u32, query_offset))?,
    };
    ctx.record_softmax_backend_sample(SoftmaxBackend::Kernel, start.elapsed());
    Ok(result)
}

fn try_apply_mps_softmax<T: TensorElement>(
    ctx: &mut Context<T>,
    cache: &mut ResourceCache,
    attn: &Tensor<T>,
    view: &MpsMatrixBatchView,
    batch: usize,
    rows: usize,
    columns: usize,
    dtype: Dtype,
) -> Result<(), MetalError> {
    ctx.prepare_tensors_for_active_cmd(&[attn])?;

    let descriptor_key = MpsMatrixDescriptorKey {
        rows,
        columns,
        row_bytes: view.row_bytes,
        matrices: view.batch,
        matrix_bytes: view.matrix_bytes,
        dtype,
    };
    let descriptor = cache.get_or_create_descriptor(descriptor_key, &ctx.device)?;
    let softmax_key = MpsSoftMaxKey { rows, columns, dtype };
    let softmax = cache.get_or_create_softmax(softmax_key, &ctx.device)?;

    let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
    let op = SoftmaxMpsOperation {
        attn: attn.clone(),
        descriptor,
        softmax,
        batch,
    };
    command_buffer.record(&op, cache)?;
    ctx.mark_tensor_pending(attn);
    Ok(())
}

struct SoftmaxMpsOperation<T: TensorElement> {
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

#[cfg(test)]
mod softmax_test;

/// Public, user-facing, zero-sized struct for the Softmax operation.
pub struct SoftmaxOp;

/// Internal struct that holds data for the Operation trait.
struct SoftmaxOperation<T: TensorElement> {
    attn: Tensor<T>,
    rows_total: u32,
    seq_q: u32,
    seq_k: u32,
    causal: u32,
    query_offset: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for SoftmaxOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, u32, u32, u32, u32, u32); // (attn, rows_total, seq_q, seq_k, causal, query_offset)

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::FusedSoftmax)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (attn, rows_total, seq_q, seq_k, causal, query_offset) = args;

        // Validate dimensions
        if attn.dims().len() < 2 {
            return Err(MetalError::InvalidShape(format!(
                "Softmax input must be at least 2D, got {:?}",
                attn.dims()
            )));
        }

        let view = attn.as_mps_matrix_batch_view()?;
        if view.rows != seq_q as usize || view.columns != seq_k as usize {
            return Err(MetalError::InvalidShape(format!(
                "Attention matrix dimensions {:?} don't match seq_q={} x seq_k={}",
                attn.dims(),
                seq_q,
                seq_k
            )));
        }

        if rows_total as usize != view.batch * view.rows {
            return Err(MetalError::InvalidShape(format!(
                "Softmax rows_total {} does not match view layout {:?}",
                rows_total,
                attn.dims()
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&[attn])?;

        let op = SoftmaxOperation {
            attn: attn.clone(),
            rows_total,
            seq_q,
            seq_k,
            causal,
            query_offset,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), attn.clone())) // Return a shallow clone since operation is in-place
    }
}

impl<T: TensorElement> Operation for SoftmaxOperation<T> {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        // Ensure at least 32 threads per threadgroup to satisfy kernel's reduction assumptions
        let native = self.pipeline.threadExecutionWidth();
        let width = if native < 32 { 32 } else { native };
        let threads_per_tg = MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: 1,
            height: self.rows_total as usize,
            depth: 1,
        };
        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.attn.buf, self.attn.offset);
        set_bytes(&encoder, 1, &self.seq_q);
        set_bytes(&encoder, 2, &self.seq_k);
        set_bytes(&encoder, 3, &self.causal);
        set_bytes(&encoder, 4, &self.query_offset);
        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
