use super::*;
use crate::metallic::cache_keys::{MpsMatrixDescriptorKey, MpsSoftMaxKey};
use crate::metallic::kernels::matmul::mps_matrix_from_buffer;
use crate::metallic::resource_cache::ResourceCache;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
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

pub fn apply_softmax(
    ctx: &mut Context,
    mut cache: Option<&mut ResourceCache>,
    attn: &Tensor,
    rows: usize,
    columns: usize,
    causal: bool,
    query_offset: u32,
    allow_mps: bool,
) -> Result<Tensor, MetalError> {
    let preference = softmax_backend_preference();
    let start = Instant::now();

    let can_use_mps = allow_mps && !causal && query_offset == 0 && !preference.forces_kernel();
    if can_use_mps {
        if let Some(cache_slot) = cache.as_mut() {
            let cache_ref: &mut ResourceCache = &mut **cache_slot;
            try_apply_mps_softmax(ctx, cache_ref, attn, rows, columns)?;
            ctx.record_softmax_backend_sample(SoftmaxBackend::Mps, start.elapsed());
            return Ok(attn.clone());
        }
    }

    let result = match cache {
        Some(cache_ref) => {
            ctx.call_with_cache::<SoftmaxOp>((attn.clone(), rows as u32, columns as u32, causal as u32, query_offset), cache_ref)?
        }
        None => ctx.call::<SoftmaxOp>((attn.clone(), rows as u32, columns as u32, causal as u32, query_offset))?,
    };
    ctx.record_softmax_backend_sample(SoftmaxBackend::Kernel, start.elapsed());
    Ok(result)
}

fn try_apply_mps_softmax(
    ctx: &mut Context,
    cache: &mut ResourceCache,
    attn: &Tensor,
    rows: usize,
    columns: usize,
) -> Result<(), MetalError> {
    let mut attn_for_prepare = attn.clone();
    ctx.prepare_tensors_for_active_cmd(&mut [&mut attn_for_prepare]);

    let descriptor_key = MpsMatrixDescriptorKey {
        rows,
        columns,
        row_bytes: columns * size_of::<f32>(),
    };
    let descriptor = cache.get_or_create_descriptor(descriptor_key, &ctx.device)?;
    let softmax_key = MpsSoftMaxKey { rows, columns };
    let softmax = cache.get_or_create_softmax(softmax_key, &ctx.device)?;

    let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
    let op = SoftmaxMpsOperation {
        attn: attn.clone(),
        descriptor,
        softmax,
    };
    command_buffer.record(&op, cache)?;
    ctx.mark_tensor_pending(attn);
    Ok(())
}

struct SoftmaxMpsOperation {
    attn: Tensor,
    descriptor: Retained<MPSMatrixDescriptor>,
    softmax: Retained<MPSMatrixSoftMax>,
}

impl Operation for SoftmaxMpsOperation {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let attn_matrix = mps_matrix_from_buffer(&self.attn.buf, self.attn.offset, &self.descriptor);
        unsafe {
            self.softmax
                .encodeToCommandBuffer_inputMatrix_resultMatrix(command_buffer, &attn_matrix, &attn_matrix);
        }
        Ok(())
    }
}

/// Public, user-facing, zero-sized struct for the Softmax operation.
pub struct SoftmaxOp;

/// Internal struct that holds data for the Operation trait.
struct SoftmaxOperation {
    attn: Tensor,
    seq_q: u32,
    seq_k: u32,
    causal: u32,
    query_offset: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for SoftmaxOp {
    type Args = (Tensor, u32, u32, u32, u32); // (attn, seq_q, seq_k, causal, query_offset)

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::FusedSoftmax)
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        let (mut attn, seq_q, seq_k, causal, query_offset) = args;

        // Validate dimensions
        if attn.dims().len() != 2 {
            return Err(MetalError::InvalidShape(format!(
                "Softmax input must be 2D [seq_q, seq_k], got {:?}",
                attn.dims()
            )));
        }

        let expected_dims = [seq_q as usize, seq_k as usize];
        if attn.dims() != expected_dims {
            return Err(MetalError::InvalidShape(format!(
                "Attention matrix dimensions {:?} don't match seq_q={} x seq_k={}",
                attn.dims(),
                seq_q,
                seq_k
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&mut [&mut attn]);

        let op = SoftmaxOperation {
            attn: attn.clone(),
            seq_q,
            seq_k,
            causal,
            query_offset,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), attn)) // Return the same tensor since operation is in-place
    }
}

impl Operation for SoftmaxOperation {
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
            height: self.seq_q as usize,
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

#[cfg(test)]
mod softmax_test {
    use super::*;

    #[test]
    fn test_softmax_logic() -> Result<(), MetalError> {
        let mut ctx = Context::new()?;
        // Create a simple test tensor [2, 3] with values that will produce recognizable softmax results
        let input_data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]; // Two rows to softmax independently
        let attn = Tensor::create_tensor_from_slice(&input_data, vec![2, 3], &ctx)?;

        // Apply softmax with no causal masking (causal=0)
        let result = ctx.call::<SoftmaxOp>((attn, 2, 3, 0, 0))?;

        // Check that each row sums to approximately 1 (property of softmax)
        let result_slice = result.as_slice();
        let row1_sum: f32 = result_slice[0..3].iter().sum();
        let row2_sum: f32 = result_slice[3..6].iter().sum();

        assert!((row1_sum - 1.0).abs() < 1e-5, "Row 1 sum should be 1.0, got {}", row1_sum);
        assert!((row2_sum - 1.0).abs() < 1e-5, "Row 2 sum should be 1.0, got {}", row2_sum);
        Ok(())
    }
}
