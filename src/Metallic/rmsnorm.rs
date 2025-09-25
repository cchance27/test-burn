use super::{Context, MetalError, Operation, Tensor, resource_cache::ResourceCache};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder as _, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice as _, MTLLibrary as _, MTLSize,
};

use crate::metallic::encoder::{
    dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state,
};

/// Ensure the RMSNorm compute pipeline is compiled and cached on the Context.
pub fn ensure_rmsnorm_pipeline(ctx: &mut Context) -> Result<(), MetalError> {
    if ctx.rmsnorm_pipeline.is_some() {
        return Ok(());
    }

    let source = r#"
    using namespace metal;

    constant float EPS = 1e-6f;

    // RMSNorm: normalize by root-mean-square and apply per-feature scale (gamma)
    kernel void rmsnorm_kernel(device float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               device float* gamma [[buffer(2)]],
                               constant uint& feature_dim [[buffer(3)]],
                               constant uint& total_elements [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
        if (gid >= total_elements) return;

        uint feature_idx = gid % feature_dim;
        uint row_idx = gid / feature_dim;

        // Compute sum of squares for this row
        float sum_sq = 0.0f;
        for (uint f = 0; f < feature_dim; ++f) {
            float v = input[row_idx * feature_dim + f];
            sum_sq += v * v;
        }

        float rms = sqrt(sum_sq / float(feature_dim) + EPS);

        float x = input[gid];

        // Normalize by RMS and apply gamma scaling
        output[gid] = (x / rms) * gamma[feature_idx];
    }
    "#;

    let source_ns = NSString::from_str(source);
    let library = ctx
        .device
        .newLibraryWithSource_options_error(&source_ns, None)
        .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;

    let fn_name = NSString::from_str("rmsnorm_kernel");
    let function = library
        .newFunctionWithName(&fn_name)
        .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
    let pipeline = ctx
        .device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|_err| MetalError::PipelineCreationFailed)?;

    ctx.rmsnorm_pipeline = Some(pipeline);
    Ok(())
}

/// An operation that runs RMS normalization over input tensors.
pub struct RMSNorm {
    pub input: Tensor,
    pub output: Tensor,
    pub gamma: Tensor,
    pub feature_dim: u32,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl RMSNorm {
    pub fn new(
        input: Tensor,
        output: Tensor,
        gamma: Tensor,
        feature_dim: u32,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ) -> Result<Self, MetalError> {
        // Validate dimensions
        if input.dims().last() != Some(&(feature_dim as usize)) {
            return Err(MetalError::InvalidShape(format!(
                "Input feature dimension {} does not match specified feature_dim {}",
                input.dims().last().unwrap_or(&0),
                feature_dim
            )));
        }
        if gamma.dims() != [feature_dim as usize] {
            return Err(MetalError::InvalidShape(format!(
                "Gamma shape {:?} does not match feature_dim {}",
                gamma.dims(),
                feature_dim
            )));
        }
        if output.dims() != input.dims() {
            return Err(MetalError::InvalidShape(format!(
                "Output shape {:?} does not match input shape {:?}",
                output.dims(),
                input.dims()
            )));
        }

        Ok(Self {
            input,
            output,
            gamma,
            feature_dim,
            pipeline,
        })
    }
}

impl Operation for RMSNorm {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let total_elements = self.input.len() as u32;
        let threads_per_tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: total_elements.div_ceil(256) as usize,
            height: 1,
            depth: 1,
        };

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(&encoder, 1, &self.output.buf, self.output.offset);
        set_buffer(&encoder, 2, &self.gamma.buf, self.gamma.offset);
        set_bytes(&encoder, 3, &self.feature_dim);
        set_bytes(&encoder, 4, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
