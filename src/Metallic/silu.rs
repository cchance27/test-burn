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

/// Ensure the SiLU compute pipeline is compiled and cached on the Context.
pub fn ensure_silu_pipeline(ctx: &mut Context) -> Result<(), MetalError> {
    if ctx.silu_pipeline.is_some() {
        return Ok(());
    }

    let source = r#"
    using namespace metal;

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    kernel void silu_kernel(device float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& total_elements [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
        if (gid >= total_elements) return;
        float x = input[gid];
        // Clamp for numerical stability
        if (x > 50.0f) {
            output[gid] = x;
            return;
        }
        if (x < -50.0f) {
            output[gid] = 0.0f;
            return;
        }
        float sig = 1.0f / (1.0f + exp(-x));
        output[gid] = x * sig;
    }
    "#;

    let source_ns = NSString::from_str(source);
    let library = ctx
        .device
        .newLibraryWithSource_options_error(&source_ns, None)
        .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;

    let fn_name = NSString::from_str("silu_kernel");
    let function = library
        .newFunctionWithName(&fn_name)
        .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
    let pipeline = ctx
        .device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|_err| MetalError::PipelineCreationFailed)?;

    ctx.silu_pipeline = Some(pipeline);
    Ok(())
}

/// An operation that runs SiLU activation over input tensors.
pub struct Silu {
    pub input: Tensor,
    pub output: Tensor,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Silu {
    pub fn new(
        input: Tensor,
        output: Tensor,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ) -> Result<Self, MetalError> {
        // Validate dimensions
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
            pipeline,
        })
    }
}

impl Operation for Silu {
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
        set_bytes(&encoder, 2, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
