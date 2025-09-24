use super::{Context, MetalError, Operation, Tensor, resource_cache::ResourceCache};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder as _, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice as _, MTLLibrary as _, MTLSize,
};

use crate::metallic::encoder::{
    dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state,
};

/// Ensure the elementwise multiply compute pipeline is compiled and cached on the Context.
pub fn ensure_mul_pipeline(ctx: &mut Context) -> Result<(), MetalError> {
    if ctx.mul_pipeline.is_some() {
        return Ok(());
    }

    let source = r#"
    using namespace metal;

    // Elementwise multiply: out[i] = a[i] * b[i]
    kernel void mul_kernel(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant uint& total_elements [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
        if (gid >= total_elements) return;
        out[gid] = a[gid] * b[gid];
    }
    "#;

    let source_ns = NSString::from_str(source);
    let library = ctx
        .device
        .newLibraryWithSource_options_error(&source_ns, None)
        .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;

    let fn_name = NSString::from_str("mul_kernel");
    let function = library
        .newFunctionWithName(&fn_name)
        .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
    let pipeline = ctx
        .device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|_err| MetalError::PipelineCreationFailed)?;

    ctx.mul_pipeline = Some(pipeline);
    Ok(())
}

/// An operation that runs elementwise multiply over input tensors.
pub struct ElemwiseMul {
    pub a: Tensor,
    pub b: Tensor,
    pub out: Tensor,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl ElemwiseMul {
    pub fn new(
        a: Tensor,
        b: Tensor,
        out: Tensor,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ) -> Result<Self, MetalError> {
        if a.dims() != b.dims() || a.dims() != out.dims() {
            return Err(MetalError::InvalidShape(format!(
                "ElemwiseMul: input shapes must match, got a={:?}, b={:?}, out={:?}",
                a.dims(),
                b.dims(),
                out.dims()
            )));
        }

        Ok(Self {
            a,
            b,
            out,
            pipeline,
        })
    }
}

impl Operation for ElemwiseMul {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let total_elements = self.a.len() as u32;
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
        set_buffer(&encoder, 0, &self.a.buf, self.a.offset);
        set_buffer(&encoder, 1, &self.b.buf, self.b.offset);
        set_buffer(&encoder, 2, &self.out.buf, self.out.offset);
        set_bytes(&encoder, 3, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
