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

/// Ensure the elementwise add compute pipeline is compiled and cached on the Context.
pub fn ensure_add_pipeline(ctx: &mut Context) -> Result<(), MetalError> {
    if ctx.add_pipeline.is_some() {
        return Ok(());
    }

    let source = r#"
    using namespace metal;

    // Elementwise add: out[i] = a[i] + b[i]
    kernel void add_kernel(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant uint& total_elements [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
        if (gid >= total_elements) return;
        out[gid] = a[gid] + b[gid];
    }

    // Broadcast add for bias: out[i] = a[i] + b[i % b_len], where b_len is the broadcast dimension (e.g., bias len)
    kernel void broadcast_add_kernel(device const float* a [[buffer(0)]],
                                     device const float* b [[buffer(1)]],
                                     device float* out [[buffer(2)]],
                                     constant uint& total_elements [[buffer(3)]],
                                     constant uint& b_len [[buffer(4)]],
                                     uint gid [[thread_position_in_grid]]) {
        if (gid >= total_elements) return;
        uint b_idx = gid % b_len;
        out[gid] = a[gid] + b[b_idx];
    }
    "#;

    let source_ns = NSString::from_str(source);
    let library = ctx
        .device
        .newLibraryWithSource_options_error(&source_ns, None)
        .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;

    let add_fn_name = NSString::from_str("add_kernel");
    let add_function = library
        .newFunctionWithName(&add_fn_name)
        .ok_or_else(|| MetalError::FunctionCreationFailed(add_fn_name.to_string()))?;
    let add_pipeline = ctx
        .device
        .newComputePipelineStateWithFunction_error(&add_function)
        .map_err(|_err| MetalError::PipelineCreationFailed)?;

    let broadcast_add_fn_name = NSString::from_str("broadcast_add_kernel");
    let broadcast_add_function = library
        .newFunctionWithName(&broadcast_add_fn_name)
        .ok_or_else(|| MetalError::FunctionCreationFailed(broadcast_add_fn_name.to_string()))?;
    let broadcast_add_pipeline = ctx
        .device
        .newComputePipelineStateWithFunction_error(&broadcast_add_function)
        .map_err(|_err| MetalError::PipelineCreationFailed)?;

    ctx.add_pipeline = Some(add_pipeline);
    ctx.broadcast_add_pipeline = Some(broadcast_add_pipeline);
    Ok(())
}

/// An operation that runs elementwise add over input tensors.
pub struct ElemwiseAdd {
    pub a: Tensor,
    pub b: Tensor,
    pub out: Tensor,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl ElemwiseAdd {
    pub fn new(
        a: Tensor,
        b: Tensor,
        out: Tensor,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ) -> Result<Self, MetalError> {
        if a.dims() != b.dims() || a.dims() != out.dims() {
            return Err(MetalError::InvalidShape(format!(
                "ElemwiseAdd: input shapes must match, got a={:?}, b={:?}, out={:?}",
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
pub struct BroadcastElemwiseAdd {
    pub a: Tensor,
    pub b: Tensor,
    pub out: Tensor,
    pub b_len: usize,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl BroadcastElemwiseAdd {
    pub fn new(
        a: Tensor,
        b: Tensor,
        out: Tensor,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ) -> Result<Self, MetalError> {
        let b_len = b.len();
        if b_len == 0 {
            return Err(MetalError::InvalidShape("Broadcast b cannot be empty".to_string()));
        }
        if a.dims() != out.dims() {
            return Err(MetalError::InvalidShape(format!(
                "BroadcastElemwiseAdd: a and out shapes must match, got a={:?}, out={:?}",
                a.dims(),
                out.dims()
            )));
        }
        // b should be 1D [b_len]
        if b.dims().len() != 1 {
            return Err(MetalError::InvalidShape(format!(
                "Broadcast b must be 1D, got {:?}",
                b.dims()
            )));
        }

        Ok(Self {
            a,
            b,
            out,
            b_len,
            pipeline,
        })
    }
}

impl Operation for BroadcastElemwiseAdd {
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
        set_bytes(&encoder, 4, &(self.b_len as u32));

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

impl Operation for ElemwiseAdd {
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
