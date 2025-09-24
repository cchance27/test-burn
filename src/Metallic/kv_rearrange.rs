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

/// Compile and cache a compute kernel that rearranges KV rows for grouped-query attention.
///
/// It maps input K/V of shape [M, kv_dim] where M = batch*seq into output shape
/// [batch * n_heads, seq, head_dim] by grouping Q heads to KV heads:
///   out[ out_batch, s, hd ] = input[ (b*seq + s) * kv_dim + kv_h * kv_head_dim + hd ]
/// where:
///   out_batch = b * n_heads + h
///   kv_h = h / group_size
pub fn ensure_kv_rearrange_pipeline(ctx: &mut Context) -> Result<(), MetalError> {
    let source = r#"
    using namespace metal;

    kernel void kv_rearrange_kernel(device const float* input [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    constant uint &kv_dim [[buffer(2)]],
                                    constant uint &kv_head_dim [[buffer(3)]],
                                    constant uint &n_heads [[buffer(4)]],
                                    constant uint &n_kv_heads [[buffer(5)]],
                                    constant uint &head_dim [[buffer(6)]],
                                    constant uint &seq [[buffer(7)]],
                                    constant uint &total_elements [[buffer(8)]],
                                    uint gid [[thread_position_in_grid]]) {
        if (gid >= total_elements) return;

        // Output layout: [batch_heads, seq, head_dim]
        uint hd = gid % head_dim;
        uint tmp = gid / head_dim;
        uint s = tmp % seq;
        uint out_batch = tmp / seq;

        uint b = out_batch / n_heads;
        uint h = out_batch % n_heads;
        uint group_size = n_heads / n_kv_heads;
        uint kv_h = h / group_size;

        // Source index into input: (b*seq + s) * kv_dim + kv_h * kv_head_dim + hd
        uint src_row = b * seq + s;
        uint src_idx = src_row * kv_dim + kv_h * kv_head_dim + hd;

        output[gid] = input[src_idx];
    }
    "#;

    let source_ns = NSString::from_str(source);
    let library = ctx
        .device
        .newLibraryWithSource_options_error(&source_ns, None)
        .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;

    let fn_name = NSString::from_str("kv_rearrange_kernel");
    let function = library
        .newFunctionWithName(&fn_name)
        .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
    let pipeline = ctx
        .device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|_err| MetalError::PipelineCreationFailed)?;

    // store in kv_rearrange_pipeline slot
    if ctx.kv_rearrange_pipeline.is_none() {
        ctx.kv_rearrange_pipeline = Some(pipeline);
    }

    Ok(())
}

/// Operation to perform device-side KV -> per-head rearrange.
pub struct KvRearrange {
    pub input: Tensor,  // [M, kv_dim]
    pub output: Tensor, // [batch*n_heads, seq, head_dim]
    pub kv_dim: u32,
    pub kv_head_dim: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub seq: u32,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[allow(clippy::too_many_arguments)]
impl KvRearrange {
    pub fn new(
        input: Tensor,
        output: Tensor,
        kv_dim: u32,
        kv_head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq: u32,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ) -> Result<Self, MetalError> {
        // Basic validation
        if input.dims().len() != 2 {
            return Err(MetalError::InvalidShape(format!(
                "KvRearrange input must be 2D [M, kv_dim], got {:?}",
                input.dims()
            )));
        }
        if output.dims().len() != 3 {
            return Err(MetalError::InvalidShape(format!(
                "KvRearrange output must be 3D [batch_heads, seq, head_dim], got {:?}",
                output.dims()
            )));
        }
        Ok(Self {
            input,
            output,
            kv_dim,
            kv_head_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            seq,
            pipeline,
        })
    }
}

impl Operation for KvRearrange {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let total_elements = self.output.len() as u32;
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
        set_bytes(&encoder, 2, &self.kv_dim);
        set_bytes(&encoder, 3, &self.kv_head_dim);
        set_bytes(&encoder, 4, &self.n_heads);
        set_bytes(&encoder, 5, &self.n_kv_heads);
        set_bytes(&encoder, 6, &self.head_dim);
        set_bytes(&encoder, 7, &self.seq);
        set_bytes(&encoder, 8, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
