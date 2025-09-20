use super::{Context, MetalError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder as _, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice as _, MTLLibrary as _, MTLSize,
};

/// Ensure the fused mask+softmax compute pipeline is compiled and cached on the Context.
pub fn ensure_fused_softmax_pipeline(ctx: &mut Context) -> Result<(), MetalError> {
    if ctx.fused_softmax_pipeline.is_some() {
        return Ok(());
    }

    let source = r#"
    using namespace metal;
    
    // Optimized version with better memory coalescing and reduced barriers
    kernel void sdpa_fused_softmax(device float* attn [[buffer(0)]],
                                   constant uint &seq_q [[buffer(1)]],
                                   constant uint &seq_k [[buffer(2)]],
                                   constant uint &causal_flag [[buffer(3)]],
                                   uint3 tg_pos [[threadgroup_position_in_grid]],
                                   uint3 tid3 [[thread_position_in_threadgroup]],
                                   uint3 tptg [[threads_per_threadgroup]]) {
        // One threadgroup processes one row. Threads stride across columns.
        uint row = tg_pos.y;
        uint lane = tid3.x;
        uint stride = tptg.x;
        uint base = row * seq_k;
        uint i_q = row % seq_q;

        // Use a more efficient shared memory size based on common hardware
        // Apple GPUs typically have good performance with 256 or 512 threads per group
        threadgroup float shared_data[256];
        
        // Phase 1: row-wise max reduction with causal masking.
        float local_max = -INFINITY;
        for (uint c = lane; c < seq_k; c += stride) {
            float xv = attn[base + c];
            // Apply causal mask
            if (causal_flag == 1u && c > i_q) { 
                xv = -INFINITY; 
            }
            local_max = fmax(local_max, xv);
        }
        
        shared_data[lane] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Optimized reduction with fewer iterations
        for (uint offset = stride / 2u; offset > 15u; offset /= 2u) {
            if (lane < offset) {
                shared_data[lane] = fmax(shared_data[lane], shared_data[lane + offset]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Final reduction for small offsets
        if (lane < 16) {
            shared_data[lane] = fmax(shared_data[lane], shared_data[lane + 16]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (lane < 8) {
            shared_data[lane] = fmax(shared_data[lane], shared_data[lane + 8]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (lane < 4) {
            shared_data[lane] = fmax(shared_data[lane], shared_data[lane + 4]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (lane < 2) {
            shared_data[lane] = fmax(shared_data[lane], shared_data[lane + 2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (lane == 0) {
            shared_data[0] = fmax(shared_data[0], shared_data[1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float maxv = shared_data[0];

        // Phase 2: compute exp(x - max) and partial sums
        float local_sum = 0.0f;
        for (uint c = lane; c < seq_k; c += stride) {
            float xv = attn[base + c];
            // Apply causal mask
            if (causal_flag == 1u && c > i_q) { 
                xv = -INFINITY; 
            }
            float e = exp(xv - maxv);
            attn[base + c] = e;
            local_sum += e;
        }
        
        shared_data[lane] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Same optimized reduction for sum
        for (uint offset = stride / 2u; offset > 15u; offset /= 2u) {
            if (lane < offset) {
                shared_data[lane] += shared_data[lane + offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        if (lane < 16) {
            shared_data[lane] += shared_data[lane + 16];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (lane < 8) {
            shared_data[lane] += shared_data[lane + 8];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (lane < 4) {
            shared_data[lane] += shared_data[lane + 4];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (lane < 2) {
            shared_data[lane] += shared_data[lane + 2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (lane == 0) {
            shared_data[0] += shared_data[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float sumv = shared_data[0];

        // Phase 3: normalize in place
        for (uint c = lane; c < seq_k; c += stride) {
            attn[base + c] = attn[base + c] / sumv;
        }
    }
    "#;

    let source_ns = NSString::from_str(source);
    let library = ctx
        .device
        .newLibraryWithSource_options_error(&source_ns, None)
        .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;

    let fn_name = NSString::from_str("sdpa_fused_softmax");
    let function = library
        .newFunctionWithName(&fn_name)
        .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
    let pipeline = ctx
        .device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|_err| MetalError::PipelineCreationFailed)?;

    ctx.fused_softmax_pipeline = Some(pipeline);
    Ok(())
}

