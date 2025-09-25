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
        threadgroup uint shared_indices[256];
        
        // Phase 1: row-wise max reduction with causal masking and index tracking.
        float local_max = -INFINITY;
        uint max_index = 0;
        for (uint c = lane; c < seq_k; c += stride) {
            float xv = attn[base + c];
            // Apply causal mask
            if (causal_flag == 1u && c > i_q) { 
                xv = -INFINITY; 
            }
            if (xv > local_max) {
                local_max = xv;
                max_index = c; // Store relative index within the row
            }
        }
        
        shared_data[lane] = local_max;
        shared_indices[lane] = max_index;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Optimized reduction with fewer iterations, tracking the index of the maximum
        for (uint offset = stride / 2u; offset > 0u; offset /= 2u) {
            if (lane < offset) {
                if (shared_data[lane + offset] > shared_data[lane]) {
                    shared_data[lane] = shared_data[lane + offset];
                    shared_indices[lane] = shared_indices[lane + offset];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        float maxv = shared_data[0];
        uint row_max_index = shared_indices[0]; // This is the relative index within the row

        // Phase 2: compute exp(x - max) and partial sums
        float local_sum = 0.0f;
        for (uint c = lane; c < seq_k; c += stride) {
            float xv = attn[base + c];
            // Apply causal mask
            if (causal_flag == 1u && c > i_q) { 
                xv = -INFINITY; 
            }
            // Compute exp(x - max) with proper handling for extreme values
            float e = 0.0f;
            if (isinf(maxv) && maxv > 0) { // maxv is +inf
                if (isinf(xv) && xv > 0) {
                    e = 1.0f;
                } else {
                    e = 0.0f;
                }
            } else if (xv != -INFINITY) {
                // For very large negative differences, exp might underflow to 0
                // This is actually the correct behavior
                float diff = xv - maxv;
                // Clamp the difference to prevent extreme values that could cause overflow/underflow
                if (diff < -80.0f) {  // Prevent underflow
                    e = 0.0f;
                } else if (diff > 80.0f) {  // Prevent overflow
                    e = exp(80.0f);  // Though this case should not happen due to max subtraction
                } else {
                    e = exp(diff);
                }
            }
            attn[base + c] = e;
            local_sum += e;
        }
        
        shared_data[lane] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Same optimized reduction for sum
        for (uint offset = stride / 2u; offset > 0u; offset /= 2u) {
            if (lane < offset) {
                shared_data[lane] += shared_data[lane + offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        float sumv = shared_data[0];

        // Phase 3: normalize in place
        for (uint c = lane; c < seq_k; c += stride) {
            // Handle case where sum is zero or invalid
            if (isnan(sumv)) {
                attn[base + c] = sumv; // Propagate NaN
            } else if (sumv > 0.0f && sumv != INFINITY) {
                attn[base + c] = attn[base + c] / sumv;
            } else {
                // If sum is zero or invalid, handle appropriately
                if (causal_flag == 1u && c > i_q) {
                    attn[base + c] = 0.0f;
                } else {
                    // When all exponentials underflow to zero, give probability 1.0 to the maximum element
                    // and 0.0 to all others
                    if (c == row_max_index) {
                        attn[base + c] = 1.0f;
                    } else {
                        attn[base + c] = 0.0f;
                    }
                }
            }
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

/// An operation that runs the fused softmax kernel over an attention matrix.
pub struct SoftmaxOperation {
    pub attn_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub attn_offset: usize,
    pub seq_q: u32,
    pub seq_k: u32,
    pub causal: u32,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Operation for SoftmaxOperation {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        // Create a compute encoder and issue the kernel
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
        set_buffer(&encoder, 0, &self.attn_buf, self.attn_offset);
        set_bytes(&encoder, 1, &self.seq_q);
        set_bytes(&encoder, 2, &self.seq_k);
        set_bytes(&encoder, 3, &self.causal);
        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
