#include <metal_stdlib>
using namespace metal;

// ElemwiseAddParams struct is injected by Foundry via struct_defs()

/// Broadcast Element-wise Add kernel for half precision.
///
/// Adds a 1D bias tensor (b) to each row of tensor (a).
/// out[gid] = a[gid] + b[gid % b_len]
kernel void broadcast_add_kernel_f16(
    const device half* a [[buffer(0)]],
    const device half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ElemwiseAddParams* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_elements = params->total_elements;
    uint b_len = params->b_len;
    
    if (gid >= total_elements) return;
    
    uint b_idx = gid % b_len;
    float a_val = (float)a[gid];
    float b_val = (float)b[b_idx];
    out[gid] = (half)(a_val + b_val);
}
