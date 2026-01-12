#include <metal_stdlib>
using namespace metal;

// OnesParams struct is injected by Foundry via struct_defs()

/// Ones kernel for half precision.
///
/// Fills output with 1.0. Each thread handles 4 elements for vectorization.
kernel void ones_kernel_f16(
    device half* output [[buffer(0)]],
    constant OnesParams* params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_elements = params->total_elements;
    uint base = gid * 4u;
    
    for (uint i = 0u; i < 4u; ++i) {
        uint idx = base + i;
        if (idx < total_elements) {
            output[idx] = (half)1.0f;
        }
    }
}
