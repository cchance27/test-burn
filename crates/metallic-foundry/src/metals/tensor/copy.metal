#include <metal_stdlib>
using namespace metal;

// CopyParams struct is injected by Foundry

kernel void copy_u32(
    const device uint* src [[buffer(0)]],
    device uint* dst [[buffer(1)]],
    constant CopyParams& params [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < params.total_elements) {
        dst[id] = src[id];
    }
}
