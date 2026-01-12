#include <metal_stdlib>
using namespace metal;

// No params struct needed for arange - just uses thread_id

/// Arange kernel for half precision.
///
/// Fills output with sequential values: out[i] = i
kernel void arange_kernel_f16(
    device half* output [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = (half)gid;
}
