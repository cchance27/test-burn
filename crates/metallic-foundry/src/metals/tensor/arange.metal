#include <metal_stdlib>
using namespace metal;

// No params struct needed for arange - just uses thread_id

/// Arange kernel for runtime output storage type.
///
/// Fills output with sequential values: out[i] = i
kernel void arange_kernel(
    device OutputStorageT* output [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    metallic_store_output(output, gid, metallic_to_accum((float)gid));
}
