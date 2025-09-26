#include <metal_stdlib>
using namespace metal;

// Kernel for permuting tensor dimensions.
// This kernel takes arrays of strides and permutation indices as constant buffers.
// NOTE: The arrays (src_strides, dst_strides, dims, permute) must be passed as proper MTLBuffers,
// not as inline bytes, because set_bytes only works for small scalar values, not arrays.
kernel void permute_kernel(device const float* src [[buffer(0)]],
                           device float* dst [[buffer(1)]],
                           constant const uint* src_strides [[buffer(2)]],
                           constant const uint* dst_strides [[buffer(3)]],
                           constant const uint* dims [[buffer(4)]],
                           constant const uint* permute [[buffer(5)]],
                           constant const uint& rank [[buffer(6)]],
                           constant const uint& num_elements [[buffer(7)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= num_elements) return;

    uint src_idx = gid;
    uint temp_idx = src_idx;

    uint src_coords[8];
    for (uint i = 0; i < rank; ++i) {
        src_coords[i] = temp_idx / src_strides[i];
        temp_idx %= src_strides[i];
    }

    uint dst_coords[8];
    for (uint i = 0; i < rank; ++i) {
        dst_coords[i] = src_coords[permute[i]];
    }

    uint dst_idx = 0;
    for (uint i = 0; i < rank; ++i) {
        dst_idx += dst_coords[i] * dst_strides[i];
    }

    dst[dst_idx] = src[src_idx];
}