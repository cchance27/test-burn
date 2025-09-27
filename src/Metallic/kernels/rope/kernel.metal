#include <metal_stdlib>
using namespace metal;

kernel void rope_kernel(device float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        device float* cos_buf [[buffer(2)]],
                        device float* sin_buf [[buffer(3)]],
                        constant uint& dim [[buffer(4)]],
                        constant uint& seq_len [[buffer(5)]],
                        constant uint& position_offset [[buffer(6)]],
                        constant uint& total_elements [[buffer(7)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) return;

    // Determine row (sequence position) and feature index within row
    uint feature_idx = gid % dim;
    uint row_idx = gid / dim;

    // Position in sequence (assume rows are arranged so that seq dimension varies fastest across rows)
    uint pos = (row_idx % seq_len) + position_offset;

    // Half-split RoPE: pair indices across the two halves of the last dimension
    uint half_dim = dim / 2u;
    uint pair = (feature_idx < half_dim) ? feature_idx : (feature_idx - half_dim);
    float cosv = cos_buf[pos * half_dim + pair];
    float sinv = sin_buf[pos * half_dim + pair];

    if (feature_idx < half_dim) {
        // first half element x_i pairs with x_j at index +half_dim
        float x_i = input[gid];
        float x_j = input[row_idx * dim + feature_idx + half_dim];
        float out_i = x_i * cosv - x_j * sinv;
        output[gid] = out_i;
    } else {
        // second half element x_j pairs with x_i at index -half_dim
        float x_j = input[gid];
        float x_i = input[row_idx * dim + (feature_idx - half_dim)];
        float out_j = x_j * cosv + x_i * sinv;
        output[gid] = out_j;
    }
}