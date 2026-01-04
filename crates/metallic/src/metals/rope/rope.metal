#include <metal_stdlib>
using namespace metal;

// RopeParams struct is injected by Foundry via struct_defs()

/// RoPE (Rotary Position Embedding) kernel for half precision.
///
/// Each thread processes one element, applying rotation based on cos/sin caches.
/// The rotation pairs feature i with feature (i + half_dim):
///   out_i = x_i * cos - x_j * sin
///   out_j = x_j * cos + x_i * sin
kernel void rope_kernel_f16(
    const device half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    const device half* cos_buf [[buffer(2)]],
    const device half* sin_buf [[buffer(3)]],
    constant RopeParamsResolved* params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dim = params->dim;
    uint seq_len = params->seq_len;
    uint position_offset = params->position_offset;
    uint total_elements = params->total_elements;
    
    if (gid >= total_elements) return;
    
    // Compute indices
    uint feature_idx = gid % dim;
    uint row_idx = gid / dim;
    uint pos = (row_idx % seq_len) + position_offset;
    uint half_dim = dim / 2u;
    uint pair = (feature_idx < half_dim) ? feature_idx : (feature_idx - half_dim);
    
    // Load cos/sin values for this position and feature pair
    float cosv = (float)cos_buf[pos * half_dim + pair];
    float sinv = (float)sin_buf[pos * half_dim + pair];
    
    if (feature_idx < half_dim) {
        // First half: out_i = x_i * cos - x_j * sin
        float x_i = (float)input[gid];
        float x_j = (float)input[row_idx * dim + feature_idx + half_dim];
        float out_i = x_i * cosv - x_j * sinv;
        output[gid] = (half)out_i;
    } else {
        // Second half: out_j = x_j * cos + x_i * sin
        float x_j = (float)input[gid];
        float x_i = (float)input[row_idx * dim + (feature_idx - half_dim)];
        float out_j = x_j * cosv + x_i * sinv;
        output[gid] = (half)out_j;
    }
}
