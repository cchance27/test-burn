#include <metal_stdlib>
#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

ALWAYS_INLINE void rope_rotate_half(
    thread float& out_i,
    thread float& out_j,
    float x_i,
    float x_j,
    float cos_v,
    float sin_v
) {
    out_i = x_i * cos_v - x_j * sin_v;
    out_j = x_j * cos_v + x_i * sin_v;
}

using namespace metal;

// RopeParams struct is injected by Foundry via struct_defs()

/// RoPE (Rotary Position Embedding) kernel for runtime storage types.
///
/// Each thread processes one element, applying rotation based on cos/sin caches.
/// The rotation pairs feature i with feature (i + half_dim):
///   out_i = x_i * cos - x_j * sin
///   out_j = x_j * cos + x_i * sin
kernel void rope_kernel(
    const device InputStorageT* input [[buffer(0)]],
    device OutputStorageT* output [[buffer(1)]],
    const device TensorStorageT* cos_buf [[buffer(2)]],
    const device TensorStorageT* sin_buf [[buffer(3)]],
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
    
    // Load cos/sin values for this position and feature pair
    // Pair index is always feature_idx if < half_dim, or feature_idx - half_dim
    uint pair = (feature_idx < half_dim) ? feature_idx : (feature_idx - half_dim);
    
    float cosv = metallic_load_tensor(cos_buf, pos * half_dim + pair);
    float sinv = metallic_load_tensor(sin_buf, pos * half_dim + pair);
    
    if (feature_idx < half_dim) {
        // First half: out_i
        float x_i = metallic_load_input(input, gid);
        float x_j = metallic_load_input(input, row_idx * dim + feature_idx + half_dim);
        
        float out_i, out_j;
        rope_rotate_half(out_i, out_j, x_i, x_j, cosv, sinv);
        
        metallic_store_output(output, gid, metallic_to_accum(out_i));
    } else {
        // Second half: out_j
        float x_j = metallic_load_input(input, gid);
        float x_i = metallic_load_input(input, row_idx * dim + (feature_idx - half_dim));
        
        float out_i, out_j;
        rope_rotate_half(out_i, out_j, x_i, x_j, cosv, sinv);
        
        metallic_store_output(output, gid, metallic_to_accum(out_j));
    }
}
