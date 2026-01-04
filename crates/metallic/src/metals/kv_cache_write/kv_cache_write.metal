#include <metal_stdlib>
using namespace metal;

/// Copies input K/V [n_kv_heads * head_dim] into cache at specified position.
/// Cache layout: [n_kv_heads, max_seq_len, head_dim]
kernel void kv_cache_write_kernel(
    const device half* input [[buffer(0)]],
    device half* cache [[buffer(1)]],
    const constant KvCacheWriteParamsResolved* params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params->total_elements) return;
    
    // Input is flat: [n_kv_heads, input_seq_len, head_dim]
    // Cache is [n_kv_heads, max_seq_len, head_dim]
    
    // Decompose gid into head, seq_pos, and dim indices
    uint input_head_stride = params->input_seq_len * params->head_dim;
    uint input_head_idx = gid / input_head_stride;
    uint remainder = gid % input_head_stride;
    uint input_seq_pos = remainder / params->head_dim;
    uint dim_idx = remainder % params->head_dim;
    
    // Cache index: head_idx * max_seq_len * head_dim + (position_offset + seq_pos) * head_dim + dim_idx
    uint cache_idx = input_head_idx * params->max_seq_len * params->head_dim 
                   + (params->position_offset + input_seq_pos) * params->head_dim 
                   + dim_idx;
    
    cache[cache_idx] = input[gid];
}
