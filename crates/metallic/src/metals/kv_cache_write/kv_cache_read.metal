#include <metal_stdlib>
using namespace metal;

/// Reads cache[0..seq_len] into a contiguous output tensor.
/// Cache layout: [n_kv_heads, max_seq_len, head_dim]
/// Output layout: [n_kv_heads, seq_len, head_dim]
kernel void kv_cache_read_kernel(
    const device half* cache [[buffer(0)]],
    device half* output [[buffer(1)]],
    const constant KvCacheReadParamsResolved* params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params->total_elements) return;
    
    // Output is contiguous: [n_kv_heads, seq_len, head_dim]
    // Decompose gid into head, pos, dim indices
    uint head_stride = params->seq_len * params->head_dim;
    uint head_idx = gid / head_stride;
    uint remainder = gid % head_stride;
    uint pos_idx = remainder / params->head_dim;
    uint dim_idx = remainder % params->head_dim;
    
    // Cache index: head_idx * max_seq_len * head_dim + pos_idx * head_dim + dim_idx
    uint cache_idx = head_idx * params->max_seq_len * params->head_dim 
                   + pos_idx * params->head_dim 
                   + dim_idx;
    
    output[gid] = cache[cache_idx];
}
