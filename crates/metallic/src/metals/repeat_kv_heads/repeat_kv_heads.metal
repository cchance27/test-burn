#include <metal_stdlib>
using namespace metal;

// RepeatKvHeadsParams struct is injected by Foundry via struct_defs()

/// RepeatKvHeads kernel for half precision.
///
/// Repeats K/V heads for GQA (Grouped Query Attention).
/// Input: [batch * n_kv_heads, cache_stride, head_dim]
/// Output: [batch * n_heads, seq, head_dim]
kernel void repeat_kv_heads_kernel_f16(
    const device half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant RepeatKvHeadsParams* params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint group_size = params->group_size;
    uint batch = params->batch;
    uint n_kv_heads = params->n_kv_heads;
    uint n_heads = params->n_heads;
    uint seq = params->seq;
    uint head_dim = params->head_dim;
    uint cache_stride = params->cache_stride;
    uint total_elements = params->total_elements;
    
    if (gid >= total_elements) return;
    
    // Decode output index
    uint dim_idx = gid % head_dim;
    uint tmp = gid / head_dim;
    uint seq_idx = tmp % seq;
    uint batch_head_idx = tmp / seq;
    uint b = batch_head_idx / n_heads;
    uint h = batch_head_idx % n_heads;
    
    // Map to KV head
    uint kv_head = h / group_size;
    uint input_batch_head = b * n_kv_heads + kv_head;
    
    // Compute source index with cache_stride
    uint input_index = ((input_batch_head * cache_stride) + seq_idx) * head_dim + dim_idx;
    
    output[gid] = input[input_index];
}
