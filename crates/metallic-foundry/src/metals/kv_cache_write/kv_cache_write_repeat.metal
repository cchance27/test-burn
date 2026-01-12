#include <metal_stdlib>
using namespace metal;

/// Writes input K/V [n_kv_heads, input_seq_len, head_dim] into an expanded cache
/// [n_heads, max_seq_len, head_dim] by repeating each KV head across `group_size` heads.
///
/// This matches the Context engine behavior where KV caches are stored already repeated
/// (so decode does not need a separate RepeatKvHeads dispatch).
kernel void kv_cache_write_repeat_kv_heads_kernel(
    const device half* input [[buffer(0)]],
    device half* cache [[buffer(1)]],
    const constant KvCacheWriteRepeatKvHeadsParamsResolved* params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params->total_elements) return;

    // Input is flat: [n_kv_heads, input_seq_len, head_dim]
    uint input_head_stride = params->input_seq_len * params->head_dim;
    uint kv_head_idx = gid / input_head_stride;
    uint remainder = gid % input_head_stride;
    uint input_seq_pos = remainder / params->head_dim;
    uint dim_idx = remainder % params->head_dim;

    // Expand KV head into group_size query heads.
    uint base_head = kv_head_idx * params->group_size;
    uint max_heads = params->n_heads;

    // Cache layout: [n_heads, max_seq_len, head_dim]
    uint cache_head_stride = params->max_seq_len * params->head_dim;
    uint cache_pos = params->position_offset + input_seq_pos;
    uint cache_row_base = cache_pos * params->head_dim + dim_idx;

    half v = input[gid];
    for (uint r = 0; r < params->group_size; ++r) {
        uint out_head = base_head + r;
        if (out_head >= max_heads) break;
        uint cache_idx = out_head * cache_head_stride + cache_row_base;
        cache[cache_idx] = v;
    }
}

