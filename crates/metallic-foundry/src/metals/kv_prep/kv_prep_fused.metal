#include <metal_stdlib>
#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

using namespace metal;

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

/// Fused KV preparation for decode/prefill:
/// - Rearrange + RoPE for Q into q_rot (layout identical to Rope(KvRearrange(Q)) output)
/// - Rearrange + RoPE for K and write into compact cache [n_kv_heads, max_seq_len, head_dim]
/// - Rearrange for V and write into compact cache [n_kv_heads, max_seq_len, head_dim]
///
/// This is intended to replace the 7-kernel chain in the DSL per layer.
kernel void kv_prep_fused_kernel_f16(
    const device half* q_in [[buffer(0)]],
    const device half* k_in [[buffer(1)]],
    const device half* v_in [[buffer(2)]],
    device half* q_rot [[buffer(3)]],
    device half* k_cache [[buffer(4)]],
    device half* v_cache [[buffer(5)]],
    const device half* cos_buf [[buffer(6)]],
    const device half* sin_buf [[buffer(7)]],
    constant KvPrepFusedParamsResolved* params [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint head_dim = params->head_dim;
    const uint seq_len = params->seq_len;
    const uint total_q = params->total_elements;
    if (gid >= total_q) return;

    // Flattening matches kv_rearrange_kernel_f16 output for Q:
    // gid -> (h, s, hd) over [n_heads, seq_len, head_dim]
    const uint hd = gid % head_dim;
    const uint tmp = gid / head_dim;
    const uint s = tmp % seq_len;
    const uint h = tmp / seq_len;

    // === Q: Rearrange then RoPE ===
    const uint d_model = params->d_model;
    const uint q_src = s * d_model + (h * head_dim + hd);

    const uint half_dim = head_dim >> 1;
    const uint pos = params->position_offset + s;
    const uint no_rope_layer_step = params->no_rope_layer_step;
    const bool apply_rope = no_rope_layer_step == 0 || (((params->layer_idx + 1u) % no_rope_layer_step) != 0u);

    float q_out = (float)q_in[q_src];
    bool first_in_pair = true;
    uint mate_hd = hd;
    float cosv = 1.0f;
    float sinv = 0.0f;

    if (apply_rope) {
        const uint rope_mode = params->rope_mode;
        const bool rope_norm = rope_mode == 1;
        const uint pair = rope_norm ? (hd >> 1) : ((hd < half_dim) ? hd : (hd - half_dim));
        cosv = (float)cos_buf[pos * half_dim + pair];
        sinv = (float)sin_buf[pos * half_dim + pair];

        first_in_pair = rope_norm ? ((hd & 1u) == 0u) : (hd < half_dim);
        mate_hd = rope_norm ? (first_in_pair ? (hd + 1) : (hd - 1)) : (first_in_pair ? (hd + half_dim) : (hd - half_dim));
        if (first_in_pair) {
            const float x_i = (float)q_in[q_src];
            const float x_j = (float)q_in[s * d_model + (h * head_dim + mate_hd)];
            q_out = x_i * cosv - x_j * sinv;
        } else {
            const float x_j = (float)q_in[q_src];
            const float x_i = (float)q_in[s * d_model + (h * head_dim + mate_hd)];
            q_out = x_j * cosv + x_i * sinv;
        }
    }
    q_rot[gid] = (half)q_out;

    // === K/V: Rearrange (+RoPE for K) and write to compact KV cache ===
    // Only one thread per KV element performs cache writes to avoid redundant repeats:
    // choose the first head in each group.
    const uint group_size = params->group_size;
    if ((h % group_size) != 0) return;

    const uint kv_h = h / group_size;
    const uint kv_dim = params->kv_dim;
    const uint kv_src_base = s * kv_dim + (kv_h * head_dim + hd);

    // Cache is [n_kv_heads, max_seq_len, head_dim]
    const uint cache_head_stride = params->max_seq_len * head_dim;
    const uint cache_pos = params->position_offset + s;
    const uint cache_row_base = cache_pos * head_dim + hd;

    // K: optionally apply RoPE.
    float k_out = (float)k_in[kv_src_base];
    if (apply_rope) {
        if (first_in_pair) {
            const float x_i = (float)k_in[kv_src_base];
            const float x_j = (float)k_in[s * kv_dim + (kv_h * head_dim + mate_hd)];
            k_out = x_i * cosv - x_j * sinv;
        } else {
            const float x_j = (float)k_in[kv_src_base];
            const float x_i = (float)k_in[s * kv_dim + (kv_h * head_dim + mate_hd)];
            k_out = x_j * cosv + x_i * sinv;
        }
    }
    const half k_half = (half)k_out;

    // V: no RoPE
    const half v_half = v_in[kv_src_base];

    // Store once per KV head in compact cache layout [n_kv_heads, max_seq_len, head_dim].
    const uint cache_idx = kv_h * cache_head_stride + cache_row_base;
    k_cache[cache_idx] = k_half;
    v_cache[cache_idx] = v_half;
}
