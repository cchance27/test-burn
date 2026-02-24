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

ALWAYS_INLINE float kv_prep_load_input(const device InputStorageT* ptr, const ulong idx) {
#if METALLIC_FASTPATH_INPUT_HALF
    return (float)((const device half*)ptr)[idx];
#else
    return (float)metallic_load_input(ptr, idx);
#endif
}

ALWAYS_INLINE float kv_prep_load_tensor(const device TensorStorageT* ptr, const ulong idx) {
#if METALLIC_FASTPATH_TENSOR_HALF
    return (float)((const device half*)ptr)[idx];
#else
    return (float)metallic_load_tensor(ptr, idx);
#endif
}

ALWAYS_INLINE void kv_prep_store_output(device OutputStorageT* ptr, const ulong idx, const float value) {
#if METALLIC_FASTPATH_OUTPUT_HALF
    ((device half*)ptr)[idx] = (half)value;
#else
    metallic_store_output(ptr, idx, metallic_to_accum(value));
#endif
}

/// Fused KV preparation for decode/prefill:
/// - Rearrange + RoPE for Q into q_rot (layout identical to Rope(KvRearrange(Q)) output)
/// - Rearrange + RoPE for K and write into compact cache [n_kv_heads, max_seq_len, head_dim]
/// - Rearrange for V and write into compact cache [n_kv_heads, max_seq_len, head_dim]
///
/// This is intended to replace the 7-kernel chain in the DSL per layer.
kernel void kv_prep_fused_kernel(
    const device InputStorageT* q_in [[buffer(0)]],
    const device InputStorageT* k_in [[buffer(1)]],
    const device InputStorageT* v_in [[buffer(2)]],
    device OutputStorageT* q_rot [[buffer(3)]],
    device OutputStorageT* k_cache [[buffer(4)]],
    device OutputStorageT* v_cache [[buffer(5)]],
    const device TensorStorageT* cos_buf [[buffer(6)]],
    const device TensorStorageT* sin_buf [[buffer(7)]],
    constant KvPrepFusedParamsResolved* params [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
#if METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_TENSOR_HALF && METALLIC_FASTPATH_OUTPUT_HALF
    const device half* q_in_h = (const device half*)q_in;
    const device half* k_in_h = (const device half*)k_in;
    const device half* v_in_h = (const device half*)v_in;
    device half* q_rot_h = (device half*)q_rot;
    device half* k_cache_h = (device half*)k_cache;
    device half* v_cache_h = (device half*)v_cache;
    const device half* cos_buf_h = (const device half*)cos_buf;
    const device half* sin_buf_h = (const device half*)sin_buf;

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

    float q_out = (float)q_in_h[q_src];
    bool first_in_pair = true;
    uint mate_hd = hd;
    float cosv = 1.0f;
    float sinv = 0.0f;

    if (apply_rope) {
        const uint rope_mode = params->rope_mode;
        const bool rope_norm = rope_mode == 1;
        const uint pair = rope_norm ? (hd >> 1) : ((hd < half_dim) ? hd : (hd - half_dim));
        cosv = (float)cos_buf_h[pos * half_dim + pair];
        sinv = (float)sin_buf_h[pos * half_dim + pair];

        first_in_pair = rope_norm ? ((hd & 1u) == 0u) : (hd < half_dim);
        mate_hd = rope_norm ? (first_in_pair ? (hd + 1) : (hd - 1)) : (first_in_pair ? (hd + half_dim) : (hd - half_dim));
        if (first_in_pair) {
            const float x_i = (float)q_in_h[q_src];
            const float x_j = (float)q_in_h[s * d_model + (h * head_dim + mate_hd)];
            q_out = x_i * cosv - x_j * sinv;
        } else {
            const float x_j = (float)q_in_h[q_src];
            const float x_i = (float)q_in_h[s * d_model + (h * head_dim + mate_hd)];
            q_out = x_j * cosv + x_i * sinv;
        }
    }
    q_rot_h[gid] = (half)q_out;

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
    float k_out = (float)k_in_h[kv_src_base];
    if (apply_rope) {
        if (first_in_pair) {
            const float x_i = (float)k_in_h[kv_src_base];
            const float x_j = (float)k_in_h[s * kv_dim + (kv_h * head_dim + mate_hd)];
            k_out = x_i * cosv - x_j * sinv;
        } else {
            const float x_j = (float)k_in_h[kv_src_base];
            const float x_i = (float)k_in_h[s * kv_dim + (kv_h * head_dim + mate_hd)];
            k_out = x_j * cosv + x_i * sinv;
        }
    }
    const half k_half = (half)k_out;

    // V: no RoPE
    const half v_half = v_in_h[kv_src_base];

    // Store once per KV head in compact cache layout [n_kv_heads, max_seq_len, head_dim].
    const uint cache_idx = kv_h * cache_head_stride + cache_row_base;
    k_cache_h[cache_idx] = k_half;
    v_cache_h[cache_idx] = v_half;
#else
    const uint head_dim = params->head_dim;
    const uint seq_len = params->seq_len;
    const uint total_q = params->total_elements;
    if (gid >= total_q) return;

    // Flattening matches kv_rearrange_kernel output for Q:
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

    float q_out = kv_prep_load_input(q_in, (ulong)q_src); // INDEX64_OK
    bool first_in_pair = true;
    uint mate_hd = hd;
    float cosv = 1.0f;
    float sinv = 0.0f;

    if (apply_rope) {
        const uint rope_mode = params->rope_mode;
        const bool rope_norm = rope_mode == 1;
        const uint pair = rope_norm ? (hd >> 1) : ((hd < half_dim) ? hd : (hd - half_dim));
        cosv = kv_prep_load_tensor(cos_buf, (ulong)pos * (ulong)half_dim + (ulong)pair); // INDEX64_OK
        sinv = kv_prep_load_tensor(sin_buf, (ulong)pos * (ulong)half_dim + (ulong)pair); // INDEX64_OK

        first_in_pair = rope_norm ? ((hd & 1u) == 0u) : (hd < half_dim);
        mate_hd = rope_norm ? (first_in_pair ? (hd + 1) : (hd - 1)) : (first_in_pair ? (hd + half_dim) : (hd - half_dim));
        if (first_in_pair) {
            const float x_i = kv_prep_load_input(q_in, (ulong)q_src); // INDEX64_OK
            const float x_j = kv_prep_load_input(q_in, (ulong)s * (ulong)d_model + (ulong)(h * head_dim + mate_hd)); // INDEX64_OK
            q_out = x_i * cosv - x_j * sinv;
        } else {
            const float x_j = kv_prep_load_input(q_in, (ulong)q_src); // INDEX64_OK
            const float x_i = kv_prep_load_input(q_in, (ulong)s * (ulong)d_model + (ulong)(h * head_dim + mate_hd)); // INDEX64_OK
            q_out = x_j * cosv + x_i * sinv;
        }
    }
    kv_prep_store_output(q_rot, (ulong)gid, q_out); // INDEX64_OK

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
    float k_out = kv_prep_load_input(k_in, (ulong)kv_src_base); // INDEX64_OK
    if (apply_rope) {
        if (first_in_pair) {
            const float x_i = kv_prep_load_input(k_in, (ulong)kv_src_base); // INDEX64_OK
            const float x_j = kv_prep_load_input(k_in, (ulong)s * (ulong)kv_dim + (ulong)(kv_h * head_dim + mate_hd)); // INDEX64_OK
            k_out = x_i * cosv - x_j * sinv;
        } else {
            const float x_j = kv_prep_load_input(k_in, (ulong)kv_src_base); // INDEX64_OK
            const float x_i = kv_prep_load_input(k_in, (ulong)s * (ulong)kv_dim + (ulong)(kv_h * head_dim + mate_hd)); // INDEX64_OK
            k_out = x_j * cosv + x_i * sinv;
        }
    }
    // V: no RoPE
    const float v_out = kv_prep_load_input(v_in, (ulong)kv_src_base); // INDEX64_OK

    // Store once per KV head in compact cache layout [n_kv_heads, max_seq_len, head_dim].
    const uint cache_idx = kv_h * cache_head_stride + cache_row_base;
    kv_prep_store_output(k_cache, (ulong)cache_idx, k_out); // INDEX64_OK
    kv_prep_store_output(v_cache, (ulong)cache_idx, v_out); // INDEX64_OK
#endif
}
