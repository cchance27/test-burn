#include <metal_stdlib>
using namespace metal;

typedef FastScalarT QkvFastScalarT;
typedef FastVec4T QkvFastVec4T;

ALWAYS_INLINE QkvFastVec4T qkv_load_input_half4(const device InputStorageT* input, const ulong idx) {
#if METALLIC_FASTPATH_INPUT_HALF
    return ((const device QkvFastVec4T*)((const device QkvFastScalarT*)input + idx))[0];
#else
    return QkvFastVec4T(metallic_load_input_vec4f(input, idx));
#endif
}

ALWAYS_INLINE QkvFastVec4T qkv_load_gamma_half4(const device GammaStorageT* gamma, const ulong idx) {
#if METALLIC_FASTPATH_GAMMA_HALF
    return ((const device QkvFastVec4T*)((const device QkvFastScalarT*)gamma + idx))[0];
#else
    return QkvFastVec4T(metallic_load_gamma_vec4f(gamma, idx));
#endif
}

ALWAYS_INLINE ulong qkv_weight_index(
    const uint row,
    const uint k,
    const uint n,
    const uint weights_per_block
) {
    return ((ulong)(k % weights_per_block))
        + ((ulong)weights_per_block)
            * (((ulong)row) + ((ulong)(k / weights_per_block)) * ((ulong)n)); // INDEX64_OK
}

template<typename PolicyQ, typename PolicyK, typename PolicyV, uint vec_width, bool has_norm>
ALWAYS_INLINE float3 run_parallel_qkv_project_stage(
    const device uchar* w_q,
    const device uchar* s_q,
    const device uchar* w_k,
    const device uchar* s_k,
    const device uchar* w_v,
    const device uchar* s_v,
    const device InputStorageT* input,
    constant uint& k_dim,
    constant uint& n_dim,
    constant uint& n_kv,
    constant uint& weights_per_block_q,
    constant uint& weights_per_block_k,
    constant uint& weights_per_block_v,
    const device GammaStorageT* gamma,
    float inv_rms,
    uint lane_id,
    uint row_idx,
    uint batch_idx
) {
    const uint blocks_per_k_q = (k_dim + weights_per_block_q - 1) / weights_per_block_q;
    const uint blocks_per_k_k = (k_dim + weights_per_block_k - 1) / weights_per_block_k;
    const uint blocks_per_k_v = (k_dim + weights_per_block_v - 1) / weights_per_block_v;
    float acc_q = 0.0f;
    float acc_k = 0.0f;
    float acc_v = 0.0f;
    uint k_base = 0;

    const ulong scale_stride_q = (ulong)row_idx * blocks_per_k_q * PolicyQ::SCALE_BYTES; // INDEX64_OK
    const ulong scale_stride_k = (ulong)row_idx * blocks_per_k_k * PolicyK::SCALE_BYTES; // INDEX64_OK
    const ulong scale_stride_v = (ulong)row_idx * blocks_per_k_v * PolicyV::SCALE_BYTES; // INDEX64_OK
    const device uchar* row_s_q = s_q + scale_stride_q;
    const device uchar* row_s_k = (row_idx < n_kv) ? (s_k + scale_stride_k) : s_k;
    const device uchar* row_s_v = (row_idx < n_kv) ? (s_v + scale_stride_v) : s_v;
    const bool same_wpb_qk = (weights_per_block_q == weights_per_block_k);
    const bool same_wpb_qv = (weights_per_block_q == weights_per_block_v);
    const bool same_wpb_kv = (weights_per_block_k == weights_per_block_v);
#if METALLIC_FASTPATH_INPUT_HALF
    const device QkvFastScalarT* input_half = (const device QkvFastScalarT*)input;
#endif
#if METALLIC_FASTPATH_GAMMA_HALF
    const device QkvFastScalarT* gamma_half = (const device QkvFastScalarT*)gamma;
#endif

    while (k_base + K_CHUNK_SIZE <= k_dim) {
        const uint k = k_base + lane_id * vec_width;
        QkvFastVec4T xv_lo;
        QkvFastVec4T xv_hi;

        if (vec_width == 4) {
            #if METALLIC_FASTPATH_INPUT_HALF
            const uint x_base = batch_idx * k_dim + k;
            const QkvFastVec4T xv_raw = *(const device QkvFastVec4T*)(input_half + x_base);
            #else
            const ulong x_base = (ulong)batch_idx * (ulong)k_dim + (ulong)k; // INDEX64_OK
            const half4 xv_raw = qkv_load_input_half4(input, x_base);
            #endif
            xv_lo = xv_raw;
            xv_hi = QkvFastVec4T((QkvFastScalarT)0.0f);
        } else {
            #if METALLIC_FASTPATH_INPUT_HALF
            const uint x_base = batch_idx * k_dim + k;
            const float4 xv_raw_f = *(const device float4*)(input_half + x_base);
            xv_lo = as_type<QkvFastVec4T>(xv_raw_f.xy);
            xv_hi = as_type<QkvFastVec4T>(xv_raw_f.zw);
            #else
            const ulong x_base = (ulong)batch_idx * (ulong)k_dim + (ulong)k; // INDEX64_OK
            xv_lo = qkv_load_input_half4(input, x_base + 0ul);
            xv_hi = qkv_load_input_half4(input, x_base + 4ul);
            #endif
        }

        if (has_norm && gamma) {
            float4 f_lo = float4(xv_lo);
            float4 f_hi = float4(xv_hi);
            f_lo *= inv_rms;
            f_hi *= inv_rms;
#if METALLIC_FASTPATH_GAMMA_HALF
            const device QkvFastScalarT* g_ptr = gamma_half + k;
            f_lo.x *= (float)g_ptr[0];
            f_lo.y *= (float)g_ptr[1];
            f_lo.z *= (float)g_ptr[2];
            f_lo.w *= (float)g_ptr[3];
            if (vec_width == 8) {
                f_hi.x *= (float)g_ptr[4];
                f_hi.y *= (float)g_ptr[5];
                f_hi.z *= (float)g_ptr[6];
                f_hi.w *= (float)g_ptr[7];
            }
#else
            f_lo *= float4(qkv_load_gamma_half4(gamma, (ulong)k + 0ul)); // INDEX64_OK
            if (vec_width == 8) {
                f_hi *= float4(qkv_load_gamma_half4(gamma, (ulong)k + 4ul)); // INDEX64_OK
            }
#endif
            xv_lo = QkvFastVec4T(f_lo);
            xv_hi = QkvFastVec4T(f_hi);
        }

        const float4 xv_f32_lo = float4(xv_lo);
        const float4 xv_f32_hi = float4(xv_hi);
        const uint block_off_q = k / weights_per_block_q;
        const uint block_off_k = same_wpb_qk ? block_off_q : (k / weights_per_block_k);
        const uint block_off_v = same_wpb_qv ? block_off_q : (k / weights_per_block_v);

        const float s_q_val = (float)PolicyQ::load_scale(row_s_q, block_off_q);
        const float s_k_val = (row_idx < n_kv) ? (float)PolicyK::load_scale(row_s_k, block_off_k) : 0.0f;
        const float s_v_val = (row_idx < n_kv) ? (float)PolicyV::load_scale(row_s_v, block_off_v) : 0.0f;
        const float a_q_val = (PolicyQ::HAS_AFFINE ? (float)PolicyQ::load_affine(row_s_q, block_off_q) : 0.0f);
        const float a_k_val = (row_idx < n_kv) ? (PolicyK::HAS_AFFINE ? (float)PolicyK::load_affine(row_s_k, block_off_k) : 0.0f) : 0.0f;
        const float a_v_val = (row_idx < n_kv) ? (PolicyV::HAS_AFFINE ? (float)PolicyV::load_affine(row_s_v, block_off_v) : 0.0f) : 0.0f;

        ulong w_idx;
#if defined(IS_CANONICAL) && IS_CANONICAL
        if (weights_per_block_q == 32) {
            const ulong U32_MAX_U = 0xFFFFFFFFul;
            if (((ulong)n_dim) * ((ulong)k_dim) <= U32_MAX_U) { // INDEX64_OK
                const uint w_idx_u32 = (k & 31u) + (row_idx << 5u) + (k & ~31u) * n_dim;
                w_idx = (ulong)w_idx_u32; // INDEX64_OK
            } else {
                w_idx = ((ulong)(k & 31u)) + (((ulong)row_idx) << 5ul) + ((ulong)(k & ~31u)) * ((ulong)n_dim); // INDEX64_OK
            }
        } else {
            w_idx = qkv_weight_index(row_idx, k, n_dim, weights_per_block_q); // INDEX64_OK
        }
#else
        w_idx = qkv_weight_index(row_idx, k, n_dim, weights_per_block_q); // INDEX64_OK
#endif

        if (vec_width == 4) {
            float w[4];
            PolicyQ::template load_weights<4>(w_q, w_idx, w);
            acc_q += s_q_val * dot(xv_f32_lo, float4(w[0], w[1], w[2], w[3]))
                + (PolicyQ::HAS_AFFINE ? (a_q_val * dot(xv_f32_lo, float4(1.0f))) : 0.0f);
        } else {
            acc_q += PolicyQ::template dot<8>(w_q, w_idx, s_q_val, xv_f32_lo, xv_f32_hi)
                + (PolicyQ::HAS_AFFINE ? (a_q_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
        }

        if (row_idx < n_kv) {
            ulong w_idx_k;
            ulong w_idx_v;
#if defined(IS_CANONICAL) && IS_CANONICAL
            if (weights_per_block_k == 32 && weights_per_block_v == 32) {
                const ulong U32_MAX_U = 0xFFFFFFFFul;
                if (((ulong)n_kv) * ((ulong)k_dim) <= U32_MAX_U) { // INDEX64_OK
                    const uint w_idx_kv_u32 = (k & 31u) + (row_idx << 5u) + (k & ~31u) * n_kv;
                    w_idx_k = (ulong)w_idx_kv_u32; // INDEX64_OK
                    w_idx_v = (ulong)w_idx_kv_u32; // INDEX64_OK
                } else {
                    const ulong w_idx_kv_u64 = ((ulong)(k & 31u)) + (((ulong)row_idx) << 5ul) + ((ulong)(k & ~31u)) * ((ulong)n_kv); // INDEX64_OK
                    w_idx_k = w_idx_kv_u64;
                    w_idx_v = w_idx_kv_u64;
                }
            } else {
                w_idx_k = qkv_weight_index(row_idx, k, n_kv, weights_per_block_k); // INDEX64_OK
                w_idx_v = same_wpb_kv ? w_idx_k : qkv_weight_index(row_idx, k, n_kv, weights_per_block_v); // INDEX64_OK
            }
#else
            w_idx_k = qkv_weight_index(row_idx, k, n_kv, weights_per_block_k); // INDEX64_OK
            w_idx_v = same_wpb_kv ? w_idx_k : qkv_weight_index(row_idx, k, n_kv, weights_per_block_v); // INDEX64_OK
#endif

            if (vec_width == 4) {
                float wk[4];
                float wv[4];
                PolicyK::template load_weights<4>(w_k, w_idx_k, wk);
                PolicyV::template load_weights<4>(w_v, w_idx_v, wv);
                acc_k += s_k_val * dot(xv_f32_lo, float4(wk[0], wk[1], wk[2], wk[3]))
                    + (PolicyK::HAS_AFFINE ? (a_k_val * dot(xv_f32_lo, float4(1.0f))) : 0.0f);
                acc_v += s_v_val * dot(xv_f32_lo, float4(wv[0], wv[1], wv[2], wv[3]))
                    + (PolicyV::HAS_AFFINE ? (a_v_val * dot(xv_f32_lo, float4(1.0f))) : 0.0f);
            } else {
                acc_k += PolicyK::template dot<8>(w_k, w_idx_k, s_k_val, xv_f32_lo, xv_f32_hi)
                    + (PolicyK::HAS_AFFINE ? (a_k_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
                acc_v += PolicyV::template dot<8>(w_v, w_idx_v, s_v_val, xv_f32_lo, xv_f32_hi)
                    + (PolicyV::HAS_AFFINE ? (a_v_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
            }
        }

        k_base += K_CHUNK_SIZE;
    }

    if (k_base < k_dim) {
        const uint k = k_base + lane_id * vec_width;
        float4 xv_raw = float4(0.0f);
        uint valid_count = 0;

        if (k + vec_width <= k_dim) {
#if METALLIC_FASTPATH_INPUT_HALF
            const uint x_base = batch_idx * k_dim + k;
            const float4 xv_raw_f = *(const device float4*)(input_half + x_base);
            QkvFastVec4T h_lo = as_type<QkvFastVec4T>(xv_raw_f.xy);
            QkvFastVec4T h_hi = as_type<QkvFastVec4T>(xv_raw_f.zw);
#else
            QkvFastVec4T h_lo = qkv_load_input_half4(input, (ulong)batch_idx * (ulong)k_dim + (ulong)k + 0ul); // INDEX64_OK
            QkvFastVec4T h_hi = (vec_width == 8)
                ? qkv_load_input_half4(input, (ulong)batch_idx * (ulong)k_dim + (ulong)k + 4ul) // INDEX64_OK
                : QkvFastVec4T((QkvFastScalarT)0.0f);
#endif
            xv_raw = float4(as_type<float2>(h_lo), as_type<float2>(h_hi));
            valid_count = vec_width;
        } else if (k < k_dim) {
#if METALLIC_FASTPATH_INPUT_HALF
            const uint x_base = batch_idx * k_dim + k;
            for (uint i = 0; i < vec_width && k + i < k_dim; ++i) {
                ((thread QkvFastScalarT*)&xv_raw)[i] = input_half[x_base + i];
                valid_count++;
            }
#else
            for (uint i = 0; i < vec_width && k + i < k_dim; ++i) {
                ((thread QkvFastScalarT*)&xv_raw)[i] = (QkvFastScalarT)metallic_load_input(
                    input,
                    (ulong)batch_idx * (ulong)k_dim + (ulong)k + (ulong)i // INDEX64_OK
                );
                valid_count++;
            }
#endif
        }

        QkvFastVec4T xv_lo = as_type<QkvFastVec4T>(xv_raw.xy);
        QkvFastVec4T xv_hi = as_type<QkvFastVec4T>(xv_raw.zw);

        if (has_norm && gamma) {
            float4 f_lo = float4(xv_lo);
            float4 f_hi = float4(xv_hi);
            f_lo *= inv_rms;
            f_hi *= inv_rms;
#if METALLIC_FASTPATH_GAMMA_HALF
            const device QkvFastScalarT* g_ptr = gamma_half + k;
            for (uint i = 0; i < 4 && k + i < k_dim; ++i) {
                f_lo[i] *= (float)g_ptr[i];
            }
            if (vec_width == 8) {
                for (uint i = 0; i < 4 && k + 4 + i < k_dim; ++i) {
                    f_hi[i] *= (float)g_ptr[4u + i];
                }
            }
#else
            for (uint i = 0; i < 4 && k + i < k_dim; ++i) {
                f_lo[i] *= (float)metallic_load_gamma(gamma, (ulong)k + (ulong)i); // INDEX64_OK
            }
            if (vec_width == 8) {
                for (uint i = 0; i < 4 && k + 4 + i < k_dim; ++i) {
                    f_hi[i] *= (float)metallic_load_gamma(gamma, (ulong)k + 4ul + (ulong)i); // INDEX64_OK
                }
            }
#endif
            xv_lo = QkvFastVec4T(f_lo);
            xv_hi = QkvFastVec4T(f_hi);
        }

        const float4 xv_f32_lo = float4(xv_lo);
        const float4 xv_f32_hi = float4(xv_hi);
        const uint block_off_q = k / weights_per_block_q;
        const uint block_off_k = same_wpb_qk ? block_off_q : (k / weights_per_block_k);
        const uint block_off_v = same_wpb_qv ? block_off_q : (k / weights_per_block_v);

        const float s_q_val = (k < k_dim) ? (float)PolicyQ::load_scale(row_s_q, block_off_q) : 0.0f;
        const float s_k_val = (row_idx < n_kv && k < k_dim) ? (float)PolicyK::load_scale(row_s_k, block_off_k) : 0.0f;
        const float s_v_val = (row_idx < n_kv && k < k_dim) ? (float)PolicyV::load_scale(row_s_v, block_off_v) : 0.0f;
        const float a_q_val = (k < k_dim) ? (PolicyQ::HAS_AFFINE ? (float)PolicyQ::load_affine(row_s_q, block_off_q) : 0.0f) : 0.0f;
        const float a_k_val = (row_idx < n_kv && k < k_dim) ? (PolicyK::HAS_AFFINE ? (float)PolicyK::load_affine(row_s_k, block_off_k) : 0.0f) : 0.0f;
        const float a_v_val = (row_idx < n_kv && k < k_dim) ? (PolicyV::HAS_AFFINE ? (float)PolicyV::load_affine(row_s_v, block_off_v) : 0.0f) : 0.0f;

        if (k < k_dim) {
            if (vec_width == 4) {
                float w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const ulong w_q_idx = qkv_weight_index(row_idx, k, n_dim, weights_per_block_q); // INDEX64_OK
                PolicyQ::template load_weights<4>(w_q, w_q_idx, w);
                for (uint i = valid_count; i < 4u; ++i) w[i] = 0.0f;
                acc_q += s_q_val * dot(xv_f32_lo, float4(w[0], w[1], w[2], w[3]))
                    + (PolicyQ::HAS_AFFINE ? (a_q_val * dot(xv_f32_lo, float4(1.0f))) : 0.0f);
            } else {
                float w[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                const ulong w_q_idx = qkv_weight_index(row_idx, k, n_dim, weights_per_block_q); // INDEX64_OK
                PolicyQ::template load_weights<8>(w_q, w_q_idx, w);
                for (uint i = valid_count; i < 8u; ++i) w[i] = 0.0f;
                acc_q += s_q_val * (dot(xv_f32_lo, float4(w[0], w[1], w[2], w[3])) + dot(xv_f32_hi, float4(w[4], w[5], w[6], w[7])))
                    + (PolicyQ::HAS_AFFINE ? (a_q_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
            }
        }

        if (row_idx < n_kv && k < k_dim) {
            if (vec_width == 4) {
                float wk[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float wv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const ulong w_k_idx = qkv_weight_index(row_idx, k, n_kv, weights_per_block_k); // INDEX64_OK
                const ulong w_v_idx = same_wpb_kv ? w_k_idx : qkv_weight_index(row_idx, k, n_kv, weights_per_block_v); // INDEX64_OK
                PolicyK::template load_weights<4>(w_k, w_k_idx, wk);
                PolicyV::template load_weights<4>(w_v, w_v_idx, wv);
                for (uint i = valid_count; i < 4u; ++i) {
                    wk[i] = 0.0f;
                    wv[i] = 0.0f;
                }
                acc_k += s_k_val * dot(xv_f32_lo, float4(wk[0], wk[1], wk[2], wk[3]))
                    + (PolicyK::HAS_AFFINE ? (a_k_val * dot(xv_f32_lo, float4(1.0f))) : 0.0f);
                acc_v += s_v_val * dot(xv_f32_lo, float4(wv[0], wv[1], wv[2], wv[3]))
                    + (PolicyV::HAS_AFFINE ? (a_v_val * dot(xv_f32_lo, float4(1.0f))) : 0.0f);
            } else {
                float wk[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                float wv[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                const ulong w_k_idx = qkv_weight_index(row_idx, k, n_kv, weights_per_block_k); // INDEX64_OK
                const ulong w_v_idx = same_wpb_kv ? w_k_idx : qkv_weight_index(row_idx, k, n_kv, weights_per_block_v); // INDEX64_OK
                PolicyK::template load_weights<8>(w_k, w_k_idx, wk);
                PolicyV::template load_weights<8>(w_v, w_v_idx, wv);
                for (uint i = valid_count; i < 8u; ++i) {
                    wk[i] = 0.0f;
                    wv[i] = 0.0f;
                }
                acc_k += s_k_val * (dot(xv_f32_lo, float4(wk[0], wk[1], wk[2], wk[3])) + dot(xv_f32_hi, float4(wk[4], wk[5], wk[6], wk[7])))
                    + (PolicyK::HAS_AFFINE ? (a_k_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
                acc_v += s_v_val * (dot(xv_f32_lo, float4(wv[0], wv[1], wv[2], wv[3])) + dot(xv_f32_hi, float4(wv[4], wv[5], wv[6], wv[7])))
                    + (PolicyV::HAS_AFFINE ? (a_v_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
            }
        }
    }

    return float3(acc_q, acc_k, acc_v);
}

template<typename Policy, uint vec_width, bool has_norm>
ALWAYS_INLINE float3 run_parallel_qkv_project_stage_uniform(
    const device uchar* w_q,
    const device uchar* s_q,
    const device uchar* w_k,
    const device uchar* s_k,
    const device uchar* w_v,
    const device uchar* s_v,
    const device InputStorageT* input,
    constant uint& k_dim,
    constant uint& n_dim,
    constant uint& n_kv,
    constant uint& weights_per_block,
    const device GammaStorageT* gamma,
    float inv_rms,
    uint lane_id,
    uint row_idx,
    uint batch_idx
) {
    return run_parallel_qkv_project_stage<Policy, Policy, Policy, vec_width, has_norm>(
        w_q,
        s_q,
        w_k,
        s_k,
        w_v,
        s_v,
        input,
        k_dim,
        n_dim,
        n_kv,
        weights_per_block,
        weights_per_block,
        weights_per_block,
        gamma,
        inv_rms,
        lane_id,
        row_idx,
        batch_idx
    );
}

ALWAYS_INLINE float3 run_qkv_reduce_stage(float3 input_var) {
    float3 qkv_sum = input_var;
    for (uint offset = 16; offset > 0; offset /= 2) {
        qkv_sum.x += simd_shuffle_down(qkv_sum.x, offset);
        qkv_sum.y += simd_shuffle_down(qkv_sum.y, offset);
        qkv_sum.z += simd_shuffle_down(qkv_sum.z, offset);
    }
    return qkv_sum;
}

ALWAYS_INLINE void run_qkv_write_stage(
    float3 input_var,
    device OutputStorageT* out_q,
    device OutputStorageT* out_k,
    device OutputStorageT* out_v,
    const device BiasStorageT* b_q,
    const device BiasStorageT* b_k,
    const device BiasStorageT* b_v,
    constant uint& has_b,
    constant uint& n_dim,
    constant uint& n_kv,
    uint lane_id,
    uint row_idx,
    uint batch_idx
) {
    if (lane_id == 0) {
#if METALLIC_FASTPATH_OUTPUT_HALF && METALLIC_FASTPATH_BIAS_HALF
        const device QkvFastScalarT* bq = (const device QkvFastScalarT*)b_q;
        const device QkvFastScalarT* bk = (const device QkvFastScalarT*)b_k;
        const device QkvFastScalarT* bv = (const device QkvFastScalarT*)b_v;
        device QkvFastScalarT* oq = (device QkvFastScalarT*)out_q;
        device QkvFastScalarT* ok = (device QkvFastScalarT*)out_k;
        device QkvFastScalarT* ov = (device QkvFastScalarT*)out_v;
        oq[batch_idx * n_dim + row_idx] = QkvFastScalarT(input_var.x + (has_b ? (float)bq[row_idx] : 0.0f));
        if (row_idx < n_kv) {
            ok[batch_idx * n_kv + row_idx] = QkvFastScalarT(input_var.y + (has_b ? (float)bk[row_idx] : 0.0f));
            ov[batch_idx * n_kv + row_idx] = QkvFastScalarT(input_var.z + (has_b ? (float)bv[row_idx] : 0.0f));
        }
#else
        const ulong q_idx = (ulong)batch_idx * (ulong)n_dim + (ulong)row_idx; // INDEX64_OK
        const float q_bias = has_b ? metallic_load_bias(b_q, (ulong)row_idx) : 0.0f; // INDEX64_OK
        metallic_store_output(out_q, q_idx, metallic_to_accum(input_var.x + q_bias));
        if (row_idx < n_kv) {
            const ulong kv_idx = (ulong)batch_idx * (ulong)n_kv + (ulong)row_idx; // INDEX64_OK
            const float k_bias = has_b ? metallic_load_bias(b_k, (ulong)row_idx) : 0.0f; // INDEX64_OK
            const float v_bias = has_b ? metallic_load_bias(b_v, (ulong)row_idx) : 0.0f; // INDEX64_OK
            metallic_store_output(out_k, kv_idx, metallic_to_accum(input_var.y + k_bias));
            metallic_store_output(out_v, kv_idx, metallic_to_accum(input_var.z + v_bias));
        }
#endif
    }
}
