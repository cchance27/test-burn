#include <metal_stdlib>
using namespace metal;

template<typename Policy, uint vec_width, bool has_norm>
ALWAYS_INLINE float3 run_parallel_qkv_project_stage(
    const device uchar* w_q,
    const device uchar* s_q,
    const device uchar* w_k,
    const device uchar* s_k,
    const device uchar* w_v,
    const device uchar* s_v,
    const device half* input,
    constant uint& k_dim,
    constant uint& n_dim,
    constant uint& n_kv,
    constant uint& weights_per_block,
    const device half* gamma,
    float inv_rms,
    uint lane_id,
    uint row_idx,
    uint batch_idx
) {
    const uint blocks_per_k = (k_dim + weights_per_block - 1) / weights_per_block;
    float acc_q = 0.0f;
    float acc_k = 0.0f;
    float acc_v = 0.0f;
    uint k_base = 0;

    const ulong scale_stride = (ulong)row_idx * blocks_per_k * Policy::SCALE_BYTES;
    const device uchar* row_s_q = s_q + scale_stride;
    const device uchar* row_s_k = s_k + scale_stride;
    const device uchar* row_s_v = s_v + scale_stride;

    while (k_base + K_CHUNK_SIZE <= k_dim) {
        const uint k = k_base + lane_id * vec_width;
        half4 xv_lo;
        half4 xv_hi;

        if (vec_width == 4) {
            const half4 xv_raw = *(const device half4*)(input + batch_idx * k_dim + k);
            xv_lo = xv_raw;
            xv_hi = half4(0.0h);
        } else {
            const float4 xv_raw_f = *(const device float4*)(input + batch_idx * k_dim + k);
            xv_lo = as_type<half4>(xv_raw_f.xy);
            xv_hi = as_type<half4>(xv_raw_f.zw);
        }

        if (has_norm && gamma) {
            float4 f_lo = float4(xv_lo);
            float4 f_hi = float4(xv_hi);
            f_lo *= inv_rms;
            f_hi *= inv_rms;
            const device half* g_ptr = gamma + k;
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
            xv_lo = half4(f_lo);
            xv_hi = half4(f_hi);
        }

        const float4 xv_f32_lo = float4(xv_lo);
        const float4 xv_f32_hi = float4(xv_hi);
        const uint block_off = k / weights_per_block;

        const float s_q_val = (float)Policy::load_scale(row_s_q, block_off);
        const float s_k_val = (row_idx < n_kv) ? (float)Policy::load_scale(row_s_k, block_off) : 0.0f;
        const float s_v_val = (row_idx < n_kv) ? (float)Policy::load_scale(row_s_v, block_off) : 0.0f;
        const float a_q_val = (Policy::HAS_AFFINE ? (float)Policy::load_affine(row_s_q, block_off) : 0.0f);
        const float a_k_val = (row_idx < n_kv) ? (Policy::HAS_AFFINE ? (float)Policy::load_affine(row_s_k, block_off) : 0.0f) : 0.0f;
        const float a_v_val = (row_idx < n_kv) ? (Policy::HAS_AFFINE ? (float)Policy::load_affine(row_s_v, block_off) : 0.0f) : 0.0f;

        ulong w_idx;
#if defined(IS_CANONICAL) && IS_CANONICAL
        if (weights_per_block == 32) {
            const ulong U32_MAX_U = 0xFFFFFFFFul;
            if (((ulong)n_dim) * ((ulong)k_dim) <= U32_MAX_U) {
                const uint w_idx_u32 = (k & 31u) + (row_idx << 5u) + (k & ~31u) * n_dim;
                w_idx = (ulong)w_idx_u32;
            } else {
                w_idx = ((ulong)(k & 31u)) + (((ulong)row_idx) << 5ul) + ((ulong)(k & ~31u)) * ((ulong)n_dim);
            }
        } else {
            w_idx = (ulong)WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
        }
#else
        w_idx = (ulong)WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
#endif

        if (vec_width == 4) {
            float w[4];
            Policy::template load_weights<4>(w_q, w_idx, w);
            acc_q += s_q_val * dot(xv_f32_lo, float4(w[0], w[1], w[2], w[3]))
                + (Policy::HAS_AFFINE ? (a_q_val * dot(xv_f32_lo, float4(1.0f))) : 0.0f);
        } else {
            acc_q += Policy::template dot<8>(w_q, w_idx, s_q_val, xv_f32_lo, xv_f32_hi)
                + (Policy::HAS_AFFINE ? (a_q_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
        }

        if (row_idx < n_kv) {
            ulong w_idx_kv;
#if defined(IS_CANONICAL) && IS_CANONICAL
            if (weights_per_block == 32) {
                const ulong U32_MAX_U = 0xFFFFFFFFul;
                if (((ulong)n_kv) * ((ulong)k_dim) <= U32_MAX_U) {
                    const uint w_idx_kv_u32 = (k & 31u) + (row_idx << 5u) + (k & ~31u) * n_kv;
                    w_idx_kv = (ulong)w_idx_kv_u32;
                } else {
                    w_idx_kv = ((ulong)(k & 31u)) + (((ulong)row_idx) << 5ul) + ((ulong)(k & ~31u)) * ((ulong)n_kv);
                }
            } else {
                w_idx_kv = (ulong)WEIGHT_INDEX(row_idx, k, k_dim, n_kv);
            }
#else
            w_idx_kv = (ulong)WEIGHT_INDEX(row_idx, k, k_dim, n_kv);
#endif

            if (vec_width == 4) {
                float wk[4];
                float wv[4];
                Policy::template load_weights<4>(w_k, w_idx_kv, wk);
                Policy::template load_weights<4>(w_v, w_idx_kv, wv);
                acc_k += s_k_val * dot(xv_f32_lo, float4(wk[0], wk[1], wk[2], wk[3]))
                    + (Policy::HAS_AFFINE ? (a_k_val * dot(xv_f32_lo, float4(1.0f))) : 0.0f);
                acc_v += s_v_val * dot(xv_f32_lo, float4(wv[0], wv[1], wv[2], wv[3]))
                    + (Policy::HAS_AFFINE ? (a_v_val * dot(xv_f32_lo, float4(1.0f))) : 0.0f);
            } else {
                acc_k += Policy::template dot<8>(w_k, w_idx_kv, s_k_val, xv_f32_lo, xv_f32_hi)
                    + (Policy::HAS_AFFINE ? (a_k_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
                acc_v += Policy::template dot<8>(w_v, w_idx_kv, s_v_val, xv_f32_lo, xv_f32_hi)
                    + (Policy::HAS_AFFINE ? (a_v_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
            }
        }

        k_base += K_CHUNK_SIZE;
    }

    if (k_base < k_dim) {
        const uint k = k_base + lane_id * vec_width;
        float4 xv_raw = float4(0.0f);
        uint valid_count = 0;

        if (k + vec_width <= k_dim) {
            xv_raw = *(const device float4*)(input + batch_idx * k_dim + k);
            valid_count = vec_width;
        } else if (k < k_dim) {
            for (uint i = 0; i < vec_width && k + i < k_dim; ++i) {
                ((thread half*)&xv_raw)[i] = input[batch_idx * k_dim + k + i];
                valid_count++;
            }
        }

        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);

        if (has_norm && gamma) {
            float4 f_lo = float4(xv_lo);
            float4 f_hi = float4(xv_hi);
            f_lo *= inv_rms;
            f_hi *= inv_rms;
            for (uint i = 0; i < 4 && k + i < k_dim; ++i) {
                f_lo[i] *= (float)gamma[k + i];
            }
            if (vec_width == 8) {
                for (uint i = 0; i < 4 && k + 4 + i < k_dim; ++i) {
                    f_hi[i] *= (float)gamma[k + 4 + i];
                }
            }
            xv_lo = half4(f_lo);
            xv_hi = half4(f_hi);
        }

        const float4 xv_f32_lo = float4(xv_lo);
        const float4 xv_f32_hi = float4(xv_hi);
        const uint block_off = k / weights_per_block;

        const float s_q_val = (k < k_dim) ? (float)Policy::load_scale(row_s_q, block_off) : 0.0f;
        const float s_k_val = (row_idx < n_kv && k < k_dim) ? (float)Policy::load_scale(row_s_k, block_off) : 0.0f;
        const float s_v_val = (row_idx < n_kv && k < k_dim) ? (float)Policy::load_scale(row_s_v, block_off) : 0.0f;
        const float a_q_val = (k < k_dim) ? (Policy::HAS_AFFINE ? (float)Policy::load_affine(row_s_q, block_off) : 0.0f) : 0.0f;
        const float a_k_val = (row_idx < n_kv && k < k_dim) ? (Policy::HAS_AFFINE ? (float)Policy::load_affine(row_s_k, block_off) : 0.0f) : 0.0f;
        const float a_v_val = (row_idx < n_kv && k < k_dim) ? (Policy::HAS_AFFINE ? (float)Policy::load_affine(row_s_v, block_off) : 0.0f) : 0.0f;

        if (k < k_dim) {
            if (vec_width == 4) {
                float w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                Policy::template load_weights<4>(w_q, WEIGHT_INDEX(row_idx, k, k_dim, n_dim), w);
                for (uint i = valid_count; i < 4u; ++i) w[i] = 0.0f;
                acc_q += s_q_val * dot(xv_f32_lo, float4(w[0], w[1], w[2], w[3]))
                    + (Policy::HAS_AFFINE ? (a_q_val * dot(xv_f32_lo, float4(1.0f))) : 0.0f);
            } else {
                float w[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                Policy::template load_weights<8>(w_q, WEIGHT_INDEX(row_idx, k, k_dim, n_dim), w);
                for (uint i = valid_count; i < 8u; ++i) w[i] = 0.0f;
                acc_q += s_q_val * (dot(xv_f32_lo, float4(w[0], w[1], w[2], w[3])) + dot(xv_f32_hi, float4(w[4], w[5], w[6], w[7])))
                    + (Policy::HAS_AFFINE ? (a_q_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
            }
        }

        if (row_idx < n_kv && k < k_dim) {
            if (vec_width == 4) {
                float wk[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float wv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                Policy::template load_weights<4>(w_k, WEIGHT_INDEX(row_idx, k, k_dim, n_kv), wk);
                Policy::template load_weights<4>(w_v, WEIGHT_INDEX(row_idx, k, k_dim, n_kv), wv);
                for (uint i = valid_count; i < 4u; ++i) {
                    wk[i] = 0.0f;
                    wv[i] = 0.0f;
                }
                acc_k += s_k_val * dot(xv_f32_lo, float4(wk[0], wk[1], wk[2], wk[3]))
                    + (Policy::HAS_AFFINE ? (a_k_val * dot(xv_f32_lo, float4(1.0f))) : 0.0f);
                acc_v += s_v_val * dot(xv_f32_lo, float4(wv[0], wv[1], wv[2], wv[3]))
                    + (Policy::HAS_AFFINE ? (a_v_val * dot(xv_f32_lo, float4(1.0f))) : 0.0f);
            } else {
                float wk[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                float wv[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                Policy::template load_weights<8>(w_k, WEIGHT_INDEX(row_idx, k, k_dim, n_kv), wk);
                Policy::template load_weights<8>(w_v, WEIGHT_INDEX(row_idx, k, k_dim, n_kv), wv);
                for (uint i = valid_count; i < 8u; ++i) {
                    wk[i] = 0.0f;
                    wv[i] = 0.0f;
                }
                acc_k += s_k_val * (dot(xv_f32_lo, float4(wk[0], wk[1], wk[2], wk[3])) + dot(xv_f32_hi, float4(wk[4], wk[5], wk[6], wk[7])))
                    + (Policy::HAS_AFFINE ? (a_k_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
                acc_v += s_v_val * (dot(xv_f32_lo, float4(wv[0], wv[1], wv[2], wv[3])) + dot(xv_f32_hi, float4(wv[4], wv[5], wv[6], wv[7])))
                    + (Policy::HAS_AFFINE ? (a_v_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
            }
        }
    }

    return float3(acc_q, acc_k, acc_v);
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
    device half* out_q,
    device half* out_k,
    device half* out_v,
    const device half* b_q,
    const device half* b_k,
    const device half* b_v,
    constant uint& has_b,
    constant uint& n_dim,
    constant uint& n_kv,
    uint lane_id,
    uint row_idx,
    uint batch_idx
) {
    if (lane_id == 0) {
        out_q[batch_idx * n_dim + row_idx] = half(input_var.x + (has_b ? (float)b_q[row_idx] : 0.0f));
        if (row_idx < n_kv) {
            out_k[batch_idx * n_kv + row_idx] = half(input_var.y + (has_b ? (float)b_k[row_idx] : 0.0f));
            out_v[batch_idx * n_kv + row_idx] = half(input_var.z + (has_b ? (float)b_v[row_idx] : 0.0f));
        }
    }
}
