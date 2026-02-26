#ifndef GEMV_V2_VECTORIZED_STAGE_METAL_H
#define GEMV_V2_VECTORIZED_STAGE_METAL_H

// Fast path vector aliases (currently F16-specialized).
typedef FastScalarT GemvFastScalarT;
typedef FastVec4T GemvFastVec4T;

ALWAYS_INLINE void gemv_apply_norm_inplace(
    thread GemvFastVec4T &xv_lo,
    thread GemvFastVec4T &xv_hi,
    const device GammaStorageT *gamma,
    const bool has_gamma,
    const float inv_rms,
    const bool has_shared_norm,
    const uint k,
    const uint k_dim,
    const uint vec_width
) {
    if (has_gamma && gamma != nullptr) {
        float4 f_lo = float4(xv_lo) * inv_rms;
        float4 f_hi = float4(xv_hi) * inv_rms;
#if METALLIC_FASTPATH_GAMMA_HALF
        const device GemvFastScalarT* gamma_half = (const device GemvFastScalarT*)gamma;
#endif
        for (uint i = 0; i < 4u && (k + i) < k_dim; ++i) {
#if METALLIC_FASTPATH_GAMMA_HALF
            f_lo[i] *= (float)gamma_half[k + i];
#else
            f_lo[i] *= (float)metallic_load_gamma(gamma, (ulong)(k + i)); // INDEX64_OK
#endif
        }
        if (vec_width == 8u) {
            for (uint i = 0; i < 4u && (k + 4u + i) < k_dim; ++i) {
#if METALLIC_FASTPATH_GAMMA_HALF
                f_hi[i] *= (float)gamma_half[k + 4u + i];
#else
                f_hi[i] *= (float)metallic_load_gamma(gamma, (ulong)(k + 4u + i)); // INDEX64_OK
#endif
            }
        }
        xv_lo = GemvFastVec4T(f_lo);
        xv_hi = GemvFastVec4T(f_hi);
    } else if (has_shared_norm) {
        xv_lo *= (GemvFastVec4T)inv_rms;
        xv_hi *= (GemvFastVec4T)inv_rms;
    }
}

ALWAYS_INLINE GemvFastScalarT gemv_load_input_half(const device InputStorageT* input, const ulong idx) {
#if METALLIC_FASTPATH_INPUT_HALF
    return ((const device GemvFastScalarT*)input)[idx];
#else
    return (GemvFastScalarT)metallic_load_input(input, idx);
#endif
}

ALWAYS_INLINE GemvFastVec4T gemv_load_input_half4(const device InputStorageT* input, const ulong idx) {
#if METALLIC_FASTPATH_INPUT_HALF
    return ((const device GemvFastVec4T*)((const device GemvFastScalarT*)input + idx))[0];
#else
    return GemvFastVec4T(metallic_load_input_vec4f(input, idx));
#endif
}

ALWAYS_INLINE float4 gemv_load_input_float4(const device InputStorageT* input, const ulong idx) {
    return metallic_load_input_vec4f(input, idx);
}

ALWAYS_INLINE void gemv_apply_norm_f32_inplace(
    thread float4 &xv_lo,
    thread float4 &xv_hi,
    const device GammaStorageT *gamma,
    const bool has_gamma,
    const float inv_rms,
    const bool has_shared_norm,
    const uint k,
    const uint k_dim,
    const uint vec_width
) {
    if (has_gamma && gamma != nullptr) {
        xv_lo *= inv_rms;
        xv_hi *= inv_rms;
        for (uint i = 0; i < 4u && (k + i) < k_dim; ++i) {
            xv_lo[i] *= (float)metallic_load_gamma(gamma, (ulong)(k + i)); // INDEX64_OK
        }
        if (vec_width == 8u) {
            for (uint i = 0; i < 4u && (k + 4u + i) < k_dim; ++i) {
                xv_hi[i] *= (float)metallic_load_gamma(gamma, (ulong)(k + 4u + i)); // INDEX64_OK
            }
        }
    } else if (has_shared_norm) {
        xv_lo *= inv_rms;
        xv_hi *= inv_rms;
    }
}

/// Canonical stage entrypoint used by derive-based Rust stage glue.
template<typename Policy>
ALWAYS_INLINE float run_gemv_canonical_stage(
    const device uchar *weights,
    const device uchar *scale_bytes,
    const device InputStorageT *input,
    const uint row_idx,
    const uint lane_id,
    const uint batch_idx,
    const uint k_dim,
    const uint n_dim,
    const uint weights_per_block
) {
    const device InputStorageT* input_row = input + batch_idx * k_dim;
    float acc = 0.0f;
    const uint k_step = 32u * 4u;
    uint k = lane_id * 4u;

    while (k < k_dim) {
        acc += gemv_dot_canonical<Policy>(
            weights,
            scale_bytes,
            input_row,
            row_idx,
            k,
            k + 4u,
            k_dim,
            n_dim,
            weights_per_block
        );
        k += k_step;
    }

    return acc;
}

/// Vectorized stage entrypoint used by derive-based Rust stage glue.
template<typename Policy, uint VEC_WIDTH, bool USE_F16_COLS8>
ALWAYS_INLINE float run_gemv_vectorized_stage(
    const device uchar *weights,
    const device uchar *scale_bytes,
    const device InputStorageT *input,
    const device GammaStorageT *gamma,
    const float inv_rms,
    const uint k_dim,
    const uint n_dim,
    const uint weights_per_block,
    const uint row_idx,
    const uint lane_id,
    const uint batch_idx,
    const uint lid_x,
    const bool has_gamma,
    const bool has_shared_norm
) {
    const uint input_row_base = batch_idx * k_dim;
    float acc = 0.0f;
    uint k_base = 0u;

    // Fast contiguous path is only valid for true FP16 weights.
#if defined(METALLIC_POLICY_WEIGHTS_FP16) && METALLIC_POLICY_WEIGHTS_FP16
    if (USE_F16_COLS8 && METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_GAMMA_HALF) {
#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
        const device GemvFastScalarT* weights_half = (const device GemvFastScalarT*)weights;
        const device GemvFastScalarT* row_ptr = weights_half + (ulong)row_idx * (ulong)k_dim; // INDEX64_OK
        const device GemvFastScalarT* w_ptr = row_ptr + lane_id * VEC_WIDTH;
        const device GemvFastScalarT* x_ptr = (const device GemvFastScalarT*)(input + input_row_base + lane_id * VEC_WIDTH);

        uint remaining = k_dim;
        while (remaining >= K_CHUNK_SIZE) {
            float4 xv_raw = *(const device float4*)(x_ptr);
            GemvFastVec4T xv_lo = as_type<GemvFastVec4T>(xv_raw.xy);
            GemvFastVec4T xv_hi = as_type<GemvFastVec4T>(xv_raw.zw);

            gemv_apply_norm_inplace(xv_lo, xv_hi, gamma, has_gamma, inv_rms, has_shared_norm, k_base + lane_id * VEC_WIDTH, k_dim, VEC_WIDTH);

            float4 w_raw = *(const device float4*)(w_ptr);
            GemvFastVec4T w_lo = as_type<GemvFastVec4T>(w_raw.xy);
            GemvFastVec4T w_hi = as_type<GemvFastVec4T>(w_raw.zw);
            acc += dot(float4(xv_lo), float4(w_lo)) + dot(float4(xv_hi), float4(w_hi));

            x_ptr += K_CHUNK_SIZE;
            w_ptr += K_CHUNK_SIZE;
            k_base += K_CHUNK_SIZE;
            remaining -= K_CHUNK_SIZE;
        }

        if (remaining > 0u) {
            const uint lane_off = lane_id * VEC_WIDTH;
            const uint valid_count = (remaining > lane_off) ? min(VEC_WIDTH, remaining - lane_off) : 0u;

            float4 xv_raw = float4(0.0f);
            if (valid_count == VEC_WIDTH) {
                xv_raw = *(const device float4*)(x_ptr);
            } else if (valid_count > 0u) {
                #pragma unroll
                for (uint i = 0; i < VEC_WIDTH; ++i) {
                    if (i < valid_count) {
                        ((thread GemvFastScalarT*)&xv_raw)[i] = x_ptr[i];
                    }
                }
            }

            GemvFastVec4T xv_lo = as_type<GemvFastVec4T>(xv_raw.xy);
            GemvFastVec4T xv_hi = as_type<GemvFastVec4T>(xv_raw.zw);
            gemv_apply_norm_inplace(xv_lo, xv_hi, gamma, has_gamma, inv_rms, has_shared_norm, k_base + lane_off, k_dim, VEC_WIDTH);

            float4 w_raw = float4(0.0f);
            if (valid_count == VEC_WIDTH) {
                w_raw = *(const device float4*)(w_ptr);
            } else if (valid_count > 0u) {
                #pragma unroll
                for (uint i = 0; i < VEC_WIDTH; ++i) {
                    if (i < valid_count) {
                        ((thread GemvFastScalarT*)&w_raw)[i] = w_ptr[i];
                    }
                }
            }

            GemvFastVec4T w_lo = as_type<GemvFastVec4T>(w_raw.xy);
            GemvFastVec4T w_hi = as_type<GemvFastVec4T>(w_raw.zw);
            if (valid_count > 0u) {
                acc += dot(float4(xv_lo), float4(w_lo)) + dot(float4(xv_hi), float4(w_hi));
            }
        }

        return acc;
#endif
    }
#endif

#if defined(METALLIC_POLICY_HAS_SCALE) && METALLIC_POLICY_HAS_SCALE
    const uint blocks_per_k = (k_dim + weights_per_block - 1u) / weights_per_block;
#else
    const uint blocks_per_k = 0u;
#endif

#if (VEC_WIDTH == 8u)
#if METALLIC_FASTPATH_INPUT_HALF && defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS && defined(METALLIC_POLICY_WEIGHTS_FP16) && METALLIC_POLICY_WEIGHTS_FP16
    threadgroup GemvFastScalarT x_tile[K_CHUNK_SIZE];
    // x_tile reduces duplicate input reads across warps, but adds two TG barriers per chunk.
    // Restrict it to the dense contiguous FP16 fastpath and larger K to avoid barrier-heavy jitter.
    const bool use_x_tile = !has_gamma && !has_shared_norm
        && ((n_dim & (WARPS_PER_TG - 1u)) == 0u)
        && (k_dim >= (K_CHUNK_SIZE * 2u));
#else
    const bool use_x_tile = false;
#endif
#else
    const bool use_x_tile = false;
#endif

    while (k_base + K_CHUNK_SIZE <= k_dim) {
#if (VEC_WIDTH == 8u)
        if (use_x_tile) {
            x_tile[lid_x] = ((const device GemvFastScalarT*)input)[input_row_base + k_base + lid_x];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#endif
        const uint k = k_base + lane_id * VEC_WIDTH;
        float4 xv_f32_lo;
        float4 xv_f32_hi;
#if METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_GAMMA_HALF
        half4 xv_lo;
        half4 xv_hi;
#endif
#if (VEC_WIDTH == 8u)
        if (use_x_tile) {
            const uint base = lane_id * 8u;
            const threadgroup GemvFastScalarT* x_ptr_tg = x_tile + base;
#if METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_GAMMA_HALF
            xv_lo = *(const threadgroup half4*)(x_ptr_tg + 0u);
            xv_hi = *(const threadgroup half4*)(x_ptr_tg + 4u);
#else
            xv_f32_lo = float4(*(const threadgroup GemvFastVec4T*)(x_ptr_tg + 0u));
            xv_f32_hi = float4(*(const threadgroup GemvFastVec4T*)(x_ptr_tg + 4u));
#endif
        } else
#endif
        if (VEC_WIDTH == 4u) {
#if METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_GAMMA_HALF
            xv_lo = *(const device half4*)((const device half*)input + input_row_base + k);
            xv_hi = half4(0.0h);
#else
            xv_f32_lo = gemv_load_input_float4(input, (ulong)(input_row_base + k)); // INDEX64_OK
            xv_f32_hi = float4(0.0f);
#endif
        } else {
#if METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_GAMMA_HALF
            float4 xv_raw = *(const device float4*)((const device half*)input + input_row_base + k);
            xv_lo = as_type<half4>(xv_raw.xy);
            xv_hi = as_type<half4>(xv_raw.zw);
#else
            const ulong x_idx = (ulong)(input_row_base + k); // INDEX64_OK
            xv_f32_lo = gemv_load_input_float4(input, x_idx + 0ul);
            xv_f32_hi = gemv_load_input_float4(input, x_idx + 4ul);
#endif
        }

#if METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_GAMMA_HALF
        gemv_apply_norm_inplace(
            xv_lo,
            xv_hi,
            gamma,
            has_gamma,
            inv_rms,
            has_shared_norm,
            k,
            k_dim,
            VEC_WIDTH
        );
        xv_f32_lo = float4(xv_lo);
        xv_f32_hi = float4(xv_hi);
#else
        gemv_apply_norm_f32_inplace(xv_f32_lo, xv_f32_hi, gamma, has_gamma, inv_rms, has_shared_norm, k, k_dim, VEC_WIDTH);
#endif

#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
#if defined(METALLIC_POLICY_WEIGHTS_FP16) && METALLIC_POLICY_WEIGHTS_FP16
        const device GemvFastScalarT* weights_row = (const device GemvFastScalarT*)weights + (ulong)row_idx * (ulong)k_dim; // INDEX64_OK
        acc += Policy::template dot<VEC_WIDTH>(weights_row, (ulong)k, 1.0f, xv_f32_lo, xv_f32_hi); // INDEX64_OK
#else
        const ulong base_idx = WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
        const ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block); // INDEX64_OK
        float scale = 1.0f;
#if defined(METALLIC_POLICY_HAS_SCALE) && METALLIC_POLICY_HAS_SCALE
        scale = (float)Policy::load_scale(scale_bytes, scale_idx);
#endif
        const float affine = Policy::HAS_AFFINE ? (float)Policy::load_affine(scale_bytes, scale_idx) : 0.0f;
        acc += Policy::template dot<VEC_WIDTH>(weights, base_idx, scale, xv_f32_lo, xv_f32_hi)
            + (Policy::HAS_AFFINE ? (affine * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
#endif
#else
        float w[VEC_WIDTH];
        #pragma unroll
        for (uint i = 0; i < VEC_WIDTH; ++i) {
            ulong idx = WEIGHT_INDEX(row_idx, k + i, k_dim, n_dim);
            float temp_w[1];
            Policy::template load_weights<1>(weights, idx, temp_w);
            w[i] = temp_w[0];
        }
        const ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block); // INDEX64_OK
        const float scale = Policy::HAS_SCALE ? (float)Policy::load_scale(scale_bytes, scale_idx) : 1.0f;
        const float affine = Policy::HAS_AFFINE ? (float)Policy::load_affine(scale_bytes, scale_idx) : 0.0f;
        if (VEC_WIDTH == 8u) {
            float4 w_lo = float4(w[0], w[1], w[2], w[3]);
            float4 w_hi = float4(w[4], w[5], w[6], w[7]);
            acc += scale * (dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi))
                + (Policy::HAS_AFFINE ? (affine * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
        } else {
            float4 w_lo = float4(w[0], w[1], w[2], w[3]);
            acc += scale * dot(xv_f32_lo, w_lo)
                + (Policy::HAS_AFFINE ? (affine * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
        }
#endif

#if (VEC_WIDTH == 8u)
        if (use_x_tile) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#endif
        k_base += K_CHUNK_SIZE;
    }

    if (k_base < k_dim) {
        const uint k = k_base + lane_id * VEC_WIDTH;
        float4 xv_f32_lo;
        float4 xv_f32_hi;
#if METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_GAMMA_HALF
        half4 xv_lo = half4(0.0h);
        half4 xv_hi = half4(0.0h);
#else
        xv_f32_lo = float4(0.0f);
        xv_f32_hi = float4(0.0f);
#endif
        if (VEC_WIDTH == 8u) {
            if (k + 8u <= k_dim) {
#if METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_GAMMA_HALF
                float4 xv_raw = *(const device float4*)((const device half*)input + input_row_base + k);
                xv_lo = as_type<half4>(xv_raw.xy);
                xv_hi = as_type<half4>(xv_raw.zw);
#else
                const ulong x_idx = (ulong)(input_row_base + k); // INDEX64_OK
                xv_f32_lo = gemv_load_input_float4(input, x_idx + 0ul);
                xv_f32_hi = gemv_load_input_float4(input, x_idx + 4ul);
#endif
            } else if (k < k_dim) {
#if METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_GAMMA_HALF
                float4 xv_raw = float4(0.0f);
                for (uint i = 0; i < 8u && k + i < k_dim; ++i) {
                    ((thread half*)&xv_raw)[i] = ((const device half*)input)[input_row_base + k + i];
                }
                xv_lo = as_type<half4>(xv_raw.xy);
                xv_hi = as_type<half4>(xv_raw.zw);
#else
                for (uint i = 0; i < 8u && k + i < k_dim; ++i) {
                    float x = (float)metallic_load_input(input, (ulong)(input_row_base + k + i)); // INDEX64_OK
                    if (i < 4u) {
                        xv_f32_lo[i] = x;
                    } else {
                        xv_f32_hi[i - 4u] = x;
                    }
                }
#endif
            }
        } else {
#if METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_GAMMA_HALF
            half x0 = (k + 0u < k_dim) ? ((const device half*)input)[input_row_base + k + 0u] : half(0.0h);
            half x1 = (k + 1u < k_dim) ? ((const device half*)input)[input_row_base + k + 1u] : half(0.0h);
            half x2 = (k + 2u < k_dim) ? ((const device half*)input)[input_row_base + k + 2u] : half(0.0h);
            half x3 = (k + 3u < k_dim) ? ((const device half*)input)[input_row_base + k + 3u] : half(0.0h);
            xv_lo = half4(x0, x1, x2, x3);
            xv_hi = half4(0.0h);
#else
            for (uint i = 0; i < 4u && (k + i) < k_dim; ++i) {
                xv_f32_lo[i] = (float)metallic_load_input(input, (ulong)(input_row_base + k + i)); // INDEX64_OK
            }
#endif
        }

#if METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_GAMMA_HALF
        gemv_apply_norm_inplace(
            xv_lo,
            xv_hi,
            gamma,
            has_gamma,
            inv_rms,
            has_shared_norm,
            k,
            k_dim,
            VEC_WIDTH
        );
        xv_f32_lo = float4(xv_lo);
        xv_f32_hi = float4(xv_hi);
#else
        gemv_apply_norm_f32_inplace(xv_f32_lo, xv_f32_hi, gamma, has_gamma, inv_rms, has_shared_norm, k, k_dim, VEC_WIDTH);
#endif

        if (k + VEC_WIDTH <= k_dim) {
#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
#if defined(METALLIC_POLICY_WEIGHTS_FP16) && METALLIC_POLICY_WEIGHTS_FP16
            const device GemvFastScalarT* weights_row = (const device GemvFastScalarT*)weights + (ulong)row_idx * (ulong)k_dim; // INDEX64_OK
            acc += Policy::template dot<VEC_WIDTH>(weights_row, (ulong)k, 1.0f, xv_f32_lo, xv_f32_hi); // INDEX64_OK
#else
            const ulong base_idx = WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
            const ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block); // INDEX64_OK
            float scale = 1.0f;
#if defined(METALLIC_POLICY_HAS_SCALE) && METALLIC_POLICY_HAS_SCALE
            scale = (float)Policy::load_scale(scale_bytes, scale_idx);
#endif
            const float affine = Policy::HAS_AFFINE ? (float)Policy::load_affine(scale_bytes, scale_idx) : 0.0f;
            acc += Policy::template dot<VEC_WIDTH>(weights, base_idx, scale, xv_f32_lo, xv_f32_hi)
                + (Policy::HAS_AFFINE ? (affine * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
#endif
#else
            float w[VEC_WIDTH];
            #pragma unroll
            for (uint i = 0; i < VEC_WIDTH; ++i) {
                const ulong idx = WEIGHT_INDEX(row_idx, k + i, k_dim, n_dim);
                float temp_w[1];
                Policy::template load_weights<1>(weights, idx, temp_w);
                w[i] = temp_w[0];
            }
            const ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block); // INDEX64_OK
            const float scale = (float)Policy::load_scale(scale_bytes, scale_idx);
            const float affine = Policy::HAS_AFFINE ? (float)Policy::load_affine(scale_bytes, scale_idx) : 0.0f;
            if (VEC_WIDTH == 8u) {
                float4 w_lo = float4(w[0], w[1], w[2], w[3]);
                float4 w_hi = float4(w[4], w[5], w[6], w[7]);
                acc += scale * (dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi))
                    + (Policy::HAS_AFFINE ? (affine * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
            } else {
                float4 w_lo = float4(w[0], w[1], w[2], w[3]);
                acc += scale * dot(xv_f32_lo, w_lo)
                    + (Policy::HAS_AFFINE ? (affine * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
            }
#endif
        } else if (k < k_dim) {
            const ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block); // INDEX64_OK
            const ComputeT scale = (ComputeT)Policy::load_scale(scale_bytes, scale_idx);
            const float affine = Policy::HAS_AFFINE ? (float)Policy::load_affine(scale_bytes, scale_idx) : 0.0f;
            float acc_tail = 0.0f;
            #pragma unroll
            for (uint i = 0; i < VEC_WIDTH; ++i) {
                if (k + i < k_dim) {
                    const ulong idx = WEIGHT_INDEX(row_idx, k + i, k_dim, n_dim);
                    float temp_w[1];
                    Policy::template load_weights<1>(weights, idx, temp_w);
                    const float x = (i < 4u) ? xv_f32_lo[i] : xv_f32_hi[i - 4u];
                    acc_tail += temp_w[0] * scale * x + affine * x;
                }
            }
            acc += acc_tail;
        }
    }

    return acc;
}

#endif // GEMV_V2_VECTORIZED_STAGE_METAL_H
