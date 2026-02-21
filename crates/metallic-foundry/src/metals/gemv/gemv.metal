#ifndef GEMV_V2_METAL_H
#define GEMV_V2_METAL_H

#include <metal_stdlib>
using namespace metal;

// NOTE: ALWAYS_INLINE is provided by policies/base.metal included via policy_f16.metal

// =============================================================================
// GemvV2 Core Functions - Optimized for Performance
// =============================================================================
// Uses warp-per-row strategy with vectorized loads for maximum throughput.
// WEIGHT_INDEX macro from LayoutStage provides NK/KN support.

// Dispatch constants (use ifndef to avoid redefinition if already set by stages)
#ifndef ELEMS_PER_THREAD
#define ELEMS_PER_THREAD 8
#endif

#if defined(METALLIC_POLICY_HAS_AFFINE) && METALLIC_POLICY_HAS_AFFINE
#define METALLIC_AFFINE_LOAD(scale_bytes, idx) ((float)Policy::load_affine((scale_bytes), (idx)))
#define METALLIC_AFFINE_XSUM(xv_lo, xv_hi) (dot((xv_lo), float4(1.0f)) + dot((xv_hi), float4(1.0f)))
#else
#define METALLIC_AFFINE_LOAD(scale_bytes, idx) (0.0f)
#define METALLIC_AFFINE_XSUM(xv_lo, xv_hi) (0.0f)
#endif

/// Optimized dot product using vectorized loads.
/// Each warp processes one output row, each thread handles ELEMS_PER_THREAD elements per chunk.
template<typename Policy>
ALWAYS_INLINE float gemv_dot_vectorized(
    const device uchar *data,
    const device uchar *scale_bytes,
    const device half *input,
    const uint row_idx,
    const uint K,
    const uint N,
    const uint weights_per_block,
    const uint lane_id  // 0-31 within warp
) {
    const uint blocks_per_k = Policy::HAS_SCALE ? ((K + weights_per_block - 1) / weights_per_block) : 0u;
    const uint k_chunk_size = 32 * ELEMS_PER_THREAD; // 256 elements per chunk
    
    float acc = 0.0f;
    uint k_base = 0;
    
    // Fast path: process 256 elements per iteration (no bounds checks)
#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
    if (!Policy::HAS_SCALE) {
        const device half *w_ptr = (const device half *)data;
        const device half *row_ptr = w_ptr + (ulong)row_idx * (ulong)K;

        while (k_base + k_chunk_size <= K) {
            uint k = k_base + lane_id * ELEMS_PER_THREAD;

            // Vector load 8 halves from input
            float4 xv_raw = *(const device float4*)(input + k);
            half4 xv_lo = as_type<half4>(xv_raw.xy);
            half4 xv_hi = as_type<half4>(xv_raw.zw);
            float4 xv_f32_lo = float4(xv_lo);
            float4 xv_f32_hi = float4(xv_hi);

            float4 w_raw = *(const device float4*)(row_ptr + k);
            half4 w_lo = as_type<half4>(w_raw.xy);
            half4 w_hi = as_type<half4>(w_raw.zw);

            acc += dot(xv_f32_lo, float4(w_lo)) + dot(xv_f32_hi, float4(w_hi));

            k_base += k_chunk_size;
        }

        // Tail: bounds-checked
        if (k_base < K) {
            uint k = k_base + lane_id * ELEMS_PER_THREAD;

            float4 xv_raw = float4(0.0f);
            uint valid_count = 0;

            if (k + 8 <= K) {
                xv_raw = *(const device float4*)(input + k);
                valid_count = 8;
            } else if (k < K) {
                for (uint i = 0; i < 8 && k + i < K; ++i) {
                    ((thread half*)&xv_raw)[i] = input[k + i];
                    valid_count++;
                }
            }

            half4 xv_lo = as_type<half4>(xv_raw.xy);
            half4 xv_hi = as_type<half4>(xv_raw.zw);
            float4 xv_f32_lo = float4(xv_lo);
            float4 xv_f32_hi = float4(xv_hi);

            float4 w_raw = float4(0.0f);
            if (k + 8 <= K) {
                w_raw = *(const device float4*)(row_ptr + k);
            } else if (k < K) {
                for (uint i = 0; i < 8 && k + i < K; ++i) {
                    ((thread half*)&w_raw)[i] = row_ptr[k + i];
                }
            }

            half4 w_lo = as_type<half4>(w_raw.xy);
            half4 w_hi = as_type<half4>(w_raw.zw);

            if (valid_count > 0) {
                acc += dot(xv_f32_lo, float4(w_lo)) + dot(xv_f32_hi, float4(w_hi));
            }
        }

        return acc;
    }
#endif
    while (k_base + k_chunk_size <= K) {
        uint k = k_base + lane_id * ELEMS_PER_THREAD;
        
        // Vector load 8 halves from input
        float4 xv_raw = *(const device float4*)(input + k);
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);
        
#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
        ulong base_idx = WEIGHT_INDEX(row_idx, k, K, N);
        ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block);
        float scale = Policy::HAS_SCALE
            ? (float)Policy::load_scale(scale_bytes, scale_idx)
            : 1.0f;
        float affine = METALLIC_AFFINE_LOAD(scale_bytes, scale_idx);
        acc += Policy::template dot<8>(data, base_idx, scale, xv_f32_lo, xv_f32_hi)
            + affine * METALLIC_AFFINE_XSUM(xv_f32_lo, xv_f32_hi);
#else
        float w[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            ulong idx = WEIGHT_INDEX(row_idx, k + i, K, N);
            float temp_w[1];
            Policy::template load_weights<1>(data, idx, temp_w);
            w[i] = temp_w[0];
        }

        // Load scale for this block
        ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block);
        float scale = Policy::HAS_SCALE
            ? (float)Policy::load_scale(scale_bytes, scale_idx)
            : 1.0f;
        float affine = METALLIC_AFFINE_LOAD(scale_bytes, scale_idx);

        float4 w_lo = float4(w[0], w[1], w[2], w[3]);
        float4 w_hi = float4(w[4], w[5], w[6], w[7]);
        
        acc += scale * (dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi))
            + affine * METALLIC_AFFINE_XSUM(xv_f32_lo, xv_f32_hi);
#endif
        
        k_base += k_chunk_size;
    }
    
    // Tail: bounds-checked
    if (k_base < K) {
        uint k = k_base + lane_id * ELEMS_PER_THREAD;
        
        float4 xv_raw = float4(0.0f);
        uint valid_count = 0;
        
        if (k + 8 <= K) {
            xv_raw = *(const device float4*)(input + k);
            valid_count = 8;
        } else if (k < K) {
            for (uint i = 0; i < 8 && k + i < K; ++i) {
                ((thread half*)&xv_raw)[i] = input[k + i];
                valid_count++;
            }
        }
        
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);
        
        if (k < K) {
#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
            if (valid_count == 8) {
                ulong base_idx = WEIGHT_INDEX(row_idx, k, K, N);
                ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block);
                float scale = Policy::HAS_SCALE
                    ? (float)Policy::load_scale(scale_bytes, scale_idx)
                    : 1.0f;
                float affine = METALLIC_AFFINE_LOAD(scale_bytes, scale_idx);
                acc += Policy::template dot<8>(data, base_idx, scale, xv_f32_lo, xv_f32_hi)
                    + affine * METALLIC_AFFINE_XSUM(xv_f32_lo, xv_f32_hi);
            } else {
                float w[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                ulong base_idx = WEIGHT_INDEX(row_idx, k, K, N);
                Policy::template load_weights<8>(data, base_idx, w);
                for (uint i = valid_count; i < 8; ++i) w[i] = 0.0f;
                ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block);
                float scale = Policy::HAS_SCALE
                    ? (float)Policy::load_scale(scale_bytes, scale_idx)
                    : 1.0f;
                float affine = METALLIC_AFFINE_LOAD(scale_bytes, scale_idx);
                float4 w_lo = float4(w[0], w[1], w[2], w[3]);
                float4 w_hi = float4(w[4], w[5], w[6], w[7]);
                acc += scale * (dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi))
                    + affine * METALLIC_AFFINE_XSUM(xv_f32_lo, xv_f32_hi);
            }
#else
            float w[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                ulong idx = WEIGHT_INDEX(row_idx, k + i, K, N);
                float temp_w[1];
                Policy::template load_weights<1>(data, idx, temp_w);
                w[i] = temp_w[0];
            }
            for (uint i = valid_count; i < 8; ++i) w[i] = 0.0f;
            ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block);
            float scale = Policy::HAS_SCALE
                ? (float)Policy::load_scale(scale_bytes, scale_idx)
                : 1.0f;
            float affine = METALLIC_AFFINE_LOAD(scale_bytes, scale_idx);
            float4 w_lo = float4(w[0], w[1], w[2], w[3]);
            float4 w_hi = float4(w[4], w[5], w[6], w[7]);
            acc += scale * (dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi))
                + affine * METALLIC_AFFINE_XSUM(xv_f32_lo, xv_f32_hi);
#endif
        }
    }
    
    return acc;
}

/// SIMD reduce sum within warp (32 threads).
ALWAYS_INLINE float gemv_simd_sum(float partial) {
    partial += simd_shuffle_xor(partial, 16);
    partial += simd_shuffle_xor(partial, 8);
    partial += simd_shuffle_xor(partial, 4);
    partial += simd_shuffle_xor(partial, 2);
    partial += simd_shuffle_xor(partial, 1);
    return partial;
}

/// Legacy compatible dot product (for non-vector cases)
template<typename Policy, typename DataPtr, typename InputPtr>
ALWAYS_INLINE float gemv_dot_canonical(
    DataPtr data,
    const device uchar *scale_bytes,
    InputPtr input,
    const uint row_idx,
    const uint k_start,
    const uint k_end,
    const uint K,
    const uint N,
    const uint weights_per_block
) {
    const uint blocks_per_k = (K + weights_per_block - 1) / weights_per_block;
    float acc = 0.0f;
    
    uint k = k_start;
    
    // 4x unrolled main loop
    while (k + 4 <= k_end && k + 4 <= K) {
        uint block_idx = k / weights_per_block;
        half scale = Policy::load_scale(scale_bytes, (ulong)row_idx * blocks_per_k + block_idx);
        float affine = METALLIC_AFFINE_LOAD(scale_bytes, (ulong)row_idx * blocks_per_k + block_idx);
        
        // Load 4 weights
        float w[4];
        
#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
        // Optimized vector load for contiguous K
        ulong base_idx = WEIGHT_INDEX(row_idx, k, K, N);
        Policy::template load_weights<4>(data, base_idx, w);
#else
        // Strided load for non-contiguous K
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            ulong idx = WEIGHT_INDEX(row_idx, k + i, K, N);
            float temp_w[1];
            Policy::template load_weights<1>(data, idx, temp_w);
            w[i] = temp_w[0];
        }
#endif
        
        // Load 4 inputs using generic pointer (supports [] operator)
        float4 x = float4((float)input[k], (float)input[k+1], (float)input[k+2], (float)input[k+3]);
        
        // Dot product with scale
        acc += (w[0] * x.x + w[1] * x.y + w[2] * x.z + w[3] * x.w) * (float)scale
            + affine * (x.x + x.y + x.z + x.w);
        
        k += 4;
    }
    
    // Handle remainder (< 4 elements)
    while (k < k_end && k < K) {
        uint block_idx = k / weights_per_block;
        half scale = Policy::load_scale(scale_bytes, (ulong)row_idx * blocks_per_k + block_idx);
        float affine = METALLIC_AFFINE_LOAD(scale_bytes, (ulong)row_idx * blocks_per_k + block_idx);
        
        float w[1];
        ulong idx = WEIGHT_INDEX(row_idx, k, K, N);
        Policy::template load_weights<1>(data, idx, w);
        
        acc += w[0] * (float)scale * (float)input[k] + affine * (float)input[k];
        ++k;
    }
    
    return acc;
}

ALWAYS_INLINE void gemv_apply_norm_inplace(
    thread half4 &xv_lo,
    thread half4 &xv_hi,
    const device half *gamma,
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
        for (uint i = 0; i < 4u && (k + i) < k_dim; ++i) {
            f_lo[i] *= (float)gamma[k + i];
        }
        if (vec_width == 8u) {
            for (uint i = 0; i < 4u && (k + 4u + i) < k_dim; ++i) {
                f_hi[i] *= (float)gamma[k + 4u + i];
            }
        }
        xv_lo = half4(f_lo);
        xv_hi = half4(f_hi);
    } else if (has_shared_norm) {
        xv_lo *= (half4)inv_rms;
        xv_hi *= (half4)inv_rms;
    }
}

/// Canonical stage entrypoint used by derive-based Rust stage glue.
template<typename Policy>
ALWAYS_INLINE float run_gemv_canonical_stage(
    const device uchar *weights,
    const device uchar *scale_bytes,
    const device half *input,
    const uint row_idx,
    const uint lane_id,
    const uint batch_idx,
    const uint k_dim,
    const uint n_dim,
    const uint weights_per_block
) {
    const device half* input_row = input + batch_idx * k_dim;
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
    const device half *input,
    const device half *gamma,
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

    if (USE_F16_COLS8) {
#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
        const device half* weights_half = (const device half*)weights;
        const device half* row_ptr = weights_half + (ulong)row_idx * (ulong)k_dim;
        const device half* w_ptr = row_ptr + lane_id * VEC_WIDTH;
        const device half* x_ptr = input + input_row_base + lane_id * VEC_WIDTH;

        uint remaining = k_dim;
        while (remaining >= K_CHUNK_SIZE) {
            float4 xv_raw = *(const device float4*)(x_ptr);
            half4 xv_lo = as_type<half4>(xv_raw.xy);
            half4 xv_hi = as_type<half4>(xv_raw.zw);

            gemv_apply_norm_inplace(xv_lo, xv_hi, gamma, has_gamma, inv_rms, has_shared_norm, k_base + lane_id * VEC_WIDTH, k_dim, VEC_WIDTH);

            float4 w_raw = *(const device float4*)(w_ptr);
            half4 w_lo = as_type<half4>(w_raw.xy);
            half4 w_hi = as_type<half4>(w_raw.zw);
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
                        ((thread half*)&xv_raw)[i] = x_ptr[i];
                    }
                }
            }

            half4 xv_lo = as_type<half4>(xv_raw.xy);
            half4 xv_hi = as_type<half4>(xv_raw.zw);
            gemv_apply_norm_inplace(xv_lo, xv_hi, gamma, has_gamma, inv_rms, has_shared_norm, k_base + lane_off, k_dim, VEC_WIDTH);

            float4 w_raw = float4(0.0f);
            if (valid_count == VEC_WIDTH) {
                w_raw = *(const device float4*)(w_ptr);
            } else if (valid_count > 0u) {
                #pragma unroll
                for (uint i = 0; i < VEC_WIDTH; ++i) {
                    if (i < valid_count) {
                        ((thread half*)&w_raw)[i] = w_ptr[i];
                    }
                }
            }

            half4 w_lo = as_type<half4>(w_raw.xy);
            half4 w_hi = as_type<half4>(w_raw.zw);
            if (valid_count > 0u) {
                acc += dot(float4(xv_lo), float4(w_lo)) + dot(float4(xv_hi), float4(w_hi));
            }
        }

        return acc;
#endif
    }

#if defined(METALLIC_POLICY_HAS_SCALE) && METALLIC_POLICY_HAS_SCALE
    const uint blocks_per_k = (k_dim + weights_per_block - 1u) / weights_per_block;
#else
    const uint blocks_per_k = 0u;
#endif

#if (VEC_WIDTH == 8u)
    threadgroup half x_tile[K_CHUNK_SIZE];
    const bool use_x_tile = ((n_dim & (WARPS_PER_TG - 1u)) == 0u);
#else
    const bool use_x_tile = false;
#endif

    while (k_base + K_CHUNK_SIZE <= k_dim) {
#if (VEC_WIDTH == 8u)
        if (use_x_tile) {
            x_tile[lid_x] = input[input_row_base + k_base + lid_x];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#endif
        const uint k = k_base + lane_id * VEC_WIDTH;
        half4 xv_lo;
        half4 xv_hi;
#if (VEC_WIDTH == 8u)
        if (use_x_tile) {
            const uint base = lane_id * 8u;
            const threadgroup half* x_ptr_tg = x_tile + base;
            xv_lo = *(const threadgroup half4*)(x_ptr_tg + 0u);
            xv_hi = *(const threadgroup half4*)(x_ptr_tg + 4u);
        } else
#endif
        if (VEC_WIDTH == 4u) {
            xv_lo = *(const device half4*)(input + input_row_base + k);
            xv_hi = half4(0.0h);
        } else {
            float4 xv_raw = *(const device float4*)(input + input_row_base + k);
            xv_lo = as_type<half4>(xv_raw.xy);
            xv_hi = as_type<half4>(xv_raw.zw);
        }

        gemv_apply_norm_inplace(xv_lo, xv_hi, gamma, has_gamma, inv_rms, has_shared_norm, k, k_dim, VEC_WIDTH);
        const float4 xv_f32_lo = float4(xv_lo);
        const float4 xv_f32_hi = float4(xv_hi);

#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
#if defined(METALLIC_POLICY_WEIGHTS_FP16) && METALLIC_POLICY_WEIGHTS_FP16
        const device half* weights_row = (const device half*)weights + (ulong)row_idx * (ulong)k_dim;
        acc += Policy::template dot<VEC_WIDTH>(weights_row, (ulong)k, 1.0f, xv_f32_lo, xv_f32_hi);
#else
        const ulong base_idx = WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
        const ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block);
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
        const ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block);
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
        half4 xv_lo = half4(0.0h);
        half4 xv_hi = half4(0.0h);
        if (VEC_WIDTH == 8u) {
            float4 xv_raw = float4(0.0f);
            if (k + 8u <= k_dim) {
                xv_raw = *(const device float4*)(input + input_row_base + k);
            } else if (k < k_dim) {
                for (uint i = 0; i < 8u && k + i < k_dim; ++i) {
                    ((thread half*)&xv_raw)[i] = input[input_row_base + k + i];
                }
            }
            xv_lo = as_type<half4>(xv_raw.xy);
            xv_hi = as_type<half4>(xv_raw.zw);
        } else {
            half x0 = (k + 0u < k_dim) ? input[input_row_base + k + 0u] : half(0.0h);
            half x1 = (k + 1u < k_dim) ? input[input_row_base + k + 1u] : half(0.0h);
            half x2 = (k + 2u < k_dim) ? input[input_row_base + k + 2u] : half(0.0h);
            half x3 = (k + 3u < k_dim) ? input[input_row_base + k + 3u] : half(0.0h);
            xv_lo = half4(x0, x1, x2, x3);
            xv_hi = half4(0.0h);
        }

        gemv_apply_norm_inplace(xv_lo, xv_hi, gamma, has_gamma, inv_rms, has_shared_norm, k, k_dim, VEC_WIDTH);
        const float4 xv_f32_lo = float4(xv_lo);
        const float4 xv_f32_hi = float4(xv_hi);

        if (k + VEC_WIDTH <= k_dim) {
#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
#if defined(METALLIC_POLICY_WEIGHTS_FP16) && METALLIC_POLICY_WEIGHTS_FP16
            const device half* weights_row = (const device half*)weights + (ulong)row_idx * (ulong)k_dim;
            acc += Policy::template dot<VEC_WIDTH>(weights_row, (ulong)k, 1.0f, xv_f32_lo, xv_f32_hi);
#else
            const ulong base_idx = WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
            const ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block);
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
            const ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block);
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
            const ulong scale_idx = (ulong)row_idx * blocks_per_k + (k / weights_per_block);
            const half scale = Policy::load_scale(scale_bytes, scale_idx);
            const float affine = Policy::HAS_AFFINE ? (float)Policy::load_affine(scale_bytes, scale_idx) : 0.0f;
            float acc_tail = 0.0f;
            #pragma unroll
            for (uint i = 0; i < VEC_WIDTH; ++i) {
                if (k + i < k_dim) {
                    const ulong idx = WEIGHT_INDEX(row_idx, k + i, k_dim, n_dim);
                    float temp_w[1];
                    Policy::template load_weights<1>(weights, idx, temp_w);
                    acc_tail += temp_w[0] * scale * (float)input[input_row_base + k + i]
                        + affine * (float)input[input_row_base + k + i];
                }
            }
            acc += acc_tail;
        }
    }

    return acc;
}

/// Thread-per-row scalar dot product helper (used by ScalarDotStage).
template<typename Policy, uint UNROLL>
ALWAYS_INLINE float run_gemv_scalar_dot_stage(
    const device uchar *weights,
    const device uchar *scale_bytes,
    const device half *input,
    const uint row_idx,
    const uint k_dim,
    const uint n_dim,
    const uint weights_per_block
) {
    float acc = 0.0f;
    uint k = 0u;
    const uint blocks_per_k_dim = (k_dim + weights_per_block - 1u) / weights_per_block;

    // Main loop with unrolling.
    while (k + UNROLL <= k_dim) {
        #pragma unroll
        for (uint i = 0u; i < UNROLL; ++i) {
            uint curr_k = k + i;

            float val_x = (float)input[curr_k];

            ulong idx = WEIGHT_INDEX(row_idx, curr_k, k_dim, n_dim);
            float w;
            Policy::template load_weights<1>(weights, idx, &w);

            ulong scale_idx = (ulong)row_idx * blocks_per_k_dim + (curr_k / weights_per_block);
            half scale = Policy::load_scale(scale_bytes, scale_idx);
            float affine = Policy::HAS_AFFINE ? (float)Policy::load_affine(scale_bytes, scale_idx) : 0.0f;

            acc += val_x * (w * (float)scale + affine);
        }
        k += UNROLL;
    }

    // Tail loop.
    while (k < k_dim) {
        float val_x = (float)input[k];
        ulong idx = WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
        float w;
        Policy::template load_weights<1>(weights, idx, &w);

        ulong scale_idx = (ulong)row_idx * blocks_per_k_dim + (k / weights_per_block);
        half scale = Policy::load_scale(scale_bytes, scale_idx);
        float affine = Policy::HAS_AFFINE ? (float)Policy::load_affine(scale_bytes, scale_idx) : 0.0f;

        acc += val_x * (w * (float)scale + affine);
        ++k;
    }

    return acc;
}

/// Apply bias and write output.
ALWAYS_INLINE void gemv_write_output(
    device half *output,
    const device half *bias,
    const uint row_idx,
    const float value,
    const bool has_bias
) {
    float result = value;
    if (has_bias) {
        result += (float)bias[row_idx];
    }
    output[row_idx] = (half)result;
}

#endif // GEMV_V2_METAL_H
