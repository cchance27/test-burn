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
