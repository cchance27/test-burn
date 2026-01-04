// GEMV Row-Major Kernel (strided K)
// Layout: element (col, k) at matrix[batch_offset + col + k * stride_w].
// NOTE: ALWAYS_INLINE is provided by policies/base.metal, which must be included.

#include <metal_stdlib>
using namespace metal;

// GemvParams struct is injected by Rust's MetalStruct derive

// Row-Major dispatch constants
constant uint COLS_PER_TG = 8;
constant uint WARP_SIZE = 32;
constant uint ELEMS_PER_THREAD = 8;
constant uint K_CHUNK_SIZE = 256;    // WARP_SIZE * ELEMS_PER_THREAD

// Default LOAD_MATRIX fallback removed in favor of template Policy

// ============================================================================
// Epilogue Helper (copied from Legacy for parity)
// ============================================================================

template<bool HasBias>
ALWAYS_INLINE float gemv_epilogue(
    float acc,
    const device half *bias,
    const device half *residual,
    float alpha,
    float beta,
    uint col
) {
    float result = acc;
    
    if (HasBias && bias) {
        result += (float)bias[col];
    }
    
    result = alpha * result;
    
    if (residual) {
        result += beta * (float)residual[col];
    }
    
    return result;
}

// Core Row-Major GEMV implementation.
template<typename Policy, bool HasBias>
void run_gemv_row_major_core(
    const device uchar *matrix,
    const device half *vector_x,
    device half *result_y,
    constant GemvParams *params,
    const device half *bias,
    const device half *residual,
    float alpha,
    float beta,
    uint3 gid,
    uint3 lid,
    const device uchar *scale_bytes
) {
    const uint lane_id = lid.x & 31u;
    const uint warp_id = lid.x / 32u;
    
    // Batch handling
    const uint batch_idx = gid.z;
    if (batch_idx >= params->batch) return;
    
    // Batch offsets
    const ulong matrix_batch_offset = (ulong)batch_idx * params->stride_a;
    const device half *curr_x = vector_x + (ulong)batch_idx * params->stride_x;
    device half *curr_y = result_y + (ulong)batch_idx * params->stride_y;
    const device half *curr_residual = residual ? (residual + (ulong)batch_idx * params->stride_y) : nullptr;
    
    // Each warp processes one output column
    const uint logical_col = gid.x * COLS_PER_TG + warp_id;
    if (logical_col >= params->N) return;
    
    // Thread's starting position within the K dimension
    const uint k_thread_offset = lane_id * ELEMS_PER_THREAD;
    
    // Stride for K dimension (row-strided: col + k * stride_w)
    const uint stride = params->stride_w;
    
    // Accumulator
    float acc = 0.0f;
    
    uint k_base = 0;
    const uint K = params->K;
    
    // Fast Path loop: no bounds checks
    const uint wpb = params->weights_per_block;
    const uint blocks_per_k = params->blocks_per_k;

    while (k_base + K_CHUNK_SIZE <= K) {
        uint k = k_base + k_thread_offset;
        
        // Load 8 contiguous halves from vector_x
        float4 xv_raw = *(const device float4*)(curr_x + k);
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);
        
        // Load 8 elements from matrix via Policy
        float w[8];
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint curr_k = k + i;
            ulong curr_offset = matrix_batch_offset + logical_col + (ulong)curr_k * stride;
            
            float w_val[1];
            Policy::template load_weights<1>(matrix, curr_offset, w_val);
            
            // For strided access, we must load scale per element unless we know they share a block.
            // Assuming block-size is along the dot (K) dimension.
            half scale = Policy::load_scale(scale_bytes, (ulong)logical_col * blocks_per_k + (curr_k / wpb));
            w[i] = w_val[0] * (float)scale;
        }
        
        float4 w_lo = float4(w[0], w[1], w[2], w[3]);
        float4 w_hi = float4(w[4], w[5], w[6], w[7]);
        
        acc += dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi);
        
        k_base += K_CHUNK_SIZE;
    }
    
    // Tail Loop: bounds-checked
    while (k_base < K) {
        uint k = k_base + k_thread_offset;
        
        // Safe load from vector_x
        float4 xv_raw = float4(0.0f);
        uint valid_count = 0;
        
        if (k + 8 <= K) {
            xv_raw = *(const device float4*)(curr_x + k);
            valid_count = 8;
        } else {
            for (uint i = 0; i < 8 && k + i < K; ++i) {
                ((thread half*)&xv_raw)[i] = curr_x[k + i];
                valid_count++;
            }
        }
        
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);
        
        // Safe load from matrix via Policy
        float w[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint i = 0; i < valid_count; ++i) {
            uint curr_k = k + i;
            ulong curr_offset = matrix_batch_offset + logical_col + (ulong)curr_k * stride;

            float w_val[1];
            Policy::template load_weights<1>(matrix, curr_offset, w_val);
            half scale = Policy::load_scale(scale_bytes, (ulong)logical_col * blocks_per_k + (curr_k / wpb));
            w[i] = w_val[0] * (float)scale;
        }
        
        float4 w_lo = float4(w[0], w[1], w[2], w[3]);
        float4 w_hi = float4(w[4], w[5], w[6], w[7]);
        
        acc += dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi);
        
        k_base += K_CHUNK_SIZE;
    }
    
    // SIMD Reduction
    acc += simd_shuffle_xor(acc, 16u);
    acc += simd_shuffle_xor(acc, 8u);
    acc += simd_shuffle_xor(acc, 4u);
    acc += simd_shuffle_xor(acc, 2u);
    acc += simd_shuffle_xor(acc, 1u);
    
    // Epilogue
    if (lane_id == 0) {
        float result = gemv_epilogue<HasBias>(
            acc, bias, curr_residual, alpha, beta, logical_col
        );
        curr_y[logical_col] = (half)result;
    }
}

// ============================================================================
// Entry Points
// ============================================================================

#ifndef FUSED_KERNEL
[[kernel]] void gemv_row_major_f16(
    const device uchar *matrix [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    const device half *vector_x [[buffer(2)]],
    device half *result_y [[buffer(3)]],
    constant GemvParams *params [[buffer(4)]],
    const device half *bias [[buffer(5)]],
    const device half *residual [[buffer(6)]],
    constant float &alpha [[buffer(7)]],
    constant float &beta [[buffer(8)]],
    constant uint &has_bias [[buffer(9)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint batch_idx = gid.z;
    if (batch_idx >= params->batch) return;

    const ulong matrix_batch_offset = (ulong)batch_idx * params->stride_a;
    const device half *curr_x = vector_x + (ulong)batch_idx * params->stride_x;
    device half *curr_y = result_y + (ulong)batch_idx * params->stride_y;
    const device half *curr_bias = has_bias ? bias : nullptr;
    const device half *curr_residual = beta != 0.0f ? (residual + (ulong)batch_idx * params->stride_y) : nullptr;

    // Standalone kernel uses PolicyF16
    using Policy = PolicyF16;

    if (has_bias != 0) {
        run_gemv_row_major_core<Policy, true>(
            matrix, vector_x, result_y, params, curr_bias, residual, alpha, beta, gid, lid, scale_bytes
        );
    } else {
        run_gemv_row_major_core<Policy, false>(
            matrix, vector_x, result_y, params, curr_bias, residual, alpha, beta, gid, lid, scale_bytes
        );
    }

}
#endif
