// GEMV Canonical Kernel (k-block-major layout)
// Layout: weights organized in blocks of `weights_per_block` elements.
// NOTE: ALWAYS_INLINE is provided by policies/base.metal, which must be included.

#include <metal_stdlib>
using namespace metal;

// GemvParams struct is injected by Rust's MetalStruct derive

// Canonical-specific dispatch constants
constant uint COLS_PER_TG = 8; // Match Legacy cols8 variant
constant uint WARP_SIZE = 32;
constant uint THREADS_PER_BLOCK_GROUP = 8;
constant uint ELEMS_PER_THREAD = 4;
constant uint BLOCKS_PER_CHUNK = 8;

// Default LOAD_MATRIX fallback removed in favor of template Policy

// ============================================================================
// Epilogue Helper
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

// Core Canonical GEMV implementation matching Legacy SimdGemvPolicyF16Canonical.
template<typename Policy, bool HasBias>
void run_gemv_canonical_core(
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
    
    // Thread subdivision within warp (matches Legacy)
    const uint block_in_group = lane_id / THREADS_PER_BLOCK_GROUP; // 0-3
    const uint sub_lane = lane_id % THREADS_PER_BLOCK_GROUP;       // 0-7
    const uint sub_offset = sub_lane * ELEMS_PER_THREAD;           // 0,4,8,...,28
    
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
    
    const uint weights_per_block = params->weights_per_block;
    const uint K = params->K;
    const uint N = params->N;
    
    // Stride between k-blocks for same column: N * weights_per_block
    const ulong stride_w = (ulong)N * weights_per_block;
    
    // Base pointer for this column's first block
    // Layout: matrix[batch_offset + col * weights_per_block + block_idx * stride_w + elem_in_block]
    const ulong col_base = matrix_batch_offset + (ulong)logical_col * weights_per_block;
    
    // Total number of k-blocks
    const uint total_blocks = (K + weights_per_block - 1u) / weights_per_block;
    
    // Accumulator
    float acc = 0.0f;
    
    uint k_chunk_base = 0;
    
    // Fast Path loop: 8 blocks per chunk
    while (k_chunk_base + BLOCKS_PER_CHUNK <= total_blocks) {
        // Each thread processes 2 blocks (block_in_group and block_in_group+4)
        uint block_idx_0 = k_chunk_base + block_in_group;
        uint block_idx_1 = block_idx_0 + 4u;
        
        // Load vector elements for block 0
        uint k_idx_0 = block_idx_0 * weights_per_block + sub_offset;
        float4 xv0 = float4(*(const device half4*)(curr_x + k_idx_0));
        
        // Load vector elements for block 1  
        uint k_idx_1 = block_idx_1 * weights_per_block + sub_offset;
        float4 xv1 = float4(*(const device half4*)(curr_x + k_idx_1));
        
        // Load weight elements for block 0
        float w0_arr[4];
        Policy::template load_weights<4>(matrix, col_base + block_idx_0 * stride_w + sub_offset, w0_arr);
        float4 w0 = float4(w0_arr[0], w0_arr[1], w0_arr[2], w0_arr[3]);
        
        // Load weight elements for block 1
        float w1_arr[4];
        Policy::template load_weights<4>(matrix, col_base + block_idx_1 * stride_w + sub_offset, w1_arr);
        float4 w1 = float4(w1_arr[0], w1_arr[1], w1_arr[2], w1_arr[3]);
        
        // Load scales
        half scale0 = Policy::load_scale(scale_bytes, (ulong)logical_col * total_blocks + block_idx_0);
        half scale1 = Policy::load_scale(scale_bytes, (ulong)logical_col * total_blocks + block_idx_1);
        
        // Compute partial dot products with block-level scaling
        float partial = 0.0f;
        partial += (float)scale0 * dot(xv0, w0);
        partial += (float)scale1 * dot(xv1, w1);
        
        // Reduce within sub-group of 8 threads (matches Legacy)
        partial += simd_shuffle_xor(partial, 4u);
        partial += simd_shuffle_xor(partial, 2u);
        partial += simd_shuffle_xor(partial, 1u);
        
        // Match Legacy: only sub_lane 0 returns non-zero, others return 0.0f
        // All lanes accumulate, but only sub_lane==0 contributes non-zero values
        float contribution = (sub_lane == 0u) ? partial : 0.0f;
        acc += contribution;
        
        k_chunk_base += BLOCKS_PER_CHUNK;
    }
    
    // Tail Loop: handle remaining blocks with bounds checking
    while (k_chunk_base < total_blocks) {
        uint block_idx_0 = k_chunk_base + block_in_group;
        uint block_idx_1 = block_idx_0 + 4u;
        
        float partial = 0.0f;
        
        // Process block 0 if valid
        if (block_idx_0 < total_blocks) {
            uint k_idx_0 = block_idx_0 * weights_per_block + sub_offset;
            float4 xv0 = float4(0.0f);
            if (k_idx_0 + 4u <= K) {
                xv0 = float4(*(const device half4*)(curr_x + k_idx_0));
            }
            
            float w0_arr[4];
            Policy::template load_weights<4>(matrix, col_base + block_idx_0 * stride_w + sub_offset, w0_arr);
            float4 w0 = float4(w0_arr[0], w0_arr[1], w0_arr[2], w0_arr[3]);
            
            half scale0 = Policy::load_scale(scale_bytes, (ulong)logical_col * total_blocks + block_idx_0);
            partial += (float)scale0 * dot(xv0, w0);
        }
        
        // Process block 1 if valid
        if (block_idx_1 < total_blocks) {
            uint k_idx_1 = block_idx_1 * weights_per_block + sub_offset;
            float4 xv1 = float4(0.0f);
            if (k_idx_1 + 4u <= K) {
                xv1 = float4(*(const device half4*)(curr_x + k_idx_1));
            }
            
            float w1_arr[4];
            Policy::template load_weights<4>(matrix, col_base + block_idx_1 * stride_w + sub_offset, w1_arr);
            float4 w1 = float4(w1_arr[0], w1_arr[1], w1_arr[2], w1_arr[3]);
            
            half scale1 = Policy::load_scale(scale_bytes, (ulong)logical_col * total_blocks + block_idx_1);
            partial += (float)scale1 * dot(xv1, w1);
        }
        
        // Reduce within sub-group of 8 threads (matches Legacy)
        partial += simd_shuffle_xor(partial, 4u);
        partial += simd_shuffle_xor(partial, 2u);
        partial += simd_shuffle_xor(partial, 1u);
        
        // Match Legacy: only sub_lane 0 returns non-zero, others return 0.0f
        float contribution = (sub_lane == 0u) ? partial : 0.0f;
        acc += contribution;
        
        k_chunk_base += BLOCKS_PER_CHUNK;
    }
    
    // Final Warp Reduction (ALL lanes participate, matching Legacy DefaultEpilogue)
    // Only lanes 0, 8, 16, 24 have non-zero acc values
    float final_sum = acc;
    final_sum += simd_shuffle_xor(final_sum, 16u);
    final_sum += simd_shuffle_xor(final_sum, 8u);
    final_sum += simd_shuffle_xor(final_sum, 4u);
    final_sum += simd_shuffle_xor(final_sum, 2u);
    final_sum += simd_shuffle_xor(final_sum, 1u);
    
    // Epilogue
    if (lane_id == 0) {
        float result = gemv_epilogue<HasBias>(
            final_sum, bias, curr_residual, alpha, beta, logical_col
        );
        curr_y[logical_col] = (half)result;
    }
}

// ============================================================================
// Entry Points
// ============================================================================

#ifndef FUSED_KERNEL
[[kernel]] void gemv_canonical_f16(
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

    // Standalone kernel uses PolicyF16
    using Policy = PolicyF16;

    if (has_bias != 0) {
        run_gemv_canonical_core<Policy, true>(
            matrix, vector_x, result_y, params, bias, residual, alpha, beta, gid, lid, scale_bytes
        );
    } else {
        run_gemv_canonical_core<Policy, false>(
            matrix, vector_x, result_y, params, bias, residual, alpha, beta, gid, lid, scale_bytes
        );
    }
}
#endif
