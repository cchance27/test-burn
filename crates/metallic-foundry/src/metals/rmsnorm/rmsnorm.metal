#ifndef RMSNORM_METAL_H
#define RMSNORM_METAL_H

// NOTE: ALWAYS_INLINE is provided by policies/base.metal, which must be included.
#include <metal_stdlib>
using namespace metal;

// constant float EPS = 1e-6f; // Removed in favor of RmsNormParams.epsilon
constant uint THREADS_PER_ROW = 256;

// RmsNormParams struct is injected by Foundry via struct_defs()

// ============================================================================
// RMSNorm helper: compute inv_rms with parallel reduction
// Pattern from legacy gemv_compute_inv_rms - threadgroup declared in kernel
// ============================================================================

template<typename Policy>
ALWAYS_INLINE float rmsnorm_compute_inv_rms(
    const device uchar *input,
    const device uchar *scale_bytes,
    const uint feature_dim,
    const uint row_idx,
    const uint lane_id,
    const uint warp_id,
    const float epsilon,
    threadgroup float *tg_inv_rms
) {
    // if (feature_dim == 0u) { // Restore original logic below

    if (feature_dim == 0u) {
        if (warp_id == 0u && lane_id == 0u) {
            tg_inv_rms[0] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return tg_inv_rms[0];
    }

    const uint row_start = row_idx * feature_dim;
    const uint num_blocks = (feature_dim + 7) / 8;

    // Each thread in warp 0 processes multiple 8-element blocks
    float sum_sq = 0.0f;
    if (warp_id == 0u) {
        uint block = lane_id;
        while (block < num_blocks) {
            uint k = block * 8;
            uint valid_count = min(8u, feature_dim - k);
            
            float vals[8];
            Policy::template load_weights<8>(input, (ulong)(row_start + k), vals);
            half scale = Policy::load_scale(scale_bytes, (ulong)(row_idx * num_blocks + block));
            
            for (uint i = 0; i < valid_count; ++i) {
                float v = vals[i] * (float)scale;
                sum_sq += v * v;
            }
            block += 32; // Stride by warp size
        }
        
        // Warp reduction using simd_sum
        sum_sq = simd_sum(sum_sq);
        
        if (lane_id == 0u) {
            tg_inv_rms[0] = rsqrt(sum_sq / (float)feature_dim + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return tg_inv_rms[0];
}

// ============================================================================
// Core RMSNorm - called after inv_rms is computed
// ============================================================================

template<typename Policy>
void rmsnorm_apply(
    const device uchar *input,
    device half *output,
    const device half *gamma,
    const device uchar *scale_bytes,
    const uint feature_dim,
    const uint row_idx,
    const float inv_rms,
    const uint thread_id
) {
    const uint row_start = row_idx * feature_dim;
    const uint num_blocks = (feature_dim + 7) / 8;
    
    // Each thread writes multiple 8-element blocks
    uint block = thread_id;
    while (block < num_blocks) {
        uint k = block * 8;
        uint valid_count = min(8u, feature_dim - k);
        
        float vals[8];
        Policy::template load_weights<8>(input, (ulong)(row_start + k), vals);
        half scale = Policy::load_scale(scale_bytes, (ulong)(row_idx * num_blocks + block));
        
        for (uint i = 0; i < valid_count; ++i) {
            float v = vals[i] * (float)scale;
            float gamma_val = (float)gamma[k + i];
            output[row_start + k + i] = (half)(v * inv_rms * gamma_val);
        }
        block += THREADS_PER_ROW;
    }
}

// ============================================================================
// Core template - orchestrates reduction then apply
// ============================================================================

template<typename Policy>
void run_rmsnorm_core(
    const device uchar *input,
    device half *output,
    const device half *gamma,
    constant RmsNormParams *params,
    const device uchar *scale_bytes,
    uint3 gid,
    uint3 lid,
    threadgroup float *tg_inv_rms
) {
    const uint row_idx = gid.x;
    const uint thread_id = lid.x;
    const uint feature_dim = params->feature_dim;
    const uint total_rows = params->total_elements / feature_dim;
    
    if (row_idx >= total_rows) return;
    
    const uint lane_id = thread_id & 31u;
    const uint warp_id = thread_id / 32u;
    
    // Phase 1: Compute inv_rms using parallel reduction
    float inv_rms = rmsnorm_compute_inv_rms<Policy>(
        input, scale_bytes, feature_dim, row_idx, lane_id, warp_id, params->epsilon, tg_inv_rms
    );
    
    // Phase 2: Apply normalization
    rmsnorm_apply<Policy>(
        input, output, gamma, scale_bytes, feature_dim, row_idx, inv_rms, thread_id
    );
}

// ============================================================================
// Kernel entrypoints (required for standalone `Kernel` dispatch)
// ============================================================================

kernel void rmsnorm_kernel_f16(
    const device uchar *input [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    device half *output [[buffer(2)]],
    const device half *gamma [[buffer(3)]],
    constant RmsNormParams *params [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    (void)scale_bytes;
    (void)tptg;

    const uint row_idx = gid.x;
    const uint thread_id = lid.x;
    const uint feature_dim = params->feature_dim;
    const uint total_rows = params->total_elements / feature_dim;

    if (row_idx >= total_rows) return;
    if (feature_dim == 0u) return;

    const uint row_start = row_idx * feature_dim;
    const uint num_blocks = (feature_dim + 7u) / 8u;
    const device half *in_h = (const device half *)input;

    // Phase 1: Compute inv_rms using warp 0.
    threadgroup float tg_inv_rms_storage;
    float sum_sq = 0.0f;
    const uint lane_id = thread_id & 31u;
    const uint warp_id = thread_id / 32u;
    if (warp_id == 0u) {
        uint block = lane_id;
        while (block < num_blocks) {
            const uint k = block * 8u;
            const uint valid_count = min(8u, feature_dim - k);
            const uint base = row_start + k;

            if (valid_count == 8u) {
                packed_half4 pv0 = *(const device packed_half4 *)(in_h + base);
                packed_half4 pv1 = *(const device packed_half4 *)(in_h + base + 4u);
                float4 f0 = float4(half4(pv0));
                float4 f1 = float4(half4(pv1));
                sum_sq += dot(f0, f0) + dot(f1, f1);
            } else {
                for (uint i = 0; i < valid_count; ++i) {
                    float v = (float)in_h[base + i];
                    sum_sq += v * v;
                }
            }

            block += 32u;
        }

        sum_sq = simd_sum(sum_sq);
        if (lane_id == 0u) {
            tg_inv_rms_storage = rsqrt(sum_sq / (float)feature_dim + params->epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = tg_inv_rms_storage;

    // Phase 2: Apply normalization.
    uint block = thread_id;
    while (block < num_blocks) {
        const uint k = block * 8u;
        const uint valid_count = min(8u, feature_dim - k);
        const uint base = row_start + k;

        if (valid_count == 8u) {
            packed_half4 pv0 = *(const device packed_half4 *)(in_h + base);
            packed_half4 pv1 = *(const device packed_half4 *)(in_h + base + 4u);
            float4 f0 = float4(half4(pv0));
            float4 f1 = float4(half4(pv1));

            const device half *g_ptr = gamma + k;
            packed_half4 pg0 = *(const device packed_half4 *)(g_ptr);
            packed_half4 pg1 = *(const device packed_half4 *)(g_ptr + 4u);
            float4 g0 = float4(half4(pg0));
            float4 g1 = float4(half4(pg1));

            float4 o0 = f0 * inv_rms * g0;
            float4 o1 = f1 * inv_rms * g1;

            *(device packed_half4 *)(output + base) = packed_half4(half4(o0));
            *(device packed_half4 *)(output + base + 4u) = packed_half4(half4(o1));
        } else {
            for (uint i = 0; i < valid_count; ++i) {
                float v = (float)in_h[base + i];
                float g = (float)gamma[k + i];
                output[base + i] = (half)(v * inv_rms * g);
            }
        }

        block += THREADS_PER_ROW;
    }
}

#endif // RMSNORM_METAL_H
