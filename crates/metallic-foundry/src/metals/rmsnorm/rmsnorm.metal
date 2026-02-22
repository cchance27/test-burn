#ifndef RMSNORM_METAL_H
#define RMSNORM_METAL_H

// NOTE: ALWAYS_INLINE is provided by policies/base.metal, which must be included.
#include <metal_stdlib>
using namespace metal;

// constant uint THREADS_PER_ROW = 256; // Removed in favor of tptg.x

// RmsNormParams struct is injected by Foundry via struct_defs()

// ============================================================================
// RMSNorm helper: compute inv_rms with parallel reduction.
// Threadgroup storage is provided by caller; initialization and barrier are handled here.
// ============================================================================

template<typename Policy>
ALWAYS_INLINE float rmsnorm_compute_inv_rms(
    const device uchar *input,
    const device uchar *scale_bytes,
    const uint feature_dim,
    const uint row_idx,
    const uint lane_id,
    const uint warp_id,
    const uint thread_id,
    const uint threads_per_row,
    const float epsilon,
    threadgroup float *tg_reduction_shared
) {
    // Phase 0: Initialize shared memory for this threadgroup.
    // We use index 0 for the final total sum, and indices 1..32 for warp partial sums.
    if (thread_id == 0u) {
        tg_reduction_shared[0] = 0.0f;
    }
    if (lane_id == 0u) {
        tg_reduction_shared[warp_id + 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (feature_dim == 0u) {
        return 0.0f;
    }

    const uint num_blocks = (feature_dim + 7) / 8;

    // Phase 1: All threads in the threadgroup compute local sum of squares
    float local_sum_sq = 0.0f;
    for (uint block = thread_id; block < num_blocks; block += threads_per_row) {
        uint k = block * 8;
        uint valid_count = min(8u, feature_dim - k);
        
        float vals[8] = {0.0f};
        // Load weights individually to handle bounds
        for (uint i = 0; i < valid_count; ++i) {
            float temp[1];
            Policy::template load_weights<1>(input, (ulong)(row_idx * feature_dim + k + i), temp);
            vals[i] = temp[0];
        }
        
        // Block-based scale/affine
        ulong scale_idx = (ulong)row_idx * num_blocks + block;
        half scale = Policy::load_scale(scale_bytes, scale_idx);
#if defined(METALLIC_POLICY_HAS_AFFINE) && METALLIC_POLICY_HAS_AFFINE
        half affine = Policy::load_affine(scale_bytes, scale_idx);
#else
        half affine = 0.0h;
#endif
        
        for (uint i = 0; i < valid_count; ++i) {
            float v = vals[i] * (float)scale + (float)affine;
            local_sum_sq += v * v;
        }
    }
    
    // Phase 2: Warp reduction
    float warp_sum = simd_sum(local_sum_sq);
    
    // Phase 3: Threadgroup reduction using shared memory
    if (lane_id == 0u) {
        tg_reduction_shared[warp_id + 1] = warp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Warp 0 finalizes the reduction
    if (warp_id == 0u) {
        const uint num_warps = (threads_per_row + 31u) / 32u;
        float final_sum = (lane_id < num_warps) ? tg_reduction_shared[lane_id + 1] : 0.0f;
        final_sum = simd_sum(final_sum);
        if (lane_id == 0u) {
            tg_reduction_shared[0] = final_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    return rsqrt(tg_reduction_shared[0] / (float)feature_dim + epsilon);
}

#define RMSNORM_COMPUTE_INV_RMS_STAGE(OUT_VAR, POLICY, INPUT, SCALE_BYTES, FEATURE_DIM, ROW_IDX, LANE_ID, WARP_ID, THREAD_ID, TPTG, EPSILON) \
    threadgroup float __rmsnorm_tg_reduction_storage[33];                                                                  \
    float OUT_VAR = rmsnorm_compute_inv_rms<POLICY>(                                                                       \
        (const device uchar*)INPUT,                                                                                         \
        (const device uchar*)SCALE_BYTES,                                                                                   \
        FEATURE_DIM,                                                                                                        \
        ROW_IDX,                                                                                                            \
        LANE_ID,                                                                                                            \
        WARP_ID,                                                                                                            \
        THREAD_ID,                                                                                                          \
        TPTG,                                                                                                               \
        EPSILON,                                                                                                            \
        __rmsnorm_tg_reduction_storage                                                                                      \
    )

#define RMSNORM_RUN_CORE_STAGE(POLICY, INPUT, OUTPUT, GAMMA, PARAMS, SCALE_BYTES, GID, LID, TPTG) \
    do {                                                                                       \
        threadgroup float __rmsnorm_tg_reduction[33];                                          \
        run_rmsnorm_core<POLICY>(                                                              \
            (const device uchar*)INPUT,                                                        \
            (device half*)OUTPUT,                                                              \
            (const device half*)GAMMA,                                                         \
            PARAMS,                                                                            \
            (const device uchar*)SCALE_BYTES,                                                  \
            GID,                                                                               \
            LID,                                                                               \
            TPTG,                                                                              \
            __rmsnorm_tg_reduction                                                             \
        );                                                                                     \
    } while (0)

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
    const uint thread_id,
    const uint threads_per_row
) {
    const uint num_blocks = (feature_dim + 7) / 8;
    
    // Each thread writes multiple 8-element blocks
    for (uint block = thread_id; block < num_blocks; block += threads_per_row) {
        uint k = block * 8;
        uint valid_count = min(8u, feature_dim - k);
        
        float vals[8] = {0.0f};
        for (uint i = 0; i < valid_count; ++i) {
            float temp[1];
            Policy::template load_weights<1>(input, (ulong)(row_idx * feature_dim + k + i), temp);
            vals[i] = temp[0];
        }
        
        ulong scale_idx = (ulong)row_idx * num_blocks + block;
        half scale = Policy::load_scale(scale_bytes, scale_idx);
#if defined(METALLIC_POLICY_HAS_AFFINE) && METALLIC_POLICY_HAS_AFFINE
        half affine = Policy::load_affine(scale_bytes, scale_idx);
#else
        half affine = 0.0h;
#endif
        
        for (uint i = 0; i < valid_count; ++i) {
            float v = vals[i] * (float)scale + (float)affine;
            float gamma_val = (float)gamma[k + i];
            output[row_idx * feature_dim + k + i] = (half)(v * inv_rms * gamma_val);
        }
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
    uint3 tptg,
    threadgroup float *tg_reduction_shared
) {
    const uint row_idx = gid.x;
    const uint thread_id = lid.x;
    const uint threads_per_row = tptg.x;
    const uint feature_dim = params->feature_dim;
    const uint total_rows = params->total_elements / feature_dim;
    
    if (row_idx >= total_rows) return;
    
    const uint lane_id = thread_id & 31u;
    const uint warp_id = thread_id / 32u;
    
    // Phase 1: Compute inv_rms using parallel reduction
    float inv_rms = rmsnorm_compute_inv_rms<Policy>(
        input, scale_bytes, feature_dim, row_idx, lane_id, warp_id, thread_id, threads_per_row, params->epsilon, tg_reduction_shared
    );
    
    // Phase 2: Apply normalization
    rmsnorm_apply<Policy>(
        input, output, gamma, scale_bytes, feature_dim, row_idx, inv_rms, thread_id, threads_per_row
    );
}

#endif // RMSNORM_METAL_H
