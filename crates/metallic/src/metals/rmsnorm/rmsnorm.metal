#include <metal_stdlib>
using namespace metal;

#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

constant float EPS = 1e-6f;
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
    threadgroup float *tg_inv_rms
) {
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
            tg_inv_rms[0] = rsqrt(sum_sq / (float)feature_dim + EPS);
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
        input, scale_bytes, feature_dim, row_idx, lane_id, warp_id, tg_inv_rms
    );
    
    // Phase 2: Apply normalization
    rmsnorm_apply<Policy>(
        input, output, gamma, scale_bytes, feature_dim, row_idx, inv_rms, thread_id
    );
}

// ============================================================================
// Entry Points (for standalone non-fused usage)
// ============================================================================

#ifndef FUSED_KERNEL
#ifndef POLICY_F16_DEFINED
#define POLICY_F16_DEFINED
struct PolicyF16 {
    static ALWAYS_INLINE half load_scale(const device uchar *scales, ulong block_idx) {
        return 1.0h;
    }

    template<int N>
    static ALWAYS_INLINE void load_weights(
        const device uchar *ptr, 
        ulong offset, 
        thread float results[N]
    ) {
        const device half *w_ptr = (const device half *)ptr;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            results[i] = (float)w_ptr[offset + i];
        }
    }
};
#endif // POLICY_F16_DEFINED

[[kernel]] void rmsnorm_kernel_f16(
    const device uchar *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    const device half *gamma [[buffer(2)]],
    constant RmsNormParams *params [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    // Threadgroup memory declared in kernel, passed to template
    threadgroup float tg_inv_rms;
    run_rmsnorm_core<PolicyF16>(input, output, gamma, params, input, gid, lid, &tg_inv_rms);
}
#endif
