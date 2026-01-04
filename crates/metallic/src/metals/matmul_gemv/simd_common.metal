#ifndef MATMUL_GEMV_SIMD_COMMON_METAL
#define MATMUL_GEMV_SIMD_COMMON_METAL

// Common SIMD GEMV template infrastructure used by Foundry.
//
// This file intentionally contains:
// - Struct defs needed for the SIMD GEMV template
// - gemv_compute_inv_rms helper (for fused RMSNorm policies)
// - Default epilogue + run_simd_gemv_template generic loop
//
// Policy implementations (F16/Q8/etc) live in separate includes to keep quant/policy
// logic centralized and avoid editing this file when adding new quantization schemes.
// NOTE: ALWAYS_INLINE is provided by policies/base.metal, which must be included.

#include <metal_stdlib>
using namespace metal;

constant float GEMV_RMSNORM_EPS = 1e-6f;

// NOTE: GemvParams is injected by GemvSimdConfig via struct_defs()
// QkvFusedParams and Q2FusedParams are also injected via MetalStruct derive (struct_defs_type in GemvSimdConfig)

ALWAYS_INLINE float gemv_compute_inv_rms(
    const device half *vector_x,
    const uint K,
    const uint lane_id,
    const uint warp_id,
    threadgroup float *tg_inv_rms,
    float epsilon
) {
    if (K == 0u) {
        if (warp_id == 0u && lane_id == 0u) {
            tg_inv_rms[0] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return tg_inv_rms[0];
    }

    float sum = 0.0f;
    if (warp_id == 0u) {
        uint k = lane_id * 4u;
        const uint stride = 32u * 4u;
        for (; k + 4u <= K; k += stride) {
            half4 hv = *(const device half4 *)(vector_x + k);
            float4 fv = float4(hv);
            sum += dot(fv, fv);
        }
        for (; k < K; k += 32u) {
            float v = (float)vector_x[k];
            sum += v * v;
        }
        sum = simd_sum(sum);
        if (lane_id == 0u) {
            tg_inv_rms[0] = rsqrt(sum / (float)K + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return tg_inv_rms[0];
}

ALWAYS_INLINE float gemv_compute_inv_rms(
    const device half *vector_x,
    const uint K,
    const uint lane_id,
    const uint warp_id,
    threadgroup float *tg_inv_rms
) {
    return gemv_compute_inv_rms(vector_x, K, lane_id, warp_id, tg_inv_rms, GEMV_RMSNORM_EPS);
}

// =================================================================================================
// Epilogue Policies
// =================================================================================================

struct DefaultEpilogue {
    template<uint HEADS>
    static void apply(
        float acc[HEADS],
        uint lane_id,
        uint logical_col,
        const uint N[HEADS],
        const device half *bias[HEADS],
        const uint has_bias_flags[HEADS],
        const float alpha,
        const float beta,
        const device half *residual,
        device half *result_y[HEADS]
    ) {
        for (uint h = 0; h < HEADS; ++h) {
            if (logical_col >= N[h]) continue;

            float val = acc[h];
            val += simd_shuffle_xor(val, 16u);
            val += simd_shuffle_xor(val, 8u);
            val += simd_shuffle_xor(val, 4u);
            val += simd_shuffle_xor(val, 2u);
            val += simd_shuffle_xor(val, 1u);

            if (lane_id == 0u) {
                float val_acc = val;
                if (has_bias_flags[h] && bias[h]) {
                    val_acc += (float)bias[h][logical_col];
                }

                const device half *res_ptr = (HEADS == 1u) ? residual : (const device half *)nullptr;
                float res = 0.0f;
                if (beta != 0.0f && res_ptr) {
                    res = (float)res_ptr[logical_col];
                }

                const float out = alpha * val_acc + beta * res;
                result_y[h][logical_col] = (half)out;
            }
        }
    }
};

// =================================================================================================
// Generic SIMD GEMV Template
// =================================================================================================

template <typename Policy, uint HEADS, uint COLS_PER_TG, bool HasBias, typename EpiloguePolicy = DefaultEpilogue>
void run_simd_gemv_template(
    typename Policy::Params params,
    const device half *vector_x,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    GemvParams gemv_params
) {
    const uint lane_id = lid.x & 31u;
    const uint warp_id = lid.x / 32u;

    const uint batch_idx = gid.z;
    if (batch_idx >= gemv_params.batch) return;

    const device half *curr_x = vector_x + (ulong)batch_idx * gemv_params.stride_x;

    device half *curr_y[HEADS];
    for (uint h = 0; h < HEADS; ++h) {
        curr_y[h] = result_y[h] + (ulong)batch_idx * gemv_params.stride_y;
    }

    const uint logical_col = gid.x * COLS_PER_TG + warp_id;

    bool head_active[HEADS];
    for (uint h = 0; h < HEADS; ++h) {
        head_active[h] = (logical_col < N[h]);
    }

    Policy policy;
    policy.template init<HEADS>(params, gid, lid, logical_col, K, N, gemv_params, batch_idx);

    float acc[HEADS];
    for (uint h = 0; h < HEADS; ++h) acc[h] = 0.0f;

    uint k_base = 0u;
    while (k_base + Policy::FAST_K_CHUNK_SIZE <= K) {
        policy.load_x_fast(curr_x, k_base);
        for (uint h = 0; h < HEADS; ++h) {
            if (head_active[h]) {
                acc[h] += policy.compute_dot(h, true);
            }
        }
        policy.template advance_pointers<HEADS>(Policy::FAST_K_CHUNK_SIZE);
        k_base += Policy::FAST_K_CHUNK_SIZE;
    }

    while (k_base < K) {
        policy.load_x_safe(curr_x, k_base, K);
        for (uint h = 0; h < HEADS; ++h) {
            if (head_active[h]) {
                acc[h] += policy.compute_dot(h, false);
            }
        }
        policy.template advance_pointers<HEADS>(Policy::SAFE_K_CHUNK_SIZE);
        k_base += Policy::SAFE_K_CHUNK_SIZE;
    }

    const device half *curr_residual = nullptr;
    if (residual) {
        curr_residual = residual + (ulong)batch_idx * gemv_params.stride_y;
    }

    EpiloguePolicy::template apply<HEADS>(
        acc, lane_id, logical_col, N, bias, has_bias_flags, alpha, beta, curr_residual, curr_y
    );
}

#endif // MATMUL_GEMV_SIMD_COMMON_METAL

