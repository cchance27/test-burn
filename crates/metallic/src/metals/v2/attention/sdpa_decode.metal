#ifndef METALLIC_V2_ATTENTION_SDPA_DECODE_METAL
#define METALLIC_V2_ATTENTION_SDPA_DECODE_METAL

#include <metal_stdlib>
#include "../../v2/simd.metal"

using namespace metal;

struct SdpaParams {
    uint kv_len;
    uint head_dim;
    float scale;
    uint stride_k_s;
    uint stride_v_s;
};

/// Vectorized (half4) SDPA Decode Loop.
///
/// Computes attention over K/V cache for a single query.
///
/// - `q_vec`: Pre-loaded/rotated Query vector for this thread (from shared mem).
/// - `k_base`: Pointer to start of K cache (global).
/// - `v_base`: Pointer to start of V cache (global).
/// - `output`: Destination for final result.
/// - `tid`: Intra-head thread index (vector index).
/// - `params`: Stride and dimension info.
/// - `reduce_shared`: Shared memory buffer for reductions.
template<uint REDUCE_SIZE>
inline void sdpa_decode_vectorized(
    half4 q_vec,
    const device half* k_base,
    const device half* v_base,
    device half* output,
    uint tid,
    constant SdpaParams& params,
    threadgroup float* reduce_shared
) {
    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float4 out_acc = float4(0.0f);

    uint kv_len = params.kv_len;
    float scale = params.scale;
    uint head_dim = params.head_dim;
    uint vec_dim = head_dim / 4;

    uint stride_k_s = params.stride_k_s;
    uint stride_v_s = params.stride_v_s;

    // Loop over KV sequence
    #pragma unroll 4
    for (uint i = 0; i < kv_len; ++i) {
        // 1. Load K vector
        const device half4* k_ptr_vec = (const device half4*)(k_base + i * stride_k_s);
        
        half4 k_val = 0.0h;
        if (tid < vec_dim) {
             k_val = k_ptr_vec[tid];
        }

        // 2. Compute Dot Product
        float partial_dot = 0.0f;
        if (tid < vec_dim) {
            partial_dot = (float)dot(q_vec, k_val);
        }

        // Reduce sum
        // active_threads = vec_dim
        float score = block_reduce_sum<float, REDUCE_SIZE>(partial_dot, reduce_shared, tid, vec_dim);
        score *= scale;

        // 3. Online Softmax Update
        float m_prev = max_score;
        float m_new = max(m_prev, score);
        float exp_score = exp(score - m_new);
        float exp_prev = exp(m_prev - m_new);

        float scale_v = exp_score;
        float scale_prev = exp_prev;

        sum_exp = sum_exp * scale_prev + scale_v;
        max_score = m_new;

        // 4. Accumulate V
        const device half4* v_ptr_vec = (const device half4*)(v_base + i * stride_v_s);
        
        if (tid < vec_dim) {
            half4 v_val = v_ptr_vec[tid];
            out_acc = out_acc * scale_prev + (float4)v_val * scale_v;
        }
    }

    // 5. Final Normalize and Store
    if (tid < vec_dim) {
        float4 final_val = (abs(sum_exp) > 1e-6) ? (out_acc / sum_exp) : out_acc;
        
        device half4* output_ptr_vec = (device half4*)output;
        output_ptr_vec[tid] = (half4)final_val;
    }
}

#endif // METALLIC_V2_ATTENTION_SDPA_DECODE_METAL
