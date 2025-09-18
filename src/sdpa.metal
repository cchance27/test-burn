#include <metal_stdlib>
using namespace metal;

kernel void sdpa_kernel(device const float *query [[buffer(0)]],
                        device const float *key [[buffer(1)]],
                        device const float *value [[buffer(2)]],
                        device float *output [[buffer(3)]],
                        constant uint &batch [[buffer(4)]],
                        constant uint &seq_q [[buffer(5)]],
                        constant uint &seq_k [[buffer(6)]],
                        constant uint &dim [[buffer(7)]],
                        uint3 tgpid [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {

    // Each threadgroup handles one (b, q_idx)
    uint b = tgpid.y;   // height dimension is batch
    uint q_idx = tgpid.x;

    if (b >= batch || q_idx >= seq_q) {
        return;
    }

    float d_k_sqrt = sqrt((float)dim);

    // TODO: make this dynamic via threadgroup memory if seq_k can exceed 1024
    threadgroup float attn_scores[1024];

    // 1) Compute attention scores for this (b, q_idx)
    for (uint k_idx = tid; k_idx < seq_k; k_idx += 32) {
        float score = 0.0f;
        for (uint d = 0; d < dim; ++d) {
            uint q_offset = b * seq_q * dim + q_idx * dim + d;
            uint k_offset = b * seq_k * dim + k_idx * dim + d;
            score += query[q_offset] * key[k_offset];
        }
        attn_scores[k_idx] = score / d_k_sqrt;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) Reduce max across k for numerical stability
    threadgroup float partial_max[32];
    threadgroup float max_score;

    float local_max = -FLT_MAX;
    for (uint k_idx = tid; k_idx < seq_k; k_idx += 32) {
        local_max = fmax(local_max, attn_scores[k_idx]);
    }
    partial_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float m = -FLT_MAX;
        for (uint i = 0; i < 32; ++i) {
            m = fmax(m, partial_max[i]);
        }
        max_score = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Exponentiate and sum with reduction
    threadgroup float partial_sum[32];
    threadgroup float total_exp_sum;

    float local_sum = 0.0f;
    for (uint k_idx = tid; k_idx < seq_k; k_idx += 32) {
        float val = exp(attn_scores[k_idx] - max_score);
        attn_scores[k_idx] = val;
        local_sum += val;
    }
    partial_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < 32; ++i) {
            s += partial_sum[i];
        }
        total_exp_sum = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4) Normalize
    for (uint k_idx = tid; k_idx < seq_k; k_idx += 32) {
        attn_scores[k_idx] /= total_exp_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5) Compute output vector for this (b, q_idx)
    for (uint d_idx = tid; d_idx < dim; d_idx += 32) {
        float out_val = 0.0f;
        for (uint k_idx = 0; k_idx < seq_k; ++k_idx) {
            uint v_offset = b * seq_k * dim + k_idx * dim + d_idx;
            out_val += attn_scores[k_idx] * value[v_offset];
        }
        uint out_offset = b * seq_q * dim + q_idx * dim + d_idx;
        output[out_offset] = out_val;
    }
}
