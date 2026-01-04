// NOTE: ALWAYS_INLINE is provided by policies/base.metal, which must be included.
#include <metal_stdlib>
using namespace metal;

// Core computation template (policy-aware)
// STAGE CONVENTION: fn<Policy>(matrix, [stage_args in buffer order], scale_bytes, gid, lid, [threadgroup])
template<typename Policy>
ALWAYS_INLINE void run_softmax_block_core(
    const device uchar *matrix,           // Buffer 0 (stage_skip - Policy provides)
    device half *output,                   // Buffer 2
    constant uint &batch,                  // Buffer 3 - scalar reference
    constant uint &seq_q,                  // Buffer 4
    constant uint &seq_k,                  // Buffer 5
    constant uint &segment_size,           // Buffer 6
    constant uint &causal,                 // Buffer 7
    constant uint &query_offset,           // Buffer 8
    const device uchar *scale_bytes,       // Buffer 1 (Policy provides)
    uint3 gid,
    uint3 lid,
    threadgroup float *shared_max,
    threadgroup float *shared_sum
) {
    // Scalars are direct references, no dereferencing needed
    
    uint tid = lid.x;
    uint row_idx = gid.x;
    
    const uint THREADS = 256;

    if (row_idx >= batch * seq_q) return;

    uint query_pos = query_offset + (row_idx % seq_q);

    // Phase 1: Find row max
    float local_max = -INFINITY;
    for (uint i = tid; i < seq_k; i += THREADS) {
        uint input_idx = row_idx * seq_k + i;

        float val_arr[1];
        Policy::template load_weights<1>(matrix, input_idx, val_arr);
        half scale = Policy::load_scale(scale_bytes, input_idx);
        float val = val_arr[0] * (float)scale;

        if (causal != 0u && i > query_pos) {
            val = -INFINITY;
        }
        if (val > local_max) {
            local_max = val;
        }
    }

    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];

    // Phase 2: Compute exp and sum
    float local_sum_exp = 0.0f;
    for (uint i = tid; i < seq_k; i += THREADS) {
        uint input_idx = row_idx * seq_k + i;
        
        float val_arr[1];
        Policy::template load_weights<1>(matrix, input_idx, val_arr);
        half scale = Policy::load_scale(scale_bytes, input_idx);
        float val = val_arr[0] * (float)scale;

        if (causal == 0u || i <= query_pos) {
            float diff = val - row_max;
            float exp_val = (val != -INFINITY) ? exp(diff) : 0.0f;
            local_sum_exp += exp_val;
        }
    }

    shared_sum[tid] = local_sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared_sum[0];

    // Phase 3: Normalize
    for (uint i = tid; i < seq_k; i += THREADS) {
        uint input_idx = row_idx * seq_k + i;
        
        float val_arr[1];
        Policy::template load_weights<1>(matrix, input_idx, val_arr);
        half scale = Policy::load_scale(scale_bytes, input_idx);
        float val = val_arr[0] * (float)scale;

        if (causal != 0u && i > query_pos) {
            output[input_idx] = half(0.0f);
        } else {
            float exp_val = (val != -INFINITY) ? exp(val - row_max) : 0.0f;
            float normalized_val = (row_sum > 0.0f) ? (exp_val / row_sum) : 0.0f;
            output[input_idx] = half(normalized_val);
        }
    }
}

// Entry point - standalone only
#ifndef FUSED_KERNEL
#ifndef POLICY_F16_DEFINED
#define POLICY_F16_DEFINED
struct PolicyF16 {
    template<int N>
    static ALWAYS_INLINE void load_weights(const device uchar *src, uint offset, thread float *dst) {
        const device half *h = reinterpret_cast<const device half*>(src + offset * sizeof(half));
        for (int i = 0; i < N; ++i) { dst[i] = float(h[i]); }
    }
    static ALWAYS_INLINE half load_scale(const device uchar *, uint) { return half(1.0h); }
};
#endif

[[kernel]] void softmax_block_f16(
    const device uchar *input [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &batch [[buffer(3)]],
    constant uint &seq_q [[buffer(4)]],
    constant uint &seq_k [[buffer(5)]],
    constant uint &segment_size [[buffer(6)]],
    constant uint &causal [[buffer(7)]],
    constant uint &query_offset [[buffer(8)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
    run_softmax_block_core<PolicyF16>(
        input, output, batch, seq_q, seq_k, segment_size, causal, query_offset,
        scale_bytes, gid, lid, shared_max, shared_sum
    );
}
#endif
