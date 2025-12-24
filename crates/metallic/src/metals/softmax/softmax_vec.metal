#include <metal_stdlib>
using namespace metal;

#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

// SoftmaxVecParams is injected by Foundry via struct_defs() - DO NOT define here!

// Core computation template (policy-aware for future Q8 support)
// STAGE CONVENTION: fn<Policy>(matrix, [stage_args in buffer order], scale_bytes, gid, lid, [threadgroup])
// - Buffers: device/const device pointers
// - Scalars: constant T& references (from setBytes)
// - Threadgroup: threadgroup T* pointers (declared in kernel, passed to template)
template<typename Policy>
ALWAYS_INLINE void run_softmax_vec_core(
    const device uchar *matrix,           // Buffer 0 (stage_skip - Policy provides)
    device half *output,                   // Buffer 2
    constant uint &seq_q,                  // Buffer 3 - scalar reference
    constant uint &seq_k,                  // Buffer 4
    constant uint &causal,                 // Buffer 5
    constant uint &query_offset,           // Buffer 6
    const device uchar *scale_bytes,       // Buffer 1 (Policy provides)
    uint3 gid,
    uint3 lid,
    threadgroup float *shared_data,
    threadgroup uint *shared_indices
) {
    const device half *input = reinterpret_cast<const device half*>(matrix);
    // seq_q, seq_k, causal, query_offset used directly as references
    
    // Each threadgroup processes one row along Y
    // Matches DispatchConfig grid layout
    uint row_idx = gid.y; // Using gid.y for rows

    const uint THREADS_PER_ROW = 32; // Fixed native width
    uint lane_id = lid.x;
    uint stride = THREADS_PER_ROW;
    uint base = row_idx * seq_k;
    uint i_q = query_offset + (row_idx % seq_q);

    // Phase 1: parallel max reduction
    float local_max = -INFINITY;
    uint local_max_index = 0u;
    for (uint c = lane_id; c < seq_k; c += stride) {
        
        // Policy-aware load with scale
        float val_arr[1];
        Policy::template load_weights<1>(matrix, base + c, val_arr);
        half scale = Policy::load_scale(scale_bytes, base + c);
        float val = val_arr[0] * (float)scale;

        if (causal != 0u && c > i_q) {
            val = -INFINITY;
        }
        if (val > local_max) {
            local_max = val;
            local_max_index = c;
        }
    }
    shared_data[lane_id] = local_max;
    shared_indices[lane_id] = local_max_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint offset = stride / 2u; offset > 0u; offset /= 2u) {
        if (lane_id < offset) {
            float cand = shared_data[lane_id + offset];
            uint cand_idx = shared_indices[lane_id + offset];
            if (cand > shared_data[lane_id]) {
                shared_data[lane_id] = cand;
                shared_indices[lane_id] = cand_idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_data[0];
    uint row_max_index = shared_indices[0];

    // Phase 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint c = lane_id; c < seq_k; c += stride) {
        // Policy-aware load with scale
        float val_arr[1];
        Policy::template load_weights<1>(matrix, base + c, val_arr);
        half scale = Policy::load_scale(scale_bytes, base + c);
        float val = val_arr[0] * (float)scale;
        
        if (causal != 0u && c > i_q) {
            val = -INFINITY;
        }
        float e = 0.0f;
        if (isinf(row_max) && row_max > 0.0f) {
            if (isinf(val) && val > 0.0f) {
                e = 1.0f;
            }
        } else if (val != -INFINITY) {
            float diff = val - row_max;
            if (diff < -80.0f) {
                e = 0.0f;
            } else if (diff > 80.0f) {
                e = exp(80.0f);
            } else {
                e = exp(diff);
            }
        }
        output[base + c] = half(e);
        local_sum += e;
    }
    shared_data[lane_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint offset = stride / 2u; offset > 0u; offset /= 2u) {
        if (lane_id < offset) {
            shared_data[lane_id] += shared_data[lane_id + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared_data[0];

    // Phase 3: normalize
    for (uint c = lane_id; c < seq_k; c += stride) {
        bool masked = (causal != 0u && c > i_q);
        if (isnan(row_sum)) {
            output[base + c] = half(row_sum);
        } else if (row_sum > 0.0f && !isinf(row_sum)) {
            float e = (float)output[base + c];
            output[base + c] = half(e / row_sum);
        } else {
            if (masked) {
                output[base + c] = half(0.0f);
            } else if (c == row_max_index) {
                output[base + c] = half(1.0f);
            } else {
                output[base + c] = half(0.0f);
            }
        }
    }
}

// Entry point - standalone only (guarded for compound kernel fusion)
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

[[kernel]] void softmax_vec_f16(
    const device uchar *input [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &seq_q [[buffer(3)]],
    constant uint &seq_k [[buffer(4)]],
    constant uint &causal [[buffer(5)]],
    constant uint &query_offset [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    // Threadgroup memory MUST be declared in kernel, not template!
    threadgroup float shared_data[256];
    threadgroup uint shared_indices[256];
    run_softmax_vec_core<PolicyF16>(
        input, output, seq_q, seq_k, causal, query_offset,
        scale_bytes,
        gid, lid, shared_data, shared_indices
    );
}
#endif
