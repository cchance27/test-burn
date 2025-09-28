#include <metal_stdlib>

#if __has_include(<metal_simdgroup>)
#include <metal_simdgroup>
#define __TB_HAS_METAL_SIMDGROUP_HEADER__ 1
#else
#define __TB_HAS_METAL_SIMDGROUP_HEADER__ 0
#endif

using namespace metal;

#if !defined(__METAL_SIMDGROUP_REDUCE_AVAILABLE__)
#define __METAL_SIMDGROUP_REDUCE_AVAILABLE__ 0
#endif

#if !defined(__METAL_VERSION__)
#define __METAL_VERSION__ 0
#endif

#if (__METAL_VERSION__ >= 310 && __TB_HAS_METAL_SIMDGROUP_HEADER__) || __METAL_SIMDGROUP_REDUCE_AVAILABLE__
#define __TB_CAN_USE_SIMDGROUP_REDUCE__ 1
#else
#define __TB_CAN_USE_SIMDGROUP_REDUCE__ 0
#endif

#if __TB_CAN_USE_SIMDGROUP_REDUCE__ && !__TB_HAS_METAL_SIMDGROUP_HEADER__
float simdgroup_reduce_max(float) __attribute__((overloadable));
float simdgroup_reduce_add(float) __attribute__((overloadable));
uint simdgroup_reduce_min(uint) __attribute__((overloadable));
#endif

template <typename T>
inline T reduce_max_simdgroup(T value) {
#if __TB_CAN_USE_SIMDGROUP_REDUCE__
    return simdgroup_reduce_max(value);
#else
    return simd_reduce_max(value);
#endif
}

template <typename T>
inline T reduce_add_simdgroup(T value) {
#if __TB_CAN_USE_SIMDGROUP_REDUCE__
    return simdgroup_reduce_add(value);
#else
    return simd_reduce_add(value);
#endif
}

inline uint reduce_min_simdgroup(uint value) {
#if __TB_CAN_USE_SIMDGROUP_REDUCE__
    return simdgroup_reduce_min(value);
#else
    return simd_reduce_min(value);
#endif
}

// Fused mask+softmax compute kernel
kernel void sdpa_fused_softmax(device float* attn [[buffer(0)]],
                               constant uint &seq_q [[buffer(1)]],
                               constant uint &seq_k [[buffer(2)]],
                               constant uint &causal_flag [[buffer(3)]],
                               constant uint &query_offset [[buffer(4)]],
                               uint3 tg_pos [[threadgroup_position_in_grid]],
                               uint3 tid3 [[thread_position_in_threadgroup]],
                               uint3 tptg [[threads_per_threadgroup]],
                               ushort simd_tid [[thread_index_in_simdgroup]],
                               ushort simd_gid [[simdgroup_index_in_threadgroup]],
                               ushort simdgroup_width [[threads_per_simdgroup]]) {
    // One threadgroup processes one row. Threads stride across columns.
    uint row = tg_pos.y;
    uint lane = tid3.x;
    uint stride = tptg.x;
    uint base = row * seq_k;
    uint i_q = (row % seq_q) + query_offset;

    constexpr uint MAX_SIMD_GROUPS = 64;
    constexpr uint INVALID_INDEX = 0xffffffffu;
    uint simdgroups_per_tg = (tptg.x + simdgroup_width - 1u) / simdgroup_width;
    simdgroups_per_tg = simdgroups_per_tg > 0u ? simdgroups_per_tg : 1u;
    uint clamped_simdgroups = simdgroups_per_tg > MAX_SIMD_GROUPS ? MAX_SIMD_GROUPS : simdgroups_per_tg;
    bool simdgroup_active = simd_gid < clamped_simdgroups;

    // Use a more efficient shared memory size based on common hardware
    // Apple GPUs typically have good performance with 256 or 512 threads per group
    threadgroup float shared_max[MAX_SIMD_GROUPS];
    threadgroup uint shared_indices[MAX_SIMD_GROUPS];
    threadgroup float shared_sum[MAX_SIMD_GROUPS];

    // Phase 1: row-wise max reduction with causal masking and index tracking.
    float local_max = -INFINITY;
    uint max_index = 0;
    for (uint c = lane; c < seq_k; c += stride) {
        float xv = attn[base + c];
        // Apply causal mask
        if (causal_flag == 1u && c > i_q) { 
            xv = -INFINITY; 
        }
        if (xv > local_max) {
            local_max = xv;
            max_index = c; // Store relative index within the row
        }
    }
    
    float group_max = reduce_max_simdgroup(local_max);
    uint candidate_index = (local_max == group_max) ? max_index : INVALID_INDEX;
    uint group_max_index = reduce_min_simdgroup(candidate_index);

    if (simd_is_first() && simdgroup_active) {
        shared_max[simd_gid] = group_max;
        shared_indices[simd_gid] = group_max_index;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0) {
        uint lane_id = static_cast<uint>(simd_tid);
        float cross_max = (lane_id < clamped_simdgroups) ? shared_max[lane_id] : -INFINITY;
        float block_max = reduce_max_simdgroup(cross_max);
        uint cross_index =
            (lane_id < clamped_simdgroups && cross_max == block_max) ? shared_indices[lane_id] : INVALID_INDEX;
        uint block_max_index = reduce_min_simdgroup(cross_index);

        if (simd_is_first()) {
            shared_max[0] = block_max;
            shared_indices[0] = block_max_index;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maxv = shared_max[0];
    uint row_max_index = shared_indices[0]; // This is the relative index within the row

    // Phase 2: compute exp(x - max) and partial sums
    float local_sum = 0.0f;
    for (uint c = lane; c < seq_k; c += stride) {
        float xv = attn[base + c];
        // Apply causal mask
        if (causal_flag == 1u && c > i_q) {
            xv = -INFINITY; 
        }
        // Compute exp(x - max) with proper handling for extreme values
        float e = 0.0f;
        if (isinf(maxv) && maxv > 0) { // maxv is +inf
            if (isinf(xv) && xv > 0) {
                e = 1.0f;
            } else {
                e = 0.0f;
            }
        } else if (xv != -INFINITY) {
            // For very large negative differences, exp might underflow to 0
            // This is actually the correct behavior
            float diff = xv - maxv;
            // Clamp the difference to prevent extreme values that could cause overflow/underflow
            if (diff < -80.0f) {  // Prevent underflow
                e = 0.0f;
            } else if (diff > 80.0f) {  // Prevent overflow
                e = exp(80.0f);  // Though this case should not happen due to max subtraction
            } else {
                e = exp(diff);
            }
        }
        attn[base + c] = e;
        local_sum += e;
    }

    float group_sum = reduce_add_simdgroup(local_sum);

    if (simd_is_first() && simdgroup_active) {
        shared_sum[simd_gid] = group_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0) {
        uint lane_id = static_cast<uint>(simd_tid);
        float cross_sum = (lane_id < clamped_simdgroups) ? shared_sum[lane_id] : 0.0f;
        float block_sum = reduce_add_simdgroup(cross_sum);

        if (simd_is_first()) {
            shared_sum[0] = block_sum;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumv = shared_sum[0];

    // Phase 3: normalize in place
    for (uint c = lane; c < seq_k; c += stride) {
        // Handle case where sum is zero or invalid
        if (isnan(sumv)) {
            attn[base + c] = sumv; // Propagate NaN
        } else if (sumv > 0.0f && sumv != INFINITY) {
            attn[base + c] = attn[base + c] / sumv;
        } else {
            // If sum is zero or invalid, handle appropriately
            if (causal_flag == 1u && c > i_q) {
                attn[base + c] = 0.0f;
            } else {
                // When all exponentials underflow to zero, give probability 1.0 to the maximum element
                // and 0.0 to all others
                if (c == row_max_index) {
                    attn[base + c] = 1.0f;
                } else {
                    attn[base + c] = 0.0f;
                }
            }
        }
    }
}
