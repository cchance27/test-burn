#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline uint simd_reduce_sum_u32(uint v, uint simd_width) {
    if (simd_width >= 32u) v += simd_shuffle_down(v, 16u);
    if (simd_width >= 16u) v += simd_shuffle_down(v, 8u);
    if (simd_width >= 8u)  v += simd_shuffle_down(v, 4u);
    if (simd_width >= 4u)  v += simd_shuffle_down(v, 2u);
    if (simd_width >= 2u)  v += simd_shuffle_down(v, 1u);
    return v;
}

inline void simd_reduce_argmax_pair(thread float &best_val, thread uint &best_idx, thread uint &best_lane, uint simd_width) {
    if (simd_width >= 32u) {
        float other_val = simd_shuffle_down(best_val, 16u);
        uint  other_idx = simd_shuffle_down(best_idx, 16u);
        uint  other_ln  = simd_shuffle_down(best_lane, 16u);
        bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
        if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
    }
    if (simd_width >= 16u) {
        float other_val = simd_shuffle_down(best_val, 8u);
        uint  other_idx = simd_shuffle_down(best_idx, 8u);
        uint  other_ln  = simd_shuffle_down(best_lane, 8u);
        bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
        if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
    }
    if (simd_width >= 8u) {
        float other_val = simd_shuffle_down(best_val, 4u);
        uint  other_idx = simd_shuffle_down(best_idx, 4u);
        uint  other_ln  = simd_shuffle_down(best_lane, 4u);
        bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
        if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
    }
    if (simd_width >= 4u) {
        float other_val = simd_shuffle_down(best_val, 2u);
        uint  other_idx = simd_shuffle_down(best_idx, 2u);
        uint  other_ln  = simd_shuffle_down(best_lane, 2u);
        bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
        if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
    }
    if (simd_width >= 2u) {
        float other_val = simd_shuffle_down(best_val, 1u);
        uint  other_idx = simd_shuffle_down(best_idx, 1u);
        uint  other_ln  = simd_shuffle_down(best_lane, 1u);
        bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
        if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
    }
}

kernel void simdgroup_unrolled_sum(
    device const uint* values [[buffer(0)]],
    device uint* per_group_sum [[buffer(1)]],
    uint tg_thread_id [[thread_position_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint threads_per_tg [[threads_per_threadgroup]])
{
    uint acc = values[tg_thread_id];
    acc = simd_reduce_sum_u32(acc, simd_size);
    if (lane_id == 0) {
        per_group_sum[simdgroup_id] = acc;
    }
}

kernel void simdgroup_unrolled_argmax(
    device const float* values [[buffer(0)]],
    device float* out_best_val [[buffer(1)]],
    device uint* out_best_idx [[buffer(2)]],
    device uint* out_best_lane [[buffer(3)]],
    uint tg_thread_id [[thread_position_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint threads_per_tg [[threads_per_threadgroup]])
{
    float best_val = values[tg_thread_id];
    uint best_idx = tg_thread_id;
    uint best_lane = lane_id;

    simd_reduce_argmax_pair(best_val, best_idx, best_lane, simd_size);

    best_val = simd_broadcast_first(best_val);
    best_idx = simd_broadcast_first(best_idx);
    best_lane = simd_broadcast_first(best_lane);

    if (lane_id == 0) {
        out_best_val[simdgroup_id] = best_val;
        out_best_idx[simdgroup_id] = best_idx;
        out_best_lane[simdgroup_id] = best_lane;
    }
}
