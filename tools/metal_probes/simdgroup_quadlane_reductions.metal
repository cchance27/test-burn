#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline uint quadlane_reduce_sum_u32(uint v, uint lane, uint simd_width) {
    // Intra-quad (4-lane) reductions first, then cross-quad.
    if (simd_width >= 4u) {
        if ((lane & 2u) == 0u) v += simd_shuffle_down(v, 2u);
        if ((lane & 1u) == 0u) v += simd_shuffle_down(v, 1u);
    } else if (simd_width >= 2u) {
        if ((lane & 1u) == 0u) v += simd_shuffle_down(v, 1u);
    }
    if (simd_width >= 8u) {
        if ((lane & 4u) == 0u) v += simd_shuffle_down(v, 4u);
    }
    if (simd_width >= 16u) {
        if ((lane & 8u) == 0u) v += simd_shuffle_down(v, 8u);
    }
    if (simd_width >= 32u) {
        if ((lane & 16u) == 0u) v += simd_shuffle_down(v, 16u);
    }
    return v;
}

inline void quadlane_reduce_argmax_pair(thread float &best_val, thread uint &best_idx, thread uint &best_lane,
                                        uint lane, uint simd_width) {
    auto step = [&](uint off){
        float other_val = simd_shuffle_down(best_val, off);
        uint  other_idx = simd_shuffle_down(best_idx, off);
        uint  other_ln  = simd_shuffle_down(best_lane, off);
        bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
        if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
    };
    if (simd_width >= 4u) {
        if ((lane & 2u) == 0u) step(2u);
        if ((lane & 1u) == 0u) step(1u);
    } else if (simd_width >= 2u) {
        if ((lane & 1u) == 0u) step(1u);
    }
    if (simd_width >= 8u) {
        if ((lane & 4u) == 0u) step(4u);
    }
    if (simd_width >= 16u) {
        if ((lane & 8u) == 0u) step(8u);
    }
    if (simd_width >= 32u) {
        if ((lane & 16u) == 0u) step(16u);
    }
}

kernel void simdgroup_quadlane_sum(
    device const uint* values [[buffer(0)]],
    device uint* per_group_sum [[buffer(1)]],
    uint tg_thread_id [[thread_position_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint threads_per_tg [[threads_per_threadgroup]])
{
    uint acc = values[tg_thread_id];
    acc = quadlane_reduce_sum_u32(acc, lane_id, simd_size);
    if (lane_id == 0) {
        per_group_sum[simdgroup_id] = acc;
    }
}

kernel void simdgroup_quadlane_argmax(
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
    uint best_lane_local = lane_id;

    quadlane_reduce_argmax_pair(best_val, best_idx, best_lane_local, lane_id, simd_size);

    best_val = simd_broadcast_first(best_val);
    best_idx = simd_broadcast_first(best_idx);
    best_lane_local = simd_broadcast_first(best_lane_local);

    if (lane_id == 0) {
        out_best_val[simdgroup_id] = best_val;
        out_best_idx[simdgroup_id] = best_idx;
        out_best_lane[simdgroup_id] = best_lane_local;
    }
}
