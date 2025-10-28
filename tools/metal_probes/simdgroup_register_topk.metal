#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Probe that keeps a per-lane shortlist purely in registers.
kernel void simdgroup_register_topk(
    device const float* values [[buffer(0)]],
    device float* lane_topk_values [[buffer(1)]],
    device uint* lane_topk_indices [[buffer(2)]],
    uint tg_thread_id [[thread_position_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint threads_per_tg [[threads_per_threadgroup]])
{
    constexpr uint ITEMS_PER_LANE = 8;
    constexpr uint KEEP = 4;

    // Caller must size buffers for threads_per_tg * ITEMS_PER_LANE inputs.
    float best_vals[KEEP];
    uint best_indices[KEEP];
    for (uint i = 0; i < KEEP; ++i) {
        best_vals[i] = -INFINITY;
        best_indices[i] = 0;
    }

    for (uint i = 0; i < ITEMS_PER_LANE; ++i) {
        uint input_index = tg_thread_id + i * threads_per_tg;
        float candidate = values[input_index];

        // Find the current worst slot in the shortlist (lowest value).
        uint worst_slot = 0;
        float worst_value = best_vals[0];
        for (uint slot = 1; slot < KEEP; ++slot) {
            if (best_vals[slot] < worst_value) {
                worst_slot = slot;
                worst_value = best_vals[slot];
            }
        }

        if (candidate > worst_value) {
            best_vals[worst_slot] = candidate;
            best_indices[worst_slot] = input_index;
        }
    }

    uint out_base = tg_thread_id * KEEP;
    for (uint slot = 0; slot < KEEP; ++slot) {
        lane_topk_values[out_base + slot] = best_vals[slot];
        lane_topk_indices[out_base + slot] = best_indices[slot];
    }
}
