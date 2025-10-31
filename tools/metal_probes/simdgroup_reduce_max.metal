#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Simple probe that reduces per-simdgroup maxima using shuffle-down intrinsics.
kernel void simdgroup_reduce_max(
    device const float* values [[buffer(0)]],
    device float* per_group_max [[buffer(1)]],
    uint tg_thread_id [[thread_position_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint threads_per_tg [[threads_per_threadgroup]])
{
    // Ensure we only read valid data; caller should size the input accordingly.
    float acc = values[tg_thread_id];

    // Shuffle-based reduction across the current SIMD-group.
    for (uint offset = simd_size >> 1; offset > 0; offset >>= 1) {
        float other = simd_shuffle_down(acc, offset);
        acc = max(acc, other);
    }

    if (lane_id == 0) {
        // Write one value per SIMD-group.
        per_group_max[simdgroup_id] = acc;
    }
}
