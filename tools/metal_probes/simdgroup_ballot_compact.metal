#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Probe that compacts predicate-passing lanes within each SIMD-group.
kernel void simdgroup_ballot_compact(
    device const float* values [[buffer(0)]],
    device uint* compacted_lanes [[buffer(1)]],
    device uint* lane_counts [[buffer(2)]],
    device ulong* lane_masks [[buffer(3)]],
    uint tg_thread_id [[thread_position_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint threads_per_tg [[threads_per_threadgroup]])
{
    constexpr float threshold = 0.25f;
    float v = values[tg_thread_id];
    bool passes = v > threshold;

    simd_vote vote = simd_ballot(passes);
    // Exercise helper predicates so we notice API changes across toolchains.
    bool any_active = vote.any();
    bool all_active = vote.all();

    // Build an explicit bit-mask via lane-local accumulation so we do not rely
    // on newer simd_vote helpers. Current GPUs expose 32-wide SIMD-groups;
    // guard the shift to avoid UB if a future device widens this.
    uint lane_bit = (passes && lane_id < 32) ? (1u << lane_id) : 0u;
    for (uint offset = simd_size >> 1; offset > 0; offset >>= 1) {
        lane_bit |= simd_shuffle_down(lane_bit, offset);
    }
    uint mask32 = simd_broadcast_first(lane_bit);
    ulong ballot_mask = static_cast<ulong>(mask32);

    // Manual reduction to count active lanes (toolchains prior to Metal 3.1
    // lack simd_vote::count_active_threads()).
    uint lane_count = passes ? 1u : 0u;
    for (uint offset = simd_size >> 1; offset > 0; offset >>= 1) {
        lane_count += simd_shuffle_down(lane_count, offset);
    }
    uint active_count = simd_broadcast_first(lane_count);

    // Compute exclusive prefix for compaction. Non-passing lanes contribute 0.
    uint prefix = simd_prefix_exclusive_sum(passes ? 1u : 0u);

    if (passes) {
        uint base = simdgroup_id * simd_size;
        compacted_lanes[base + prefix] = lane_id;
    }

    if (lane_id == 0) {
        lane_counts[simdgroup_id] = active_count;
        ulong meta = ballot_mask;
        // Encode the any/all flags in the top bits for quick inspection.
        meta |= static_cast<ulong>(any_active) << 62;
        meta |= static_cast<ulong>(all_active) << 63;
        lane_masks[simdgroup_id] = meta;
    }
}
