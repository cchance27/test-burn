#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;



struct LogitPair { float value; uint index; };

inline bool pair_worse(const thread LogitPair& a, const thread LogitPair& b) {
    return (a.value < b.value) || ((a.value == b.value) && (a.index > b.index));
}

struct RNG {
    uint state;
    inline uint mix(uint x) {
        // 32-bit mix (avalanche) to decorrelate sequential seeds.
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }
    inline void seed(uint s) {
        uint v = s ? s : 1u;
        state = mix(v);
        if (state == 0u) state = 1u;
    }
    inline float next() {
        // Fast xorshift RNG (hot path). We already mix once in `seed()` to decorrelate nearby seeds.
        uint x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        union { uint u; float f; } temp;
        temp.u = (x >> 9) | 0x3f800000u;
        return temp.f - 1.0f;
    }
};

constant uint MAX_TPTG = 1024;
constant uint TG_OUTPUT_K = 256;
constant uint MIN_SIMDGROUP_SIZE = 32;
constant uint MAX_SIMDGROUP_SIZE = 64;
constant uint MAX_SIMDGROUPS = MAX_TPTG / MIN_SIMDGROUP_SIZE;
constant uint TG_MERGE_CAP = 4032;

// Min-heap functions
inline void heap_sift_down(thread LogitPair* heap, uint size, uint pos) {
    while (true) {
        uint left = 2 * pos + 1;
        uint right = 2 * pos + 2;
        uint smallest = pos;
        if (left < size && pair_worse(heap[left], heap[smallest])) smallest = left;
        if (right < size && pair_worse(heap[right], heap[smallest])) smallest = right;
        if (smallest == pos) break;
        LogitPair temp = heap[pos];
        heap[pos] = heap[smallest];
        heap[smallest] = temp;
        pos = smallest;
    }
}

inline void heap_insert(thread LogitPair* heap, thread uint* heap_size, uint max_size, float val, uint idx) {
    if (*heap_size < max_size) {
        uint pos = (*heap_size)++;
        heap[pos].value = val;
        heap[pos].index = idx;
        while (pos > 0) {
            uint parent = (pos - 1) / 2;
            if (!pair_worse(heap[pos], heap[parent])) break;
            LogitPair temp = heap[pos];
            heap[pos] = heap[parent];
            heap[parent] = temp;
            pos = parent;
        }
    } else {
        const LogitPair candidate = LogitPair{val, idx};
        if (!pair_worse(heap[0], candidate)) {
            return;
        }
        heap[0] = candidate;
        heap_sift_down(heap, max_size, 0);
    }
}

inline void heap_to_sorted(thread LogitPair* heap, uint size) {
    // Convert min-heap into an array sorted in descending order (largest first)
    for (uint i = size; i > 1; --i) {
        LogitPair temp = heap[0];
        heap[0] = heap[i - 1];
        heap[i - 1] = temp;
        heap_sift_down(heap, i - 1, 0);
    }
}

inline uint simd_reduce_sum_u32(uint v, uint simd_width) {
    // Unrolled reduction tree to minimize loop overhead.
    if (simd_width >= 64u) v += simd_shuffle_down(v, 32u);
    if (simd_width >= 32u) v += simd_shuffle_down(v, 16u);
    if (simd_width >= 16u) v += simd_shuffle_down(v, 8u);
    if (simd_width >= 8u)  v += simd_shuffle_down(v, 4u);
    if (simd_width >= 4u)  v += simd_shuffle_down(v, 2u);
    if (simd_width >= 2u)  v += simd_shuffle_down(v, 1u);
    return v;
}

inline uint quadlane_reduce_sum_u32(uint v, uint lane, uint simd_width) {
    // Perform intra-quad reductions (2,1) first, then cross-quad (4,8,16).
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
    if (simd_width >= 64u) {
        if ((lane & 32u) == 0u) v += simd_shuffle_down(v, 32u);
    }
    return v;
}

inline void quadlane_reduce_argmax_pair(thread float &best_val, thread uint &best_idx, thread uint &best_lane,
                                        uint lane, uint simd_width) {
    if (simd_width >= 4u) {
        if ((lane & 2u) == 0u) {
            float other_val = simd_shuffle_down(best_val, 2u);
            uint  other_idx = simd_shuffle_down(best_idx, 2u);
            uint  other_ln  = simd_shuffle_down(best_lane, 2u);
            bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
            if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
        }
        if ((lane & 1u) == 0u) {
            float other_val = simd_shuffle_down(best_val, 1u);
            uint  other_idx = simd_shuffle_down(best_idx, 1u);
            uint  other_ln  = simd_shuffle_down(best_lane, 1u);
            bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
            if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
        }
    } else if (simd_width >= 2u) {
        if ((lane & 1u) == 0u) {
            float other_val = simd_shuffle_down(best_val, 1u);
            uint  other_idx = simd_shuffle_down(best_idx, 1u);
            uint  other_ln  = simd_shuffle_down(best_lane, 1u);
            bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
            if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
        }
    }
    if (simd_width >= 8u) {
        if ((lane & 4u) == 0u) {
            float other_val = simd_shuffle_down(best_val, 4u);
            uint  other_idx = simd_shuffle_down(best_idx, 4u);
            uint  other_ln  = simd_shuffle_down(best_lane, 4u);
            bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
            if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
        }
    }
    if (simd_width >= 16u) {
        if ((lane & 8u) == 0u) {
            float other_val = simd_shuffle_down(best_val, 8u);
            uint  other_idx = simd_shuffle_down(best_idx, 8u);
            uint  other_ln  = simd_shuffle_down(best_lane, 8u);
            bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
            if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
        }
    }
    if (simd_width >= 32u) {
        if ((lane & 16u) == 0u) {
            float other_val = simd_shuffle_down(best_val, 16u);
            uint  other_idx = simd_shuffle_down(best_idx, 16u);
            uint  other_ln  = simd_shuffle_down(best_lane, 16u);
            bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
            if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
        }
    }
    if (simd_width >= 64u) {
        if ((lane & 32u) == 0u) {
            float other_val = simd_shuffle_down(best_val, 32u);
            uint  other_idx = simd_shuffle_down(best_idx, 32u);
            uint  other_ln  = simd_shuffle_down(best_lane, 32u);
            bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx));
            if (take_other) { best_val = other_val; best_idx = other_idx; best_lane = other_ln; }
        }
    }
}

// Redirect the existing reducers to quad-lane versions within kernels
#define simd_reduce_sum_u32(v, sw) quadlane_reduce_sum_u32((v), simd_lane, (sw))
#define simd_reduce_argmax_pair(bv, bi, bl, sw) quadlane_reduce_argmax_pair((bv), (bi), (bl), simd_lane, (sw))

// f16 logits variant (accumulated in float)
kernel void sample_topk_fused_f16(
    const device half*   logits             [[buffer(0)]],
    device uint*         out_token          [[buffer(1)]],
    constant SampleParams& params           [[buffer(2)]],
    uint tid                                 [[thread_position_in_threadgroup]],
    uint tptg                                [[threads_per_threadgroup]],
    uint tg_id                               [[threadgroup_position_in_grid]],
    uint num_tgs                             [[threadgroups_per_grid]],
    uint simd_lane_id                        [[thread_index_in_simdgroup]],
    uint simd_group_in_tg                    [[simdgroup_index_in_threadgroup]],
    uint simd_width_attr                     [[threads_per_simdgroup]]
) {
    const uint V     = params.vocab_size;
    const float denom = fmax(params.temperature, 1e-6f);
    const float invT = static_cast<float>(1.0f / denom);
    const uint M     = params.per_thread_m;

    // Per-thread candidate capacity. Keep this >= the runtime `per_thread_m` default
    // (now min(top_k, 40)) so concentrated-lane distributions retain full Top-K fidelity.
    constexpr uint MAX_LANE_HEAP = 40;
    constexpr uint SIMDGROUP_SHORTLIST = 64;
    constexpr uint SIMDGROUP_SHORTLIST_CAP = MAX_SIMDGROUPS * SIMDGROUP_SHORTLIST;

    LogitPair lane_heap[MAX_LANE_HEAP];
    const uint lane_target = min(M, (uint)MAX_LANE_HEAP);
    uint lane_heap_size = 0;

    const uint items_per_tg = V / num_tgs + 1;
    const uint block_start = tg_id * items_per_tg;
    const uint block_end = min(block_start + items_per_tg, V);

    uint scalar_prefix_end = block_start;
    if ((block_start & 3u) != 0u) {
        const uint align_gap = min(4u - (block_start & 3u), block_end - block_start);
        scalar_prefix_end += align_gap;
    }
    scalar_prefix_end = min(scalar_prefix_end, block_end);
    for (uint i = block_start + tid; i < scalar_prefix_end; i += tptg) {
        float v = static_cast<float>(logits[i]);
        v *= invT;
        heap_insert(lane_heap, &lane_heap_size, lane_target, static_cast<float>(v), i);
    }

    const uint vectorizable_len = (block_end > scalar_prefix_end) ? ((block_end - scalar_prefix_end) & ~3u) : 0u;
    const uint vector_count = vectorizable_len >> 2;
    if (vector_count > 0u) {
        using Vec4 = metal::vec<half, 4>;
        const device Vec4* logits4 = reinterpret_cast<const device Vec4*>(logits + scalar_prefix_end);
        for (uint group = tid; group < vector_count; group += tptg) {
            const Vec4 raw = logits4[group];
            const uint base_idx = scalar_prefix_end + (group << 2);
            for (uint lane_step = 0; lane_step < 4u; ++lane_step) {
                const uint idx = base_idx + lane_step;
                float v = static_cast<float>(raw[lane_step]);
                v *= invT;
                heap_insert(lane_heap, &lane_heap_size, lane_target, static_cast<float>(v), idx);
            }
        }
    }
    const uint tail_start = scalar_prefix_end + (vector_count << 2);
    for (uint i = tail_start + tid; i < block_end; i += tptg) {
        float v = static_cast<float>(logits[i]);
        v *= invT;
        heap_insert(lane_heap, &lane_heap_size, lane_target, static_cast<float>(v), i);
    }

    heap_to_sorted(lane_heap, lane_heap_size);

    const uint simd_width = clamp(simd_width_attr, (uint)MIN_SIMDGROUP_SIZE, (uint)MAX_SIMDGROUP_SIZE);
    const uint simd_lane = simd_lane_id;
    const uint simd_group_id = simd_group_in_tg;
    const uint simd_group_count = (tptg + simd_width - 1) / simd_width;

    threadgroup LogitPair sg_shortlists[SIMDGROUP_SHORTLIST_CAP];
    threadgroup uint sg_counts[MAX_SIMDGROUPS];
    threadgroup uint sg_offsets[MAX_SIMDGROUPS];
    threadgroup uint shortlist_total_entries;
    if (tid < MAX_SIMDGROUPS) {
        sg_counts[tid] = 0u;
        sg_offsets[tid] = 0u;
    }
    if (tid == 0) {
        shortlist_total_entries = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lane_count = simd_reduce_sum_u32(lane_heap_size, simd_width);
    const uint total_candidates = simd_broadcast_first(lane_count);
    const uint groups_active = min(simd_group_count, (uint)MAX_SIMDGROUPS);
    const uint per_group_quota = ((params.k + groups_active - 1u) / max(groups_active, 1u)) * 2u;
    const uint min_quota = max(max(lane_target, 1u), 16u);
    const uint shortlist_limit = clamp(max(per_group_quota, min_quota), min_quota, (uint)SIMDGROUP_SHORTLIST);
    const uint shortlist_target = min(shortlist_limit, total_candidates);

    uint lane_cursor = 0;
    LogitPair sg_local[SIMDGROUP_SHORTLIST];
    uint sg_count = 0;

    for (uint iter = 0; iter < shortlist_target; ++iter) {
        const bool has_candidate = lane_cursor < lane_heap_size;
        float candidate_val = has_candidate ? lane_heap[lane_cursor].value : -INFINITY;
        uint candidate_idx = has_candidate ? lane_heap[lane_cursor].index : 0u;

        float best_val = candidate_val;
        uint best_idx = candidate_idx;
        uint best_lane = simd_lane;

        simd_reduce_argmax_pair(best_val, best_idx, best_lane, simd_width);

        best_val = simd_broadcast_first(best_val);
        best_idx = simd_broadcast_first(best_idx);
        best_lane = simd_broadcast_first(best_lane);

        if (!isfinite(best_val)) {
            break;
        }

        if (simd_lane == best_lane && has_candidate) {
            lane_cursor += 1;
        }

        if (simd_lane == 0 && sg_count < SIMDGROUP_SHORTLIST) {
            sg_local[sg_count] = LogitPair{best_val, best_idx};
            sg_count += 1;
        }
    }

    const bool sg_is_valid = simd_group_id < groups_active;
    if (simd_lane == 0 && sg_is_valid) {
        sg_counts[simd_group_id] = sg_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint offset = 0;
        for (uint sg = 0; sg < groups_active; ++sg) {
            const uint count = sg_counts[sg];
            sg_offsets[sg] = offset;
            offset += count;
        }
        shortlist_total_entries = offset;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane == 0 && sg_is_valid) {
        const uint base = sg_offsets[simd_group_id];
        for (uint i = 0; i < sg_count; ++i) {
            const uint dst = base + i;
            if (dst < SIMDGROUP_SHORTLIST_CAP) {
                sg_shortlists[dst] = sg_local[i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        const uint out_k = min(params.k, TG_OUTPUT_K);
        LogitPair final_heap[TG_OUTPUT_K];
        uint final_heap_size = 0;
        const uint shortlist_total = min(shortlist_total_entries, (uint)SIMDGROUP_SHORTLIST_CAP);
        for (uint i = 0; i < shortlist_total; ++i) {
            const LogitPair cand = sg_shortlists[i];
            if (isfinite(cand.value) && cand.index < V) {
                heap_insert(final_heap, &final_heap_size, out_k, cand.value, cand.index);
            }
        }
        heap_to_sorted(final_heap, final_heap_size);
        if (final_heap_size == 0) {
            out_token[0] = 0u;
            return;
        }
        float maxv = final_heap[0].value;
        float sum = 0.0f;
        for (uint i = 0; i < final_heap_size; ++i) {
            float e = metal::fast::exp(final_heap[i].value - maxv);
            final_heap[i].value = e;
            sum += e;
        }
        if (sum <= 0.0f || !isfinite(sum)) {
            out_token[0] = final_heap[0].index < V ? final_heap[0].index : 0u;
            return;
        }
        const float top_p = clamp(params.top_p, 0.0f, 1.0f);
        const float min_p = clamp(params.min_p, 0.0f, 1.0f);
        const float max_prob = final_heap[0].value / sum;
        const float min_p_cut = (min_p > 0.0f) ? (min_p * max_prob) : 0.0f;

        uint kept = 0;
        float cumulative = 0.0f;
        for (uint i = 0; i < final_heap_size; ++i) {
            float prob = final_heap[i].value / sum;
            if (min_p_cut > 0.0f && prob < min_p_cut) {
                continue;
            }
            final_heap[kept].index = final_heap[i].index;
            final_heap[kept].value = prob;
            cumulative += prob;
            kept += 1;
            if (cumulative >= top_p) {
                break;
            }
        }

        if (kept == 0) {
            out_token[0] = final_heap[0].index < V ? final_heap[0].index : 0u;
            return;
        }

        const float cutoff_sum = cumulative;
        const uint cutoff = kept - 1;
        const float renorm = (cutoff_sum > 0.0f && isfinite(cutoff_sum)) ? (1.0f / cutoff_sum) : 1.0f;
        for (uint i = 0; i <= cutoff; ++i) {
            final_heap[i].value *= renorm;
        }
        RNG rng;
        rng.seed(params.seed);
        float r = rng.next();
        float acc = 0.0f;
        uint chosen = final_heap[0].index;
        for (uint i = 0; i <= cutoff; ++i) {
            acc += final_heap[i].value;
            if (r <= acc) {
                chosen = final_heap[i].index;
                break;
            }
        }
        if (chosen >= V) {
            chosen = (V > 0) ? (V - 1) : 0;
        }
        out_token[0] = chosen;
    }
}
