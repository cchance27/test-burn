#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16)

struct SampleParams {
    uint   vocab_size;
    uint   k;
    float  top_p;
    float  temperature;
    uint   seed;
    uint   per_thread_m;
    uint   num_threadgroups;
};

struct LogitPair { float value; uint index; };

struct RNG {
    uint state;
    inline void seed(uint s) { state = s ? s : 1u; }
    inline float next() {
        state = state * 1664525u + 1013904223u;
        union { uint u; float f; } temp;
        temp.u = (state >> 9) | 0x3f800000u;
        return temp.f - 1.0f;
    }
};

constant uint MAX_TPTG = 256;
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
        
        if (left < size && heap[left].value < heap[smallest].value) smallest = left;
        if (right < size && heap[right].value < heap[smallest].value) smallest = right;
        
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
            if (heap[parent].value <= heap[pos].value) break;
            LogitPair temp = heap[pos];
            heap[pos] = heap[parent];
            heap[parent] = temp;
            pos = parent;
        }
    } else if (val > heap[0].value) {
        heap[0].value = val;
        heap[0].index = idx;
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
    // Note: no final reverse; result is descending order, suitable for assuming index 0 is the max
}

// Partials kernel - writes to separate buffers
#define DEFINE_SAMPLE_TOPK_PARTIALS_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void sample_topk_partials_##SUFFIX( \
    const device SCALAR*  logits             [[buffer(0)]], \
    device float*         partials_values    [[buffer(1)]], \
    device uint*          partials_indices   [[buffer(2)]], \
    constant SampleParams& params            [[buffer(3)]], \
    uint tid                                  [[thread_position_in_threadgroup]], \
    uint tptg                                 [[threads_per_threadgroup]], \
    uint tg_id                                [[threadgroup_position_in_grid]], \
    uint num_tgs                              [[threadgroups_per_grid]], \
    uint simd_lane_id                         [[thread_index_in_simdgroup]], \
    uint simd_group_in_tg                     [[simdgroup_index_in_threadgroup]], \
    uint simd_width_attr                      [[threads_per_simdgroup]] \
) { \
    const uint V     = params.vocab_size; \
    const float denom = fmax(params.temperature, 1e-6f); \
    const ACCUM invT = static_cast<ACCUM>(1.0f / denom); \
    const uint M     = params.per_thread_m; \
    \
    constexpr uint MAX_LANE_HEAP = 4; \
    constexpr uint SIMDGROUP_SHORTLIST = 64; \
    constexpr uint SIMDGROUP_SHORTLIST_CAP = MAX_SIMDGROUPS * SIMDGROUP_SHORTLIST; \
    \
    LogitPair lane_heap[MAX_LANE_HEAP]; \
    const uint lane_target = min(M, (uint)MAX_LANE_HEAP); \
    uint lane_heap_size = 0; \
    \
    const uint items_per_tg = V / num_tgs + 1; \
    const uint block_start = tg_id * items_per_tg; \
    const uint block_end = min(block_start + items_per_tg, V); \
    \
    uint scalar_prefix_end = block_start; \
    if ((block_start & 3u) != 0u) { \
        const uint align_gap = min(4u - (block_start & 3u), block_end - block_start); \
        scalar_prefix_end += align_gap; \
    } \
    scalar_prefix_end = min(scalar_prefix_end, block_end); \
    for (uint i = block_start + tid; i < scalar_prefix_end; i += tptg) { \
        ACCUM v = static_cast<ACCUM>(logits[i]); \
        v *= invT; \
        heap_insert(lane_heap, &lane_heap_size, lane_target, static_cast<float>(v), i); \
    } \
    \
    const uint vectorizable_len = (block_end > scalar_prefix_end) ? ((block_end - scalar_prefix_end) & ~3u) : 0u; \
    const uint vector_count = vectorizable_len >> 2; \
    if (vector_count > 0u) { \
        using Vec4 = metal::vec<SCALAR, 4>; \
        const device Vec4* logits4 = reinterpret_cast<const device Vec4*>(logits + scalar_prefix_end); \
        for (uint group = tid; group < vector_count; group += tptg) { \
            const Vec4 raw = logits4[group]; \
            const uint base_idx = scalar_prefix_end + (group << 2); \
            for (uint lane_step = 0; lane_step < 4u; ++lane_step) { \
                const uint idx = base_idx + lane_step; \
                ACCUM v = static_cast<ACCUM>(raw[lane_step]); \
                v *= invT; \
                heap_insert(lane_heap, &lane_heap_size, lane_target, static_cast<float>(v), idx); \
            } \
        } \
    } \
    const uint tail_start = scalar_prefix_end + (vector_count << 2); \
    for (uint i = tail_start + tid; i < block_end; i += tptg) { \
        ACCUM v = static_cast<ACCUM>(logits[i]); \
        v *= invT; \
        heap_insert(lane_heap, &lane_heap_size, lane_target, static_cast<float>(v), i); \
    } \
    \
    heap_to_sorted(lane_heap, lane_heap_size); \
    \
    const uint simd_width = clamp(simd_width_attr, (uint)MIN_SIMDGROUP_SIZE, (uint)MAX_SIMDGROUP_SIZE); \
    const uint simd_lane = simd_lane_id; \
    const uint simd_group_id = simd_group_in_tg; \
    const uint simd_group_count = (tptg + simd_width - 1) / simd_width; \
    \
    threadgroup LogitPair sg_shortlists[SIMDGROUP_SHORTLIST_CAP]; \
    threadgroup uint sg_counts[MAX_SIMDGROUPS]; \
    threadgroup uint sg_offsets[MAX_SIMDGROUPS]; \
    threadgroup uint shortlist_total_entries; \
    if (tid < MAX_SIMDGROUPS) { \
        sg_counts[tid] = 0u; \
        sg_offsets[tid] = 0u; \
    } \
    if (tid == 0) { \
        shortlist_total_entries = 0u; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    uint lane_count = lane_heap_size; \
    for (uint offset = simd_width >> 1; offset > 0; offset >>= 1) { \
        lane_count += simd_shuffle_down(lane_count, offset); \
    } \
    const uint total_candidates = simd_broadcast_first(lane_count); \
    const uint groups_active = min(simd_group_count, (uint)MAX_SIMDGROUPS); \
    const uint per_group_quota = ((params.k + groups_active - 1u) / max(groups_active, 1u)) * 2u; \
    const uint min_quota = max(max(lane_target, 1u), 16u); \
    const uint shortlist_limit = clamp(max(per_group_quota, min_quota), min_quota, (uint)SIMDGROUP_SHORTLIST); \
    const uint shortlist_target = min(shortlist_limit, total_candidates); \
    \
    uint lane_cursor = 0; \
    LogitPair sg_local[SIMDGROUP_SHORTLIST]; \
    uint sg_count = 0; \
    \
    for (uint iter = 0; iter < shortlist_target; ++iter) { \
        const bool has_candidate = lane_cursor < lane_heap_size; \
        float candidate_val = has_candidate ? lane_heap[lane_cursor].value : -INFINITY; \
        uint candidate_idx = has_candidate ? lane_heap[lane_cursor].index : 0u; \
        \
        float best_val = candidate_val; \
        uint best_idx = candidate_idx; \
        uint best_lane = simd_lane; \
        \
        for (uint offset = simd_width >> 1; offset > 0; offset >>= 1) { \
            float other_val = simd_shuffle_down(best_val, offset); \
            uint other_idx = simd_shuffle_down(best_idx, offset); \
            uint other_lane = simd_shuffle_down(best_lane, offset); \
            const bool take_other = (other_val > best_val) || ((other_val == best_val) && (other_idx < best_idx)); \
            if (take_other) { \
                best_val = other_val; \
                best_idx = other_idx; \
                best_lane = other_lane; \
            } \
        } \
        \
        best_val = simd_broadcast_first(best_val); \
        best_idx = simd_broadcast_first(best_idx); \
        best_lane = simd_broadcast_first(best_lane); \
        \
        if (!isfinite(best_val)) { \
            break; \
        } \
        \
        if (simd_lane == best_lane && has_candidate) { \
            lane_cursor += 1; \
        } \
        \
        if (simd_lane == 0 && sg_count < SIMDGROUP_SHORTLIST) { \
            sg_local[sg_count] = LogitPair{best_val, best_idx}; \
            sg_count += 1; \
        } \
    } \
    \
    const bool sg_is_valid = simd_group_id < groups_active; \
    if (simd_lane == 0 && sg_is_valid) { \
        sg_counts[simd_group_id] = sg_count; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    if (tid == 0) { \
        uint offset = 0; \
        for (uint sg = 0; sg < groups_active; ++sg) { \
            const uint count = sg_counts[sg]; \
            sg_offsets[sg] = offset; \
            offset += count; \
        } \
        shortlist_total_entries = offset; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    if (simd_lane == 0 && sg_is_valid) { \
        const uint base = sg_offsets[simd_group_id]; \
        for (uint i = 0; i < sg_count; ++i) { \
            const uint dst = base + i; \
            if (dst < SIMDGROUP_SHORTLIST_CAP) { \
                sg_shortlists[dst] = sg_local[i]; \
            } \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    if (tid == 0) { \
        const uint out_k = min(params.k, TG_OUTPUT_K); \
        LogitPair final_heap[TG_OUTPUT_K]; \
        uint final_heap_size = 0; \
        const uint shortlist_total = min(shortlist_total_entries, (uint)SIMDGROUP_SHORTLIST_CAP); \
        for (uint i = 0; i < shortlist_total; ++i) { \
            const LogitPair cand = sg_shortlists[i]; \
            if (isfinite(cand.value) && cand.index < V) { \
                heap_insert(final_heap, &final_heap_size, out_k, cand.value, cand.index); \
            } \
        } \
        heap_to_sorted(final_heap, final_heap_size); \
        const uint output_base = tg_id * TG_OUTPUT_K; \
        for (uint i = 0; i < TG_OUTPUT_K; ++i) { \
            if (i < final_heap_size) { \
                partials_values[output_base + i] = final_heap[i].value; \
                partials_indices[output_base + i] = final_heap[i].index; \
            } else { \
                partials_values[output_base + i] = -INFINITY; \
                partials_indices[output_base + i] = 0u; \
            } \
        } \
    } \
}

// Merge kernel - NO ENTROPY HASHING, use seed directly

#define DEFINE_SAMPLE_TOPK_MERGE_AND_SAMPLE_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void sample_topk_merge_and_sample_##SUFFIX( \
    device float*         partials_values    [[buffer(0)]], \
    device uint*          partials_indices   [[buffer(1)]], \
    device uint*          out_token          [[buffer(2)]], \
    constant SampleParams& params            [[buffer(3)]], \
    uint tid                                  [[thread_position_in_threadgroup]], \
    uint tcount                               [[threads_per_threadgroup]], \
    uint tg_id                                [[threadgroup_position_in_grid]], \
    uint simd_lane_id                         [[thread_index_in_simdgroup]], \
    uint simd_group_in_tg                     [[simdgroup_index_in_threadgroup]], \
    uint simd_width_attr                      [[threads_per_simdgroup]] \
) { \
    const uint V       = params.vocab_size; \
    const uint K_req   = (params.k == 0u) ? 1u : params.k; \
    const uint K       = min(K_req, V); \
    const ACCUM top_p  = clamp(static_cast<ACCUM>(params.top_p), static_cast<ACCUM>(0.0f), static_cast<ACCUM>(1.0f)); \
    const uint num_tgs = params.num_threadgroups; \
    const uint N_cand = num_tgs * TG_OUTPUT_K; \
    \
    constexpr uint PER_THREAD_K = 16; \
    const uint requested_k = min(K, (uint)PER_THREAD_K); \
    const uint max_slots_per_thread = max(1u, (uint)(TG_MERGE_CAP / tcount)); \
    const uint heap_capacity = max(1u, min(requested_k, max_slots_per_thread)); \
    LogitPair local_heap[PER_THREAD_K]; \
    uint local_heap_size = 0; \
    \
    for (uint i = tid; i < N_cand; i += tcount) { \
        float v = partials_values[i]; \
        uint idx = partials_indices[i]; \
        if (idx < V && isfinite(v)) { \
            heap_insert(local_heap, &local_heap_size, heap_capacity, v, idx); \
        } \
    } \
    \
    heap_to_sorted(local_heap, local_heap_size); \
    \
    threadgroup LogitPair tg_shared[TG_MERGE_CAP]; \
    threadgroup LogitPair sg_best_vals[MAX_SIMDGROUPS]; \
    threadgroup uint sg_best_slots[MAX_SIMDGROUPS]; \
    threadgroup float sg_reduce_scalars[MAX_SIMDGROUPS]; \
    threadgroup uint selection_count_shared; \
    \
    const uint output_base = tg_id * TG_OUTPUT_K; \
    device float* shortlist_vals = partials_values + output_base; \
    device uint*  shortlist_indices = partials_indices + output_base; \
    \
    for (uint i = 0; i < heap_capacity; ++i) { \
        const uint sidx = tid * heap_capacity + i; \
        if (sidx < TG_MERGE_CAP) { \
            tg_shared[sidx] = (i < local_heap_size) ? local_heap[i] : LogitPair{-INFINITY, 0u}; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    const uint usable_candidates = min(tcount * heap_capacity, (uint)TG_MERGE_CAP); \
    const uint simd_width = clamp(simd_width_attr, (uint)MIN_SIMDGROUP_SIZE, (uint)MAX_SIMDGROUP_SIZE); \
    const uint simd_lane = simd_lane_id; \
    const uint simd_group_id = simd_group_in_tg; \
    const uint simd_group_count = (tcount + simd_width - 1) / simd_width; \
    constexpr uint INVALID_SLOT = 0xffffffffu; \
    \
    if (tid == 0) { \
        selection_count_shared = 0u; \
    } \
    if (tid < simd_group_count && tid < MAX_SIMDGROUPS) { \
        sg_best_slots[tid] = INVALID_SLOT; \
        sg_best_vals[tid] = LogitPair{-INFINITY, 0u}; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    constexpr uint FINAL_K = 256; \
    const uint selection_budget = min((uint)FINAL_K, max(K, 2u * params.k)); \
    for (uint iter = 0; iter < selection_budget; ++iter) { \
        float local_best_val = -INFINITY; \
        uint local_best_idx = 0u; \
        uint local_best_slot = INVALID_SLOT; \
        for (uint slot = tid; slot < usable_candidates; slot += tcount) { \
            const LogitPair cand = tg_shared[slot]; \
            if (cand.index < V && isfinite(cand.value)) { \
                const bool take = (local_best_slot == INVALID_SLOT) || \
                                  (cand.value > local_best_val) || \
                                  ((cand.value == local_best_val) && (cand.index < local_best_idx)); \
                if (take) { \
                    local_best_val = cand.value; \
                    local_best_idx = cand.index; \
                    local_best_slot = slot; \
                } \
            } \
        } \
        float sg_best_val = local_best_val; \
        uint sg_best_idx = local_best_idx; \
        uint sg_best_slot = local_best_slot; \
        for (uint offset = simd_width >> 1; offset > 0; offset >>= 1) { \
            float other_val = simd_shuffle_down(sg_best_val, offset); \
            uint other_idx = simd_shuffle_down(sg_best_idx, offset); \
            uint other_slot = simd_shuffle_down(sg_best_slot, offset); \
            const bool take_other = (other_slot != INVALID_SLOT) && \
                                    ((sg_best_slot == INVALID_SLOT) || \
                                     (other_val > sg_best_val) || \
                                     ((other_val == sg_best_val) && (other_idx < sg_best_idx))); \
            if (take_other) { \
                sg_best_val = other_val; \
                sg_best_idx = other_idx; \
                sg_best_slot = other_slot; \
            } \
        } \
        if (simd_lane == 0 && simd_group_id < MAX_SIMDGROUPS) { \
            sg_best_vals[simd_group_id] = (sg_best_slot != INVALID_SLOT) ? LogitPair{sg_best_val, sg_best_idx} : LogitPair{-INFINITY, 0u}; \
            sg_best_slots[simd_group_id] = sg_best_slot; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (tid == 0) { \
            LogitPair chosen = LogitPair{-INFINITY, 0u}; \
            uint chosen_slot = INVALID_SLOT; \
            for (uint sg = 0; sg < simd_group_count && sg < MAX_SIMDGROUPS; ++sg) { \
                const uint candidate_slot = sg_best_slots[sg]; \
                const LogitPair candidate = sg_best_vals[sg]; \
                const bool take_candidate = (candidate_slot != INVALID_SLOT) && \
                                            ((chosen_slot == INVALID_SLOT) || \
                                             (candidate.value > chosen.value) || \
                                             ((candidate.value == chosen.value) && (candidate.index < chosen.index))); \
                if (take_candidate) { \
                    chosen = candidate; \
                    chosen_slot = candidate_slot; \
                } \
            } \
            if (chosen_slot == INVALID_SLOT) { \
                break; \
            } \
            if (selection_count_shared < selection_budget) { \
                shortlist_vals[selection_count_shared] = chosen.value; \
                shortlist_indices[selection_count_shared] = chosen.index; \
                selection_count_shared += 1; \
            } \
            if (chosen_slot < TG_MERGE_CAP) { \
                tg_shared[chosen_slot].value = -INFINITY; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    const uint selection_count = min(selection_count_shared, selection_budget); \
    if (selection_count == 0) { \
        if (tid == 0) { \
            shortlist_vals[0] = -INFINITY; \
            shortlist_indices[0] = 0u; \
            out_token[0] = 0u; \
            for (uint i = 1; i < TG_OUTPUT_K; ++i) { \
                shortlist_vals[i] = -INFINITY; \
                shortlist_indices[i] = 0u; \
            } \
        } \
        return; \
    } \
    \
    float local_max = -INFINITY; \
    for (uint i = tid; i < selection_count; i += tcount) { \
        local_max = fmax(local_max, shortlist_vals[i]); \
    } \
    float sg_max = local_max; \
    for (uint offset = simd_width >> 1; offset > 0; offset >>= 1) { \
        sg_max = fmax(sg_max, simd_shuffle_down(sg_max, offset)); \
    } \
    if (simd_lane == 0 && simd_group_id < MAX_SIMDGROUPS) { \
        sg_reduce_scalars[simd_group_id] = sg_max; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    float maxv = -INFINITY; \
    if (tid == 0) { \
        for (uint sg = 0; sg < simd_group_count && sg < MAX_SIMDGROUPS; ++sg) { \
            maxv = fmax(maxv, sg_reduce_scalars[sg]); \
        } \
        sg_reduce_scalars[0] = maxv; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    maxv = sg_reduce_scalars[0]; \
    \
    float local_sum = 0.0f; \
    for (uint i = tid; i < selection_count; i += tcount) { \
        const float e = exp(shortlist_vals[i] - maxv); \
        shortlist_vals[i] = e; \
        local_sum += e; \
    } \
    float sg_sum = local_sum; \
    for (uint offset = simd_width >> 1; offset > 0; offset >>= 1) { \
        sg_sum += simd_shuffle_down(sg_sum, offset); \
    } \
    if (simd_lane == 0 && simd_group_id < MAX_SIMDGROUPS) { \
        sg_reduce_scalars[simd_group_id] = sg_sum; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    float total_sum = 0.0f; \
    if (tid == 0) { \
        for (uint sg = 0; sg < simd_group_count && sg < MAX_SIMDGROUPS; ++sg) { \
            total_sum += sg_reduce_scalars[sg]; \
        } \
        sg_reduce_scalars[0] = total_sum; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    total_sum = sg_reduce_scalars[0]; \
    \
    if (total_sum <= 0.0f || !isfinite(total_sum)) { \
        if (tid == 0) { \
            const uint fallback = shortlist_indices[0] < V ? shortlist_indices[0] : 0u; \
            out_token[0] = fallback; \
            shortlist_vals[0] = 1.0f; \
            for (uint i = 1; i < TG_OUTPUT_K; ++i) { \
                shortlist_vals[i] = 0.0f; \
                shortlist_indices[i] = 0u; \
            } \
        } \
        return; \
    } \
    \
    if (tid == 0) { \
        float cumulative = 0.0f; \
        uint cutoff = selection_count - 1; \
        float cutoff_sum = 0.0f; \
        for (uint i = 0; i < selection_count; ++i) { \
            float prob = shortlist_vals[i] / total_sum; \
            cumulative += prob; \
            shortlist_vals[i] = prob; \
            if (cumulative >= static_cast<float>(top_p)) { \
                cutoff = i; \
                cutoff_sum = cumulative; \
                break; \
            } \
        } \
        if (cutoff == selection_count - 1 && cutoff_sum == 0.0f) { \
            cutoff_sum = cumulative; \
        } \
        const float renorm = (cutoff_sum > 0.0f && isfinite(cutoff_sum)) ? (1.0f / cutoff_sum) : 1.0f; \
        for (uint i = 0; i <= cutoff; ++i) { \
            shortlist_vals[i] *= renorm; \
        } \
        RNG rng; \
        rng.seed(params.seed); \
        float r = rng.next(); \
        float acc = 0.0f; \
        uint chosen = shortlist_indices[0]; \
        for (uint i = 0; i <= cutoff; ++i) { \
            acc += shortlist_vals[i]; \
            if (r <= acc) { \
                chosen = shortlist_indices[i]; \
                break; \
            } \
        } \
        if (chosen >= V) { \
            chosen = (V > 0) ? (V - 1) : 0; \
        } \
        out_token[0] = chosen; \
        for (uint i = cutoff + 1; i < selection_count; ++i) { \
            shortlist_vals[i] = 0.0f; \
        } \
        for (uint i = selection_count; i < TG_OUTPUT_K; ++i) { \
            shortlist_vals[i] = -INFINITY; \
            shortlist_indices[i] = 0u; \
        } \
    } \
}

FOR_EACH_FLOAT_TYPE(DEFINE_SAMPLE_TOPK_PARTIALS_KERNEL)
FOR_EACH_FLOAT_TYPE(DEFINE_SAMPLE_TOPK_MERGE_AND_SAMPLE_KERNEL)
#define DEFINE_SAMPLE_TOPK_MERGE_AND_SAMPLE_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void sample_topk_merge_and_sample_##SUFFIX( \
    device float*         partials_values    [[buffer(0)]], \
    device uint*          partials_indices   [[buffer(1)]], \
    device uint*          out_token          [[buffer(2)]], \
    constant SampleParams& params            [[buffer(3)]], \
    uint tid                                  [[thread_position_in_threadgroup]], \
    uint tcount                               [[threads_per_threadgroup]], \
    uint tg_id                                [[threadgroup_position_in_grid]], \
    uint simd_lane_id                         [[thread_index_in_simdgroup]], \
    uint simd_group_in_tg                     [[simdgroup_index_in_threadgroup]], \
    uint simd_width_attr                      [[threads_per_simdgroup]] \
) { \
    const uint V       = params.vocab_size; \
    const uint K_req   = (params.k == 0u) ? 1u : params.k; \
    const uint K       = min(K_req, V); \
    const ACCUM top_p  = clamp(static_cast<ACCUM>(params.top_p), static_cast<ACCUM>(0.0f), static_cast<ACCUM>(1.0f)); \
    const uint num_tgs = params.num_threadgroups; \
    const uint N_cand = num_tgs * TG_OUTPUT_K; \
    \
    constexpr uint PER_THREAD_K = 16; \
    const uint requested_k = min(K, (uint)PER_THREAD_K); \
    const uint max_slots_per_thread = max(1u, (uint)(TG_MERGE_CAP / tcount)); \
    const uint heap_capacity = max(1u, min(requested_k, max_slots_per_thread)); \
    LogitPair local_heap[PER_THREAD_K]; \
    uint local_heap_size = 0; \
    \
    for (uint i = tid; i < N_cand; i += tcount) { \
        float v = partials_values[i]; \
        uint idx = partials_indices[i]; \
        if (idx < V && isfinite(v)) { \
            heap_insert(local_heap, &local_heap_size, heap_capacity, v, idx); \
        } \
    } \
    \
    heap_to_sorted(local_heap, local_heap_size); \
    \
    threadgroup LogitPair tg_shared[TG_MERGE_CAP]; \
    threadgroup LogitPair sg_best_vals[MAX_SIMDGROUPS]; \
    threadgroup uint sg_best_slots[MAX_SIMDGROUPS]; \
    threadgroup float sg_reduce_scalars[MAX_SIMDGROUPS]; \
    threadgroup uint selection_count_shared; \
    \
    const uint output_base = tg_id * TG_OUTPUT_K; \
    device float* shortlist_vals = partials_values + output_base; \
    device uint*  shortlist_indices = partials_indices + output_base; \
    \
    for (uint i = 0; i < heap_capacity; ++i) { \
        const uint sidx = tid * heap_capacity + i; \
        if (sidx < TG_MERGE_CAP) { \
            tg_shared[sidx] = (i < local_heap_size) ? local_heap[i] : LogitPair{-INFINITY, 0u}; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    const uint usable_candidates = min(tcount * heap_capacity, (uint)TG_MERGE_CAP); \
    const uint simd_width = clamp(simd_width_attr, (uint)MIN_SIMDGROUP_SIZE, (uint)MAX_SIMDGROUP_SIZE); \
    const uint simd_lane = simd_lane_id; \
    const uint simd_group_id = simd_group_in_tg; \
    const uint simd_group_count = (tcount + simd_width - 1) / simd_width; \
    constexpr uint INVALID_SLOT = 0xffffffffu; \
    \
    if (tid == 0) { \
        selection_count_shared = 0u; \
    } \
    if (tid < simd_group_count && tid < MAX_SIMDGROUPS) { \
        sg_best_slots[tid] = INVALID_SLOT; \
        sg_best_vals[tid] = LogitPair{-INFINITY, 0u}; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    constexpr uint FINAL_K = 256; \
    const uint selection_budget = min((uint)FINAL_K, max(K, 2u * params.k)); \
    for (uint iter = 0; iter < selection_budget; ++iter) { \
        float local_best_val = -INFINITY; \
        uint local_best_idx = 0u; \
        uint local_best_slot = INVALID_SLOT; \
        for (uint slot = tid; slot < usable_candidates; slot += tcount) { \
            const LogitPair cand = tg_shared[slot]; \
            if (cand.index < V && isfinite(cand.value)) { \
                const bool take = (local_best_slot == INVALID_SLOT) || \
                                  (cand.value > local_best_val) || \
                                  ((cand.value == local_best_val) && (cand.index < local_best_idx)); \
                if (take) { \
                    local_best_val = cand.value; \
                    local_best_idx = cand.index; \
                    local_best_slot = slot; \
                } \
            } \
        } \
        float sg_best_val = local_best_val; \
        uint sg_best_idx = local_best_idx; \
        uint sg_best_slot = local_best_slot; \
        for (uint offset = simd_width >> 1; offset > 0; offset >>= 1) { \
            float other_val = simd_shuffle_down(sg_best_val, offset); \
            uint other_idx = simd_shuffle_down(sg_best_idx, offset); \
            uint other_slot = simd_shuffle_down(sg_best_slot, offset); \
            const bool take_other = (other_slot != INVALID_SLOT) && \
                                    ((sg_best_slot == INVALID_SLOT) || \
                                     (other_val > sg_best_val) || \
                                     ((other_val == sg_best_val) && (other_idx < sg_best_idx))); \
            if (take_other) { \
                sg_best_val = other_val; \
                sg_best_idx = other_idx; \
                sg_best_slot = other_slot; \
            } \
        } \
        if (simd_lane == 0 && simd_group_id < MAX_SIMDGROUPS) { \
            sg_best_vals[simd_group_id] = (sg_best_slot != INVALID_SLOT) ? LogitPair{sg_best_val, sg_best_idx} : LogitPair{-INFINITY, 0u}; \
            sg_best_slots[simd_group_id] = sg_best_slot; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (tid == 0) { \
            LogitPair chosen = LogitPair{-INFINITY, 0u}; \
            uint chosen_slot = INVALID_SLOT; \
            for (uint sg = 0; sg < simd_group_count && sg < MAX_SIMDGROUPS; ++sg) { \
                const uint candidate_slot = sg_best_slots[sg]; \
                const LogitPair candidate = sg_best_vals[sg]; \
                const bool take_candidate = (candidate_slot != INVALID_SLOT) && \
                                            ((chosen_slot == INVALID_SLOT) || \
                                             (candidate.value > chosen.value) || \
                                             ((candidate.value == chosen.value) && (candidate.index < chosen.index))); \
                if (take_candidate) { \
                    chosen = candidate; \
                    chosen_slot = candidate_slot; \
                } \
            } \
            if (chosen_slot == INVALID_SLOT) { \
                break; \
            } \
            if (selection_count_shared < selection_budget) { \
                shortlist_vals[selection_count_shared] = chosen.value; \
                shortlist_indices[selection_count_shared] = chosen.index; \
                selection_count_shared += 1; \
            } \
            if (chosen_slot < TG_MERGE_CAP) { \
                tg_shared[chosen_slot].value = -INFINITY; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    const uint selection_count = min(selection_count_shared, selection_budget); \
    if (selection_count == 0) { \
        if (tid == 0) { \
            shortlist_vals[0] = -INFINITY; \
            shortlist_indices[0] = 0u; \
            out_token[0] = 0u; \
            for (uint i = 1; i < TG_OUTPUT_K; ++i) { \
                shortlist_vals[i] = -INFINITY; \
                shortlist_indices[i] = 0u; \
            } \
        } \
        return; \
    } \
    \
    float local_max = -INFINITY; \
    for (uint i = tid; i < selection_count; i += tcount) { \
        local_max = fmax(local_max, shortlist_vals[i]); \
    } \
    float sg_max = local_max; \
    for (uint offset = simd_width >> 1; offset > 0; offset >>= 1) { \
        sg_max = fmax(sg_max, simd_shuffle_down(sg_max, offset)); \
    } \
    if (simd_lane == 0 && simd_group_id < MAX_SIMDGROUPS) { \
        sg_reduce_scalars[simd_group_id] = sg_max; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    float maxv = -INFINITY; \
    if (tid == 0) { \
        for (uint sg = 0; sg < simd_group_count && sg < MAX_SIMDGROUPS; ++sg) { \
            maxv = fmax(maxv, sg_reduce_scalars[sg]); \
        } \
        sg_reduce_scalars[0] = maxv; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    maxv = sg_reduce_scalars[0]; \
    \
    float local_sum = 0.0f; \
    for (uint i = tid; i < selection_count; i += tcount) { \
        const float e = exp(shortlist_vals[i] - maxv); \
        shortlist_vals[i] = e; \
        local_sum += e; \
    } \
    float sg_sum = local_sum; \
    for (uint offset = simd_width >> 1; offset > 0; offset >>= 1) { \
        sg_sum += simd_shuffle_down(sg_sum, offset); \
    } \
    if (simd_lane == 0 && simd_group_id < MAX_SIMDGROUPS) { \
        sg_reduce_scalars[simd_group_id] = sg_sum; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    float total_sum = 0.0f; \
    if (tid == 0) { \
        for (uint sg = 0; sg < simd_group_count && sg < MAX_SIMDGROUPS; ++sg) { \
            total_sum += sg_reduce_scalars[sg]; \
        } \
        sg_reduce_scalars[0] = total_sum; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    total_sum = sg_reduce_scalars[0]; \
    \
    if (total_sum <= 0.0f || !isfinite(total_sum)) { \
        if (tid == 0) { \
            const uint fallback = shortlist_indices[0] < V ? shortlist_indices[0] : 0u; \
            out_token[0] = fallback; \
            shortlist_vals[0] = 1.0f; \
            for (uint i = 1; i < TG_OUTPUT_K; ++i) { \
                shortlist_vals[i] = 0.0f; \
                shortlist_indices[i] = 0u; \
            } \
        } \
        return; \
    } \
    \
    if (tid == 0) { \
        float cumulative = 0.0f; \
        uint cutoff = selection_count - 1; \
        float cutoff_sum = 0.0f; \
        for (uint i = 0; i < selection_count; ++i) { \
            float prob = shortlist_vals[i] / total_sum; \
            cumulative += prob; \
            shortlist_vals[i] = prob; \
            if (cumulative >= static_cast<float>(top_p)) { \
                cutoff = i; \
                cutoff_sum = cumulative; \
                break; \
            } \
        } \
        if (cutoff == selection_count - 1 && cutoff_sum == 0.0f) { \
            cutoff_sum = cumulative; \
        } \
        const float renorm = (cutoff_sum > 0.0f && isfinite(cutoff_sum)) ? (1.0f / cutoff_sum) : 1.0f; \
        for (uint i = 0; i <= cutoff; ++i) { \
            shortlist_vals[i] *= renorm; \
        } \
        RNG rng; \
        rng.seed(params.seed); \
        float r = rng.next(); \
        float acc = 0.0f; \
        uint chosen = shortlist_indices[0]; \
        for (uint i = 0; i <= cutoff; ++i) { \
            acc += shortlist_vals[i]; \
            if (r <= acc) { \
                chosen = shortlist_indices[i]; \
                break; \
            } \
        } \
        if (chosen >= V) { \
            chosen = (V > 0) ? (V - 1) : 0; \
        } \
        out_token[0] = chosen; \
        for (uint i = cutoff + 1; i < selection_count; ++i) { \
            shortlist_vals[i] = 0.0f; \
        } \
        for (uint i = selection_count; i < TG_OUTPUT_K; ++i) { \
            shortlist_vals[i] = -INFINITY; \
            shortlist_indices[i] = 0u; \
        } \
    } \
}
