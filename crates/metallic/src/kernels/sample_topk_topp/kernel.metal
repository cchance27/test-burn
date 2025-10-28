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
    constexpr uint SIMDGROUP_CANDIDATE_CAP = MAX_SIMDGROUP_SIZE * MAX_LANE_HEAP; \
    constexpr uint SIMDGROUP_SHORTLIST_CAP = MAX_SIMDGROUPS * SIMDGROUP_SHORTLIST; \
    constexpr uint SIMDGROUP_CANDIDATE_CAP_TOTAL = MAX_SIMDGROUPS * SIMDGROUP_CANDIDATE_CAP; \
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
    threadgroup LogitPair sg_candidates[SIMDGROUP_CANDIDATE_CAP_TOTAL]; \
    threadgroup LogitPair sg_shortlists[SIMDGROUP_SHORTLIST_CAP]; \
    for (uint idx = tid; idx < SIMDGROUP_CANDIDATE_CAP_TOTAL; idx += tptg) { \
        sg_candidates[idx] = LogitPair{-INFINITY, 0u}; \
    } \
    for (uint idx = tid; idx < SIMDGROUP_SHORTLIST_CAP; idx += tptg) { \
        sg_shortlists[idx] = LogitPair{-INFINITY, 0u}; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    if (simd_group_id < MAX_SIMDGROUPS) { \
        const uint lane_base = simd_group_id * SIMDGROUP_CANDIDATE_CAP + simd_lane * MAX_LANE_HEAP; \
        for (uint m = 0; m < MAX_LANE_HEAP; ++m) { \
            const uint idx = lane_base + m; \
            if (idx < SIMDGROUP_CANDIDATE_CAP_TOTAL) { \
                sg_candidates[idx] = (m < lane_heap_size) ? lane_heap[m] : LogitPair{-INFINITY, 0u}; \
            } \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    if (simd_group_id < min(simd_group_count, (uint)MAX_SIMDGROUPS) && simd_lane == 0) { \
        LogitPair sg_heap[SIMDGROUP_SHORTLIST]; \
        uint sg_heap_size = 0; \
        const uint shortlist_base = simd_group_id * SIMDGROUP_SHORTLIST; \
        const uint candidate_base = simd_group_id * SIMDGROUP_CANDIDATE_CAP; \
        const uint lanes_in_group = min(simd_width, tptg - simd_group_id * simd_width); \
        const uint candidate_count = min(lanes_in_group * MAX_LANE_HEAP, (uint)SIMDGROUP_CANDIDATE_CAP); \
        for (uint i = 0; i < candidate_count; ++i) { \
            const uint idx = candidate_base + i; \
            if (idx < SIMDGROUP_CANDIDATE_CAP_TOTAL) { \
                const LogitPair cand = sg_candidates[idx]; \
                if (isfinite(cand.value) && cand.index < V) { \
                    heap_insert(sg_heap, &sg_heap_size, SIMDGROUP_SHORTLIST, cand.value, cand.index); \
                } \
            } \
        } \
        heap_to_sorted(sg_heap, sg_heap_size); \
        for (uint i = 0; i < SIMDGROUP_SHORTLIST; ++i) { \
            const uint idx = shortlist_base + i; \
            if (idx < SIMDGROUP_SHORTLIST_CAP) { \
                sg_shortlists[idx] = (i < sg_heap_size) ? sg_heap[i] : LogitPair{-INFINITY, 0u}; \
            } \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    if (tid == 0) { \
        const uint out_k = min(params.k, TG_OUTPUT_K); \
        LogitPair final_heap[TG_OUTPUT_K]; \
        uint final_heap_size = 0; \
        const uint shortlist_total = min(simd_group_count, (uint)MAX_SIMDGROUPS) * SIMDGROUP_SHORTLIST; \
        for (uint i = 0; i < shortlist_total; ++i) { \
            if (i < SIMDGROUP_SHORTLIST_CAP) { \
                const LogitPair cand = sg_shortlists[i]; \
                if (isfinite(cand.value) && cand.index < V) { \
                    heap_insert(final_heap, &final_heap_size, out_k, cand.value, cand.index); \
                } \
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
    uint tcount                               [[threads_per_threadgroup]] \
) { \
    const uint V       = params.vocab_size; \
    const uint K_req   = (params.k == 0u) ? 1u : params.k; \
    const uint K       = min(K_req, V); \
    const ACCUM top_p  = clamp(static_cast<ACCUM>(params.top_p), static_cast<ACCUM>(0.0f), static_cast<ACCUM>(1.0f)); \
    const uint num_tgs = params.num_threadgroups; \
    const uint N_cand = num_tgs * TG_OUTPUT_K; \
    \
    constexpr uint PER_THREAD_K = 16; \
    const uint local_k = min(K, (uint)PER_THREAD_K); \
    LogitPair local_heap[PER_THREAD_K]; \
    uint local_heap_size = 0; \
    \
    for (uint i = tid; i < N_cand; i += tcount) { \
        float v = partials_values[i]; \
        uint idx = partials_indices[i]; \
        if (idx < V && isfinite(v)) { \
            heap_insert(local_heap, &local_heap_size, local_k, v, idx); \
        } \
    } \
    \
    heap_to_sorted(local_heap, local_heap_size); \
    \
    threadgroup LogitPair tg_shared[4096]; \
    for (uint i = 0; i < local_k; ++i) { \
        const uint sidx = tid * local_k + i; \
        if (sidx < 4096) { \
            tg_shared[sidx] = (i < local_heap_size) ? local_heap[i] : LogitPair{-INFINITY, 0u}; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    if (tid == 0) { \
        constexpr uint FINAL_K = 256; \
        const uint final_k = min((uint)FINAL_K, max(K, 2u * params.k)); \
        LogitPair global_heap[FINAL_K]; \
        uint global_heap_size = 0; \
        \
        const uint total = min(tcount * local_k, (uint)4096); \
        for (uint i = 0; i < total; ++i) { \
            if (isfinite(tg_shared[i].value)) { \
                heap_insert(global_heap, &global_heap_size, final_k, tg_shared[i].value, tg_shared[i].index); \
            } \
        } \
        \
        heap_to_sorted(global_heap, global_heap_size); \
        \
        if (global_heap_size == 0) { \
            out_token[0] = 0u; \
            return; \
        } \
        \
        float maxv = global_heap[0].value; \
        float sum = 0.0f; \
        for (uint i = 0; i < global_heap_size; ++i) { \
            float e = exp(global_heap[i].value - maxv); \
            global_heap[i].value = e; \
            sum += e; \
        } \
        \
        if (sum <= 0.0f || !isfinite(sum)) { \
            out_token[0] = global_heap[0].index; \
            return; \
        } \
        \
        float cumulative = 0.0f; \
        uint cutoff = global_heap_size - 1; \
        float cutoff_sum = 0.0f; \
        for (uint i = 0; i < global_heap_size; ++i) { \
            float prob = global_heap[i].value / sum; \
            cumulative += prob; \
            global_heap[i].value = prob; \
            if (cumulative >= static_cast<float>(top_p)) { \
                cutoff = i; \
                cutoff_sum = cumulative; \
                break; \
            } \
        } \
        if (cutoff == global_heap_size - 1 && cutoff_sum == 0.0f) { \
            cutoff_sum = cumulative; \
        } \
        const float renorm = (cutoff_sum > 0.0f && isfinite(cutoff_sum)) ? (1.0f / cutoff_sum) : 1.0f; \
        for (uint i = 0; i <= cutoff; ++i) { \
            global_heap[i].value *= renorm; \
        } \
        \
        RNG rng; \
        rng.seed(params.seed); \
        float r = rng.next(); \
        \
        float acc = 0.0f; \
        uint chosen = global_heap[0].index; \
        for (uint i = 0; i <= cutoff; ++i) { \
            acc += global_heap[i].value; \
            if (r <= acc) { \
                chosen = global_heap[i].index; \
                break; \
            } \
        } \
        if (chosen >= V) { \
            chosen = V > 0 ? (V - 1) : 0; \
        } \
        out_token[0] = chosen; \
    } \
}

FOR_EACH_FLOAT_TYPE(DEFINE_SAMPLE_TOPK_PARTIALS_KERNEL)
FOR_EACH_FLOAT_TYPE(DEFINE_SAMPLE_TOPK_MERGE_AND_SAMPLE_KERNEL)
#define SIMDGROUP_CANDIDATE_CAP (MAX_SIMDGROUP_SIZE * MAX_LANE_HEAP)
#define SIMDGROUP_CANDIDATE_CAP_TOTAL (MAX_SIMDGROUPS * SIMDGROUP_CANDIDATE_CAP)
