#include <metal_stdlib>
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
    uint num_tgs                              [[threadgroups_per_grid]] \
) { \
    const uint V     = params.vocab_size; \
    const float denom = fmax(params.temperature, 1e-6f); \
    const ACCUM invT = static_cast<ACCUM>(1.0f / denom); \
    const uint M     = params.per_thread_m; \
    \
    constexpr uint MAX_M = 32; \
    LogitPair heap[MAX_M]; \
    const uint M_use = min(M, (uint)MAX_M); \
    uint heap_size = 0; \
    \
    const uint items_per_tg = V / num_tgs + 1; \
    const uint block_start = tg_id * items_per_tg; \
    const uint block_end = min(block_start + items_per_tg, V); \
    \
    for (uint i = block_start + tid; i < block_end; i += tptg) { \
        ACCUM v = static_cast<ACCUM>(logits[i]); \
        v *= invT; \
        heap_insert(heap, &heap_size, M_use, static_cast<float>(v), i); \
    } \
    \
    heap_to_sorted(heap, heap_size); \
    \
    threadgroup LogitPair tg_shared[4096]; \
    const uint max_per_thread = max((uint)1, 4096u / tptg); \
    const uint write_count = min((uint)M_use, max_per_thread); \
    for (uint m = 0; m < write_count; ++m) { \
        const uint idx = tid * write_count + m; \
        if (idx < 4096) { \
            tg_shared[idx] = (m < heap_size) ? heap[m] : LogitPair{-INFINITY, 0u}; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    const uint merge_threads = min(tptg, (uint)32); \
    \
    if (tid < merge_threads) { \
        const uint out_k = min(params.k, TG_OUTPUT_K); \
        LogitPair my_heap[TG_OUTPUT_K]; \
        uint my_heap_size = 0; \
        \
        const uint total_items = min(tptg * write_count, (uint)4096); \
        for (uint i = tid; i < total_items; i += merge_threads) { \
            if (isfinite(tg_shared[i].value)) { \
                heap_insert(my_heap, &my_heap_size, out_k, tg_shared[i].value, tg_shared[i].index); \
            } \
        } \
        \
        heap_to_sorted(my_heap, my_heap_size); \
        \
        const uint my_base = tid * out_k; \
        for (uint i = 0; i < out_k; ++i) { \
            const uint idx = my_base + i; \
            if (idx < 4096) { \
                tg_shared[idx] = (i < my_heap_size) ? my_heap[i] : LogitPair{-INFINITY, 0u}; \
            } \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    if (tid == 0) { \
        const uint out_k = min(params.k, TG_OUTPUT_K); \
        LogitPair final_heap[TG_OUTPUT_K]; \
        uint final_heap_size = 0; \
        \
        const uint merge_items = min(merge_threads * out_k, (uint)4096); \
        for (uint i = 0; i < merge_items; ++i) { \
            if (isfinite(tg_shared[i].value)) { \
                heap_insert(final_heap, &final_heap_size, out_k, tg_shared[i].value, tg_shared[i].index); \
            } \
        } \
        \
        heap_to_sorted(final_heap, final_heap_size); \
        \
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
        float inv_sum = 1.0f / sum; \
        for (uint i = 0; i < global_heap_size; ++i) { \
            global_heap[i].value *= inv_sum; \
        } \
        \
        float cumulative = 0.0f; \
        uint cutoff = global_heap_size - 1; \
        for (uint i = 0; i < global_heap_size; ++i) { \
            cumulative += global_heap[i].value; \
            if (cumulative >= static_cast<float>(top_p)) { \
                cutoff = i; \
                break; \
            } \
        } \
        \
        RNG rng; \
        rng.seed(params.seed); \
        float r = rng.next() * cumulative; \
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