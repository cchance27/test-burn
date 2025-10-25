#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16)

// ------------------------- Common structs -------------------------
struct SampleParams {
    uint   vocab_size;
    uint   k;
    float  top_p;
    float  temperature;
    uint   seed;
    uint   per_thread_m;     // set to 1 for lean path
    uint   num_threadgroups; // filled by host for kernel1
};

struct LogitPair { float value; uint index; };

struct RNG {
    uint state;
    inline void seed(uint s) { state = s ? s : 1u; }
    inline float next() {
        state = state * 1664525u + 1013904223u;
        // Convert uint32 to float in [0, 1) with high precision
        // by using the top 23 bits of the state for the mantissa.
        union {
            uint  u;
            float f;
        } temp;
        temp.u = (state >> 9) | 0x3f800000u; // A float in [1.0, 2.0)
        return temp.f - 1.0f; // A float in [0.0, 1.0)
    }
};

constant uint MAX_TPTG = 256;

inline void k_insert_sorted_dynamic(thread LogitPair* arr, uint M, float x, uint idx) {
    if (M == 0u) return;
    if (x <= arr[M - 1].value) return;
    int pos = int(M) - 1;
    while (pos > 0 && x > arr[pos - 1].value) {
        arr[pos] = arr[pos - 1];
        pos--;
    }
    arr[pos].value = x;
    arr[pos].index = idx;
}

// ------------------------- Kernel templates -------------------------
#define DEFINE_SAMPLE_TOPK_PARTIALS_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void sample_topk_partials_##SUFFIX( \
    const device SCALAR*  logits             [[buffer(0)]], \
    device LogitPair*     partials           [[buffer(1)]], /* size: num_tgs * threads_per_tg * per_thread_m */ \
    constant SampleParams& params           [[buffer(2)]], \
    uint3 tid_tg_vec                        [[thread_position_in_threadgroup]], \
    uint3 tg_size_vec                       [[threads_per_threadgroup]], \
    uint3 tg_pos                            [[threadgroup_position_in_grid]], \
    uint3 tg_cnt                            [[threadgroups_per_grid]] \
) { \
    const uint tid   = tid_tg_vec.x; \
    const uint tptg  = tg_size_vec.x; \
    const uint tg_id = tg_pos.x; \
    const uint num_tgs = tg_cnt.x; \
\
    const uint V     = params.vocab_size; \
    const ACCUM invT = (params.temperature > 0.f) ? (static_cast<ACCUM>(1.f) / static_cast<ACCUM>(params.temperature)) : static_cast<ACCUM>(0.f); \
    const uint M     = params.per_thread_m; /* elements per thread - should be set to top_k or reasonable limit */ \
\
    /* Global thread id and total threads across ALL TGs */ \
    const uint gtid     = tg_id * tptg + tid; \
    const uint total_thr = tptg * num_tgs; \
\
    /* Per-thread top-M array */ \
    constexpr uint MAX_M = 32; /* Reduce to fit TG memory on M1-class GPUs */ \
    LogitPair thread_top[MAX_M]; \
    \
    /* Initialize with -INFINITY */ \
    for (uint i = 0; i < M && i < MAX_M; ++i) { \
        thread_top[i].value = -INFINITY; \
        thread_top[i].index = 0u; \
    } \
\
    /* Process elements assigned to this thread */ \
    for (uint i = gtid; i < V; i += total_thr) { \
        ACCUM v = static_cast<ACCUM>(logits[i]); \
        if (invT > static_cast<ACCUM>(0.0f)) v *= invT; \
        \
        /* Insert into sorted array to maintain top-M */ \
        if (M == 1) { \
            /* Special case for efficiency */ \
            if (v > thread_top[0].value) { \
                thread_top[0].value = static_cast<float>(v); \
                thread_top[0].index = i; \
            } \
        } else { \
            /* Insert into sorted array to maintain top-M (clamped to buffer size) */ \
            const uint M_use = (M < MAX_M) ? M : MAX_M; \
            k_insert_sorted_dynamic(thread_top, M_use, static_cast<float>(v), i); \
        } \
    } \
\
    /* Write M candidates per thread */ \
    /* Layout: [gtid * M + m] for thread gtid, element m */ \
    for (uint m = 0; m < M && m < MAX_M; ++m) { \
        partials[gtid * M + m] = thread_top[m]; \
    } \
}

// ------------------------- Kernel 2: merge + sample -------------------------
#define DEFINE_SAMPLE_TOPK_MERGE_AND_SAMPLE_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void sample_topk_merge_and_sample_##SUFFIX( \
    device LogitPair*     partials           [[buffer(0)]], \
    device uint*          out_token          [[buffer(1)]], \
    constant SampleParams& params            [[buffer(2)]], \
    uint3 tid_tg_vec                         [[thread_position_in_threadgroup]], \
    uint3 tg_size_vec                        [[threads_per_threadgroup]] \
) { \
    const uint tid_tg = tid_tg_vec.x; \
    const uint tcount = tg_size_vec.x; \
\
    const uint V       = params.vocab_size; \
    const uint K_req   = (params.k == 0u) ? 1u : params.k; \
    const uint K       = (K_req <= V) ? K_req : V; \
    const ACCUM top_p  = clamp(static_cast<ACCUM>(params.top_p), static_cast<ACCUM>(0.0f), static_cast<ACCUM>(1.0f)); \
    const uint num_tgs = params.num_threadgroups; \
    const uint M       = params.per_thread_m; /* Use the parameter passed from partials kernel */ \
\
    const uint N_cand = num_tgs * tcount * M; \
\
    constexpr uint KLOCAL_MAX = 12; \
    const uint Klocal = (K <= KLOCAL_MAX) ? K : KLOCAL_MAX; \
\
    LogitPair local_top[KLOCAL_MAX]; \
    for (uint i = 0; i < Klocal; ++i) { \
        local_top[i].value = -INFINITY; \
        local_top[i].index = 0u; \
    } \
\
    for (uint i = tid_tg; i < N_cand; i += tcount) { \
        const float v = partials[i].value; \
        const uint  ix = partials[i].index; \
        if (v > local_top[Klocal - 1].value) { \
            k_insert_sorted_dynamic(local_top, Klocal, v, ix); \
        } \
    } \
\
    /* Store per-thread local_top into threadgroup memory and merge from there */ \
    threadgroup LogitPair tg_local[MAX_TPTG * KLOCAL_MAX]; \
    const uint base = tid_tg * Klocal; \
    for (uint i = 0; i < Klocal; ++i) { tg_local[base + i] = local_top[i]; } \
\
    threadgroup_barrier(mem_flags::mem_threadgroup); \
\
    /* Parallel pre-reduction within merge kernel */ \
    constexpr uint K_MAX = 64; \
    const uint K_use = (K <= K_MAX) ? K : K_MAX; \
    constexpr uint L2 = 3; \
\
    LogitPair local2[L2]; \
    for (uint i = 0; i < L2; ++i) { local2[i].value = -INFINITY; local2[i].index = 0u; } \
    const uint total_small = tcount * Klocal; \
    for (uint i = tid_tg; i < total_small; i += tcount) { \
        const float v = tg_local[i].value; \
        const uint ix = tg_local[i].index; \
        if (v > local2[L2 - 1].value) { k_insert_sorted_dynamic(local2, L2, v, ix); } \
    } \
\
    threadgroup LogitPair tg_pre[L2 * MAX_TPTG]; \
    const uint base2 = tid_tg * L2; \
    for (uint i = 0; i < L2; ++i) { tg_pre[base2 + i] = local2[i]; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
\
    if (tid_tg == 0) { \
        LogitPair global_top[K_MAX]; \
        for (uint i = 0; i < K_use; ++i) { global_top[i].value = -INFINITY; global_top[i].index = 0u; } \
\
        const uint reduced = tcount * L2; \
        for (uint i = 0; i < reduced; ++i) { \
            const float v = tg_pre[i].value; \
            const uint  ix = tg_pre[i].index; \
            if (v > global_top[K_use - 1].value) { \
                k_insert_sorted_dynamic(global_top, K_use, v, ix); \
            } \
        } \
\
        /* Softmax over K */ \
        float maxv = global_top[0].value; \
        float sum  = 0.0f; \
        for (uint i = 0; i < K_use; ++i) { \
            const float e = exp(global_top[i].value - maxv); \
            global_top[i].value = e; \
            sum += e; \
        } \
        if (sum <= 0.0f || !isfinite(sum)) { \
            out_token[0] = global_top[0].index; \
            return; \
        } \
        const float inv_sum = 1.0f / sum; \
        for (uint i = 0; i < K_use; ++i) { \
            global_top[i].value *= inv_sum; \
        } \
\
        /* Top-p cutoff */ \
        float cumulative = 0.0f; \
        uint cutoff = (K_use > 0u) ? (K_use - 1u) : 0u; \
        for (uint i = 0; i < K_use; ++i) { \
            cumulative += global_top[i].value; \
            if (cumulative >= static_cast<float>(top_p)) { cutoff = i; break; } \
        } \
        if (cumulative <= 0.0f) { \
            out_token[0] = global_top[0].index; \
            return; \
        } \
\
        /* Sample */ \
        RNG rng; rng.seed(params.seed); \
        const float r = rng.next() * cumulative; \
\
        float acc = 0.0f; \
        uint chosen = global_top[0].index; \
        for (uint i = 0; i <= cutoff; ++i) { \
            acc += global_top[i].value; \
            if (r <= acc) { chosen = global_top[i].index; break; } \
        } \
        out_token[0] = chosen; \
    } \
}

// Apply the template macros to generate kernels for each type
FOR_EACH_FLOAT_TYPE(DEFINE_SAMPLE_TOPK_PARTIALS_KERNEL)
FOR_EACH_FLOAT_TYPE(DEFINE_SAMPLE_TOPK_MERGE_AND_SAMPLE_KERNEL)

#undef DEFINE_SAMPLE_TOPK_PARTIALS_KERNEL
#undef DEFINE_SAMPLE_TOPK_MERGE_AND_SAMPLE_KERNEL
#undef FOR_EACH_FLOAT_TYPE