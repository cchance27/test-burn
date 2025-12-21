#ifndef MATMUL_GEMV_COMMON_DEFS
#define MATMUL_GEMV_COMMON_DEFS

#include <metal_stdlib>
using namespace metal;

#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE inline __attribute__((always_inline))
#endif

struct GemvParams {
    uint K;
    uint N;
    uint blocks_per_k;
    uint weights_per_block;
    uint batch;
    uint stride_x;
    uint stride_y;
    uint stride_a;
    uint stride_w;
    uint stride_scale;
};

enum GemvLoaderMode : uint {
    GemvLoaderDense = 0,
    GemvLoaderDenseBias = 1,
    GemvLoaderQ8Canonical = 2,
    GemvLoaderQ8CanonicalBias = 3,
    GemvLoaderQ8CanonicalDebug = 4,
    GemvLoaderDenseCanonical = 5,
    GemvLoaderDenseCanonicalBias = 6,
    GemvLoaderDenseStrided = 7,
    GemvLoaderDenseStridedBias = 8,
};

struct QkvFusedParams {
    uint K;
    uint Nq;
    uint Nk;
    uint Nv;
    uint blocks_per_k;
    uint weights_per_block;
    uint has_bias_q;
    uint has_bias_k;
    uint has_bias_v;
};

struct Q2FusedParams {
    uint K;
    uint N0;
    uint N1;
    uint blocks_per_k;
    uint weights_per_block;
    uint has_bias0;
    uint has_bias1;
};

template <uint COLS>
struct Q8FusedHeadOut {
    device half *out;
    const device half *bias;
    uint N;
    uint has_bias;
};

ALWAYS_INLINE bool q8_should_use_wide(uint K, uint max_cols) {
    return (K < 4096u) || (max_cols >= 896u);
}

#endif // MATMUL_GEMV_COMMON_DEFS
