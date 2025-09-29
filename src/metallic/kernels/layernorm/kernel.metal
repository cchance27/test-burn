#include <metal_stdlib>
using namespace metal;

constant float EPS = 1e-5f;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16) \
    OP(bfloat, float, bf16)

#define DEFINE_LAYERNORM_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void layernorm_kernel_##SUFFIX( \
    device SCALAR* input [[buffer(0)]], \
    device SCALAR* output [[buffer(1)]], \
    device SCALAR* gamma [[buffer(2)]], \
    device SCALAR* beta [[buffer(3)]], \
    constant uint& feature_dim [[buffer(4)]], \
    constant uint& total_elements [[buffer(5)]], \
    uint gid [[thread_position_in_grid]]) { \
    if (gid >= total_elements) { \
        return; \
    } \
    uint feature_idx = gid % feature_dim; \
    uint row_idx = gid / feature_dim; \
    ACCUM sum = static_cast<ACCUM>(0.0f); \
    ACCUM sum_sq = static_cast<ACCUM>(0.0f); \
    ACCUM feature_dim_acc = static_cast<ACCUM>(feature_dim); \
    for (uint f = 0; f < feature_dim; ++f) { \
        ACCUM val = static_cast<ACCUM>(input[row_idx * feature_dim + f]); \
        sum += val; \
        sum_sq += val * val; \
    } \
    ACCUM mean = sum / feature_dim_acc; \
    ACCUM var = (sum_sq - mean * sum) / feature_dim_acc; \
    ACCUM x = static_cast<ACCUM>(input[gid]); \
    ACCUM normalized = (x - mean) / sqrt(var + static_cast<ACCUM>(EPS)); \
    ACCUM gamma_val = static_cast<ACCUM>(gamma[feature_idx]); \
    ACCUM beta_val = static_cast<ACCUM>(beta[feature_idx]); \
    output[gid] = static_cast<SCALAR>(normalized * gamma_val + beta_val); \
}

FOR_EACH_FLOAT_TYPE(DEFINE_LAYERNORM_KERNEL)

#undef DEFINE_LAYERNORM_KERNEL
#undef FOR_EACH_FLOAT_TYPE
