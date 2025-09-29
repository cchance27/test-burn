#include <metal_stdlib>
using namespace metal;

constant float EPS = 1e-6f;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16) \
    OP(bfloat, float, bf16)

#define DEFINE_RMSNORM_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void rmsnorm_kernel_##SUFFIX(device SCALAR* input [[buffer(0)]], \
                                    device SCALAR* output [[buffer(1)]], \
                                    device SCALAR* gamma [[buffer(2)]], \
                                    constant uint& feature_dim [[buffer(3)]], \
                                    constant uint& total_elements [[buffer(4)]], \
                                    uint gid [[thread_position_in_grid]]) { \
    if (gid >= total_elements) return; \

    uint feature_idx = gid % feature_dim; \
    uint row_idx = gid / feature_dim; \

    ACCUM sum_sq = static_cast<ACCUM>(0.0f); \
    ACCUM feature_dim_acc = static_cast<ACCUM>(feature_dim); \
    for (uint f = 0; f < feature_dim; ++f) { \
        ACCUM v = static_cast<ACCUM>(input[row_idx * feature_dim + f]); \
        sum_sq += v * v; \
    } \

    ACCUM rms = sqrt(sum_sq / feature_dim_acc + static_cast<ACCUM>(EPS)); \
    ACCUM x = static_cast<ACCUM>(input[gid]); \
    ACCUM gamma_val = static_cast<ACCUM>(gamma[feature_idx]); \

    output[gid] = static_cast<SCALAR>((x / rms) * gamma_val); \
}

FOR_EACH_FLOAT_TYPE(DEFINE_RMSNORM_KERNEL)

#undef DEFINE_RMSNORM_KERNEL
#undef FOR_EACH_FLOAT_TYPE