#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16) \
    OP(bfloat, float, bf16)

#define DEFINE_GELU_KERNEL(SCALAR, ACCUM, SUFFIX) \
// More numerically stable GELU implementation \
kernel void gelu_kernel_##SUFFIX( \
    device SCALAR* input [[buffer(0)]], \
    device SCALAR* output [[buffer(1)]], \
    constant uint& total_elements [[buffer(2)]], \
    uint gid [[thread_position_in_grid]]) { \
    if (gid >= total_elements) { \
        return; \
    } \
    ACCUM x = static_cast<ACCUM>(input[gid]); \
    if (x > static_cast<ACCUM>(5.0f)) { \
        output[gid] = static_cast<SCALAR>(x); \
        return; \
    } \
    if (x < static_cast<ACCUM>(-5.0f)) { \
        output[gid] = static_cast<SCALAR>(0.0f); \
        return; \
    } \
    ACCUM abs_x = fabs(x); \
    ACCUM x3; \
    if (abs_x < static_cast<ACCUM>(1e-10f)) { \
        x3 = static_cast<ACCUM>(0.0f); \
    } else if (abs_x > static_cast<ACCUM>(10.0f)) { \
        x3 = copysign(exp(static_cast<ACCUM>(3.0f) * log(abs_x)), x); \
    } else { \
        x3 = x * x * x; \
    } \
    ACCUM inner = sqrt(static_cast<ACCUM>(2.0f / 3.141592653589793f)) * (x + static_cast<ACCUM>(0.044715f) * x3); \
    ACCUM tanh_inner = tanh(inner); \
    output[gid] = static_cast<SCALAR>(static_cast<ACCUM>(0.5f) * x * (static_cast<ACCUM>(1.0f) + tanh_inner)); \
}

FOR_EACH_FLOAT_TYPE(DEFINE_GELU_KERNEL)

#undef DEFINE_GELU_KERNEL
#undef FOR_EACH_FLOAT_TYPE
