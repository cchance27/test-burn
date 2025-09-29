#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16) \
    OP(bfloat, float, bf16)

#define DEFINE_SILU_KERNEL(SCALAR, ACCUM, SUFFIX) \
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) \
kernel void silu_kernel_##SUFFIX(device SCALAR* input [[buffer(0)]], \
                                device SCALAR* output [[buffer(1)]], \
                                constant uint& total_elements [[buffer(2)]], \
                                uint gid [[thread_position_in_grid]]) { \
    if (gid >= total_elements) return; \
    ACCUM x = static_cast<ACCUM>(input[gid]); \
    if (x > static_cast<ACCUM>(20.0f)) { \
        if (x > static_cast<ACCUM>(80.0f)) { \
            output[gid] = static_cast<SCALAR>(x); \
        } else { \
            ACCUM sig = static_cast<ACCUM>(1.0f) / (static_cast<ACCUM>(1.0f) + exp(-x)); \
            output[gid] = static_cast<SCALAR>(x * sig); \
        } \
    } else if (x < static_cast<ACCUM>(-20.0f)) { \
        if (x < static_cast<ACCUM>(-80.0f)) { \
            output[gid] = static_cast<SCALAR>(0.0f); \
        } else { \
            ACCUM sig = static_cast<ACCUM>(1.0f) / (static_cast<ACCUM>(1.0f) + exp(-x)); \
            output[gid] = static_cast<SCALAR>(x * sig); \
        } \
    } else { \
        ACCUM sig = static_cast<ACCUM>(1.0f) / (static_cast<ACCUM>(1.0f) + exp(-x)); \
        output[gid] = static_cast<SCALAR>(x * sig); \
    } \
}

FOR_EACH_FLOAT_TYPE(DEFINE_SILU_KERNEL)

#undef DEFINE_SILU_KERNEL
#undef FOR_EACH_FLOAT_TYPE