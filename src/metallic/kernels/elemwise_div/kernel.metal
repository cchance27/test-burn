#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16) 
#define DEFINE_ELEMWISE_DIV_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void div_kernel_##SUFFIX(device const SCALAR* a [[buffer(0)]], \
                                device const SCALAR* b [[buffer(1)]], \
                                device SCALAR* out [[buffer(2)]], \
                                constant uint& total_elements [[buffer(3)]], \
                                uint gid [[thread_position_in_grid]]) { \
    if (gid >= total_elements) return; \
    ACCUM a_val = static_cast<ACCUM>(a[gid]); \
    ACCUM b_val = static_cast<ACCUM>(b[gid]); \
    out[gid] = static_cast<SCALAR>(a_val / b_val); \
}

FOR_EACH_FLOAT_TYPE(DEFINE_ELEMWISE_DIV_KERNEL)

#undef DEFINE_ELEMWISE_DIV_KERNEL
#undef FOR_EACH_FLOAT_TYPE
    