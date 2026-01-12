#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16) 
    
#define DEFINE_ELEMWISE_ABS_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void abs_kernel_##SUFFIX(device const SCALAR* a [[buffer(0)]], \
                                device SCALAR* out [[buffer(1)]], \
                                constant uint& total_elements [[buffer(2)]], \
                                uint gid [[thread_position_in_grid]]) { \
    if (gid >= total_elements) return; \
    ACCUM a_val = static_cast<ACCUM>(a[gid]); \
    out[gid] = static_cast<SCALAR>(fabs(a_val)); \
}

FOR_EACH_FLOAT_TYPE(DEFINE_ELEMWISE_ABS_KERNEL)

#undef DEFINE_ELEMWISE_ABS_KERNEL
#undef FOR_EACH_FLOAT_TYPE