#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, f32) \
    OP(half, f16) \
    OP(bfloat, bf16)

#define DEFINE_PERMUTE_KERNEL(SCALAR, SUFFIX) \
kernel void permute_kernel_##SUFFIX(device const SCALAR* src [[buffer(0)]], \
                                    device SCALAR* dst [[buffer(1)]], \
                                    constant const uint* src_strides [[buffer(2)]], \
                                    constant const uint* dst_strides [[buffer(3)]], \
                                    constant const uint* dims [[buffer(4)]], \
                                    constant const uint* permute [[buffer(5)]], \
                                    constant const uint& rank [[buffer(6)]], \
                                    constant const uint& num_elements [[buffer(7)]], \
                                    uint gid [[thread_position_in_grid]]) { \
    if (gid >= num_elements) return; \

    uint src_idx = gid; \
    uint temp_idx = src_idx; \

    uint src_coords[8]; \
    for (uint i = 0; i < rank; ++i) { \
        src_coords[i] = temp_idx / src_strides[i]; \
        temp_idx %= src_strides[i]; \
    } \

    uint dst_coords[8]; \
    for (uint i = 0; i < rank; ++i) { \
        dst_coords[i] = src_coords[permute[i]]; \
    } \

    uint dst_idx = 0; \
    for (uint i = 0; i < rank; ++i) { \
        dst_idx += dst_coords[i] * dst_strides[i]; \
    } \

    dst[dst_idx] = src[src_idx]; \
}

FOR_EACH_FLOAT_TYPE(DEFINE_PERMUTE_KERNEL)

#undef DEFINE_PERMUTE_KERNEL
#undef FOR_EACH_FLOAT_TYPE