#include <metal_stdlib>

using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, half, f16) 
    
#define DEFINE_CAST_TO_F16_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void cast_to_f16_kernel_##SUFFIX( \
    device const SCALAR* input [[buffer(0)]], \
    device half* output [[buffer(1)]], \
    constant uint& total_elements [[buffer(2)]], \
    uint gid [[thread_position_in_grid]] \
) { \
    if (gid >= total_elements) { \
        return; \
    } \
    ACCUM value = static_cast<ACCUM>(input[gid]); \
    output[gid] = static_cast<half>(value); \
}

#define DEFINE_CAST_FROM_F16_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void cast_from_f16_kernel_##SUFFIX( \
    device const half* input [[buffer(0)]], \
    device SCALAR* output [[buffer(1)]], \
    constant uint& total_elements [[buffer(2)]], \
    uint gid [[thread_position_in_grid]] \
) { \
    if (gid >= total_elements) { \
        return; \
    } \
    half raw = input[gid]; \
    ACCUM value = static_cast<ACCUM>(raw); \
    output[gid] = static_cast<SCALAR>(value); \
}

#define DEFINE_CAST_TO_F32_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void cast_to_f32_kernel_##SUFFIX( \
    device const SCALAR* input [[buffer(0)]], \
    device float* output [[buffer(1)]], \
    constant uint& total_elements [[buffer(2)]], \
    uint gid [[thread_position_in_grid]] \
) { \
    if (gid >= total_elements) { \
        return; \
    } \
    ACCUM value = static_cast<ACCUM>(input[gid]); \
    output[gid] = static_cast<float>(value); \
}

#define DEFINE_CAST_FROM_F32_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void cast_from_f32_kernel_##SUFFIX( \
    device const float* input [[buffer(0)]], \
    device SCALAR* output [[buffer(1)]], \
    constant uint& total_elements [[buffer(2)]], \
    uint gid [[thread_position_in_grid]] \
) { \
    if (gid >= total_elements) { \
        return; \
    } \
    float raw = input[gid]; \
    ACCUM value = static_cast<ACCUM>(raw); \
    output[gid] = static_cast<SCALAR>(value); \
}

#define INSTANTIATE_CAST_KERNELS(SCALAR, ACCUM, SUFFIX) \
    DEFINE_CAST_TO_F16_KERNEL(SCALAR, ACCUM, SUFFIX) \
    DEFINE_CAST_FROM_F16_KERNEL(SCALAR, ACCUM, SUFFIX) \
    DEFINE_CAST_TO_F32_KERNEL(SCALAR, ACCUM, SUFFIX) \
    DEFINE_CAST_FROM_F32_KERNEL(SCALAR, ACCUM, SUFFIX)

FOR_EACH_FLOAT_TYPE(INSTANTIATE_CAST_KERNELS)

#undef INSTANTIATE_CAST_KERNELS
#undef DEFINE_CAST_FROM_F16_KERNEL
#undef DEFINE_CAST_TO_F16_KERNEL
#undef DEFINE_CAST_FROM_F32_KERNEL
#undef DEFINE_CAST_TO_F32_KERNEL
#undef FOR_EACH_FLOAT_TYPE
