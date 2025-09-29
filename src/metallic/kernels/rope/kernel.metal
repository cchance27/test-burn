#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16) \
    OP(bfloat, float, bf16)

#define DEFINE_ROPE_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void rope_kernel_##SUFFIX(device SCALAR* input [[buffer(0)]], \
                                 device SCALAR* output [[buffer(1)]], \
                                 device SCALAR* cos_buf [[buffer(2)]], \
                                 device SCALAR* sin_buf [[buffer(3)]], \
                                 constant uint& dim [[buffer(4)]], \
                                 constant uint& seq_len [[buffer(5)]], \
                                 constant uint& position_offset [[buffer(6)]], \
                                 constant uint& total_elements [[buffer(7)]], \
                                 uint gid [[thread_position_in_grid]]) { \
    if (gid >= total_elements) return; \

    uint feature_idx = gid % dim; \
    uint row_idx = gid / dim; \
    uint pos = (row_idx % seq_len) + position_offset; \

    uint half_dim = dim / 2u; \
    uint pair = (feature_idx < half_dim) ? feature_idx : (feature_idx - half_dim); \
    ACCUM cosv = static_cast<ACCUM>(cos_buf[pos * half_dim + pair]); \
    ACCUM sinv = static_cast<ACCUM>(sin_buf[pos * half_dim + pair]); \

    if (feature_idx < half_dim) { \
        ACCUM x_i = static_cast<ACCUM>(input[gid]); \
        ACCUM x_j = static_cast<ACCUM>(input[row_idx * dim + feature_idx + half_dim]); \
        ACCUM out_i = x_i * cosv - x_j * sinv; \
        output[gid] = static_cast<SCALAR>(out_i); \
    } else { \
        ACCUM x_j = static_cast<ACCUM>(input[gid]); \
        ACCUM x_i = static_cast<ACCUM>(input[row_idx * dim + (feature_idx - half_dim)]); \
        ACCUM out_j = x_j * cosv + x_i * sinv; \
        output[gid] = static_cast<SCALAR>(out_j); \
    } \
}

FOR_EACH_FLOAT_TYPE(DEFINE_ROPE_KERNEL)

#undef DEFINE_ROPE_KERNEL
#undef FOR_EACH_FLOAT_TYPE