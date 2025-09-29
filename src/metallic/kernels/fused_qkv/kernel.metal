#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16) \
    OP(bfloat, float, bf16)

#define DEFINE_FUSED_QKV_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void fused_qkv_bias_split_##SUFFIX( \
    device const SCALAR* qkv_linear [[buffer(0)]], \
    device const SCALAR* bias [[buffer(1)]], \
    device SCALAR* q_out [[buffer(2)]], \
    device SCALAR* k_out [[buffer(3)]], \
    device SCALAR* v_out [[buffer(4)]], \
    constant uint& rows [[buffer(5)]], \
    constant uint& q_cols [[buffer(6)]], \
    constant uint& kv_cols [[buffer(7)]], \
    uint gid [[thread_position_in_grid]]) \
{ \
    uint total_cols = q_cols + 2 * kv_cols; \
    uint total = rows * total_cols; \
    if (gid >= total) { \
        return; \
    } \

    uint row = gid / total_cols; \
    uint col = gid % total_cols; \

    ACCUM value = static_cast<ACCUM>(qkv_linear[gid]) + static_cast<ACCUM>(bias[col]); \

    if (col < q_cols) { \
        uint out_idx = row * q_cols + col; \
        q_out[out_idx] = static_cast<SCALAR>(value); \
    } else if (col < q_cols + kv_cols) { \
        uint offset = col - q_cols; \
        uint out_idx = row * kv_cols + offset; \
        k_out[out_idx] = static_cast<SCALAR>(value); \
    } else { \
        uint offset = col - q_cols - kv_cols; \
        uint out_idx = row * kv_cols + offset; \
        v_out[out_idx] = static_cast<SCALAR>(value); \
    } \
}

FOR_EACH_FLOAT_TYPE(DEFINE_FUSED_QKV_KERNEL)

#undef DEFINE_FUSED_QKV_KERNEL
#undef FOR_EACH_FLOAT_TYPE
