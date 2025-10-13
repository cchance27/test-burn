#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, f32) \
    OP(half, f16)

#define DEFINE_KV_REARRANGE_KERNEL(SCALAR, SUFFIX) \
kernel void kv_rearrange_kernel_##SUFFIX( \
    device const SCALAR* input [[buffer(0)]], \
    device SCALAR* output [[buffer(1)]], \
    constant uint& kv_dim [[buffer(2)]], \
    constant uint& row_stride [[buffer(3)]], \
    constant uint& kv_head_dim [[buffer(4)]], \
    constant uint& n_heads [[buffer(5)]], \
    constant uint& n_kv_heads [[buffer(6)]], \
    constant uint& head_dim [[buffer(7)]], \
    constant uint& seq [[buffer(8)]], \
    constant uint& total_elements [[buffer(9)]], \
    uint gid [[thread_position_in_grid]]) { \
    if (gid >= total_elements) { \
        return; \
    } \
    uint hd = gid % head_dim; \
    uint tmp = gid / head_dim; \
    uint s = tmp % seq; \
    uint out_batch = tmp / seq; \
    uint b = out_batch / n_heads; \
    uint h = out_batch % n_heads; \
    uint group_size = n_heads / n_kv_heads; \
    uint kv_h = h / group_size; \
    uint base_offset = kv_h * kv_head_dim + hd; \
    if (base_offset >= kv_dim) { \
        return; \
    } \
    uint src_row = b * seq + s; \
    uint src_idx = src_row * row_stride + base_offset; \
    output[gid] = input[src_idx]; \
}

FOR_EACH_FLOAT_TYPE(DEFINE_KV_REARRANGE_KERNEL)

#undef DEFINE_KV_REARRANGE_KERNEL
#undef FOR_EACH_FLOAT_TYPE
