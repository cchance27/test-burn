#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, f32) \
    OP(half, f16)

#define DEFINE_REPEAT_KV_KERNEL(SCALAR, SUFFIX) \
kernel void repeat_kv_heads_kernel_##SUFFIX( \
    device const SCALAR* input [[buffer(0)]], \
    device SCALAR* output [[buffer(1)]], \
    constant uint& group_size [[buffer(2)]], \
    constant uint& batch [[buffer(3)]], \
    constant uint& n_kv_heads [[buffer(4)]], \
    constant uint& n_heads [[buffer(5)]], \
    constant uint& seq [[buffer(6)]], \
    constant uint& head_dim [[buffer(7)]], \
    constant uint& cache_stride [[buffer(8)]], \
    constant uint& dest_offset [[buffer(9)]], \
    constant uint& output_stride [[buffer(10)]], \
    constant uint& total_elements [[buffer(11)]], \
    uint gid [[thread_position_in_grid]]) { \
    if (gid >= total_elements) { \
        return; \
    } \
    uint dim_idx = gid % head_dim; \
    uint tmp = gid / head_dim; \
    uint seq_idx = tmp % seq; \
    uint batch_head_idx = tmp / seq; \
    uint b = batch_head_idx / n_heads; \
    uint h = batch_head_idx % n_heads; \
    uint kv_head = h / group_size; \
    uint input_batch_head = b * n_kv_heads + kv_head; \
    uint input_index = ((input_batch_head * cache_stride) + seq_idx) * head_dim + dim_idx; \
    uint output_index = ((batch_head_idx * output_stride) + dest_offset + seq_idx) * head_dim + dim_idx; \
    output[output_index] = input[input_index]; \
}

FOR_EACH_FLOAT_TYPE(DEFINE_REPEAT_KV_KERNEL)

#undef DEFINE_REPEAT_KV_KERNEL
#undef FOR_EACH_FLOAT_TYPE
