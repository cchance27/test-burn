#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, f32) \
    OP(half, f16)

#define DEFINE_KV_CACHE_WRITE_KERNEL(SCALAR, SUFFIX) \
kernel void kv_cache_write_kernel_##SUFFIX( \
    device const SCALAR* k_src [[buffer(0)]], \
    device const SCALAR* v_src [[buffer(1)]], \
    device SCALAR* k_dst [[buffer(2)]], \
    device SCALAR* v_dst [[buffer(3)]], \
    constant uint& canonical_heads [[buffer(4)]], \
    constant uint& head_dim [[buffer(5)]], \
    constant uint& seq_len [[buffer(6)]], \
    constant uint& step [[buffer(7)]], \
    constant uint& group_size [[buffer(8)]], \
    constant uint& src_head_stride [[buffer(9)]], \
    constant uint& src_seq_stride [[buffer(10)]], \
    constant uint& dst_head_stride [[buffer(11)]], \
    constant uint& dst_seq_stride [[buffer(12)]], \
    constant uint& total_threads [[buffer(13)]], \
    constant uint& repeated_heads [[buffer(14)]], \
    uint gid [[thread_position_in_grid]]) { \
    if (gid >= total_threads) { \
        return; \
    } \
    uint head_idx = gid / head_dim; \
    if (head_idx >= canonical_heads) { \
        return; \
    } \
    if (repeated_heads < canonical_heads * group_size) { \
        return; \
    } \
    uint dim_idx = gid % head_dim; \
    for (uint seq_idx = 0; seq_idx < seq_len; ++seq_idx) { \
        uint k_src_index = head_idx * src_head_stride + seq_idx * src_seq_stride + dim_idx; \
        uint v_src_index = head_idx * src_head_stride + seq_idx * src_seq_stride + dim_idx; \
        SCALAR k_value = k_src[k_src_index]; \
        SCALAR v_value = v_src[v_src_index]; \
        uint cache_step = step + seq_idx; \
        for (uint group = 0; group < group_size; ++group) { \
            uint dst_head = head_idx * group_size + group; \
            uint dst_k_index = dst_head * dst_head_stride + cache_step * dst_seq_stride + dim_idx; \
            uint dst_v_index = dst_head * dst_head_stride + cache_step * dst_seq_stride + dim_idx; \
            k_dst[dst_k_index] = k_value; \
            v_dst[dst_v_index] = v_value; \
        } \
    } \
}

FOR_EACH_FLOAT_TYPE(DEFINE_KV_CACHE_WRITE_KERNEL)

#undef DEFINE_KV_CACHE_WRITE_KERNEL
#undef FOR_EACH_FLOAT_TYPE
