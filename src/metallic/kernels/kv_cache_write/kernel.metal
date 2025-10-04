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
    device SCALAR* repeated_k_dst [[buffer(4)]], \
    device SCALAR* repeated_v_dst [[buffer(5)]], \
    constant uint& canonical_heads [[buffer(6)]], \
    constant uint& head_dim [[buffer(7)]], \
    constant uint& seq_len [[buffer(8)]], \
    constant uint& step [[buffer(9)]], \
    constant uint& group_size [[buffer(10)]], \
    constant uint& src_head_stride [[buffer(11)]], \
    constant uint& src_seq_stride [[buffer(12)]], \
    constant uint& dst_head_stride [[buffer(13)]], \
    constant uint& dst_seq_stride [[buffer(14)]], \
    constant uint& repeated_head_stride [[buffer(15)]], \
    constant uint& repeated_seq_stride [[buffer(16)]], \
    constant uint& has_repeated [[buffer(17)]], \
    constant uint& total_threads [[buffer(18)]], \
    uint gid [[thread_position_in_grid]]) { \
    if (gid >= total_threads) { \
        return; \
    } \
    uint head_idx = gid / head_dim; \
    if (head_idx >= canonical_heads) { \
        return; \
    } \
    uint dim_idx = gid % head_dim; \
    for (uint seq_idx = 0; seq_idx < seq_len; ++seq_idx) { \
        uint k_src_index = head_idx * src_head_stride + seq_idx * src_seq_stride + dim_idx; \
        uint v_src_index = head_idx * src_head_stride + seq_idx * src_seq_stride + dim_idx; \
        SCALAR k_value = k_src[k_src_index]; \
        SCALAR v_value = v_src[v_src_index]; \
        uint cache_step = step + seq_idx; \
        uint k_dst_index = head_idx * dst_head_stride + cache_step * dst_seq_stride + dim_idx; \
        uint v_dst_index = head_idx * dst_head_stride + cache_step * dst_seq_stride + dim_idx; \
        k_dst[k_dst_index] = k_value; \
        v_dst[v_dst_index] = v_value; \
        if (has_repeated != 0) { \
            for (uint group = 0; group < group_size; ++group) { \
                uint dst_head = head_idx * group_size + group; \
                uint repeated_k_index = dst_head * repeated_head_stride + cache_step * repeated_seq_stride + dim_idx; \
                uint repeated_v_index = dst_head * repeated_head_stride + cache_step * repeated_seq_stride + dim_idx; \
                repeated_k_dst[repeated_k_index] = k_value; \
                repeated_v_dst[repeated_v_index] = v_value; \
            } \
        } \
    } \
}

FOR_EACH_FLOAT_TYPE(DEFINE_KV_CACHE_WRITE_KERNEL)

#undef DEFINE_KV_CACHE_WRITE_KERNEL
#undef FOR_EACH_FLOAT_TYPE
