#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16)

#define DEFINE_FUSED_QKV_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void fused_qkv_kernel_##SUFFIX( \
    device const SCALAR* fused [[buffer(0)]], \
    device SCALAR* q_out [[buffer(1)]], \
    device SCALAR* k_out [[buffer(2)]], \
    device SCALAR* v_out [[buffer(3)]], \
    device const SCALAR* cos_buf [[buffer(4)]], \
    device const SCALAR* sin_buf [[buffer(5)]], \
    constant uint& row_stride [[buffer(6)]], \
    constant uint& d_model [[buffer(7)]], \
    constant uint& kv_dim [[buffer(8)]], \
    constant uint& head_dim [[buffer(9)]], \
    constant uint& kv_head_dim [[buffer(10)]], \
    constant uint& n_heads [[buffer(11)]], \
    constant uint& n_kv_heads [[buffer(12)]], \
    constant uint& seq [[buffer(13)]], \
    constant uint& apply_rope [[buffer(14)]], \
    constant uint& position_offset [[buffer(15)]], \
    constant uint& total_q [[buffer(16)]], \
    constant uint& total_k [[buffer(17)]], \
    constant uint& total_v [[buffer(18)]], \
    uint gid [[thread_position_in_grid]]) { \
    uint total = total_q + total_k + total_v; \
    if (gid >= total) { \
        return; \
    } \
\
    if (gid < total_q) { \
        uint local = gid; \
        uint feature = local % head_dim; \
        uint tmp = local / head_dim; \
        uint s = tmp % seq; \
        uint out_batch = tmp / seq; \
        uint b = out_batch / n_heads; \
        uint h = out_batch % n_heads; \
        uint row = b * seq + s; \
        uint head_base = h * head_dim; \
        uint base_offset = head_base + feature; \
        uint src_idx = row * row_stride + base_offset; \
        SCALAR raw = fused[src_idx]; \
        if (apply_rope != 0 && head_dim >= 2) { \
            uint half_dim = head_dim / 2u; \
            uint pair = (feature < half_dim) ? feature : (feature - half_dim); \
            uint cos_idx = (position_offset + s) * half_dim + pair; \
            ACCUM cosv = static_cast<ACCUM>(cos_buf[cos_idx]); \
            ACCUM sinv = static_cast<ACCUM>(sin_buf[cos_idx]); \
            uint mate_offset = (feature < half_dim) ? (feature + half_dim) : (feature - half_dim); \
            uint mate_idx = row * row_stride + head_base + mate_offset; \
            ACCUM x_self = static_cast<ACCUM>(raw); \
            ACCUM x_mate = static_cast<ACCUM>(fused[mate_idx]); \
            ACCUM rotated = (feature < half_dim) ? (x_self * cosv - x_mate * sinv) : (x_self * cosv + x_mate * sinv); \
            q_out[gid] = static_cast<SCALAR>(rotated); \
        } else { \
            q_out[gid] = raw; \
        } \
        return; \
    } \
\
    uint offset = gid - total_q; \
    if (offset < total_k) { \
        uint local = offset; \
        uint feature = local % kv_head_dim; \
        uint tmp = local / kv_head_dim; \
        uint s = tmp % seq; \
        uint out_batch = tmp / seq; \
        uint b = out_batch / n_kv_heads; \
        uint h = out_batch % n_kv_heads; \
        uint row = b * seq + s; \
        uint head_base = h * kv_head_dim; \
        uint base_offset = head_base + feature; \
        uint src_idx = row * row_stride + d_model + base_offset; \
        SCALAR raw = fused[src_idx]; \
        if (apply_rope != 0 && kv_head_dim >= 2) { \
            uint half_dim = kv_head_dim / 2u; \
            uint pair = (feature < half_dim) ? feature : (feature - half_dim); \
            uint cos_idx = (position_offset + s) * half_dim + pair; \
            ACCUM cosv = static_cast<ACCUM>(cos_buf[cos_idx]); \
            ACCUM sinv = static_cast<ACCUM>(sin_buf[cos_idx]); \
            uint mate_offset = (feature < half_dim) ? (feature + half_dim) : (feature - half_dim); \
            uint mate_idx = row * row_stride + d_model + head_base + mate_offset; \
            ACCUM x_self = static_cast<ACCUM>(raw); \
            ACCUM x_mate = static_cast<ACCUM>(fused[mate_idx]); \
            ACCUM rotated = (feature < half_dim) ? (x_self * cosv - x_mate * sinv) : (x_self * cosv + x_mate * sinv); \
            k_out[offset] = static_cast<SCALAR>(rotated); \
        } else { \
            k_out[offset] = raw; \
        } \
        return; \
    } \
\
    uint v_local = offset - total_k; \
    uint feature = v_local % kv_head_dim; \
    uint tmp = v_local / kv_head_dim; \
    uint s = tmp % seq; \
    uint out_batch = tmp / seq; \
    uint b = out_batch / n_kv_heads; \
    uint h = out_batch % n_kv_heads; \
    uint row = b * seq + s; \
    uint head_base = h * kv_head_dim; \
    uint base_offset = head_base + feature; \
    uint src_idx = row * row_stride + d_model + kv_dim + base_offset; \
    v_out[v_local] = fused[src_idx]; \
}

FOR_EACH_FLOAT_TYPE(DEFINE_FUSED_QKV_KERNEL)

#undef DEFINE_FUSED_QKV_KERNEL
#undef FOR_EACH_FLOAT_TYPE
