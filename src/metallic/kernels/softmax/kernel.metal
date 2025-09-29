#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16) \
    OP(bfloat, float, bf16)

#define DEFINE_SOFTMAX_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void sdpa_fused_softmax_##SUFFIX( \
    device SCALAR* attn [[buffer(0)]], \
    constant uint& seq_q [[buffer(1)]], \
    constant uint& seq_k [[buffer(2)]], \
    constant uint& causal_flag [[buffer(3)]], \
    constant uint& query_offset [[buffer(4)]], \
    uint3 tg_pos [[threadgroup_position_in_grid]], \
    uint3 tid3 [[thread_position_in_threadgroup]], \
    uint3 tptg [[threads_per_threadgroup]]) { \
    uint row = tg_pos.y; \
    uint lane = tid3.x; \
    uint stride = tptg.x; \
    uint base = row * seq_k; \
    uint i_q = (row % seq_q) + query_offset; \
    threadgroup ACCUM shared_data[256]; \
    threadgroup uint shared_indices[256]; \
    ACCUM local_max = -INFINITY; \
    uint max_index = 0; \
    for (uint c = lane; c < seq_k; c += stride) { \
        ACCUM xv = static_cast<ACCUM>(attn[base + c]); \
        if (causal_flag == 1u && c > i_q) { \
            xv = -INFINITY; \
        } \
        if (xv > local_max) { \
            local_max = xv; \
            max_index = c; \
        } \
    } \
    shared_data[lane] = local_max; \
    shared_indices[lane] = max_index; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint offset = stride / 2u; offset > 0u; offset /= 2u) { \
        if (lane < offset) { \
            if (shared_data[lane + offset] > shared_data[lane]) { \
                shared_data[lane] = shared_data[lane + offset]; \
                shared_indices[lane] = shared_indices[lane + offset]; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    ACCUM maxv = shared_data[0]; \
    uint row_max_index = shared_indices[0]; \
    ACCUM local_sum = static_cast<ACCUM>(0.0f); \
    for (uint c = lane; c < seq_k; c += stride) { \
        ACCUM xv = static_cast<ACCUM>(attn[base + c]); \
        if (causal_flag == 1u && c > i_q) { \
            xv = -INFINITY; \
        } \
        ACCUM e = static_cast<ACCUM>(0.0f); \
        if (isinf(maxv) && maxv > static_cast<ACCUM>(0.0f)) { \
            if (isinf(xv) && xv > static_cast<ACCUM>(0.0f)) { \
                e = static_cast<ACCUM>(1.0f); \
            } \
        } else if (xv != -INFINITY) { \
            ACCUM diff = xv - maxv; \
            if (diff < static_cast<ACCUM>(-80.0f)) { \
                e = static_cast<ACCUM>(0.0f); \
            } else if (diff > static_cast<ACCUM>(80.0f)) { \
                e = exp(static_cast<ACCUM>(80.0f)); \
            } else { \
                e = exp(diff); \
            } \
        } \
        attn[base + c] = static_cast<SCALAR>(e); \
        local_sum += e; \
    } \
    shared_data[lane] = local_sum; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint offset = stride / 2u; offset > 0u; offset /= 2u) { \
        if (lane < offset) { \
            shared_data[lane] += shared_data[lane + offset]; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    ACCUM sumv = shared_data[0]; \
    for (uint c = lane; c < seq_k; c += stride) { \
        if (isnan(sumv)) { \
            attn[base + c] = static_cast<SCALAR>(sumv); \
        } else if (sumv > static_cast<ACCUM>(0.0f) && sumv != INFINITY) { \
            ACCUM val = static_cast<ACCUM>(attn[base + c]) / sumv; \
            attn[base + c] = static_cast<SCALAR>(val); \
        } else { \
            if (causal_flag == 1u && c > i_q) { \
                attn[base + c] = static_cast<SCALAR>(0.0f); \
            } else if (c == row_max_index) { \
                attn[base + c] = static_cast<SCALAR>(1.0f); \
            } else { \
                attn[base + c] = static_cast<SCALAR>(0.0f); \
            } \
        } \
    } \
}

FOR_EACH_FLOAT_TYPE(DEFINE_SOFTMAX_KERNEL)

#undef DEFINE_SOFTMAX_KERNEL
#undef FOR_EACH_FLOAT_TYPE
