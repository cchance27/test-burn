#include <metal_stdlib>
using namespace metal;

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16)

#define DEFINE_SWIGLU_FUSED_ACTIVATION_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void swiglu_fused_activation_##SUFFIX( \
    device const SCALAR* gate [[buffer(0)]], \
    device SCALAR* up_inout [[buffer(1)]], \
    device const SCALAR* gate_bias [[buffer(2)]], \
    device const SCALAR* up_bias [[buffer(3)]], \
    constant uint& total_elements [[buffer(4)]], \
    constant uint& bias_len [[buffer(5)]], \
    constant uint& vector_width [[buffer(6)]], \
    constant uint& gate_leading_stride [[buffer(7)]], \
    constant uint& up_leading_stride [[buffer(8)]], \
    uint gid [[thread_position_in_grid]]) { \
    const uint kVectorWidth = 4; \
    if (bias_len == 0u) return; \
    const uint row_length = bias_len; \
    if (vector_width == kVectorWidth) { \
        const uint vecs_per_row = row_length / kVectorWidth; \
        if (vecs_per_row == 0u) { \
            return; \
        } \
        const uint total_rows = row_length == 0u ? 0u : total_elements / row_length; \
        const uint total_vector_threads = total_rows * vecs_per_row; \
        const uint remainder = total_elements - total_vector_threads * kVectorWidth; \
        if (gid < total_vector_threads) { \
            using ScalarVec = vec<SCALAR, kVectorWidth>; \
            using AccumVec = vec<ACCUM, kVectorWidth>; \
 \
            const uint row = gid / vecs_per_row; \
            const uint col_vec = gid % vecs_per_row; \
            const uint col = col_vec * kVectorWidth; \
 \
            const uint gate_index = row * gate_leading_stride + col; \
            const uint up_index = row * up_leading_stride + col; \
 \
            device const ScalarVec* gate_vec_ptr = \
                reinterpret_cast<device const ScalarVec*>(gate + gate_index); \
            device ScalarVec* up_vec_ptr = \
                reinterpret_cast<device ScalarVec*>(up_inout + up_index); \
            device const ScalarVec* gate_bias_vec = \
                reinterpret_cast<device const ScalarVec*>(gate_bias); \
            device const ScalarVec* up_bias_vec = \
                reinterpret_cast<device const ScalarVec*>(up_bias); \
 \
            const AccumVec gate_vals = static_cast<AccumVec>(gate_vec_ptr[0]) + \
                static_cast<AccumVec>(gate_bias_vec[col_vec]); \
            const AccumVec up_vals = static_cast<AccumVec>(up_vec_ptr[0]) + \
                static_cast<AccumVec>(up_bias_vec[col_vec]); \
 \
            const AccumVec one(static_cast<ACCUM>(1)); \
            const AccumVec sigmoid = one / (one + metal::fast::exp(-gate_vals)); \
            const AccumVec activated = gate_vals * sigmoid; \
            up_vec_ptr[0] = static_cast<ScalarVec>(activated * up_vals); \
            return; \
        } else if (gid < total_vector_threads + remainder) { \
            const uint scalar_linear = total_vector_threads * kVectorWidth + (gid - total_vector_threads); \
            const uint row = scalar_linear / row_length; \
            const uint col = scalar_linear % row_length; \
            const uint gate_index = row * gate_leading_stride + col; \
            const uint up_index = row * up_leading_stride + col; \
            ACCUM gate_val = static_cast<ACCUM>(gate[gate_index]) + static_cast<ACCUM>(gate_bias[col]); \
            ACCUM up_val = static_cast<ACCUM>(up_inout[up_index]) + static_cast<ACCUM>(up_bias[col]); \
            ACCUM sigmoid = static_cast<ACCUM>(1) / (static_cast<ACCUM>(1) + exp(-gate_val)); \
            ACCUM activated = gate_val * sigmoid; \
            ACCUM result = activated * up_val; \
            up_inout[up_index] = static_cast<SCALAR>(result); \
            return; \
        } else { \
            return; \
        } \
    } \
    if (gid >= total_elements) return; \
    const uint row = gid / row_length; \
    const uint col = gid % row_length; \
    const uint gate_index = row * gate_leading_stride + col; \
    const uint up_index = row * up_leading_stride + col; \
    ACCUM gate_val = static_cast<ACCUM>(gate[gate_index]) + static_cast<ACCUM>(gate_bias[col]); \
    ACCUM up_val = static_cast<ACCUM>(up_inout[up_index]) + static_cast<ACCUM>(up_bias[col]); \
    ACCUM sigmoid = static_cast<ACCUM>(1) / (static_cast<ACCUM>(1) + exp(-gate_val)); \
    ACCUM activated = gate_val * sigmoid; \
    ACCUM result = activated * up_val; \
    up_inout[up_index] = static_cast<SCALAR>(result); \
}

FOR_EACH_FLOAT_TYPE(DEFINE_SWIGLU_FUSED_ACTIVATION_KERNEL)

#undef DEFINE_SWIGLU_FUSED_ACTIVATION_KERNEL
#undef FOR_EACH_FLOAT_TYPE
