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
    uint gid [[thread_position_in_grid]]) { \
    const uint kVectorWidth = 4; \
    if (vector_width == kVectorWidth) { \
        const uint total_vector_threads = total_elements / kVectorWidth; \
        const uint remainder = total_elements - total_vector_threads * kVectorWidth; \
        if (gid < total_vector_threads) { \
            using ScalarVec = vec<SCALAR, kVectorWidth>; \
            using AccumVec = vec<ACCUM, kVectorWidth>; \
 \
            const uint bias_vec_len = bias_len / kVectorWidth; \
            const uint bias_vec_idx = gid % bias_vec_len; \
 \
            device const ScalarVec* gate_vec = \
                reinterpret_cast<device const ScalarVec*>(gate); \
            device ScalarVec* up_vec = \
                reinterpret_cast<device ScalarVec*>(up_inout); \
            device const ScalarVec* gate_bias_vec = \
                reinterpret_cast<device const ScalarVec*>(gate_bias); \
            device const ScalarVec* up_bias_vec = \
                reinterpret_cast<device const ScalarVec*>(up_bias); \
 \
            const AccumVec gate_vals = static_cast<AccumVec>(gate_vec[gid]) + \
                static_cast<AccumVec>(gate_bias_vec[bias_vec_idx]); \
            const AccumVec up_vals = static_cast<AccumVec>(up_vec[gid]) + \
                static_cast<AccumVec>(up_bias_vec[bias_vec_idx]); \
 \
            const AccumVec one(static_cast<ACCUM>(1)); \
            const AccumVec sigmoid = one / (one + metal::fast::exp(-gate_vals)); \
            const AccumVec activated = gate_vals * sigmoid; \
            up_vec[gid] = static_cast<ScalarVec>(activated * up_vals); \
            return; \
        } else if (gid < total_vector_threads + remainder) { \
            uint scalar_idx = total_vector_threads * kVectorWidth + (gid - total_vector_threads); \
            uint bias_idx = scalar_idx % bias_len; \
            ACCUM gate_val = static_cast<ACCUM>(gate[scalar_idx]) + static_cast<ACCUM>(gate_bias[bias_idx]); \
            ACCUM up_val = static_cast<ACCUM>(up_inout[scalar_idx]) + static_cast<ACCUM>(up_bias[bias_idx]); \
            ACCUM sigmoid = static_cast<ACCUM>(1) / (static_cast<ACCUM>(1) + exp(-gate_val)); \
            ACCUM activated = gate_val * sigmoid; \
            ACCUM result = activated * up_val; \
            up_inout[scalar_idx] = static_cast<SCALAR>(result); \
            return; \
        } else { \
            return; \
        } \
    } \
    if (gid >= total_elements) return; \
    uint bias_idx = gid % bias_len; \
    ACCUM gate_val = static_cast<ACCUM>(gate[gid]) + static_cast<ACCUM>(gate_bias[bias_idx]); \
    ACCUM up_val = static_cast<ACCUM>(up_inout[gid]) + static_cast<ACCUM>(up_bias[bias_idx]); \
    ACCUM sigmoid = static_cast<ACCUM>(1) / (static_cast<ACCUM>(1) + exp(-gate_val)); \
    ACCUM activated = gate_val * sigmoid; \
    ACCUM result = activated * up_val; \
    up_inout[gid] = static_cast<SCALAR>(result); \
}

FOR_EACH_FLOAT_TYPE(DEFINE_SWIGLU_FUSED_ACTIVATION_KERNEL)

#undef DEFINE_SWIGLU_FUSED_ACTIVATION_KERNEL
#undef FOR_EACH_FLOAT_TYPE
