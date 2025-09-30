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
        uint total_vector_threads = total_elements / kVectorWidth; \
        uint remainder = total_elements - total_vector_threads * kVectorWidth; \
        if (gid < total_vector_threads) { \
            uint base_idx = gid * kVectorWidth; \
            uint bias_idx = base_idx % bias_len; \
            uint bias_vec_idx = bias_idx / kVectorWidth; \
            device const vec<SCALAR, kVectorWidth>* gate_vec = \
                reinterpret_cast<device const vec<SCALAR, kVectorWidth>*>(gate); \
            device vec<SCALAR, kVectorWidth>* up_vec = \
                reinterpret_cast<device vec<SCALAR, kVectorWidth>*>(up_inout); \
            device const vec<SCALAR, kVectorWidth>* gate_bias_vec = \
                reinterpret_cast<device const vec<SCALAR, kVectorWidth>*>(gate_bias); \
            device const vec<SCALAR, kVectorWidth>* up_bias_vec = \
                reinterpret_cast<device const vec<SCALAR, kVectorWidth>*>(up_bias); \
            vec<SCALAR, kVectorWidth> gate_values = gate_vec[gid]; \
            vec<SCALAR, kVectorWidth> gate_bias_values = gate_bias_vec[bias_vec_idx]; \
            vec<SCALAR, kVectorWidth> up_values = up_vec[gid]; \
            vec<SCALAR, kVectorWidth> up_bias_values = up_bias_vec[bias_vec_idx]; \
            ACCUM results[kVectorWidth]; \
            for (uint lane = 0; lane < kVectorWidth; ++lane) { \
                ACCUM gate_val = static_cast<ACCUM>(gate_values[lane]) + \
                    static_cast<ACCUM>(gate_bias_values[lane]); \
                ACCUM up_val = static_cast<ACCUM>(up_values[lane]) + \
                    static_cast<ACCUM>(up_bias_values[lane]); \
                ACCUM sigmoid = static_cast<ACCUM>(1) / (static_cast<ACCUM>(1) + exp(-gate_val)); \
                ACCUM activated = gate_val * sigmoid; \
                results[lane] = activated * up_val; \
            } \
            vec<SCALAR, kVectorWidth> result_vec = vec<SCALAR, kVectorWidth>( \
                static_cast<SCALAR>(results[0]), \
                static_cast<SCALAR>(results[1]), \
                static_cast<SCALAR>(results[2]), \
                static_cast<SCALAR>(results[3])); \
            up_vec[gid] = result_vec; \
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
