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
    uint gid [[thread_position_in_grid]]) { \
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
