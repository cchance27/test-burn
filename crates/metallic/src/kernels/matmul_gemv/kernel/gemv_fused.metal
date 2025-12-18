// Fused GEMV kernels (SwiGLU, etc)
#include "gemv_simd_impl.metal"

// =================================================================================================
// SwiGLU Epilogue
// =================================================================================================
// Fuses Gate and Up projections + Silu Activation.
// - Input: acc[0] = Gate (before act), acc[1] = Up
// - Output: result_y[0] = (Gate * Sigmoid(Gate)) * Up 
// - result_y[1] is unused.

struct SwiGluEpilogue {
    template<uint HEADS>
    static void apply(
        float acc[HEADS],
        uint lane_id,
        uint logical_col,
        const uint N[HEADS],
        const device half *bias[HEADS],
        const uint has_bias_flags[HEADS],
        const float alpha,
        const float beta,
        const device half *residual,
        device half *result_y[HEADS]
    ) {
        // HEADS must be >= 2 for SwiGlu (Gate, Up)
        // If HEADS=3, maybe Gate, Up, something else? usually just 2.
        
        // We only care if we are within range of N[0] (Gate) and N[1] (Up).
        // Usually N[0] == N[1].
        if (logical_col >= N[0]) return;
        
        // 1. Warp Reduce Gate
        float gate = acc[0];
        gate += simd_shuffle_xor(gate, 16u);
        gate += simd_shuffle_xor(gate, 8u);
        gate += simd_shuffle_xor(gate, 4u);
        gate += simd_shuffle_xor(gate, 2u);
        gate += simd_shuffle_xor(gate, 1u);
        
        // 2. Warp Reduce Up
        float up = acc[1];
        up += simd_shuffle_xor(up, 16u);
        up += simd_shuffle_xor(up, 8u);
        up += simd_shuffle_xor(up, 4u);
        up += simd_shuffle_xor(up, 2u);
        up += simd_shuffle_xor(up, 1u);
        
        if (lane_id == 0) {
            // Apply Bias if present
            if (has_bias_flags[0] && bias[0]) gate += (float)bias[0][logical_col];
            if (has_bias_flags[1] && bias[1]) up += (float)bias[1][logical_col];
            
            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            float silu_gate = gate / (1.0f + exp(-gate));
            
            float val = silu_gate * up;
            
            // Apply residual if needed (SwiGLU usually doesn't have residual in standard MLP, but maybe?)
            // We'll support simple write-back.
            
            result_y[0][logical_col] = (half)val;
        }
    }
};

// =================================================================================================
// Kernel Definitions
// =================================================================================================

// Q8 SwiGLU Kernel
// - Inputs: params for Gate and Up weights.
// - Outputs: Single result buffer (written to result_y[0]).

[[kernel]] void gemv_q8_swiglu_f16(
    const device uchar *data_g [[buffer(0)]],
    const device uchar *data_u [[buffer(1)]],
    const device half *vector_x [[buffer(2)]],
    device half *out_res [[buffer(3)]],
    const constant Q2FusedParams *params [[buffer(4)]],
    const device uchar *scales_g [[buffer(5)]],
    const device uchar *scales_u [[buffer(6)]],
    const device half *bias_g [[buffer(7)]],
    const device half *bias_u [[buffer(8)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    const device uchar *data_arr[2] = {data_g, data_u};
    const device uchar *scale_arr[2] = {scales_g, scales_u};
    device half *res_arr[2] = {out_res, nullptr}; // Only 1 output
    const uint N_arr[2] = {params->N0, params->N1};
    const device half *bias_arr[2] = {bias_g, bias_u};
    const uint bias_flags[2] = {params->has_bias0, params->has_bias1};
    
    SimdGemvPolicyQ8::Params p = { 
        (const device uchar**)data_arr, 
        (const device uchar**)scale_arr, 
        params->weights_per_block 
    };

    run_simd_gemv_template<SimdGemvPolicyQ8, 2, 4, true, SwiGluEpilogue>(
        p, vector_x, res_arr, N_arr, params->K, bias_arr, bias_flags, 1.0f, 0.0f, nullptr, gid, lid
    );
}

