#include <metal_stdlib>
using namespace metal;

#ifndef ACTIVATION
#define ACTIVATION ActivationSiLU
#endif

// SwigluParams struct is injected by Foundry via struct_defs()

// Fast vectorized path aliases (currently F16-specialized).
typedef FastScalarT SwigluFastScalarT;
typedef FastVec4T SwigluFastVec4T;

// ============================================================================
// SIMD GEMV Epilogue (for fused decode path)
// ============================================================================

/// SIMD GEMV epilogue that applies SwiGLU activation to gate+up accumulator outputs.
/// Used by CompoundKernel fusions for fused RMSNorm + Gate/Up GEMV + SwiGLU.
struct SwiGluEpilogue {
    template<uint HEADS>
    static void apply(
        float acc[HEADS],
        uint lane_id,
        uint logical_col,
        const uint N[HEADS],
        const device BiasStorageT *bias[HEADS],
        const uint has_bias_flags[HEADS],
        const float alpha,
        const float beta,
        const device ResidualStorageT *residual,
        device OutputStorageT *result_y[HEADS]
    ) {
        if (logical_col >= N[0]) return;

        float gate = acc[0];
        gate += simd_shuffle_xor(gate, 16u);
        gate += simd_shuffle_xor(gate, 8u);
        gate += simd_shuffle_xor(gate, 4u);
        gate += simd_shuffle_xor(gate, 2u);
        gate += simd_shuffle_xor(gate, 1u);

        float up = acc[1];
        up += simd_shuffle_xor(up, 16u);
        up += simd_shuffle_xor(up, 8u);
        up += simd_shuffle_xor(up, 4u);
        up += simd_shuffle_xor(up, 2u);
        up += simd_shuffle_xor(up, 1u);

        if (lane_id == 0) {
            if (has_bias_flags[0] && bias[0]) gate += (float)metallic_load_bias(bias[0], (ulong)logical_col);
            if (has_bias_flags[1] && bias[1]) up += (float)metallic_load_bias(bias[1], (ulong)logical_col);

            float agg_gate = ACTIVATION::apply(gate);
            float val = agg_gate * up;

            metallic_store_output(result_y[0], (ulong)logical_col, metallic_to_accum(val));
        }
    }
};

template<typename ACTIVATION>
ALWAYS_INLINE void run_swiglu_write_stage(
    float2 input_var,
    device OutputStorageT* output,
    const device BiasStorageT* b_gate,
    const device BiasStorageT* b_up,
    uint has_b_gate,
    uint has_b_up,
    uint lane_id,
    uint row_idx,
    uint batch_idx,
    uint n_dim
) {
    if (lane_id == 0) {
        float gate = input_var.x;
        float up = input_var.y;
        if (has_b_gate != 0) {
            gate += (float)metallic_load_bias(b_gate, (ulong)row_idx);
        }
        if (has_b_up != 0) {
            up += (float)metallic_load_bias(b_up, (ulong)row_idx);
        }
        float activated_gate = ACTIVATION::apply(gate);
        float val = activated_gate * up;
        const ulong out_idx = (ulong)batch_idx * (ulong)n_dim + (ulong)row_idx;
        metallic_store_output(output, out_idx, metallic_to_accum(val));
    }
}

// ============================================================================
// Standalone Activation Kernel (for non-fused path)
// Only compiled when NOT in a fused compound kernel context.
// ============================================================================

template<typename ACTIVATION, typename TGate, typename TOutput, typename TBias>
ALWAYS_INLINE void run_swiglu_stage(
    const device TGate* gate,
    device TOutput* up_inout,
    const device TBias* gate_bias,
    const device TBias* up_bias,
    constant SwigluParams* params,
    uint3 gid,
    uint3 lid,
    uint3 tptg
) {
    const uint total_elements = params->total_elements;
    const uint bias_len = params->bias_len;
    const uint vector_width = params->vector_width;
    const uint gate_leading_stride = params->gate_leading_stride;
    const uint up_leading_stride = params->up_leading_stride;
    const uint global_id = gid.x * tptg.x + lid.x;

    if (bias_len == 0u) return;
    if (global_id >= total_elements) return;

    const bool use_f16_vectorized = (vector_width == 4u)
        && METALLIC_FASTPATH_INPUT_HALF
        && METALLIC_FASTPATH_OUTPUT_HALF
        && METALLIC_FASTPATH_BIAS_HALF;

    if (use_f16_vectorized) {
        const device SwigluFastScalarT* gate_half = (const device SwigluFastScalarT*)gate;
        device SwigluFastScalarT* up_half = (device SwigluFastScalarT*)up_inout;
        const device SwigluFastScalarT* gate_bias_half = (const device SwigluFastScalarT*)gate_bias;
        const device SwigluFastScalarT* up_bias_half = (const device SwigluFastScalarT*)up_bias;

        const uint row_length = bias_len;
        const uint vecs_per_row = row_length / 4u;
        if (vecs_per_row == 0u) return;

        const uint total_rows = total_elements / row_length;
        const uint total_vector_threads = total_rows * vecs_per_row;
        const uint remainder = total_elements - total_vector_threads * 4u;

        if (global_id < total_vector_threads) {
            using ScalarVec = SwigluFastVec4T;
            using AccumVec = float4;

            const uint row = global_id / vecs_per_row;
            const uint col_vec = global_id % vecs_per_row;
            const uint col = col_vec * 4u;

            const uint gate_index = row * gate_leading_stride + col;
            const uint up_index = row * up_leading_stride + col;

            const device ScalarVec* gate_vec_ptr = reinterpret_cast<const device ScalarVec*>(gate_half + gate_index);
            device ScalarVec* up_vec_ptr = reinterpret_cast<device ScalarVec*>(up_half + up_index);
            const device ScalarVec* gate_bias_vec = reinterpret_cast<const device ScalarVec*>(gate_bias_half);
            const device ScalarVec* up_bias_vec = reinterpret_cast<const device ScalarVec*>(up_bias_half);

            const AccumVec gate_vals = (AccumVec)(gate_vec_ptr[0]) + (AccumVec)(gate_bias_vec[col_vec]);
            const AccumVec up_vals = (AccumVec)(up_vec_ptr[0]) + (AccumVec)(up_bias_vec[col_vec]);
            const AccumVec activated = ACTIVATION::apply(gate_vals);
            up_vec_ptr[0] = (ScalarVec)(activated * up_vals);
            return;
        }

        if (global_id < total_vector_threads + remainder) {
            const uint scalar_linear = total_vector_threads * 4u + (global_id - total_vector_threads);
            const uint row = scalar_linear / row_length;
            const uint col = scalar_linear % row_length;
            const uint gate_index = row * gate_leading_stride + col;
            const uint up_index = row * up_leading_stride + col;

            const float gate_val = (float)gate_half[gate_index] + (float)gate_bias_half[col];
            const float up_val = (float)up_half[up_index] + (float)up_bias_half[col];
            const float activated = ACTIVATION::apply(gate_val);
            up_half[up_index] = (SwigluFastScalarT)(activated * up_val);
        }
        return;
    }

    const uint row = global_id / bias_len;
    const uint col = global_id % bias_len;
    const uint gate_index = row * gate_leading_stride + col;
    const uint up_index = row * up_leading_stride + col;
    const float gate_val = (float)gate[gate_index] + (float)gate_bias[col];
    const float up_val = (float)up_inout[up_index] + (float)up_bias[col];
    const float activated = ACTIVATION::apply(gate_val);
    up_inout[up_index] = (TOutput)(activated * up_val);
}

#ifndef FUSED_KERNEL

/// SwiGLU Fused Activation kernel for runtime storage/bias/output types.
///
/// Computes: output = ACTIVATION(gate + gate_bias) * (up + up_bias)
///
/// This kernel uses vectorization (SwigluFastVec4T) when bias_len is aligned.
kernel void swiglu_fused_activation(
    const device InputStorageT* gate [[buffer(0)]],
    device OutputStorageT* up_inout [[buffer(1)]],
    const device BiasStorageT* gate_bias [[buffer(2)]],
    const device BiasStorageT* up_bias [[buffer(3)]],
    constant SwigluParams* params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    run_swiglu_stage<ACTIVATION>(
        gate,
        up_inout,
        gate_bias,
        up_bias,
        params,
        uint3(gid, 0u, 0u),
        uint3(0u, 0u, 0u),
        uint3(1u, 1u, 1u)
    );
}

#endif // FUSED_KERNEL
