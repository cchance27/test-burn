#include <metal_stdlib>
using namespace metal;

// SwigluParams struct is injected by Foundry via struct_defs()

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
        const device half *bias[HEADS],
        const uint has_bias_flags[HEADS],
        const float alpha,
        const float beta,
        const device half *residual,
        device half *result_y[HEADS]
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
            if (has_bias_flags[0] && bias[0]) gate += (float)bias[0][logical_col];
            if (has_bias_flags[1] && bias[1]) up += (float)bias[1][logical_col];

            float agg_gate = ACTIVATION::apply(gate);
            float val = agg_gate * up;

            result_y[0][logical_col] = (half)val;
        }
    }
};

template<typename ACTIVATION>
ALWAYS_INLINE void run_swiglu_write_stage(
    float2 input_var,
    device half* output,
    const device half* b_gate,
    const device half* b_up,
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
            gate += (float)b_gate[row_idx];
        }
        if (has_b_up != 0) {
            up += (float)b_up[row_idx];
        }
        float activated_gate = ACTIVATION::apply(gate);
        float val = activated_gate * up;
        output[batch_idx * n_dim + row_idx] = (half)val;
    }
}

// ============================================================================
// Standalone Activation Kernel (for non-fused path)
// Only compiled when NOT in a fused compound kernel context.
// ============================================================================

template<typename ACTIVATION>
ALWAYS_INLINE void run_swiglu_stage(
    const device half* gate,
    device half* up_inout,
    const device half* gate_bias,
    const device half* up_bias,
    constant SwigluParams* params,
    uint3 gid,
    uint3 lid,
    uint3 tptg
) {
    uint total_elements = params->total_elements;
    uint bias_len = params->bias_len;
    uint vector_width = params->vector_width;
    uint gate_leading_stride = params->gate_leading_stride;
    uint up_leading_stride = params->up_leading_stride;
    
    // Calculate global thread index from CompoundKernel standard arguments
    uint global_id = gid.x * tptg.x + lid.x; 
    
    if (bias_len == 0u) return; 
    
    const uint kVectorWidth = 4;
    const uint row_length = bias_len;
    
    if (vector_width == kVectorWidth) {
        const uint vecs_per_row = row_length / kVectorWidth;
        if (vecs_per_row == 0u) return;
        
        const uint total_rows = total_elements / row_length;
        const uint total_vector_threads = total_rows * vecs_per_row;
        const uint remainder = total_elements - total_vector_threads * kVectorWidth;
        
        if (global_id < total_vector_threads) {
            using ScalarVec = half4;
            using AccumVec = float4;
            
            const uint row = global_id / vecs_per_row;
            const uint col_vec = global_id % vecs_per_row;
            const uint col = col_vec * kVectorWidth;
            
            const uint gate_index = row * gate_leading_stride + col;
            const uint up_index = row * up_leading_stride + col;
            
            const device ScalarVec* gate_vec_ptr = 
                reinterpret_cast<const device ScalarVec*>(gate + gate_index);
            device ScalarVec* up_vec_ptr = 
                reinterpret_cast<device ScalarVec*>(up_inout + up_index);
            const device ScalarVec* gate_bias_vec = 
                reinterpret_cast<const device ScalarVec*>(gate_bias);
            const device ScalarVec* up_bias_vec = 
                reinterpret_cast<const device ScalarVec*>(up_bias);
                
            // Load and add biases
            AccumVec gate_vals = (AccumVec)(gate_vec_ptr[0]) + (AccumVec)(gate_bias_vec[col_vec]);
            AccumVec up_vals = (AccumVec)(up_vec_ptr[0]) + (AccumVec)(up_bias_vec[col_vec]);
            
            AccumVec activated = ACTIVATION::apply(gate_vals);
            
            // Output
            up_vec_ptr[0] = (ScalarVec)(activated * up_vals);
            
        } else if (global_id < total_vector_threads + remainder) {
            const uint scalar_linear = total_vector_threads * kVectorWidth + (global_id - total_vector_threads);
            const uint row = scalar_linear / row_length;
            const uint col = scalar_linear % row_length;
            
            const uint gate_index = row * gate_leading_stride + col;
            const uint up_index = row * up_leading_stride + col;
            
            float gate_val = (float)gate[gate_index] + (float)gate_bias[col];
            float up_val = (float)up_inout[up_index] + (float)up_bias[col];
            float activated = ACTIVATION::apply(gate_val);
            up_inout[up_index] = (half)(activated * up_val);
        }
    } else {
        // Scalar fallback
        if (global_id < total_elements) {
            const uint row = global_id / row_length;
            const uint col = global_id % row_length;
            const uint gate_index = row * gate_leading_stride + col;
            const uint up_index = row * up_leading_stride + col;
            
            float gate_val = (float)gate[gate_index] + (float)gate_bias[col];
            float up_val = (float)up_inout[up_index] + (float)up_bias[col];
            float activated = ACTIVATION::apply(gate_val);
            up_inout[up_index] = (half)(activated * up_val);
        }
    }
}

#ifndef FUSED_KERNEL

/// SwiGLU Fused Activation kernel for half precision.
///
/// Computes: output = ACTIVATION(gate + gate_bias) * (up + up_bias)
///
/// This kernel uses vectorization (half4) when bias_len is aligned.
kernel void swiglu_fused_activation_f16(
    const device half* gate [[buffer(0)]],
    device half* up_inout [[buffer(1)]],
    const device half* gate_bias [[buffer(2)]],
    const device half* up_bias [[buffer(3)]],
    constant SwigluParams* params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_elements = params->total_elements;
    uint bias_len = params->bias_len;
    uint vector_width = params->vector_width;
    uint gate_leading_stride = params->gate_leading_stride;
    uint up_leading_stride = params->up_leading_stride;
    
    if (bias_len == 0u) return;
    
    const uint kVectorWidth = 4;
    const uint row_length = bias_len;
    
    // Vectorized path
    if (vector_width == kVectorWidth) {
        const uint vecs_per_row = row_length / kVectorWidth;
        if (vecs_per_row == 0u) return;
        
        const uint total_rows = total_elements / row_length;
        const uint total_vector_threads = total_rows * vecs_per_row;
        const uint remainder = total_elements - total_vector_threads * kVectorWidth;
        
        if (gid < total_vector_threads) {
            using ScalarVec = half4;
            using AccumVec = float4;
            
            const uint row = gid / vecs_per_row;
            const uint col_vec = gid % vecs_per_row;
            const uint col = col_vec * kVectorWidth;
            
            const uint gate_index = row * gate_leading_stride + col;
            const uint up_index = row * up_leading_stride + col;
            
            const device ScalarVec* gate_vec_ptr = 
                reinterpret_cast<const device ScalarVec*>(gate + gate_index);
            device ScalarVec* up_vec_ptr = 
                reinterpret_cast<device ScalarVec*>(up_inout + up_index);
            const device ScalarVec* gate_bias_vec = 
                reinterpret_cast<const device ScalarVec*>(gate_bias);
            const device ScalarVec* up_bias_vec = 
                reinterpret_cast<const device ScalarVec*>(up_bias);
            
            // Load and add biases
            AccumVec gate_vals = (AccumVec)(gate_vec_ptr[0]) + (AccumVec)(gate_bias_vec[col_vec]);
            AccumVec up_vals = (AccumVec)(up_vec_ptr[0]) + (AccumVec)(up_bias_vec[col_vec]);
            
            // Apply Activation
            AccumVec activated = ACTIVATION::apply(gate_vals);
            
            // Output
            up_vec_ptr[0] = (ScalarVec)(activated * up_vals);
            return;
        } else if (gid < total_vector_threads + remainder) {
            // Handle remainder elements
            const uint scalar_linear = total_vector_threads * kVectorWidth + (gid - total_vector_threads);
            const uint row = scalar_linear / row_length;
            const uint col = scalar_linear % row_length;
            
            const uint gate_index = row * gate_leading_stride + col;
            const uint up_index = row * up_leading_stride + col;
            
            float gate_val = (float)gate[gate_index] + (float)gate_bias[col];
            float up_val = (float)up_inout[up_index] + (float)up_bias[col];
            float activated = ACTIVATION::apply(gate_val);
            up_inout[up_index] = (half)(activated * up_val);
            return;
        } else {
            return;
        }
    }
    
    // Scalar fallback
    if (gid >= total_elements) return;
    
    const uint row = gid / row_length;
    const uint col = gid % row_length;
    const uint gate_index = row * gate_leading_stride + col;
    const uint up_index = row * up_leading_stride + col;
    
    float gate_val = (float)gate[gate_index] + (float)gate_bias[col];
    float up_val = (float)up_inout[up_index] + (float)up_bias[col];
    float activated = ACTIVATION::apply(gate_val);
    up_inout[up_index] = (half)(activated * up_val);
}

#endif // FUSED_KERNEL
