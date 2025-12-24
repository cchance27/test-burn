#include <metal_stdlib>
using namespace metal;

// SwigluParams struct is injected by Foundry via struct_defs()

/// SwiGLU Fused Activation kernel for half precision.
///
/// Computes: output = SiLU(gate + gate_bias) * (up + up_bias)
/// where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
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
            
            // SiLU: x * sigmoid(x)
            AccumVec one(1.0f);
            AccumVec sigmoid = one / (one + metal::fast::exp(-gate_vals));
            AccumVec activated = gate_vals * sigmoid;
            
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
            float sigmoid = 1.0f / (1.0f + exp(-gate_val));
            float activated = gate_val * sigmoid;
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
    float sigmoid = 1.0f / (1.0f + exp(-gate_val));
    float activated = gate_val * sigmoid;
    up_inout[up_index] = (half)(activated * up_val);
}
