use crate::{
    compound::{BufferArg, Stage}, metals::swiglu::{SwigluParams, SwigluParamsResolved}, policy::activation::Activation
};

/// Stage for SwiGLU fused activation.
///
/// Computes: up = SiLU(gate + gate_bias) * (up + up_bias)
///
/// Handles both vectorized and scalar paths based on params.
#[derive(Debug, Clone)]
pub struct SwigluStage {
    #[allow(dead_code)]
    params: SwigluParamsResolved,
    activation: Activation,
}

impl SwigluStage {
    pub fn new(params: SwigluParamsResolved) -> Self {
        Self {
            params,
            activation: Activation::SiLU,
        }
    }

    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
}

impl Stage for SwigluStage {
    fn includes(&self) -> Vec<&'static str> {
        // Included manually via struct_defs to ensure correct order with params
        vec![self.activation.header()]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "gate",
                metal_type: "const device half*",
                buffer_index: 0,
            },
            BufferArg {
                name: "up_inout",
                metal_type: "device half*",
                buffer_index: 1,
            },
            BufferArg {
                name: "gate_bias",
                metal_type: "const device half*",
                buffer_index: 2,
            },
            BufferArg {
                name: "up_bias",
                metal_type: "const device half*",
                buffer_index: 3,
            },
            BufferArg {
                name: "params",
                metal_type: "constant SwigluParams*",
                buffer_index: 4,
            },
        ]
    }

    fn struct_defs(&self) -> String {
        format!(
            "#define FUSED_KERNEL 1\n#define ACTIVATION {}\n{}\n{}",
            self.activation.struct_name(),
            SwigluParams::METAL_STRUCT_DEF,
            include_str!("swiglu.metal")
        )
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let activation = self.activation.struct_name();
        let code = format!(
            "
    uint total_elements = params->total_elements;
    uint bias_len = params->bias_len;
    uint vector_width = params->vector_width;
    uint gate_leading_stride = params->gate_leading_stride;
    uint up_leading_stride = params->up_leading_stride;
    
    // Calculate global thread index from CompoundKernel standard arguments
    uint global_id = gid.x * tptg.x + lid.x; 
    
    if (bias_len == 0u) return; // Should return, but we are inside void main()
    
    const uint kVectorWidth = 4;
    const uint row_length = bias_len;
    
    if (vector_width == kVectorWidth) {{
        const uint vecs_per_row = row_length / kVectorWidth;
        if (vecs_per_row == 0u) return;
        
        const uint total_rows = total_elements / row_length;
        const uint total_vector_threads = total_rows * vecs_per_row;
        const uint remainder = total_elements - total_vector_threads * kVectorWidth;
        
        if (global_id < total_vector_threads) {{
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
                
            // Check bounds for bias access? (Assumes properly padded/aligned)
            
            // Load and add biases
            AccumVec gate_vals = (AccumVec)(gate_vec_ptr[0]) + (AccumVec)(gate_bias_vec[col_vec]);
            AccumVec up_vals = (AccumVec)(up_vec_ptr[0]) + (AccumVec)(up_bias_vec[col_vec]);
            
            AccumVec activated = {activation}::apply(gate_vals);
            
            // Output
            up_vec_ptr[0] = (ScalarVec)(activated * up_vals);
            
        }} else if (global_id < total_vector_threads + remainder) {{
             // Remainder logic
            const uint scalar_linear = total_vector_threads * kVectorWidth + (global_id - total_vector_threads);
            const uint row = scalar_linear / row_length;
            const uint col = scalar_linear % row_length;
            
            const uint gate_index = row * gate_leading_stride + col;
            const uint up_index = row * up_leading_stride + col;
            
            float gate_val = (float)gate[gate_index] + (float)gate_bias[col];
            float up_val = (float)up_inout[up_index] + (float)up_bias[col];
            float activated = {activation}::apply(gate_val);
            up_inout[up_index] = (half)(activated * up_val);
        }}
        }} else {{
        // Scalar fallback
        if (global_id < total_elements) {{
            const uint row = global_id / row_length;
            const uint col = global_id % row_length;
            const uint gate_index = row * gate_leading_stride + col;
            const uint up_index = row * up_leading_stride + col;
            
            float gate_val = (float)gate[gate_index] + (float)gate_bias[col];
            float up_val = (float)up_inout[up_index] + (float)up_bias[col];
            float activated = {activation}::apply(gate_val);
            up_inout[up_index] = (half)(activated * up_val);
        }}
    }}
"
        );

        ("swiglu_output".to_string(), code)
    }
}
