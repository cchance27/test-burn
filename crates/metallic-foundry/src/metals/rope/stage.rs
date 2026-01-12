use metallic_macros::KernelArgs;

use crate::{
    compound::{BufferArg, Stage},
    metals::rope::RopeParams, // Reuse RopeParams from parent module
    types::TensorArg,
};

/// Applies Rotary Position Embedding (RoPE) to Q (and optionally K) as a pipeline Stage.
///
/// This stage injects logic to load Q into threadgroup memory and rotate it in-place using `rope_common.metal`.
/// It assumes previously resolved layout pointers (q_ptr) and provides `q_ptr_roped` (threadgroup) to subsequent stages.
///
/// Assumed Headers:
/// - "rope/rope_common.metal" (emitted via includes)
#[derive(KernelArgs, Clone, Debug)]
pub struct RopeStage {
    pub cos: TensorArg,
    pub sin: TensorArg,
    pub params_rope: RopeParams,
}

impl Stage for RopeStage {
    fn includes(&self) -> Vec<&'static str> {
        vec!["rope/rope_common.metal"]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "cos_buf", // Metal code uses cos_buf/sin_buf
                metal_type: "const device half*",
                buffer_index: 12,
            },
            BufferArg {
                name: "sin_buf",
                metal_type: "const device half*",
                buffer_index: 13,
            },
            BufferArg {
                name: "params_rope",
                metal_type: "constant RopeParams&",
                buffer_index: 14,
            },
        ]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        (
            "q_ptr_roped".to_string(), // Output name? Unused but standard return.
            r#"
    // RoPE Stage Prologue
    // 1. Define Shared Memory for Q (Vector Aligned)
    #define MAX_HEAD_DIM_VEC 64 // 256 / 4
    threadgroup half4 q_shared[MAX_HEAD_DIM_VEC]; 
    
    // 2. Load and Rotate Q (Vectorized)
    uint head_dim = params_rope.dim;
    uint rope_vec_dim = head_dim / 4;
    
    // Range check (tid is vector index 0..rope_vec_dim-1)
    if (tid < rope_vec_dim) {
        // uint half_dim = head_dim / 2; 
        uint half_vec = rope_vec_dim / 2; // half_dim / 4
        
        // Only threads covering the lower half of head_dim do the pair processing
        if (tid < half_vec) {
            uint pos = params_rope.position_offset;
            
            // Pointers as vector pointers
            const device half4* q_ptr_vec = (const device half4*)q_ptr;
            const device half4* cos_buf_vec = (const device half4*)cos_buf;
            const device half4* sin_buf_vec = (const device half4*)sin_buf;
            
            // Load pairs (i, i + half_dim)
            half4 x_low = q_ptr_vec[tid];
            half4 x_high = q_ptr_vec[tid + half_vec];
            
            // Load Cos/Sin for 'low' indices
            // cos_buf width is half_dim. Vectors correspond to 0..half_vec-1
            half4 cos_v = cos_buf_vec[pos * half_vec + tid];
            half4 sin_v = sin_buf_vec[pos * half_vec + tid];
            
            // Rotate (Vector math: component-wise) with Float precision
            float4 x_l = (float4)x_low;
            float4 x_h = (float4)x_high;
            float4 c_v = (float4)cos_v;
            float4 s_v = (float4)sin_v;

            float4 out_l = x_l * c_v - x_h * s_v;
            float4 out_h = x_l * s_v + x_h * c_v;
            
            q_shared[tid] = (half4)out_l;
            q_shared[tid + half_vec] = (half4)out_h;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 3. Expose pointers for next stage
    const threadgroup half4* q_shared_ptr = q_shared; // SdpaOnlineStage uses q_shared directly though
           "#
            .to_string(),
        )
    }

    fn struct_defs(&self) -> String {
        r#"
        struct RopeParams {
            uint dim;
            uint seq_len;
            uint position_offset;
            uint total_elements;
        };
        "#
        .to_string()
    }
}

impl RopeStage {
    pub fn new(cos: TensorArg, sin: TensorArg, params_rope: RopeParams) -> Self {
        Self { cos, sin, params_rope }
    }
}
