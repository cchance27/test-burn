//! QKV Fused Stages - Componentized N-way dot product for QKV fusion.

use super::stages::VectorWidth;
use crate::compound::{BufferArg, Stage, stages::Quantization};

/// Metal definitions for GEMV helper functions.
const GEMV_METAL: &str = include_str!("gemv.metal");

/// Parallel projection stage that computes Q, K, and V projections in a single pass.
///
/// This stage maximizes register reuse by loading the input vector `x` once
/// and performing multiple dot products against different weight matrices.
///
/// It supports GQA by allowing different N dimensions for K and V.
#[derive(Debug, Clone)]
pub struct ParallelProjectStage {
    /// Quantization type (F16, Q8)
    quantization: Quantization,
    /// Vector width for loads (Vec4 or Vec8)
    vector_width: VectorWidth,
    /// Optional shared memory variable name for normalization (e.g. "tg_inv_rms")
    norm_shared_name: Option<String>,
    /// Optional Gamma tensor index (for RMSNorm fusion)
    gamma_buffer: Option<usize>,
}

impl ParallelProjectStage {
    pub fn new(quantization: Quantization) -> Self {
        Self {
            quantization,
            vector_width: VectorWidth::Vec8,
            norm_shared_name: None,
            gamma_buffer: None,
        }
    }

    pub fn with_norm(mut self, gamma_idx: usize, shared_name: &str) -> Self {
        self.norm_shared_name = Some(shared_name.to_string());
        self.gamma_buffer = Some(gamma_idx);
        self
    }
}

impl Stage for ParallelProjectStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![self.quantization.include_path()]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        // Standardized: all weights/scales are uchar*, Policy handles casting

        let mut args = vec![
            // Q Weights & Scales
            BufferArg {
                name: "w_q",
                metal_type: "const device uchar*",
                buffer_index: 0,
            },
            BufferArg {
                name: "s_q",
                metal_type: "const device uchar*",
                buffer_index: 1,
            },
            // K Weights & Scales
            BufferArg {
                name: "w_k",
                metal_type: "const device uchar*",
                buffer_index: 2,
            },
            BufferArg {
                name: "s_k",
                metal_type: "const device uchar*",
                buffer_index: 3,
            },
            // V Weights & Scales
            BufferArg {
                name: "w_v",
                metal_type: "const device uchar*",
                buffer_index: 4,
            },
            BufferArg {
                name: "s_v",
                metal_type: "const device uchar*",
                buffer_index: 5,
            },
            // Common input and dims
            BufferArg {
                name: "input",
                metal_type: "const device half*",
                buffer_index: 6,
            },
            BufferArg {
                name: "k_dim",
                metal_type: "constant uint&",
                buffer_index: 7,
            },
            BufferArg {
                name: "n_dim",
                metal_type: "constant uint&",
                buffer_index: 8,
            },
            BufferArg {
                name: "n_kv",
                metal_type: "constant uint&",
                buffer_index: 9,
            },
            BufferArg {
                name: "weights_per_block",
                metal_type: "constant uint&",
                buffer_index: 10,
            },
        ];

        if let Some(idx) = self.gamma_buffer {
            args.push(BufferArg {
                name: "gamma",
                metal_type: "const device half*",
                buffer_index: idx as u32,
            });
        }

        args
    }

    fn struct_defs(&self) -> String {
        GEMV_METAL.to_string()
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let policy = self.quantization.policy_name();
        let vec_width = self.vector_width.elements();

        // Scale stride: Q8 uses 2 bytes per block for scale lookup
        // F16 ignores scales but we still calculate offset (Policy ignores it)
        let scale_stride = "(ulong)row_idx * blocks_per_k * 2";

        // Byte scale: WEIGHT_INDEX returns element offsets, uchar* uses byte arithmetic
        // F16 = 2 bytes per element (half), Q8 = 1 byte per weight
        let byte_scale = match self.quantization {
            Quantization::F16 => 2usize,
            Quantization::Q8 => 1usize,
        };

        let norm_logic = if self.gamma_buffer.is_some() {
            r#"
        // --- Fused Norm Application ---
        if (gamma) {
             float4 f_lo = float4(xv_lo);
             float4 f_hi = float4(xv_hi);
             f_lo *= inv_rms;
             f_hi *= inv_rms;
             const device half* g_ptr = gamma + k;
             f_lo.x *= (float)g_ptr[0]; f_lo.y *= (float)g_ptr[1]; 
             f_lo.z *= (float)g_ptr[2]; f_lo.w *= (float)g_ptr[3];
             f_hi.x *= (float)g_ptr[4]; f_hi.y *= (float)g_ptr[5]; 
             f_hi.z *= (float)g_ptr[6]; f_hi.w *= (float)g_ptr[7];
             xv_lo = half4(f_lo);
             xv_hi = half4(f_hi);
        }
            "#
        } else {
            ""
        };

        let code = format!(
            r#"
    // Parallel QKV Projection ({policy})
    const uint blocks_per_k = (k_dim + weights_per_block - 1) / weights_per_block;
    float acc_q = 0.0f, acc_k = 0.0f, acc_v = 0.0f;
    uint k_base = 0;

    // Use absolute indexing inside the loop to support blocked layouts (Canonical)
    // Scale pointers can still use row offsets as they are simple
    const device uchar* row_s_q = s_q + {scale_stride};
    const device uchar* row_s_k = s_k + {scale_stride};
    const device uchar* row_s_v = s_v + {scale_stride};

    while (k_base + K_CHUNK_SIZE <= k_dim) {{
        uint k = k_base + lane_id * {vec_width}u;
        float4 xv_raw = *(const device float4*)(input + k);
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);

        {norm_logic}
        
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);
        
        uint block_off = k / weights_per_block;
        // Optimization: Use half scales directly and only load once per block
        // (Compiler will likely optimize these further)
        float s_q_val = (float){policy}::load_scale(row_s_q, block_off);
        float s_k_val = (row_idx < n_kv) ? (float){policy}::load_scale(row_s_k, block_off) : 0.0f;
        float s_v_val = (row_idx < n_kv) ? (float){policy}::load_scale(row_s_v, block_off) : 0.0f;

        // Q Dot
        {{
            float w[{vec_width}];
            const device uchar* w_ptr = w_q + WEIGHT_INDEX(row_idx, k, k_dim, n_dim) * {byte_scale};
            {policy}::template load_weights<{vec_width}>(w_ptr, 0, w);
            acc_q += s_q_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])));
        }}
        
        // K & V Dot
        if (row_idx < n_kv) {{
            {{
                float w[{vec_width}];
                const device uchar* w_ptr = w_k + WEIGHT_INDEX(row_idx, k, k_dim, n_kv) * {byte_scale};
                {policy}::template load_weights<{vec_width}>(w_ptr, 0, w);
                acc_k += s_k_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])));
            }}
            {{
                float w[{vec_width}];
                const device uchar* w_ptr = w_v + WEIGHT_INDEX(row_idx, k, k_dim, n_kv) * {byte_scale};
                {policy}::template load_weights<{vec_width}>(w_ptr, 0, w);
                acc_v += s_v_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])));
            }}
        }}

        k_base += K_CHUNK_SIZE;
    }}
    
    // Tail: bounds-checked
    if (k_base < k_dim) {{
        uint k = k_base + lane_id * {vec_width}u;
        float4 xv_raw = float4(0.0f);
        uint valid_count = 0;
        
        if (k + {vec_width}u <= k_dim) {{
            xv_raw = *(const device float4*)(input + k);
            valid_count = {vec_width}u;
        }} else if (k < k_dim) {{
            for (uint i = 0; i < {vec_width}u && k + i < k_dim; ++i) {{
                ((thread half*)&xv_raw)[i] = input[k + i];
                valid_count++;
            }}
        }}
        
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);

        {norm_logic}
        
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);
        
        uint block_off = k / weights_per_block;
        float s_q_val = (k < k_dim) ? (float){policy}::load_scale(row_s_q, block_off) : 0.0f;
        float s_k_val = (row_idx < n_kv && k < k_dim) ? (float){policy}::load_scale(row_s_k, block_off) : 0.0f;
        float s_v_val = (row_idx < n_kv && k < k_dim) ? (float){policy}::load_scale(row_s_v, block_off) : 0.0f;

        // Q Dot
        if (k < k_dim) {{
            float w[{vec_width}] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
            const device uchar* w_ptr = w_q + WEIGHT_INDEX(row_idx, k, k_dim, n_dim) * {byte_scale};
            {policy}::template load_weights<{vec_width}>(w_ptr, 0, w);
            for (uint i = valid_count; i < {vec_width}u; ++i) w[i] = 0.0f;
            acc_q += s_q_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])));
        }}
        
        // K & V Dot
        if (row_idx < n_kv && k < k_dim) {{
            {{
                float w[{vec_width}] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
                const device uchar* w_ptr = w_k + WEIGHT_INDEX(row_idx, k, k_dim, n_kv) * {byte_scale};
                {policy}::template load_weights<{vec_width}>(w_ptr, 0, w);
                for (uint i = valid_count; i < {vec_width}u; ++i) w[i] = 0.0f;
                acc_k += s_k_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])));
            }}
            {{
                float w[{vec_width}] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
                const device uchar* w_ptr = w_v + WEIGHT_INDEX(row_idx, k, k_dim, n_kv) * {byte_scale};
                {policy}::template load_weights<{vec_width}>(w_ptr, 0, w);
                for (uint i = valid_count; i < {vec_width}u; ++i) w[i] = 0.0f;
                acc_v += s_v_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])));
            }}
        }}
    }}
    
    float3 qkv_partial = float3(acc_q, acc_k, acc_v);
"#,
            policy = policy,
            vec_width = vec_width,
            norm_logic = norm_logic,
            byte_scale = byte_scale,
            scale_stride = scale_stride
        );

        ("qkv_partial".to_string(), code)
    }
}

/// Reduction stage for multiple partial sums (Q, K, V).
#[derive(Debug, Clone, Default)]
pub struct MultiWarpReduceStage;

impl Stage for MultiWarpReduceStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }
    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![]
    }
    fn emit(&self, input_var: &str) -> (String, String) {
        let code = format!(
            r#"
    uint mask = 0xFFFFFFFF;
    float3 qkv_sum = {input_var};
    for (uint offset = 16; offset > 0; offset /= 2) {{
        qkv_sum.x += simd_shuffle_down(qkv_sum.x, offset);
        qkv_sum.y += simd_shuffle_down(qkv_sum.y, offset);
        qkv_sum.z += simd_shuffle_down(qkv_sum.z, offset);
    }}
    float3 qkv_final = qkv_sum;
"#,
            input_var = input_var
        );
        ("qkv_final".to_string(), code)
    }
}

/// Specialized write stage for QKV fused output.
#[derive(Debug, Clone)]
pub struct MultiWriteOutputStage;

impl Stage for MultiWriteOutputStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }
    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "out_q",
                metal_type: "device half*",
                buffer_index: 11,
            },
            BufferArg {
                name: "out_k",
                metal_type: "device half*",
                buffer_index: 12,
            },
            BufferArg {
                name: "out_v",
                metal_type: "device half*",
                buffer_index: 13,
            },
            // Biases
            BufferArg {
                name: "b_q",
                metal_type: "const device half*",
                buffer_index: 14,
            },
            BufferArg {
                name: "b_k",
                metal_type: "const device half*",
                buffer_index: 15,
            },
            BufferArg {
                name: "b_v",
                metal_type: "const device half*",
                buffer_index: 16,
            },
            BufferArg {
                name: "has_b",
                metal_type: "constant uint&",
                buffer_index: 17,
            },
        ]
    }
    fn emit(&self, input_var: &str) -> (String, String) {
        let code = format!(
            r#"
    if (lane_id == 0) {{
        out_q[row_idx] = half({input_var}.x + (has_b ? (float)b_q[row_idx] : 0.0f));
        if (row_idx < n_kv) {{
            out_k[row_idx] = half({input_var}.y + (has_b ? (float)b_k[row_idx] : 0.0f));
            out_v[row_idx] = half({input_var}.z + (has_b ? (float)b_v[row_idx] : 0.0f));
        }}
    }}
"#,
            input_var = input_var
        );
        ("void".to_string(), code)
    }
}
