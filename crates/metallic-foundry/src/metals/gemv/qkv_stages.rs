//! QKV Fused Stages - Componentized N-way dot product for QKV fusion.

use std::sync::Arc;

use super::stages::VectorWidth;
use crate::{
    compound::{BufferArg, Stage}, fusion::MetalPolicy
};

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
    /// Policy for quantization (F16, Q8)
    policy: Arc<dyn MetalPolicy>,
    /// Vector width for loads (matches layout stride)
    vector_width: VectorWidth,
    /// Optional shared memory variable name for normalization (e.g. "tg_inv_rms")
    norm_shared_name: Option<String>,
    /// Optional Gamma tensor index (for RMSNorm fusion)
    gamma_buffer: Option<usize>,
}

impl ParallelProjectStage {
    pub fn new(policy: Arc<dyn MetalPolicy>) -> Self {
        Self {
            policy,
            vector_width: VectorWidth::Vec8, // Default safe
            norm_shared_name: None,
            gamma_buffer: None,
        }
    }

    pub fn with_vector_width(mut self, width: VectorWidth) -> Self {
        self.vector_width = width;
        self
    }

    pub fn with_norm(mut self, gamma_idx: usize, shared_name: &str) -> Self {
        self.norm_shared_name = Some(shared_name.to_string());
        self.gamma_buffer = Some(gamma_idx);
        self
    }
}

impl Stage for ParallelProjectStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![self.policy.header()]
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
        let policy = self.policy.struct_name();
        let vec_width = self.vector_width.elements();

        // Scale stride is policy-owned so affine formats can widen block metadata.
        let scale_stride = format!("(ulong)row_idx * blocks_per_k * {}::SCALE_BYTES", policy);

        let load_input = match self.vector_width {
            VectorWidth::Vec4 => {
                r#"
                half4 xv_raw = *(const device half4*)(input + batch_idx * k_dim + k);
                half4 xv_lo = xv_raw;
                half4 xv_hi = half4(0.0h); // Unused
            "#
            }
            VectorWidth::Vec8 => {
                r#"
                float4 xv_raw_f = *(const device float4*)(input + batch_idx * k_dim + k);
                half4 xv_lo = as_type<half4>(xv_raw_f.xy);
                half4 xv_hi = as_type<half4>(xv_raw_f.zw);
            "#
            }
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
             // Note: For Vec4, hi part is dummy/unused, but logic safely applies 0 or unused
             if (blocks_per_k > 0) { // Check ensures valid compilation, optimization removes dead code
                 f_hi.x *= (float)g_ptr[4]; f_hi.y *= (float)g_ptr[5]; 
                 f_hi.z *= (float)g_ptr[6]; f_hi.w *= (float)g_ptr[7];
             }
             xv_lo = half4(f_lo);
             xv_hi = half4(f_hi);
        }
            "#
        } else {
            ""
        };

        // Unified: always use Policy::load_scale (F16 returns 1.0h, Q8 returns actual scale)
        let scales_logic = format!(
            r#"
        float s_q_val = (float){policy}::load_scale(row_s_q, block_off);
        float s_k_val = (row_idx < n_kv) ? (float){policy}::load_scale(row_s_k, block_off) : 0.0f;
        float s_v_val = (row_idx < n_kv) ? (float){policy}::load_scale(row_s_v, block_off) : 0.0f;
        float a_q_val = ({policy}::HAS_AFFINE ? (float){policy}::load_affine(row_s_q, block_off) : 0.0f);
        float a_k_val = (row_idx < n_kv) ? ({policy}::HAS_AFFINE ? (float){policy}::load_affine(row_s_k, block_off) : 0.0f) : 0.0f;
        float a_v_val = (row_idx < n_kv) ? ({policy}::HAS_AFFINE ? (float){policy}::load_affine(row_s_v, block_off) : 0.0f) : 0.0f;
                "#,
            policy = policy
        );

        // Unified tail: always use Policy::load_scale (F16 returns 1.0h, Q8 returns actual scale)
        let tail_scales_logic = format!(
            r#"
        float s_q_val = (k < k_dim) ? (float){policy}::load_scale(row_s_q, block_off) : 0.0f;
        float s_k_val = (row_idx < n_kv && k < k_dim) ? (float){policy}::load_scale(row_s_k, block_off) : 0.0f;
        float s_v_val = (row_idx < n_kv && k < k_dim) ? (float){policy}::load_scale(row_s_v, block_off) : 0.0f;
        float a_q_val = (k < k_dim) ? ({policy}::HAS_AFFINE ? (float){policy}::load_affine(row_s_q, block_off) : 0.0f) : 0.0f;
        float a_k_val = (row_idx < n_kv && k < k_dim) ? ({policy}::HAS_AFFINE ? (float){policy}::load_affine(row_s_k, block_off) : 0.0f) : 0.0f;
        float a_v_val = (row_idx < n_kv && k < k_dim) ? ({policy}::HAS_AFFINE ? (float){policy}::load_affine(row_s_v, block_off) : 0.0f) : 0.0f;
                "#,
            policy = policy
        );

        // Helper macro for optimized indexing
        // Checks if WPB==32 (constant fold if specialized) or runtime check (uniform)
        let calc_idx = r#"
        ulong w_idx;
        #if defined(IS_CANONICAL) && IS_CANONICAL
        if (weights_per_block == 32) {
            // Canonical layout fast path:
            // idx = (k % 32) + 32 * (row + (k / 32) * N)
            // Use uint arithmetic when safe, otherwise fall back to 64-bit indexing.
            const ulong U32_MAX_U = 0xFFFFFFFFul;
            if (((ulong)n_dim) * ((ulong)k_dim) <= U32_MAX_U) {
                uint w_idx_u32 = (k & 31u) + (row_idx << 5u) + (k & ~31u) * n_dim;
                w_idx = (ulong)w_idx_u32;
            } else {
                w_idx = ((ulong)(k & 31u)) + (((ulong)row_idx) << 5ul) + ((ulong)(k & ~31u)) * ((ulong)n_dim);
            }
        } else {
            w_idx = (ulong)WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
        }
        #else
        w_idx = (ulong)WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
        #endif
        "#;

        let calc_idx_kv = r#"
        ulong w_idx_kv;
        #if defined(IS_CANONICAL) && IS_CANONICAL
        if (weights_per_block == 32) {
            const ulong U32_MAX_U = 0xFFFFFFFFul;
            if (((ulong)n_kv) * ((ulong)k_dim) <= U32_MAX_U) {
                uint w_idx_kv_u32 = (k & 31u) + (row_idx << 5u) + (k & ~31u) * n_kv;
                w_idx_kv = (ulong)w_idx_kv_u32;
            } else {
                w_idx_kv = ((ulong)(k & 31u)) + (((ulong)row_idx) << 5ul) + ((ulong)(k & ~31u)) * ((ulong)n_kv);
            }
        } else {
            w_idx_kv = (ulong)WEIGHT_INDEX(row_idx, k, k_dim, n_kv);
        }
        #else
        w_idx_kv = (ulong)WEIGHT_INDEX(row_idx, k, k_dim, n_kv);
        #endif
        "#;

        // Tail generation helpers
        let zero_init = (0..self.vector_width as usize).map(|_| "0.0f").collect::<Vec<_>>().join(", ");

        let dot_logic = match self.vector_width {
            VectorWidth::Vec4 => format!(
                "acc_q += s_q_val * dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + ({policy}::HAS_AFFINE ? (a_q_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);",
                policy = policy
            ),
            VectorWidth::Vec8 => format!(
                "acc_q += s_q_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7]))) + ({policy}::HAS_AFFINE ? (a_q_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);",
                policy = policy
            ),
        };
        let dot_logic_k = match self.vector_width {
            VectorWidth::Vec4 => format!(
                "acc_k += s_k_val * dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + ({policy}::HAS_AFFINE ? (a_k_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);",
                policy = policy
            ),
            VectorWidth::Vec8 => format!(
                "acc_k += s_k_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7]))) + ({policy}::HAS_AFFINE ? (a_k_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);",
                policy = policy
            ),
        };
        let dot_logic_v = match self.vector_width {
            VectorWidth::Vec4 => format!(
                "acc_v += s_v_val * dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + ({policy}::HAS_AFFINE ? (a_v_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);",
                policy = policy
            ),
            VectorWidth::Vec8 => format!(
                "acc_v += s_v_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7]))) + ({policy}::HAS_AFFINE ? (a_v_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);",
                policy = policy
            ),
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
        {load_input}

        {norm_logic}
        
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);
        
        uint block_off = k / weights_per_block; // Keep division for scales (typically Q8 only, effectively per-warp constant)
        
        // Optimization: Use half scales directly and only load once per block
        // (Compiler will likely optimize these further)
        {scales_logic}

        // Q Dot
            {{
                {calc_idx}
            acc_q += {policy}::template dot<{vec_width}>(w_q, w_idx, s_q_val, xv_f32_lo, xv_f32_hi)
                + ({policy}::HAS_AFFINE ? (a_q_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
            }}
        
        // K & V Dot
        if (row_idx < n_kv) {{
            {calc_idx_kv}
            
            {{
                acc_k += {policy}::template dot<{vec_width}>(w_k, w_idx_kv, s_k_val, xv_f32_lo, xv_f32_hi)
                    + ({policy}::HAS_AFFINE ? (a_k_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
            }}
            {{
                acc_v += {policy}::template dot<{vec_width}>(w_v, w_idx_kv, s_v_val, xv_f32_lo, xv_f32_hi)
                    + ({policy}::HAS_AFFINE ? (a_v_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
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
            xv_raw = *(const device float4*)(input + batch_idx * k_dim + k);
            valid_count = {vec_width}u;
        }} else if (k < k_dim) {{
            for (uint i = 0; i < {vec_width}u && k + i < k_dim; ++i) {{
                ((thread half*)&xv_raw)[i] = input[batch_idx * k_dim + k + i];
                valid_count++;
            }}
        }}
        
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);

        {norm_logic}
        
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);
        
        uint block_off = k / weights_per_block;
        {tail_scales_logic}

        // Helper to formatting array init
        
        // Q Dot
        if (k < k_dim) {{
            float w[{vec_width}] = {{{zero_init}}};
            {policy}::template load_weights<{vec_width}>(w_q, WEIGHT_INDEX(row_idx, k, k_dim, n_dim), w);
            for (uint i = valid_count; i < {vec_width}u; ++i) w[i] = 0.0f;
            {dot_logic}
        }}
        
        // K & V Dot
        if (row_idx < n_kv && k < k_dim) {{
            {{
                float w[{vec_width}] = {{{zero_init}}};
                {policy}::template load_weights<{vec_width}>(w_k, WEIGHT_INDEX(row_idx, k, k_dim, n_kv), w);
                for (uint i = valid_count; i < {vec_width}u; ++i) w[i] = 0.0f;
                {dot_logic_k}
            }}
            {{
                float w[{vec_width}] = {{{zero_init}}};
                {policy}::template load_weights<{vec_width}>(w_v, WEIGHT_INDEX(row_idx, k, k_dim, n_kv), w);
                for (uint i = valid_count; i < {vec_width}u; ++i) w[i] = 0.0f;
                {dot_logic_v}
            }}
        }}
    }}
    
    float3 qkv_partial = float3(acc_q, acc_k, acc_v);
"#,
            policy = policy,
            vec_width = vec_width,
            load_input = load_input,
            norm_logic = norm_logic,
            scale_stride = scale_stride,
            scales_logic = scales_logic,
            tail_scales_logic = tail_scales_logic,
            zero_init = zero_init,
            dot_logic = dot_logic,
            dot_logic_k = dot_logic_k,
            dot_logic_v = dot_logic_v
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
        out_q[batch_idx * n_dim + row_idx] = half({input_var}.x + (has_b ? (float)b_q[row_idx] : 0.0f));
        if (row_idx < n_kv) {{
            out_k[batch_idx * n_kv + row_idx] = half({input_var}.y + (has_b ? (float)b_k[row_idx] : 0.0f));
            out_v[batch_idx * n_kv + row_idx] = half({input_var}.z + (has_b ? (float)b_v[row_idx] : 0.0f));
        }}
    }}
"#,
            input_var = input_var
        );
        ("void".to_string(), code)
    }
}
