//! Fused FFN stages: RMSNorm + Gate/Up projections + SwiGLU writeback.

use std::sync::Arc;

use crate::{
    compound::{BufferArg, Stage}, fusion::MetalPolicy, metals::gemv::stages::VectorWidth, policy::activation::Activation
};

/// Metal definitions for GEMV helper functions.
const GEMV_METAL: &str = include_str!("../gemv/gemv.metal");
const SWIGLU_METAL: &str = include_str!("swiglu.metal");

/// Dual projection stage for FFN (gate + up) with optional RMSNorm fusion.
#[derive(Debug, Clone)]
pub struct FfnDualProjectStage {
    policy: Arc<dyn MetalPolicy>,
    vector_width: VectorWidth,
    norm_shared_name: Option<String>,
    gamma_buffer: Option<usize>,
}

impl FfnDualProjectStage {
    pub fn new(policy: Arc<dyn MetalPolicy>) -> Self {
        Self {
            policy,
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

impl Stage for FfnDualProjectStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![self.policy.header()]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        let mut args = vec![
            BufferArg {
                name: "w_gate",
                metal_type: "const device uchar*",
                buffer_index: 0,
            },
            BufferArg {
                name: "s_gate",
                metal_type: "const device uchar*",
                buffer_index: 1,
            },
            BufferArg {
                name: "w_up",
                metal_type: "const device uchar*",
                buffer_index: 2,
            },
            BufferArg {
                name: "s_up",
                metal_type: "const device uchar*",
                buffer_index: 3,
            },
            BufferArg {
                name: "input",
                metal_type: "const device half*",
                buffer_index: 4,
            },
            BufferArg {
                name: "k_dim",
                metal_type: "constant uint&",
                buffer_index: 6,
            },
            BufferArg {
                name: "n_dim",
                metal_type: "constant uint&",
                buffer_index: 7,
            },
            BufferArg {
                name: "weights_per_block",
                metal_type: "constant uint&",
                buffer_index: 8,
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
        let norm_var = self.norm_shared_name.as_deref().unwrap_or("inv_rms");

        let scale_stride = "(ulong)row_idx * blocks_per_k * 2";

        let norm_logic = if self.norm_shared_name.is_some() {
            format!(
                r#"
        if (gamma) {{
            float4 f_lo = float4(xv_lo);
            float4 f_hi = float4(xv_hi);
            f_lo *= {norm_var};
            f_hi *= {norm_var};
            const device half* g_ptr = gamma + k;
            f_lo.x *= (float)g_ptr[0]; f_lo.y *= (float)g_ptr[1];
            f_lo.z *= (float)g_ptr[2]; f_lo.w *= (float)g_ptr[3];
            f_hi.x *= (float)g_ptr[4]; f_hi.y *= (float)g_ptr[5];
            f_hi.z *= (float)g_ptr[6]; f_hi.w *= (float)g_ptr[7];
            xv_lo = half4(f_lo);
            xv_hi = half4(f_hi);
        }}
            "#,
                norm_var = norm_var
            )
        } else {
            String::new()
        };

        // Unified: always use Policy::load_scale (F16 returns 1.0h, Q8 returns actual scale)
        let scales_logic = format!(
            r#"
        float s_gate_val = (float){policy}::load_scale(row_s_gate, block_off);
        float s_up_val = (float){policy}::load_scale(row_s_up, block_off);
                "#,
            policy = policy
        );

        let code = format!(
            r#"
    // FFN Dual Projection ({policy})
    const uint blocks_per_k = (k_dim + weights_per_block - 1) / weights_per_block;
    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    uint k_base = 0;

    const device uchar* row_s_gate = s_gate + {scale_stride};
    const device uchar* row_s_up = s_up + {scale_stride};

    while (k_base + K_CHUNK_SIZE <= k_dim) {{
        uint k = k_base + lane_id * {vec_width}u;
        float4 xv_raw = *(const device float4*)(input + batch_idx * k_dim + k);
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);

        {norm_logic}

        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);

        uint block_off = k / weights_per_block;
        {scales_logic}

        {{
            float w[{vec_width}];
            {policy}::template load_weights<{vec_width}>(w_gate, WEIGHT_INDEX(row_idx, k, k_dim, n_dim), w);
            acc_gate += s_gate_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])));
        }}

        {{
            float w[{vec_width}];
            {policy}::template load_weights<{vec_width}>(w_up, WEIGHT_INDEX(row_idx, k, k_dim, n_dim), w);
            acc_up += s_up_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])));
        }}

        k_base += K_CHUNK_SIZE;
    }}

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
        {scales_logic}

        {{
            float w[{vec_width}];
            {policy}::template load_weights<{vec_width}>(w_gate, WEIGHT_INDEX(row_idx, k, k_dim, n_dim), w);
            acc_gate += s_gate_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])));
        }}

        {{
            float w[{vec_width}];
            {policy}::template load_weights<{vec_width}>(w_up, WEIGHT_INDEX(row_idx, k, k_dim, n_dim), w);
            acc_up += s_up_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])));
        }}

        k_base += K_CHUNK_SIZE;
    }}

    float2 gu_partial = float2(acc_gate, acc_up);
"#,
            policy = policy,
            vec_width = vec_width,
            norm_logic = norm_logic,
            scale_stride = scale_stride,
            scales_logic = scales_logic
        );

        ("gu_partial".to_string(), code)
    }
}

/// Warp reduction stage for float2 accumulators.
#[derive(Debug, Clone, Default)]
pub struct FfnWarpReduceStage;

impl Stage for FfnWarpReduceStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![]
    }

    fn emit(&self, input_var: &str) -> (String, String) {
        let code = format!(
            r#"
    float2 gu_sum = {input_var};
    for (uint offset = 16; offset > 0; offset /= 2) {{
        gu_sum.x += simd_shuffle_down(gu_sum.x, offset);
        gu_sum.y += simd_shuffle_down(gu_sum.y, offset);
    }}
    float2 gu_final = gu_sum;
"#,
            input_var = input_var
        );
        ("gu_final".to_string(), code)
    }
}

/// Write stage for SwiGLU output.
#[derive(Debug, Clone)]
pub struct FfnSwigluWriteStage {
    activation: Activation,
}

impl FfnSwigluWriteStage {
    pub fn new() -> Self {
        Self {
            activation: Activation::SiLU,
        }
    }

    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
}

impl Stage for FfnSwigluWriteStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![self.activation.header()]
    }

    fn struct_defs(&self) -> String {
        format!(
            "#define FUSED_KERNEL 1\n#define ACTIVATION {}\n{}",
            self.activation.struct_name(),
            SWIGLU_METAL
        )
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "output",
                metal_type: "device half*",
                buffer_index: 5,
            },
            BufferArg {
                name: "b_gate",
                metal_type: "const device half*",
                buffer_index: 10,
            },
            BufferArg {
                name: "b_up",
                metal_type: "const device half*",
                buffer_index: 11,
            },
            BufferArg {
                name: "has_b_gate",
                metal_type: "constant uint&",
                buffer_index: 12,
            },
            BufferArg {
                name: "has_b_up",
                metal_type: "constant uint&",
                buffer_index: 13,
            },
        ]
    }

    fn emit(&self, input_var: &str) -> (String, String) {
        let activation = self.activation.struct_name();
        let code = format!(
            r#"
    if (lane_id == 0) {{
        float gate = {input_var}.x;
        float up = {input_var}.y;
        if (has_b_gate != 0) {{
            gate += (float)b_gate[row_idx];
        }}
        if (has_b_up != 0) {{
            up += (float)b_up[row_idx];
        }}
        float activated_gate = {activation}::apply(gate);
        float val = activated_gate * up;
        output[batch_idx * n_dim + row_idx] = (half)val;
    }}
"#,
            input_var = input_var,
            activation = activation
        );
        ("void".to_string(), code)
    }
}
