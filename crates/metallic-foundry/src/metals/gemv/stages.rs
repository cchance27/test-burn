//! GemvV2 Stages - Full-featured stages for GEMV using Stage composition.
//!
//! Features:
//! - Canonical 4x/8x unrolling for maximum throughput
//! - NK/KN layout support via WEIGHT_INDEX macro from LayoutStage
//! - Policy templates for transparent Q8/F16 dequantization
//! - Composable via CompoundKernel

use crate::compound::{BufferArg, Stage, stages::Quantization};

/// Metal definitions for GemvV2.
const GEMV_METAL: &str = include_str!("gemv.metal");

// =============================================================================
// VectorizedDotStage - Unified, quant-agnostic vectorized dot product
// =============================================================================

/// Vector load width for dot product stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum VectorWidth {
    /// 4 elements per load (half4)
    Vec4,
    /// 8 elements per load (float4 reinterpreted as 8 halves)
    #[default]
    Vec8,
}

impl VectorWidth {
    /// Number of elements loaded per thread per iteration.
    pub fn elements(&self) -> u32 {
        match self {
            VectorWidth::Vec4 => 4,
            VectorWidth::Vec8 => 8,
        }
    }
}


/// Unified vectorized dot product stage for warp-per-row GEMV.
///
/// Features:
/// - Parameterized by `Quantization` (F16, Q8) - no separate stages per quant
/// - Uses vectorized Policy::load_weights<8>() for maximum throughput
/// - Designed for use with `WarpLayoutStage` (warp-per-row dispatch)
/// - Each lane loads 8 elements per K chunk, all lanes cover 256 elements
#[derive(Debug, Clone)]
pub struct VectorizedDotStage {
    /// Quantization type (determines which Policy to use)
    quantization: Quantization,
    /// Vector width for loads
    vector_width: VectorWidth,
    /// Optional Gamma tensor (for RMSNorm fusion)
    gamma_buffer: Option<usize>,
    /// Optional shared memory variable name for normalization (e.g. "tg_inv_rms")
    norm_shared_name: Option<String>,
}

impl VectorizedDotStage {
    pub fn new(quantization: Quantization) -> Self {
        Self {
            quantization,
            vector_width: VectorWidth::Vec8,
            gamma_buffer: None,
            norm_shared_name: None,
        }
    }

    pub fn with_norm(mut self, gamma_idx: usize, shared_name: &str) -> Self {
        self.gamma_buffer = Some(gamma_idx);
        self.norm_shared_name = Some(shared_name.to_string());
        self
    }

    pub fn with_vector_width(mut self, width: VectorWidth) -> Self {
        self.vector_width = width;
        self
    }
}

impl Stage for VectorizedDotStage {
    fn includes(&self) -> Vec<&'static str> {
        // Include the appropriate policy file based on quantization
        vec![self.quantization.include_path()]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        // Buffer args for dot product - weights, scales, input, dimensions
        // Standardized: all weights/scales are uchar*, Policy handles casting
        let mut args = vec![
            BufferArg {
                name: "weights",
                metal_type: "const device uchar*",
                buffer_index: 0,
            },
            BufferArg {
                name: "scale_bytes",
                metal_type: "const device uchar*",
                buffer_index: 1,
            },
            BufferArg {
                name: "input",
                metal_type: "const device half*",
                buffer_index: 2,
            },
            BufferArg {
                name: "k_dim",
                metal_type: "constant uint&",
                buffer_index: 4,
            },
            BufferArg {
                name: "n_dim",
                metal_type: "constant uint&",
                buffer_index: 5,
            },
            BufferArg {
                name: "weights_per_block",
                metal_type: "constant uint&",
                buffer_index: 6,
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
        // Include gemv.metal for helper functions
        GEMV_METAL.to_string()
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let policy = self.quantization.policy_name();
        let vec_width = self.vector_width.elements();

        let fast_norm_logic = if self.gamma_buffer.is_some() {
            r#"
        // --- Fused Norm Application (Gamma + inv_rms) ---
        if (gamma) {
             float4 f_lo = float4(xv_lo);
             float4 f_hi = float4(xv_hi);
             
             // Intermediate float32 normalization
             f_lo *= inv_rms;
             f_hi *= inv_rms;

             // Scalar loads for gamma (to handle non-vector alignment)
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

        let shared_norm_logic = if let Some(name) = &self.norm_shared_name {
            if self.gamma_buffer.is_some() {
                // Already handled in fast_norm_logic for better precision
                String::new()
            } else {
                format!(
                    r#"
        // --- Apply Inverse RMS (Shared) ---
        xv_lo *= (half4){name};
        xv_hi *= (half4){name};
                "#
                )
            }
        } else {
            String::new()
        };

        // Optional Fused Norm Logic (Tail Path)
        let tail_norm_logic = if self.gamma_buffer.is_some() {
            r#"
        // --- Fused Norm Application (Gamma + inv_rms - Tail) ---
        if (gamma) {
             float4 f_lo = float4(xv_lo);
             float4 f_hi = float4(xv_hi);
             
             // Intermediate float32 normalization
             f_lo *= inv_rms;
             f_hi *= inv_rms;

             // Scalar load for gamma
             for (uint i = 0; i < 4 && k + i < k_dim; ++i) {
                 f_lo[i] *= (float)gamma[k + i];
             }
             for (uint i = 0; i < 4 && k + 4 + i < k_dim; ++i) {
                 f_hi[i] *= (float)gamma[k + 4 + i];
             }
             
             xv_lo = half4(f_lo);
             xv_hi = half4(f_hi);
        }
            "#.to_string()
        } else {
            String::new()
        };

        // Generate vectorized dot product code
        let code = format!(
            r#"
    // Vectorized Dot Product ({policy}, vec_width={vec_width})
    // Each lane loads {vec_width} elements per K chunk
    // K_CHUNK_SIZE = SIMD_WIDTH * {vec_width} = 256 elements
    const uint blocks_per_k = (k_dim + weights_per_block - 1) / weights_per_block;
    
    float acc = 0.0f;
    uint k_base = 0;
    const uint input_row_base = batch_idx * k_dim;
    
    // Fast path: no bounds checks (K_CHUNK_SIZE = 256)
    while (k_base + K_CHUNK_SIZE <= k_dim) {{
        uint k = k_base + lane_id * {vec_width}u;
        
        // Vector load {vec_width} halves from input
        float4 xv_raw = *(const device float4*)(input + input_row_base + k);
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);

        {fast_norm_logic}
        {shared_norm_logic}
        
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);
        
        // Load {vec_width} weights via Policy (vectorized or gather)
        float w[{vec_width}];
#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
        ulong base_idx = WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
        {policy}::template load_weights<{vec_width}>(weights, base_idx, w);
#else
        #pragma unroll
        for (int i = 0; i < {vec_width}; ++i) {{
            ulong idx = WEIGHT_INDEX(row_idx, k + i, k_dim, n_dim);
            float temp_w[1];
            {policy}::template load_weights<1>(weights, idx, temp_w);
            w[i] = temp_w[0];
        }}
#endif
        
        // Load scale via Policy
        half scale = {policy}::load_scale(scale_bytes, (ulong)row_idx * blocks_per_k + (k / weights_per_block));
        
        // Accumulate with dot product
        float4 w_lo = float4(w[0], w[1], w[2], w[3]);
        float4 w_hi = float4(w[4], w[5], w[6], w[7]);
        acc += (float)scale * (dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi));
        
        k_base += K_CHUNK_SIZE;
    }}
    
    // Tail: bounds-checked
    if (k_base < k_dim) {{
        uint k = k_base + lane_id * {vec_width}u;
        
        float4 xv_raw = float4(0.0f);
        uint valid_count = 0;
        
        if (k + {vec_width}u <= k_dim) {{
            xv_raw = *(const device float4*)(input + input_row_base + k);
            valid_count = {vec_width}u;
        }} else if (k < k_dim) {{
            for (uint i = 0; i < {vec_width}u && k + i < k_dim; ++i) {{
                ((thread half*)&xv_raw)[i] = input[input_row_base + k + i];
                valid_count++;
            }}
        }}
        
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);

        {tail_norm_logic}
        {shared_norm_logic}
        
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);
        
        float w[{vec_width}] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
        if (k < k_dim) {{
#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
            ulong base_idx = WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
            {policy}::template load_weights<{vec_width}>(weights, base_idx, w);
#else
            #pragma unroll
            for (int i = 0; i < {vec_width}; ++i) {{
                if (k + i < k_dim) {{
                    ulong idx = WEIGHT_INDEX(row_idx, k + i, k_dim, n_dim);
                    float temp_w[1];
                    {policy}::template load_weights<1>(weights, idx, temp_w);
                    w[i] = temp_w[0];
                }}
            }}
#endif
            for (uint i = valid_count; i < {vec_width}u; ++i) w[i] = 0.0f;
        }}
        
        half scale = (k < k_dim) ? {policy}::load_scale(scale_bytes, (ulong)row_idx * blocks_per_k + (k / weights_per_block)) : half(0.0h);
        
        float4 w_lo = float4(w[0], w[1], w[2], w[3]);
        float4 w_hi = float4(w[4], w[5], w[6], w[7]);
        acc += (float)scale * (dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi));
    }}
    
    float partial_dot = acc;

"#,
            policy = policy,
            vec_width = vec_width,
            fast_norm_logic = fast_norm_logic,
            shared_norm_logic = shared_norm_logic,
            tail_norm_logic = tail_norm_logic
        );

        ("partial_dot".to_string(), code)
    }
}

/// Canonical dot product stage (Legacy V1 compatibility).
///
/// Uses `gemv_dot_canonical` which implements a 4-way unrolled loop.
/// This acts as a robust fallback or alternative to the vectorized stage.
#[derive(Debug, Clone)]
pub struct CanonicalDotStage {
    quantization: Quantization,
}

impl CanonicalDotStage {
    pub fn new(quantization: Quantization) -> Self {
        Self { quantization }
    }
}

impl Stage for CanonicalDotStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![self.quantization.include_path()]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        // Same arguments as VectorizedDotStage
        // Standardized: all weights/scales are uchar*, Policy handles casting
        vec![
            BufferArg {
                name: "weights",
                metal_type: "const device uchar*",
                buffer_index: 0,
            },
            BufferArg {
                name: "scale_bytes",
                metal_type: "const device uchar*",
                buffer_index: 1,
            },
            BufferArg {
                name: "input",
                metal_type: "const device half*",
                buffer_index: 2,
            },
            BufferArg {
                name: "k_dim",
                metal_type: "constant uint&",
                buffer_index: 4,
            },
            BufferArg {
                name: "n_dim",
                metal_type: "constant uint&",
                buffer_index: 5,
            },
            BufferArg {
                name: "weights_per_block",
                metal_type: "constant uint&",
                buffer_index: 6,
            },
        ]
    }

    fn struct_defs(&self) -> String {
        GEMV_METAL.to_string()
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let policy = self.quantization.policy_name();

        let code = format!(
            r#"
    // Canonical Dot Product ({policy})
    // Uses 4-way unrolling with warp-interleaved access
    // Each thread processes 4 elements, stride = 32 * 4 = 128
    const uint blocks_per_k = (k_dim + weights_per_block - 1) / weights_per_block;
    const device half* input_row = input + batch_idx * k_dim;
    
    float acc = 0.0f;
    uint k_step = 32u * 4u; // 128
    
    uint k = lane_id * 4u;
    
    // Main interleaved loop
    while (k < k_dim) {{
         // Process 4 elements (or remainder)
         // gemv_dot_canonical handles bounds check against k_dim
         acc += gemv_dot_canonical<{policy}>(
             weights,
             scale_bytes,
             input_row,
             row_idx,
             k,
             k + 4u,
             k_dim,
             n_dim,
             weights_per_block
         );
         
         k += k_step;
    }}
    
    // Reduce happens in next stage
    float partial_dot = acc;
"#
        );

        ("partial_dot".to_string(), code)
    }
}

/// Stage that writes the reduced result to output with optional bias.
/// Designed for warp-per-row dispatch where only lane 0 writes.
#[derive(Debug, Clone)]
pub struct WarpWriteOutputStage;

impl WarpWriteOutputStage {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WarpWriteOutputStage {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage for WarpWriteOutputStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "output",
                metal_type: "device half*",
                buffer_index: 3,
            },
            BufferArg {
                name: "bias",
                metal_type: "const device half*",
                buffer_index: 7,
            },
            BufferArg {
                name: "has_bias",
                metal_type: "constant uint&",
                buffer_index: 8,
            },
            BufferArg {
                name: "alpha",
                metal_type: "constant float&",
                buffer_index: 9,
            },
            BufferArg {
                name: "residual",
                metal_type: "const device half*",
                buffer_index: 10,
            },
            BufferArg {
                name: "has_residual",
                metal_type: "constant uint&",
                buffer_index: 11,
            },
            BufferArg {
                name: "beta",
                metal_type: "constant float&",
                buffer_index: 12,
            },
        ]
    }

    fn struct_defs(&self) -> String {
        GEMV_METAL.to_string()
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let code = r#"
    // Write output (only lane 0 of each warp)
    if (lane_id == 0) {
        // Apply alpha scaling to the reduced sum
        float scaled_sum = row_sum * alpha;
        float result = scaled_sum;
        if (has_bias != 0) {
            result += (float)bias[row_idx];
        }
        if (has_residual != 0) {
            result += ((float)residual[batch_idx * n_dim + row_idx]) * beta;
        }
        output[batch_idx * n_dim + row_idx] = (half)result;
    }
"#
        .to_string();

        ("void".to_string(), code)
    }
}

// =============================================================================
// ScalarDotStage - Thread-per-row dot product (Large N optimized)
// =============================================================================

/// Dot product stage for thread-per-row dispatch.
///
/// Designed for Layout::ColMajor (KxN weights) where N is the contiguous dimension.
/// Threads in a warp access adjacent weights (W[k, n], W[k, n+1]...), enabling
/// full memory coalescing without implicit vector loads.
#[derive(Debug, Clone)]
pub struct ScalarDotStage {
    quantization: Quantization,
    unroll: usize,
}

impl ScalarDotStage {
    pub fn new(quantization: Quantization) -> Self {
        Self { quantization, unroll: 8 }
    }
}

impl Stage for ScalarDotStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![self.quantization.include_path()]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        // Same arguments as VectorizedDotStage
        vec![
            BufferArg {
                name: "weights",
                metal_type: "const device uchar*",
                buffer_index: 0,
            },
            BufferArg {
                name: "scale_bytes",
                metal_type: "const device uchar*",
                buffer_index: 1,
            },
            BufferArg {
                name: "input",
                metal_type: "const device half*",
                buffer_index: 2,
            },
            BufferArg {
                name: "k_dim",
                metal_type: "constant uint&",
                buffer_index: 4,
            },
            BufferArg {
                name: "n_dim",
                metal_type: "constant uint&",
                buffer_index: 5,
            },
            BufferArg {
                name: "weights_per_block",
                metal_type: "constant uint&",
                buffer_index: 6,
            },
        ]
    }

    fn struct_defs(&self) -> String {
        // Scalar dot doesn't need extra helper functions beyond standard policy loads
        "".to_string()
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let policy = self.quantization.policy_name();

        let code = format!(
            r#"
    // Scalar Dot Product ({{policy}})
    // Thread-per-row: One thread computes one output element.
    // Relies on warp coalescing for ColMajor weights.
    
    float acc = 0.0f;
    uint k = 0;
    
    // Main loop with unrolling
    while (k + {unroll} <= k_dim) {{
        #pragma unroll
        for (int i = 0; i < {unroll}; ++i) {{
            uint curr_k = k + i;
            
            // Load input (broadcast)
            // Note: input is half*, casting to float
            float val_x = (float)input[curr_k];
            
            // Load weight (coalesced)
            // Use scalar load<1> because each thread loads 1 scalar weight
            // But collectively the warp loads a vector (coalesced)
            ulong idx = WEIGHT_INDEX(row_idx, curr_k, k_dim, n_dim);
            float w;
            {policy}::template load_weights<1>(weights, idx, &w);
            
            // Load scale
            // Scale logic: one scale per block of K.
            uint blocks_per_k_dim = (k_dim + weights_per_block - 1) / weights_per_block;
            half scale = {policy}::load_scale(scale_bytes, (ulong)row_idx * blocks_per_k_dim + (curr_k / weights_per_block));
            
            acc += val_x * w * (float)scale;
        }}
        k += {unroll};
    }}
    
    // Tail loop
    while (k < k_dim) {{
        float val_x = (float)input[k];
        ulong idx = WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
        float w;
        {policy}::template load_weights<1>(weights, idx, &w);
        
        uint blocks_per_k_dim = (k_dim + weights_per_block - 1) / weights_per_block;
        half scale = {policy}::load_scale(scale_bytes, (ulong)row_idx * blocks_per_k_dim + (k / weights_per_block));
        
        acc += val_x * w * (float)scale;
        k++;
    }}
    
    float row_sum = acc;     // Result is full sum
"#,
            policy = policy,
            unroll = self.unroll
        );

        ("row_sum".to_string(), code)
    }
}
