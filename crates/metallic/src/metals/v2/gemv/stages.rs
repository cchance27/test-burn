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
pub enum VectorWidth {
    /// 4 elements per load (half4)
    Vec4,
    /// 8 elements per load (float4 reinterpreted as 8 halves)
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

impl Default for VectorWidth {
    fn default() -> Self {
        VectorWidth::Vec8
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
}

impl VectorizedDotStage {
    pub fn new(quantization: Quantization) -> Self {
        Self {
            quantization,
            vector_width: VectorWidth::Vec8,
        }
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
        // Include gemv.metal for helper functions
        GEMV_METAL.to_string()
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let policy = self.quantization.policy_name();
        let vec_width = self.vector_width.elements();

        // Generate vectorized dot product code
        let code = format!(
            r#"
    // Vectorized Dot Product ({policy}, vec_width={vec_width})
    // Each lane loads {vec_width} elements per K chunk
    // K_CHUNK_SIZE = SIMD_WIDTH * {vec_width} = 256 elements
    const uint blocks_per_k = (k_dim + weights_per_block - 1) / weights_per_block;
    
    float acc = 0.0f;
    uint k_base = 0;
    
    // Fast path: no bounds checks (K_CHUNK_SIZE = 256)
    while (k_base + K_CHUNK_SIZE <= k_dim) {{
        uint k = k_base + lane_id * {vec_width}u;
        
        // Vector load {vec_width} halves from input
        float4 xv_raw = *(const device float4*)(input + k);
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);
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
            vec_width = vec_width
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
        ]
    }

    fn struct_defs(&self) -> String {
        GEMV_METAL.to_string()
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let code = r#"
    // Write output (only lane 0 of each warp)
    if (lane_id == 0) {
        gemv_write_output(output, bias, row_idx, row_sum, has_bias != 0);
    }
"#
        .to_string();

        ("void".to_string(), code)
    }
}
