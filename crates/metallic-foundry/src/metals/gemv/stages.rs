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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
    /// Use F16-specific cols8 SIMD path (RowMajor only).
    f16_cols8: bool,
}

impl VectorizedDotStage {
    pub fn new(quantization: Quantization) -> Self {
        Self {
            quantization,
            vector_width: VectorWidth::Vec8,
            gamma_buffer: None,
            norm_shared_name: None,
            f16_cols8: false,
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

    pub fn with_f16_cols8(mut self, enabled: bool) -> Self {
        self.f16_cols8 = enabled;
        self
    }
}

impl Stage for VectorizedDotStage {
    fn includes(&self) -> Vec<&'static str> {
        // Include the appropriate policy file based on quantization
        vec![self.quantization.include_path()]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        // Buffer args for dot product - weights, scales, input, dimensions.
        // Type weights as `half*` for F16 so PolicyF16 overloads can avoid uchar* aliasing.
        let weights_type = match self.quantization {
            Quantization::F16 => "const device half*",
            _ => "const device uchar*",
        };

        let mut args = vec![
            BufferArg {
                name: "weights",
                metal_type: weights_type,
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
        let use_f16_cols8 = self.f16_cols8 && self.quantization == Quantization::F16 && matches!(self.vector_width, VectorWidth::Vec8);

        let (fast_norm_logic, tail_norm_logic) = if self.gamma_buffer.is_some() {
            match self.vector_width {
                VectorWidth::Vec8 => (
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
            "#,
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
            "#,
                ),
                VectorWidth::Vec4 => (
                    r#"
        // --- Fused Norm Application (Gamma + inv_rms) ---
        if (gamma) {
             float4 f_lo = float4(xv_lo);
             
             // Intermediate float32 normalization
             f_lo *= inv_rms;

             // Scalar loads for gamma (Vec4)
             const device half* g_ptr = gamma + k;
             f_lo.x *= (float)g_ptr[0]; f_lo.y *= (float)g_ptr[1];
             f_lo.z *= (float)g_ptr[2]; f_lo.w *= (float)g_ptr[3];
             
             xv_lo = half4(f_lo);
        }
            "#,
                    r#"
        // --- Fused Norm Application (Gamma + inv_rms - Tail) ---
        if (gamma) {
             float4 f_lo = float4(xv_lo);
             
             // Intermediate float32 normalization
             f_lo *= inv_rms;

             // Scalar load for gamma (Vec4 tail)
             for (uint i = 0; i < 4 && k + i < k_dim; ++i) {
                 f_lo[i] *= (float)gamma[k + i];
             }
             
             xv_lo = half4(f_lo);
        }
            "#,
                ),
            }
        } else {
            ("", "")
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

        let tail_norm_logic = tail_norm_logic.to_string();

        // Generate vectorized dot product code
        let code = if use_f16_cols8 {
            format!(
                r#"
    // Vectorized Dot Product (F16 cols8 path, vec_width={vec_width})
    // Mirrors Context SIMD GEMV pointer arithmetic for RowMajor.
    const uint input_row_base = batch_idx * k_dim;
    float acc = 0.0f;
    uint k_base = 0;

#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
    const device half* row_ptr = weights + (ulong)row_idx * (ulong)k_dim;
    const device half* w_ptr = row_ptr + lane_id * {vec_width}u;
    const device half* x_ptr = input + input_row_base + lane_id * {vec_width}u;

    // Fast path: no bounds checks (K_CHUNK_SIZE = 256)
    uint remaining = k_dim;
    while (remaining >= K_CHUNK_SIZE) {{
        float4 xv_raw = *(const device float4*)(x_ptr);
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);

        {fast_norm_logic}
        {shared_norm_logic}

        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);

        float4 w_raw = *(const device float4*)(w_ptr);
        half4 w_lo = as_type<half4>(w_raw.xy);
        half4 w_hi = as_type<half4>(w_raw.zw);

        acc += dot(xv_f32_lo, float4(w_lo)) + dot(xv_f32_hi, float4(w_hi));

        x_ptr += K_CHUNK_SIZE;
        w_ptr += K_CHUNK_SIZE;
        k_base += K_CHUNK_SIZE;
        remaining -= K_CHUNK_SIZE;
    }}

    // Tail: bounds-checked
    if (remaining > 0) {{
        const uint lane_off = lane_id * {vec_width}u;
        const uint valid_count = (remaining > lane_off) ? min({vec_width}u, remaining - lane_off) : 0u;

        float4 xv_raw = float4(0.0f);
        if (valid_count == {vec_width}u) {{
            xv_raw = *(const device float4*)(x_ptr);
        }} else if (valid_count > 0u) {{
            #pragma unroll
            for (uint i = 0; i < {vec_width}u; ++i) {{
                if (i < valid_count) {{
                    ((thread half*)&xv_raw)[i] = x_ptr[i];
                }}
            }}
        }}

        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);

        {tail_norm_logic}

        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);

        float4 w_raw = float4(0.0f);
        if (valid_count == {vec_width}u) {{
            w_raw = *(const device float4*)(w_ptr);
        }} else if (valid_count > 0u) {{
            #pragma unroll
            for (uint i = 0; i < {vec_width}u; ++i) {{
                if (i < valid_count) {{
                    ((thread half*)&w_raw)[i] = w_ptr[i];
                }}
            }}
        }}

        half4 w_lo = as_type<half4>(w_raw.xy);
        half4 w_hi = as_type<half4>(w_raw.zw);

        if (valid_count > 0u) {{
            acc += dot(xv_f32_lo, float4(w_lo)) + dot(xv_f32_hi, float4(w_hi));
        }}
    }}
#else
    // Fallback to standard vectorized dot for non-contiguous layouts.
    const uint blocks_per_k = {policy}::HAS_SCALE ? ((k_dim + weights_per_block - 1) / weights_per_block) : 0u;
    while (k_base + K_CHUNK_SIZE <= k_dim) {{
        uint k = k_base + lane_id * {vec_width}u;
        float4 xv_raw = *(const device float4*)(input + input_row_base + k);
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);
        {fast_norm_logic}
        {shared_norm_logic}
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);
        float w[{vec_width}];
        #pragma unroll
        for (int i = 0; i < {vec_width}; ++i) {{
            ulong idx = WEIGHT_INDEX(row_idx, k + i, k_dim, n_dim);
            float temp_w[1];
            {policy}::template load_weights<1>(weights, idx, temp_w);
            w[i] = temp_w[0];
        }}
        float4 w_lo = float4(w[0], w[1], w[2], w[3]);
        float4 w_hi = float4(w[4], w[5], w[6], w[7]);
        acc += dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi);
        k_base += K_CHUNK_SIZE;
    }}
    if (k_base < k_dim) {{
        uint k = k_base + lane_id * {vec_width}u;
        float4 xv_raw = float4(0.0f);
        if (k + {vec_width}u <= k_dim) {{
            xv_raw = *(const device float4*)(input + input_row_base + k);
        }} else if (k < k_dim) {{
            for (uint i = 0; i < {vec_width}u && k + i < k_dim; ++i) {{
                ((thread half*)&xv_raw)[i] = input[input_row_base + k + i];
            }}
        }}
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);
        {tail_norm_logic}
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);
        float w[{vec_width}] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
        if (k < k_dim) {{
            #pragma unroll
            for (int i = 0; i < {vec_width}; ++i) {{
                ulong idx = WEIGHT_INDEX(row_idx, k + i, k_dim, n_dim);
                float temp_w[1];
                {policy}::template load_weights<1>(weights, idx, temp_w);
                w[i] = temp_w[0];
            }}
        }}
        float4 w_lo = float4(w[0], w[1], w[2], w[3]);
        float4 w_hi = float4(w[4], w[5], w[6], w[7]);
        acc += dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi);
    }}
#endif

    float partial_dot = acc;

"#,
                policy = policy,
                vec_width = vec_width,
                fast_norm_logic = fast_norm_logic,
                shared_norm_logic = shared_norm_logic,
                tail_norm_logic = tail_norm_logic,
            )
        } else {
            format!(
                r#"
    // Vectorized Dot Product ({policy}, vec_width={vec_width})
    // Each lane loads {vec_width} elements per K chunk
    // K_CHUNK_SIZE = SIMD_WIDTH * {vec_width} = 256 elements
#if defined(METALLIC_POLICY_HAS_SCALE) && METALLIC_POLICY_HAS_SCALE
    const uint blocks_per_k = (k_dim + weights_per_block - 1) / weights_per_block;
#else
    const uint blocks_per_k = 0u;
#endif
    
	    float acc = 0.0f;
	    uint k_base = 0;
	    const uint input_row_base = batch_idx * k_dim;

// FP16-only optimization:
// When `n_dim` is a multiple of `WARPS_PER_TG` (so the warp-per-row layout never produces out-of-bounds rows),
// we can safely use threadgroup barriers to cooperatively stage the input vector once per K chunk and reuse it
// across all warps in the threadgroup. This avoids re-reading the same X values WARPS_PER_TG times from device
// memory. This is only emitted for the FP16-weight policy; for quantized weights we avoid emitting the
// threadgroup staging code entirely to keep occupancy unchanged and prevent compilation issues.
#if defined(METALLIC_POLICY_WEIGHTS_FP16) && METALLIC_POLICY_WEIGHTS_FP16 && ({vec_width}u == 8u)
	    threadgroup half x_tile[K_CHUNK_SIZE];
	    const bool use_x_tile = ((n_dim & (WARPS_PER_TG - 1u)) == 0u);
#else
	    const bool use_x_tile = false;
#endif
	    
	    // Fast path: no bounds checks (K_CHUNK_SIZE = 256)
	    while (k_base + K_CHUNK_SIZE <= k_dim) {{
#if defined(METALLIC_POLICY_WEIGHTS_FP16) && METALLIC_POLICY_WEIGHTS_FP16 && ({vec_width}u == 8u)
	        if (use_x_tile) {{
	            // Cooperative load: each thread loads 1 half (256 threads -> 256 halfs).
	            x_tile[lid.x] = input[input_row_base + k_base + lid.x];
	            threadgroup_barrier(mem_flags::mem_threadgroup);
	        }}
#endif
	        uint k = k_base + lane_id * {vec_width}u;
	        
	        // Vector load {vec_width} halves from input
	        half4 xv_lo;
	        half4 xv_hi;
#if defined(METALLIC_POLICY_WEIGHTS_FP16) && METALLIC_POLICY_WEIGHTS_FP16 && ({vec_width}u == 8u)
	        if (use_x_tile) {{
	            // Read the staged 8 halfs for this lane from threadgroup memory.
	            const uint base = lane_id * 8u;
	            const threadgroup half* x_ptr_tg = x_tile + base;
	            xv_lo = *(const threadgroup half4*)(x_ptr_tg + 0u);
	            xv_hi = *(const threadgroup half4*)(x_ptr_tg + 4u);
	        }} else
#endif
	        if ({vec_width}u == 4u) {{
	            xv_lo = *(const device half4*)(input + input_row_base + k);
	            xv_hi = half4(0.0h);
	        }} else {{
	            float4 xv_raw = *(const device float4*)(input + input_row_base + k);
	            xv_lo = as_type<half4>(xv_raw.xy);
	            xv_hi = as_type<half4>(xv_raw.zw);
	        }}

	        {fast_norm_logic}
	        {shared_norm_logic}
	        
	        float4 xv_f32_lo = float4(xv_lo);
	        float4 xv_f32_hi = float4(xv_hi);
        
#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
#if defined(METALLIC_POLICY_WEIGHTS_FP16) && METALLIC_POLICY_WEIGHTS_FP16
        const device half* weights_row = weights + (ulong)row_idx * (ulong)k_dim;
        acc += {policy}::template dot<{vec_width}>(weights_row, (ulong)k, 1.0f, xv_f32_lo, xv_f32_hi);
#else
        ulong base_idx = WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
        float scale = 1.0f;
#if defined(METALLIC_POLICY_HAS_SCALE) && METALLIC_POLICY_HAS_SCALE
        scale = (float){policy}::load_scale(scale_bytes, (ulong)row_idx * blocks_per_k + (k / weights_per_block));
#endif
        acc += {policy}::template dot<{vec_width}>(weights, base_idx, scale, xv_f32_lo, xv_f32_hi);
#endif
#else
        // Strided K: gather weights explicitly.
        float w[{vec_width}];
        #pragma unroll
        for (int i = 0; i < {vec_width}; ++i) {{
            ulong idx = WEIGHT_INDEX(row_idx, k + i, k_dim, n_dim);
            float temp_w[1];
            {policy}::template load_weights<1>(weights, idx, temp_w);
            w[i] = temp_w[0];
        }}

        // Load scale via Policy
        float scale = {policy}::HAS_SCALE
            ? (float){policy}::load_scale(scale_bytes, (ulong)row_idx * blocks_per_k + (k / weights_per_block))
            : 1.0f;
        
        // Accumulate with dot product
        if ({vec_width}u == 8u) {{
            float4 w_lo = float4(w[0], w[1], w[2], w[3]);
            float4 w_hi = float4(w[4], w[5], w[6], w[7]);
            acc += scale * (dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi));
	        }} else {{
            float4 w_lo = float4(w[0], w[1], w[2], w[3]);
            acc += scale * dot(xv_f32_lo, w_lo);
        }}
	#endif

#if defined(METALLIC_POLICY_WEIGHTS_FP16) && METALLIC_POLICY_WEIGHTS_FP16 && ({vec_width}u == 8u)
	        if (use_x_tile) {{
	            // Ensure all warps are done reading x_tile before overwriting it in the next chunk.
	            threadgroup_barrier(mem_flags::mem_threadgroup);
	        }}
#endif
	        k_base += K_CHUNK_SIZE;
	    }}
    
    // Tail: bounds-checked
    if (k_base < k_dim) {{
        uint k = k_base + lane_id * {vec_width}u;

        half4 xv_lo = half4(0.0h);
        half4 xv_hi = half4(0.0h);
        if ({vec_width}u == 8u) {{
            float4 xv_raw = float4(0.0f);

            if (k + 8u <= k_dim) {{
                xv_raw = *(const device float4*)(input + input_row_base + k);
            }} else if (k < k_dim) {{
                for (uint i = 0; i < 8u && k + i < k_dim; ++i) {{
                    ((thread half*)&xv_raw)[i] = input[input_row_base + k + i];
                }}
            }}
            xv_lo = as_type<half4>(xv_raw.xy);
            xv_hi = as_type<half4>(xv_raw.zw);
        }} else {{
            // Vec4 tail: scalar gather into half4 (avoid over-reads).
            half x0 = (k + 0u < k_dim) ? input[input_row_base + k + 0u] : half(0.0h);
            half x1 = (k + 1u < k_dim) ? input[input_row_base + k + 1u] : half(0.0h);
            half x2 = (k + 2u < k_dim) ? input[input_row_base + k + 2u] : half(0.0h);
            half x3 = (k + 3u < k_dim) ? input[input_row_base + k + 3u] : half(0.0h);
            xv_lo = half4(x0, x1, x2, x3);
            xv_hi = half4(0.0h);
        }}

        {tail_norm_logic}
        {shared_norm_logic}
        
        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);

        if (k + {vec_width}u <= k_dim) {{
            // Full vector tail (aligned for this lane).
#if defined(IS_K_CONTIGUOUS) && IS_K_CONTIGUOUS
#if defined(METALLIC_POLICY_WEIGHTS_FP16) && METALLIC_POLICY_WEIGHTS_FP16
            const device half* weights_row = weights + (ulong)row_idx * (ulong)k_dim;
            acc += {policy}::template dot<{vec_width}>(weights_row, (ulong)k, 1.0f, xv_f32_lo, xv_f32_hi);
#else
            ulong base_idx = WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
            float scale = 1.0f;
#if defined(METALLIC_POLICY_HAS_SCALE) && METALLIC_POLICY_HAS_SCALE
            scale = (float){policy}::load_scale(scale_bytes, (ulong)row_idx * blocks_per_k + (k / weights_per_block));
#endif
            acc += {policy}::template dot<{vec_width}>(weights, base_idx, (float)scale, xv_f32_lo, xv_f32_hi);
#endif
#else
            float w[{vec_width}];
            #pragma unroll
            for (int i = 0; i < {vec_width}; ++i) {{
                ulong idx = WEIGHT_INDEX(row_idx, k + i, k_dim, n_dim);
                float temp_w[1];
                {policy}::template load_weights<1>(weights, idx, temp_w);
                w[i] = temp_w[0];
            }}
            half scale = {policy}::load_scale(scale_bytes, (ulong)row_idx * blocks_per_k + (k / weights_per_block));
            if ({vec_width}u == 8u) {{
                float4 w_lo = float4(w[0], w[1], w[2], w[3]);
                float4 w_hi = float4(w[4], w[5], w[6], w[7]);
                acc += (float)scale * (dot(xv_f32_lo, w_lo) + dot(xv_f32_hi, w_hi));
            }} else {{
                float4 w_lo = float4(w[0], w[1], w[2], w[3]);
                acc += (float)scale * dot(xv_f32_lo, w_lo);
            }}
#endif
        }} else if (k < k_dim) {{
            // Partial tail (non-multiple-of-vec_width K): safe scalar gather.
            half scale = {policy}::load_scale(scale_bytes, (ulong)row_idx * blocks_per_k + (k / weights_per_block));
            float acc_tail = 0.0f;
            #pragma unroll
            for (uint i = 0; i < {vec_width}u; ++i) {{
                if (k + i < k_dim) {{
                    ulong idx = WEIGHT_INDEX(row_idx, k + i, k_dim, n_dim);
                    float temp_w[1];
                    {policy}::template load_weights<1>(weights, idx, temp_w);
                    acc_tail += temp_w[0] * scale * (float)input[input_row_base + k + i];
                }}
            }}
            acc += acc_tail;
        }}
    }}
    
    float partial_dot = acc;

"#,
                policy = policy,
                vec_width = vec_width,
                fast_norm_logic = fast_norm_logic,
                shared_norm_logic = shared_norm_logic,
                tail_norm_logic = tail_norm_logic,
            )
        };

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
        vec![
            BufferArg {
                name: "weights",
                metal_type: match self.quantization {
                    Quantization::F16 => "const device half*",
                    _ => "const device uchar*",
                },
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
                metal_type: match self.quantization {
                    Quantization::F16 => "const device half*",
                    _ => "const device uchar*",
                },
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
