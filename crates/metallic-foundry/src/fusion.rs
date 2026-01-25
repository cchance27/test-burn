use crate::policy::OptimizationMetadata;

/// Defines a Metal kernel policy (e.g., input loading strategy).
/// This corresponds to a C++ template class or struct in Metal.
pub trait MetalPolicy: Send + Sync + std::fmt::Debug {
    /// The name of the header file to include (e.g., "policy_q8.metal").
    fn header(&self) -> &'static str;

    /// The name of the Metal struct implementing the policy (e.g., "PolicyQ8").
    fn struct_name(&self) -> &'static str;

    /// Short name for kernel naming (e.g., "f16", "q8").
    fn short_name(&self) -> &'static str {
        "unknown"
    }

    /// Size of a single element in bytes (e.g., 2 for F16, 1 for Q8).
    fn element_size(&self) -> usize {
        2 // Default to F16
    }

    /// Size of a quantization block in bytes (e.g., 34 for Q8_0).
    /// For unquantized types, this is the element size.
    fn block_size_bytes(&self) -> usize {
        self.element_size()
    }

    /// Number of logical weights per quantization block (e.g., 32 for Q8_0, 1 for F16).
    fn weights_per_block(&self) -> usize {
        1
    }

    /// Convert an elements expression to a bytes expression for Metal code.
    /// Uses element_size() to compute the multiplication factor.
    fn bytes(&self, elements: &str) -> String {
        match self.element_size() {
            1 => format!("(ulong)({})", elements),
            2 => format!("(ulong)({}) * 2", elements),
            4 => format!("(ulong)({}) * 4", elements),
            n => format!("(ulong)({}) * {}", elements, n),
        }
    }

    /// Optimization hints for kernel compilation and dispatch.
    fn optimization_hints(&self) -> OptimizationMetadata {
        OptimizationMetadata::default()
    }

    /// Whether this policy uses a separate scales buffer.
    fn has_scale(&self) -> bool {
        true
    }

    /// Validate the layout of a weight tensor for this policy.
    fn validate_weight_layout(&self, dims: &[usize], bytes: usize) -> Result<(), crate::error::MetalError> {
        use crate::error::MetalError;

        // Only apply structural validation for policies that use companion resources (e.g. scales).
        // Unquantized policies (F16/F32) should be layout-agnostic here.
        if !self.has_scale() {
            return Ok(());
        }

        let wpb = self.weights_per_block().max(1);
        match dims {
            // Row/col-major: we keep logical 2D dims, so we can validate K directly.
            [_, k] => {
                if wpb > 1 && (k % wpb) != 0 {
                    return Err(MetalError::InvalidShape(format!(
                        "quantized weight K dimension ({k}) must be divisible by {wpb}"
                    )));
                }
            }
            // Canonical (or other packed layouts): weights may be stored as a flat byte/element stream.
            // In that case, we can only validate block-alignment, not logical K/N.
            [len] => {
                if wpb > 1 && (len % wpb) != 0 {
                    return Err(MetalError::InvalidShape(format!(
                        "quantized weight buffer length ({len}) must be divisible by {wpb}"
                    )));
                }
            }
            _ => {
                return Err(MetalError::InvalidShape(format!(
                    "quantized weight tensor must be 1D or 2D, got dims={dims:?}"
                )));
            }
        }

        // Optional size sanity-check (some tests validate shape without an attached buffer).
        if bytes != 0 {
            let expected_elems = dims.iter().copied().fold(1usize, |acc, v| acc.saturating_mul(v));
            let expected_min = expected_elems.saturating_mul(self.element_size());
            if bytes < expected_min {
                return Err(MetalError::InvalidShape(format!(
                    "quantized weight buffer too small: bytes={bytes} expected_at_least={expected_min} dims={dims:?}"
                )));
            }
        }

        Ok(())
    }

    /// Bind any associated resources (e.g. scales) for a weight tensor.
    fn bind_associated_resources(&self, _encoder: &crate::types::ComputeCommandEncoder, _weights: &crate::types::TensorArg) {}

    /// Metal code to initialize policy params from kernel args.
    /// E.g., "pp.matrix = matrix; pp.scales = scale_bytes;"
    fn init_params_code(&self) -> &'static str {
        ""
    }

    /// Buffer argument name to Metal type mappings.
    /// Used for generating kernel signatures with correct types.
    /// E.g., &[("matrix", "const device uchar*"), ("scale_bytes", "const device uchar*")]
    fn buffer_types(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    /// Any macro definitions required (key, value).
    fn defines(&self) -> Vec<(&'static str, String)> {
        vec![]
    }

    /// Generate the LOAD_MATRIX macro definition for this policy.
    ///
    /// This allows policies to control exactly how the kernel loads data.
    /// The macro signature is: LOAD_MATRIX(matrix, batch_offset, col, row, stride)
    ///
    /// Default implementation returns None (using kernel's default F16 loader).
    fn define_loader(&self) -> Option<String> {
        None
    }

    /// Generate inline Metal code to load data and dequant to float4.
    ///
    /// This replaces Metal C++ templates with Rust code generation.
    /// The policy handles all dtype-specific logic (Q8 dequant, F16 cast, etc.)
    ///
    /// # Arguments
    /// * `data_ptr` - Name of the data buffer pointer variable
    /// * `scale_ptr` - Name of the scale buffer pointer variable (for quantized types)
    /// * `offset` - Offset expression into the buffer
    ///
    /// # Returns
    /// * `output_var` - Name of the float4 variable containing dequanted weights
    /// * `code` - Metal code string to emit
    fn load_and_dequant(&self, data_ptr: &str, _scale_ptr: &str, offset: &str) -> (&'static str, String) {
        // Default: no-op, subclasses override
        (
            "w_vec",
            format!("float4 w_vec = float4(*(const device half4*)({data_ptr} + {offset}));"),
        )
    }

    /// Chunk size for the fast (no bounds check) iteration path.
    fn fast_chunk_size(&self) -> usize {
        256 // Default: 32 threads * 8 halves per thread
    }

    /// Number of columns per threadgroup for dispatch.
    fn cols_per_threadgroup(&self) -> usize {
        4 // Default: 4 warps
    }

    /// Threadgroup width for dispatch.
    fn threadgroup_width(&self) -> usize {
        128 // Default: 4 warps * 32 lanes
    }
}

impl PartialEq for dyn MetalPolicy + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.short_name() == other.short_name()
    }
}

impl Eq for dyn MetalPolicy + '_ {}

impl std::hash::Hash for dyn MetalPolicy + '_ {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.short_name().hash(state);
    }
}

/// Trait for types that have Metal argument info (generated by KernelArgs derive).
/// This enables default implementations for signature generation.
pub trait HasMetalArgs {
    /// Metal argument signature parts: (name, buffer_index, metal_type)
    const METAL_ARGS: &'static [(&'static str, u64, &'static str)];

    /// Stage-compatible args (excludes stage_skip buffers like PolicyStage provides)
    /// Default: same as METAL_ARGS (no skips)
    const STAGE_METAL_ARGS: &'static [(&'static str, u64, &'static str)] = Self::METAL_ARGS;

    /// Generate Metal kernel function signature with policy type overrides.
    ///
    /// Default implementation uses METAL_ARGS with policy.buffer_types() overrides.
    fn metal_signature<P: MetalPolicy>(func_name: &str, policy: &P) -> String {
        generate_metal_signature(
            func_name,
            Self::METAL_ARGS,
            policy.buffer_types(),
            true, // Include standard thread position args
        )
    }
}

/// Defines a Metal kernel epilogue (e.g., Activation, Norm).
pub trait Epilogue: Send + Sync {
    /// The name of the header file to include (e.g., "epilogue_rmsnorm.metal").
    fn header(&self) -> &'static str;

    /// The name of the Metal struct implementing the epilogue (e.g., "EpilogueRmsNorm").
    fn struct_name(&self) -> &'static str;

    /// Any macro definitions required.
    fn defines(&self) -> Vec<(&'static str, String)> {
        vec![]
    }
}

/// A builder for constructing a Metal kernel source string dynamically.
pub struct SourceBuilder {
    includes: Vec<String>,
    defines: Vec<(String, String)>,
    signature: String,
    body: String,
}

impl Default for SourceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceBuilder {
    pub fn new() -> Self {
        Self {
            includes: Vec::new(),
            defines: Vec::new(),
            signature: String::new(),
            body: String::new(),
        }
    }

    pub fn include(mut self, header: &str) -> Self {
        self.includes.push(header.to_string());
        self
    }

    pub fn define(mut self, key: &str, value: &str) -> Self {
        self.defines.push((key.to_string(), value.to_string()));
        self
    }

    pub fn signature(mut self, sig: &str) -> Self {
        self.signature = sig.to_string();
        self
    }

    pub fn body(mut self, body: &str) -> Self {
        self.body = body.to_string();
        self
    }

    pub fn build(&self) -> String {
        let mut s = String::new();
        // Includes can be injected here or handled by the caller (e.g. via `Kernel::includes`).

        for (k, v) in &self.defines {
            s.push_str(&format!("#define {} {}\n", k, v));
        }
        for inc in &self.includes {
            s.push_str(&format!("#include \"{}\"\n", inc));
        }
        s.push('\n');
        s.push_str(&format!("kernel void {} {{\n", self.signature));
        s.push_str(&self.body);
        s.push_str("\n}\n");
        s
    }
}

/// Standard Metal thread position arguments (commonly needed by most kernels).
pub const THREAD_ARGS: &[&str] = &[
    "uint3 gid [[threadgroup_position_in_grid]]",
    "uint3 lid [[thread_position_in_threadgroup]]",
    "uint3 tptg [[threads_per_threadgroup]]",
];

/// Generate a Metal kernel signature from METAL_ARGS and policy buffer_types().
///
/// # Arguments
/// * `func_name` - The kernel function name
/// * `metal_args` - &[(name, buffer_index, default_metal_type)] from METAL_ARGS const
/// * `policy_types` - Policy-specific type overrides from buffer_types()
/// * `include_thread_args` - Whether to append standard thread position args
pub fn generate_metal_signature(
    func_name: &str,
    metal_args: &[(&str, u64, &str)],
    policy_types: &[(&str, &str)],
    include_thread_args: bool,
) -> String {
    let mut args: Vec<String> = Vec::new();

    for (name, idx, default_type) in metal_args {
        // Check if policy overrides this buffer's type
        let metal_type = policy_types
            .iter()
            .find(|(n, _)| *n == *name)
            .map(|(_, t)| *t)
            .unwrap_or(*default_type);

        args.push(format!("    {} {} [[buffer({})]]", metal_type, name, idx));
    }

    if include_thread_args {
        for arg in THREAD_ARGS {
            args.push(format!("    {}", arg));
        }
    }

    format!("{}(\n{}\n)", func_name, args.join(",\n"))
}
