use std::marker::PhantomData;

use crate::{
    compound::{BufferArg, Stage}, fusion::HasMetalArgs
};

pub const MATMUL_GEMV_SIMD_INCLUDES: &[&str] = &["matmul_gemv/simd_common.metal"];

pub trait GemvConfig: Send + Sync + 'static {
    type Args: HasMetalArgs;

    const HEADS: usize;
    const COLS_PER_TG: usize;
    const FAST_PATH: bool;

    fn base_includes() -> &'static [&'static str] {
        MATMUL_GEMV_SIMD_INCLUDES
    }

    /// Optional per-head scale pointers (Q8 canonical uses half scales stored as 2 bytes).
    /// When non-empty, `scale_arr[HEADS]` is declared before hook code runs.
    fn scale_ptrs() -> &'static [&'static str] {
        &[]
    }

    fn data_ptrs() -> &'static [&'static str];
    fn result_ptrs() -> &'static [&'static str];
    fn n_exprs() -> &'static [&'static str];
    fn bias_ptrs() -> &'static [&'static str];
    fn has_bias_flags() -> &'static [&'static str];
    fn gemv_params_n0_expr() -> &'static str;

    /// Return Metal struct definitions needed by the kernel args.
    /// Default: empty string. Override if your Args contains MetalStruct fields.
    fn struct_defs() -> String {
        String::new()
    }
}

pub trait GemvHook: Send + Sync + 'static {
    fn id() -> &'static str;
    fn includes() -> &'static [&'static str] {
        &[]
    }
    fn policy_struct() -> &'static str;
    fn preamble_code() -> &'static str {
        ""
    }
    fn policy_params_code() -> &'static str;
}

/// Prologue stage for SIMD GEMV — runs before hook/main.
///
/// Prologues set up threadgroup state (e.g., computing inv_rms once per TG).
/// They do NOT produce an output variable — their effects live in threadgroup memory.
pub trait GemvPrologue: Send + Sync + 'static {
    /// Additional includes for prologue computation.
    fn includes() -> &'static [&'static str] {
        &[]
    }

    /// Buffer args contributed by prologue.
    /// Indices will be auto-assigned AFTER main args.
    fn buffer_args() -> Vec<BufferArg> {
        vec![]
    }

    /// Metal code emitted at start of kernel (before hook preamble).
    /// Sets up threadgroup-shared state used by subsequent stages.
    fn emit() -> String;
}

/// No-op prologue for kernels that don't need pre-main setup.
pub struct NoPrologue;

impl GemvPrologue for NoPrologue {
    fn emit() -> String {
        // Provide dummy values for unified policy compatibility.
        // When no RMSNorm prologue is used, inv_rms = 1.0 means no scaling,
        // and gamma = nullptr means skip gamma multiplication.
        r#"    const device half *gamma = nullptr;
    const float inv_rms = 1.0f;
"#
        .to_string()
    }
}

pub trait GemvEpilogue: Send + Sync + 'static {
    fn id() -> &'static str;
    fn includes() -> &'static [&'static str] {
        &[]
    }
    fn template_arg() -> Option<&'static str> {
        None
    }
    /// Returns Metal code for SIMD reduction ladder.
    /// Format: variable declarations and simd_shuffle_xor reductions.
    fn simd_reduce_code() -> &'static str {
        ""
    }
}

pub struct GemvStage<P, C, H, E> {
    _pd: PhantomData<(P, C, H, E)>,
}

impl<P, C, H, E> Default for GemvStage<P, C, H, E> {
    fn default() -> Self {
        Self { _pd: PhantomData }
    }
}

impl<P, C, H, E> Stage for GemvStage<P, C, H, E>
where
    P: GemvPrologue,
    C: GemvConfig,
    H: GemvHook,
    E: GemvEpilogue,
{
    fn includes(&self) -> Vec<&'static str> {
        let mut incs = Vec::new();
        incs.extend_from_slice(P::includes());
        incs.extend_from_slice(C::base_includes());
        incs.extend_from_slice(H::includes());
        incs.extend_from_slice(E::includes());
        incs
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        let mut args: Vec<BufferArg> = C::Args::METAL_ARGS
            .iter()
            .map(|(name, idx, metal_type)| BufferArg {
                name,
                metal_type,
                buffer_index: *idx as u32,
            })
            .collect();

        // Prologue args auto-assigned after main args
        let next_idx = args.iter().map(|a| a.buffer_index).max().unwrap_or(0) + 1;
        for (i, arg) in P::buffer_args().into_iter().enumerate() {
            args.push(BufferArg {
                buffer_index: next_idx + i as u32,
                ..arg
            });
        }
        args
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        let mut code = String::new();

        // 1. Prologue code (sets up threadgroup state)
        let prologue = P::emit();
        if !prologue.is_empty() {
            code.push_str(&prologue);
            if !prologue.ends_with('\n') {
                code.push('\n');
            }
        }

        // 2. Hook preamble (e.g. compute inv_rms for RMSNorm policy).
        let preamble = H::preamble_code();
        if !preamble.is_empty() {
            code.push_str(preamble);
            if !preamble.ends_with('\n') {
                code.push('\n');
            }
        }

        // NOTE: We standardize the local `data_arr` element type to `uchar*` so hooks/policies
        // can cast to whatever view they need (`half*`, custom packed formats, etc).
        code.push_str(&emit_ptr_array_cast(
            "const device uchar *",
            "const device uchar*",
            "data_arr",
            C::HEADS,
            C::data_ptrs(),
        ));
        if !C::scale_ptrs().is_empty() {
            code.push_str(&emit_ptr_array_cast(
                "const device uchar *",
                "const device uchar*",
                "scale_arr",
                C::HEADS,
                C::scale_ptrs(),
            ));
        }
        code.push_str(&emit_ptr_array("device half *", "res_arr", C::HEADS, C::result_ptrs()));
        code.push_str(&emit_uint_array("N_arr", C::HEADS, C::n_exprs()));
        code.push_str(&emit_ptr_array("const device half *", "bias_arr", C::HEADS, C::bias_ptrs()));
        code.push_str(&emit_uint_array("bias_flags", C::HEADS, C::has_bias_flags()));
        code.push('\n');

        code.push_str(H::policy_params_code());
        if !code.ends_with('\n') {
            code.push('\n');
        }
        code.push('\n');

        let fast = if C::FAST_PATH { "true" } else { "false" };
        let mut template_args = format!("{}, {}, {}, {}", H::policy_struct(), C::HEADS, C::COLS_PER_TG, fast);
        if let Some(ep) = E::template_arg() {
            template_args.push_str(", ");
            template_args.push_str(ep);
        }

        code.push_str(&format!("    run_simd_gemv_template<{}>(\n", template_args));
        code.push_str("        p, vector_x, res_arr, N_arr, params->K, bias_arr, bias_flags, 1.0f, 0.0f, nullptr, gid, lid,\n");
        code.push_str(&format!(
            "        GemvParams {{ params->K, {}, (params->K + params->weights_per_block - 1) / params->weights_per_block, params->weights_per_block, 1, 0, 0, 0, 0 }}\n",
            C::gemv_params_n0_expr()
        ));
        code.push_str("    );\n");

        ("output".to_string(), code)
    }

    fn struct_defs(&self) -> String {
        C::struct_defs()
    }
}

fn emit_ptr_array(prefix: &str, var: &str, heads: usize, elems: &[&'static str]) -> String {
    debug_assert_eq!(elems.len(), heads);
    let mut code = String::new();
    code.push_str(&format!("    {}{}[{}] = {{", prefix, var, heads));
    for (i, elem) in elems.iter().enumerate() {
        if i != 0 {
            code.push_str(", ");
        }
        code.push_str(elem);
    }
    code.push_str("};\n");
    code
}

fn emit_ptr_array_cast(prefix: &str, cast: &str, var: &str, heads: usize, elems: &[&'static str]) -> String {
    debug_assert_eq!(elems.len(), heads);
    let mut code = String::new();
    code.push_str(&format!("    {}{}[{}] = {{", prefix, var, heads));
    for (i, elem) in elems.iter().enumerate() {
        if i != 0 {
            code.push_str(", ");
        }
        code.push_str(&format!("({cast}){elem}"));
    }
    code.push_str("};\n");
    code
}

fn emit_uint_array(var: &str, heads: usize, elems: &[&'static str]) -> String {
    debug_assert_eq!(elems.len(), heads);
    let mut code = String::new();
    code.push_str(&format!("    const uint {}[{}] = {{", var, heads));
    for (i, elem) in elems.iter().enumerate() {
        if i != 0 {
            code.push_str(", ");
        }
        code.push_str(elem);
    }
    code.push_str("};\n");
    code
}
