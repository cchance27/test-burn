#![allow(clippy::all)]
use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Ident, Span};
use quote::quote;
use syn::{
    Attribute, Data, DeriveInput, Expr, ExprLit, ExprPath, Fields, Lit, Meta, Token, Type, parse_macro_input, punctuated::Punctuated, spanned::Spanned
};

mod conditional;
mod derive_compiled_step_impl;
mod derive_compound_kernel_impl;
mod derive_conditional_kernel_impl;
mod derive_gguf_block_quant_runtime_impl;
mod derive_kernel_args_impl;
mod derive_kernel_impl;
mod derive_metal_policy_impl;
mod derive_metal_struct_impl;
mod derive_stage_impl;

// --- Shared Helpers for Macros ---

fn validate_metal_template(template: &str, span: Span) -> syn::Result<()> {
    let mut brace_balance = 0;
    let mut paren_balance = 0;
    let mut bracket_balance = 0;
    let mut in_string = false;
    let mut escaped = false;

    for (i, c) in template.chars().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }

        match c {
            '{' => brace_balance += 1,
            '}' => brace_balance -= 1,
            '(' => paren_balance += 1,
            ')' => paren_balance -= 1,
            '[' => bracket_balance += 1,
            ']' => bracket_balance -= 1,
            _ => {}
        }
        if brace_balance < 0 {
            return Err(syn::Error::new(
                span,
                format!("Unbalanced closing brace '}}' in Metal template (char {i})"),
            ));
        }
        if paren_balance < 0 {
            return Err(syn::Error::new(
                span,
                format!("Unbalanced closing parenthesis ')' in Metal template (char {i})"),
            ));
        }
        if bracket_balance < 0 {
            return Err(syn::Error::new(
                span,
                format!("Unbalanced closing bracket ']' in Metal template (char {i})"),
            ));
        }
    }
    if brace_balance != 0 {
        return Err(syn::Error::new(span, "Unbalanced opening brace '{' in Metal template"));
    }
    if paren_balance != 0 {
        return Err(syn::Error::new(span, "Unbalanced opening parenthesis '(' in Metal template"));
    }
    if bracket_balance != 0 {
        return Err(syn::Error::new(span, "Unbalanced opening bracket '[' in Metal template"));
    }
    if in_string {
        return Err(syn::Error::new(span, "Unterminated string literal in Metal template"));
    }

    // Basic "must end with semicolon" check if it looks like code and doesn't end with a brace
    let trimmed = template.trim();
    if !trimmed.is_empty() && !trimmed.ends_with('}') && !trimmed.ends_with(';') && !trimmed.ends_with(')') {
        // High priority: warn about missing semicolon in templates
        // We allow some flexibility if it's just an expression, but usually they are statements
        // For now, let's keep it advisory or just check for very obvious things.
    }

    Ok(())
}

fn template_placeholder_names(template: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = template.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] == b'{' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'}' {
                j += 1;
            }
            if j < bytes.len() {
                let inner = &template[i + 1..j];
                if !inner.is_empty()
                    && inner.chars().all(|c| c == '_' || c.is_ascii_alphanumeric())
                    && inner.chars().next().is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
                {
                    out.push(inner.to_string());
                }
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }

    out
}

fn foundry_crate() -> proc_macro2::TokenStream {
    // 1. Check if we are compiling metallic-foundry itself
    if let Ok(pkg_name) = std::env::var("CARGO_PKG_NAME") {
        if pkg_name == "metallic-foundry" {
            return quote::quote! { crate };
        }
        // If compiling the facade 'metallic', foundry is likely at crate::foundry
        if pkg_name == "metallic" {
            return quote::quote! { crate::foundry };
        }
    }

    // 2. Try to find metallic-foundry directly
    if let Ok(found) = crate_name("metallic-foundry") {
        return match found {
            FoundCrate::Itself => quote::quote! { crate },
            FoundCrate::Name(n) => {
                let ident = Ident::new(&n, Span::call_site());
                quote::quote! { ::#ident }
            }
        };
    }

    // 3. Try metallic (facade) and access foundry module
    if let Ok(found) = crate_name("metallic") {
        return match found {
            FoundCrate::Itself => quote::quote! { crate::foundry },
            FoundCrate::Name(n) => {
                let ident = Ident::new(&n, Span::call_site());
                quote::quote! { ::#ident::foundry }
            }
        };
    }

    // Fallback - assume ::metallic_foundry exists or fail gracefully?
    // Let's try one more common alias
    if let Ok(found) = crate_name("metallic_foundry") {
        return match found {
            FoundCrate::Itself => quote::quote! { crate },
            FoundCrate::Name(n) => {
                let ident = Ident::new(&n, Span::call_site());
                return quote::quote! { ::#ident };
            }
        };
    }

    // Fallback
    let pkg = std::env::var("CARGO_PKG_NAME").unwrap_or_else(|_| "UNSET".to_string());
    // If we can't find it, we might be inside a crate that defines it but hasn't published it in a way we see.
    // Default to ::metallic_foundry as a safe bet for external consumers?
    // Or ::metallic::foundry?
    // Let's error out to be safe, or default to crate if we think we might be in it (covered by step 1).
    let msg =
        format!("Failed to resolve 'metallic-foundry' crate path. CARGO_PKG_NAME={pkg}. Please ensure metallic-foundry is a dependency.");
    quote::quote! { compile_error!(#msg); }
}

fn loader_crate() -> proc_macro2::TokenStream {
    if let Ok(pkg_name) = std::env::var("CARGO_PKG_NAME") {
        if pkg_name == "metallic-loader" {
            return quote::quote! { crate };
        }
    }

    if let Ok(found) = crate_name("metallic-loader") {
        return match found {
            FoundCrate::Itself => quote::quote! { crate },
            FoundCrate::Name(n) => {
                let ident = Ident::new(&n, Span::call_site());
                quote::quote! { ::#ident }
            }
        };
    }

    if let Ok(found) = crate_name("metallic_loader") {
        return match found {
            FoundCrate::Itself => quote::quote! { crate },
            FoundCrate::Name(n) => {
                let ident = Ident::new(&n, Span::call_site());
                quote::quote! { ::#ident }
            }
        };
    }

    quote::quote! { ::metallic_loader }
}

// --- syn::Type Utilities ---

fn get_path_ident(ty: &syn::Type) -> Option<Ident> {
    if let syn::Type::Path(ty_path) = ty {
        if let Some(segment) = ty_path.path.segments.last() {
            return Some(segment.ident.clone());
        }
    }
    None
}

fn is_type_match(ty: &syn::Type, name: &str) -> bool {
    get_path_ident(ty).is_some_and(|ident| ident == name)
}

fn extract_inner_generic(ty: &syn::Type) -> Option<syn::Type> {
    if let syn::Type::Path(ty_path) = ty {
        if let Some(segment) = ty_path.path.segments.last() {
            if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                    return Some(inner_ty.clone());
                }
            }
        }
    }
    None
}

fn is_tensor_arg(ty: &syn::Type) -> bool {
    // 1. Check if it's a direct TensorArg, Tensor, or similar
    if is_type_match(ty, "TensorArg") || is_type_match(ty, "Tensor") {
        return true;
    }
    // 2. Check if it's a reference &Tensor
    if let syn::Type::Reference(ty_ref) = ty {
        return is_tensor_arg(&ty_ref.elem);
    }
    // 3. Check if it's Option<TensorArg> - unwrap and recurse
    if let Some(inner) = extract_option_inner(ty) {
        return is_tensor_arg(&inner);
    }
    // 4. Optional: handle path aliasing if needed, but segments.last() covers most cases
    false
}

fn extract_option_inner(ty: &syn::Type) -> Option<syn::Type> {
    if let syn::Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
            if seg.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return Some(inner.clone());
                    }
                }
            }
        }
    }
    None
}

/// Extract the inner type T from `DynamicValue`<T> pattern.
/// Returns None if not a `DynamicValue`.
fn extract_dynamic_value_inner(ty: &syn::Type) -> Option<syn::Type> {
    if is_type_match(ty, "DynamicValue") {
        return extract_inner_generic(ty);
    }
    None
}

// Helper to map Rust types to Metal types for MetalStruct
fn rust_type_to_metal(ty: &syn::Type) -> String {
    // Check for DynamicValue<T> - extract T and map to Metal
    if let Some(inner) = extract_dynamic_value_inner(ty) {
        return rust_type_to_metal(&inner);
    }

    // Use robust type matching for f16 (handles half::f16, ::half::f16, etc.)
    if is_type_match(ty, "f16") {
        return "half".to_string();
    }

    let ident = get_path_ident(ty).map(|i| i.to_string()).unwrap_or_default();
    match ident.as_str() {
        "u8" => "uchar".to_string(),
        "i8" => "char".to_string(),
        "u16" => "ushort".to_string(),
        "i16" => "short".to_string(),
        "u32" => "uint".to_string(),
        "i32" => "int".to_string(),
        "u64" => "ulong".to_string(),
        "i64" => "long".to_string(),
        "f32" => "float".to_string(),
        "f16" => "half".to_string(), // Simple case caught above, but keep for completeness
        _ => {
            // Fallback: assume it's a struct name
            if !ident.is_empty() && ident.chars().next().unwrap().is_uppercase() {
                return ident;
            }
            let type_str = quote::quote!(#ty).to_string();
            unreachable!("Unsupported type: '{}'", type_str)
        }
    }
}

// Collect signature info for METAL_ARGS generation
struct ArgInfo {
    name: String,
    name_ident: Option<Ident>,
    rust_type: String,
    rust_type_actual: Type,
    metal_type: Option<String>,
    buffer_index: u64,
    is_output: bool,
    is_buffer: bool,
    is_option: bool,
    stage_skip: bool,
    scale_for: Option<String>,
    attrs: Vec<Attribute>,
    serde_attrs: Vec<proc_macro2::TokenStream>, // New field for generated serde attributes
    is_meta: bool,
}

// Helper to infer Metal type from Rust type string
fn infer_metal_type(ty: &syn::Type, is_buffer: bool, is_output: bool) -> String {
    // TensorArg, &Tensor, Tensor<...> → device pointer
    if is_tensor_arg(ty) {
        return if is_output {
            "device OutputStorageT*".to_string()
        } else {
            "const device InputStorageT*".to_string()
        };
    }

    let ident = get_path_ident(ty).map(|i| i.to_string()).unwrap_or_default();
    // Primitive types → constant reference
    match ident.as_str() {
        "u32" => return "constant uint&".to_string(),
        "i32" => return "constant int&".to_string(),
        "f32" => return "constant float&".to_string(),
        "u64" => return "constant ulong&".to_string(),
        "i64" => return "constant long&".to_string(),
        _ => {}
    }

    // Structs with PascalCase names (likely have METAL_STRUCT_DEF) → const constant Struct*
    if !ident.is_empty() && ident.chars().next().unwrap().is_uppercase() {
        // It's a struct type - assume it has METAL_STRUCT_DEF
        return format!("const constant {ident}*");
    }

    // Fallback for buffer types
    if is_buffer {
        if is_output {
            "device OutputStorageT*".to_string()
        } else {
            "const device InputStorageT*".to_string()
        }
    } else {
        "constant uint&".to_string()
    }
}

// Helper to collect ArgInfo from fields
fn collect_arg_infos(fields: &Fields) -> (Vec<ArgInfo>, Vec<proc_macro2::TokenStream>) {
    let mut arg_infos = Vec::new();
    let mut bindings = Vec::new();
    let mut auto_buffer_idx: u64 = 0; // Auto-increment counter for buffer indices

    let root = foundry_crate();

    if let Fields::Named(fields) = fields {
        for f in &fields.named {
            let name = &f.ident;
            let ty = &f.ty;
            let mut buffer_index: Option<u64> = None;
            let mut is_output = false;
            let mut explicit_bytes = false;
            let mut explicit_skip = false;
            let mut stage_skip = false;
            let mut metal_type: Option<String> = None;
            let mut serde_attrs = Vec::new();

            // Check the type to determine if it's a buffer type
            let is_option = extract_option_inner(ty).is_some();
            let inner_ty = extract_option_inner(ty).unwrap_or_else(|| ty.clone());
            let is_buffer_type = is_tensor_arg(&inner_ty);

            let mut scale_for: Option<String> = None;

            for attr in &f.attrs {
                if attr.path().is_ident("arg") {
                    if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                        for meta in &nested {
                            match meta {
                                Meta::NameValue(nv) => {
                                    if nv.path.is_ident("buffer") {
                                        if let Expr::Lit(expr_lit) = &nv.value {
                                            if let Lit::Int(lit) = &expr_lit.lit {
                                                buffer_index = Some(lit.base10_parse::<u64>().unwrap());
                                            }
                                        }
                                        // Still allow explicit kind="bytes" for override
                                        if let Expr::Lit(expr_lit) = &nv.value {
                                            if let Lit::Str(lit) = &expr_lit.lit {
                                                match lit.value().as_str() {
                                                    "bytes" => explicit_bytes = true,
                                                    "skip" => explicit_skip = true,
                                                    _ => {}
                                                }
                                            }
                                        }
                                    } else if nv.path.is_ident("metal_type") {
                                        if let Expr::Lit(expr_lit) = &nv.value {
                                            if let Lit::Str(lit) = &expr_lit.lit {
                                                metal_type = Some(lit.value());
                                            }
                                        }
                                    } else if nv.path.is_ident("scale_for") {
                                        if let Expr::Lit(expr_lit) = &nv.value {
                                            if let Lit::Str(lit) = &expr_lit.lit {
                                                scale_for = Some(lit.value());
                                            }
                                        }
                                    } else if nv.path.is_ident("serde_default") {
                                        if let Expr::Lit(expr_lit) = &nv.value {
                                            if let Lit::Str(lit) = &expr_lit.lit {
                                                serde_attrs.push(quote::quote! { #[serde(default = #lit)] });
                                            } else if let Lit::Bool(lit) = &expr_lit.lit {
                                                if lit.value {
                                                    serde_attrs.push(quote::quote! { #[serde(default)] });
                                                }
                                            }
                                        }
                                    } else if nv.path.is_ident("serde_with") {
                                        if let Expr::Lit(expr_lit) = &nv.value {
                                            if let Lit::Str(lit) = &expr_lit.lit {
                                                serde_attrs.push(quote::quote! { #[serde(with = #lit)] });
                                            }
                                        }
                                    }
                                }
                                Meta::Path(path) => {
                                    if path.is_ident("output") {
                                        is_output = true;
                                    } else if path.is_ident("stage_skip") {
                                        stage_skip = true;
                                    } else if path.is_ident("serde_default") {
                                        serde_attrs.push(quote::quote! { #[serde(default)] });
                                    } else if path.is_ident("serde_skip") {
                                        serde_attrs.push(quote::quote! { #[serde(skip)] });
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            // All fields are args by default. Use #[arg(skip)] to exclude.
            if !explicit_skip {
                let idx = buffer_index.unwrap_or_else(|| {
                    let idx = auto_buffer_idx;
                    auto_buffer_idx += 1;
                    idx
                });
                // If explicit index was provided, update auto counter to be past it
                if buffer_index.is_some() {
                    auto_buffer_idx = auto_buffer_idx.max(idx + 1);
                }

                // Auto-detect: if it's not a buffer type, treat as bytes
                let is_bytes = explicit_bytes || !is_buffer_type;

                let mut is_meta = false; // Default false, parsed from attrs if needed

                // Collect attributes to propagate (non-arg)
                let attrs: Vec<Attribute> = f.attrs.iter().filter(|a| !a.path().is_ident("arg")).cloned().collect();

                // Check for is_meta in arg attributes
                for attr in &f.attrs {
                    if attr.path().is_ident("arg") {
                        if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                            for meta in &nested {
                                if let Meta::Path(path) = meta {
                                    if path.is_ident("meta") {
                                        is_meta = true;
                                    }
                                }
                            }
                        }
                    }
                }

                // Collect arg info for signature generation
                arg_infos.push(ArgInfo {
                    name: name.as_ref().map(std::string::ToString::to_string).unwrap_or_default(),
                    name_ident: name.clone(),
                    buffer_index: idx,
                    metal_type: metal_type.clone(),
                    rust_type: quote::quote!(#ty).to_string(),
                    rust_type_actual: ty.clone(),
                    is_output,
                    is_buffer: is_buffer_type && !explicit_bytes && !is_meta, // Meta fields are never buffers
                    is_option,
                    stage_skip,
                    scale_for,
                    attrs,
                    serde_attrs,
                    is_meta,
                });

                // Generate binding code
                if !is_meta {
                    if is_bytes {
                        bindings.push(quote! {
                            encoder.set_bytes(#idx as u32, &self.#name);
                        });
                    } else {
                        let binding = if is_option {
                            quote! {
                                encoder.set_buffer_opt(
                                    #idx as u32,
                                    self.#name.as_ref().map(|v| #root::types::KernelArg::buffer(v)),
                                    self.#name.as_ref().map(|v| #root::types::KernelArg::offset(v)).unwrap_or(0)
                                );
                            }
                        } else {
                            quote! {
                                encoder.set_buffer(
                                    #idx as u32,
                                    #root::types::KernelArg::buffer(&self.#name),
                                    #root::types::KernelArg::offset(&self.#name)
                                );
                            }
                        };
                        bindings.push(binding);
                    }
                }
            }
        }
    }
    (arg_infos, bindings)
}

/// Derive macro to generate a Metal struct definition from a Rust struct.
///
/// # Example
/// ```ignore
/// #[derive(MetalStruct)]
/// #[repr(C)]
/// #[metal(name = "GemvParams")]  // Optional: override struct name
/// pub struct GemvParams {
///     #[metal(name = "K")]  // Optional: override field name
///     pub k: u32,
///     pub n: u32,
/// }
/// ```
///
/// Generates a const `METAL_STRUCT_DEF` containing the Metal struct definition:
/// ```metal
/// struct GemvParams {
///     uint K;
///     uint n;
/// };
/// ```
#[proc_macro_derive(MetalStruct, attributes(metal))]
pub fn derive_metal_struct(input: TokenStream) -> TokenStream {
    derive_metal_struct_impl::derive_metal_struct(input)
}

/// Derive macro to generate `MetalPolicy` trait implementation from struct annotations.
///
/// # Example
/// ```ignore
/// #[derive(MetalPolicy)]
/// #[policy(header = "policies/policy_q8.metal", struct_name = "PolicyQ8")]
/// pub struct PolicyQ8 {
///     #[param(from = "matrix")]
///     matrix: DevicePtr<u8>,
///     
///     #[param(from = "scale_bytes")]
///     scales: DevicePtr<u8>,
///     
///     #[param(from = "params.weights_per_block")]
///     weights_per_block: u32,
/// }
/// ```
///
/// Generates:
/// - `header()` → "`policies/policy_q8.metal`"
/// - `struct_name()` → "`PolicyQ8`"
/// - `init_params_code()` → "pp.matrix = matrix; pp.scales = `scale_bytes`; `pp.weights_per_block` = params->weights_per_block;"
#[proc_macro_derive(MetalPolicy, attributes(policy, param))]
pub fn derive_metal_policy(input: TokenStream) -> TokenStream {
    derive_metal_policy_impl::derive_metal_policy(input)
}

/// Derive boilerplate for GGUF block-quant policy runtime wiring.
///
/// This derive generates:
/// - `impl BlockQuantCodec`
/// - `impl LoaderStage`
/// - `impl MetalPolicyRuntime`
///
/// All args are typed expressions/paths (no stringly-typed dtype names).
#[proc_macro_derive(GgufBlockQuantRuntime, attributes(gguf_runtime))]
pub fn derive_gguf_block_quant_runtime(input: TokenStream) -> TokenStream {
    derive_gguf_block_quant_runtime_impl::derive_gguf_block_quant_runtime(input)
}

#[proc_macro_derive(KernelArgs, attributes(arg))]
pub fn derive_kernel_args(input: TokenStream) -> TokenStream {
    derive_kernel_args_impl::derive_kernel_args(input)
}

#[proc_macro_derive(Kernel, attributes(kernel))]
pub fn derive_kernel(input: TokenStream) -> TokenStream {
    derive_kernel_impl::derive_kernel(input)
}

/// Derive macro to generate a compound kernel from stage definitions.
///
/// # Example
/// ```ignore
/// #[derive(CompoundKernel)]
/// #[compound(name = "gemv_q8")]
/// pub struct GemvQ8Compound {
///     // Stages (order matters: prologues first, then main, then epilogues)
///     #[prologue]
///     pub policy: PolicyStage<PolicyQ8>,
///     #[main]
///     pub gemv: GemvCoreStage,
///     #[epilogue]
///     pub epilogue: EpilogueStage<EpilogueNone>,
/// }
/// ```
///
/// Generates:
/// - `impl Kernel` for the struct with source generation from stages
/// - Runtime validation that stages chain correctly

#[proc_macro_derive(CompoundKernel, attributes(compound, prologue, main, epilogue))]
pub fn derive_compound_kernel(input: TokenStream) -> TokenStream {
    derive_compound_kernel_impl::derive_compound_kernel(input)
}

/// Derive macro for conditional kernel dispatch with compile-time coverage analysis.
///
/// # Example
/// ```ignore
/// #[derive(ConditionalKernel, Clone)]
/// #[conditional(selector = "batch: u32")]
/// pub enum MatmulDispatch {
///     #[when(batch == 1)]
///     Gemv(GemvKernel),
///
///     #[when(batch > 1)]
///     Gemm(GemmKernel),
/// }
/// ```
///
/// Generates:
/// - `select(batch: u32) -> Self` method for runtime dispatch
/// - `impl Kernel` that delegates to the selected variant
/// - Compile-time errors for coverage gaps or overlapping conditions
#[proc_macro_derive(ConditionalKernel, attributes(conditional, when))]
pub fn derive_conditional_kernel(input: TokenStream) -> TokenStream {
    derive_conditional_kernel_impl::derive_conditional_kernel(input)
}

/// Derive macro for auto-generating `CompiledStep` boilerplate.
///
/// Use this on Step structs that manually implement `Step` trait.
/// It generates:
/// 1. A `Compiled{Step}` struct with Ref fields converted to `usize` indices
/// 2. `Step::compile()` implementation that maps Ref names to symbol indices
/// 3. `CompiledStep::execute()` implementation that fetches tensors from `FastBindings`
///
/// # Required Attributes
/// - `#[compiled_step(kernel = "KernelType")]` - The kernel type to construct and run
/// - `#[compiled_step(name = "StepName")]` - The step name for typetag
///
/// # Field Attributes  
/// - `#[ref_field]` - Mark Ref fields that need index compilation
///
/// # Example
/// ```ignore
/// #[derive(CompiledStep)]
/// #[compiled_step(kernel = "Softmax", name = "Softmax")]
/// pub struct SoftmaxStep {
///     #[ref_field]
///     pub input: Ref,
///     #[ref_field]
///     pub output: Ref,
///     pub rows_total: u32,
/// }
/// ```
#[proc_macro_derive(CompiledStep, attributes(compiled_step, ref_field))]
pub fn derive_compiled_step(input: TokenStream) -> TokenStream {
    derive_compiled_step_impl::derive_compiled_step(input)
}

/// Derive macro for Stage trait.
///
/// Generates `Stage` implementation from struct annotations.
///
/// # Example
/// ```ignore
/// #[derive(Stage, Clone)]
/// #[stage(
///     include = "v2/softmax/softmax.metal",
///     emit = r#"float local_max = find_row_max(matrix, ...);"#,
///     out_var = "local_max",
///     struct_defs = "SoftmaxParams"  // Optional: MetalStruct type to include
/// )]
/// pub struct SoftmaxMaxStage {
///     #[arg(buffer = 0)]
///     pub matrix: TensorArg,
///     #[arg(buffer = 3)]
///     pub params: SoftmaxParams,
/// }
/// ```
#[proc_macro_derive(Stage, attributes(stage, arg))]
pub fn derive_stage(input: TokenStream) -> TokenStream {
    derive_stage_impl::derive_stage(input)
}
