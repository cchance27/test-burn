use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Expr, Fields, Lit, Meta, Token, parse_macro_input, punctuated::Punctuated};

// --- Shared Helpers for Macros ---

// Helper to map Rust types to Metal types for MetalStruct
fn rust_type_to_metal(ty_str: &str) -> &'static str {
    match ty_str.trim() {
        "u8" => "uchar",
        "i8" => "char",
        "u16" => "ushort",
        "i16" => "short",
        "u32" => "uint",
        "i32" => "int",
        "u64" => "ulong",
        "i64" => "long",
        "f32" => "float",
        // Note: half::f16 shows up as "f16" in quote! output
        "f16" | "half :: f16" => "half",
        _ => unreachable!("Unsupported type: {}", ty_str),
    }
}

// Collect signature info for METAL_ARGS generation
struct ArgInfo {
    name: String,
    name_ident: Option<syn::Ident>,
    buffer_index: u64,
    metal_type: Option<String>,
    rust_type: String, // For auto-detection of Metal type
    rust_type_actual: syn::Type,
    is_output: bool,
    is_buffer: bool,
    stage_skip: bool,
}

// Helper to infer Metal type from Rust type string
fn infer_metal_type(type_str: &str, is_buffer: bool, is_output: bool) -> String {
    let trimmed = type_str.replace(' ', "");

    // TensorArg, &Tensor, Tensor<...> → device pointer
    if trimmed.contains("TensorArg") || trimmed.contains("Tensor<") || trimmed.starts_with("&Tensor") {
        return if is_output {
            "device half*".to_string()
        } else {
            "const device half*".to_string()
        };
    }

    // Primitive types → constant reference
    match trimmed.as_str() {
        "u32" => return "constant uint&".to_string(),
        "i32" => return "constant int&".to_string(),
        "f32" => return "constant float&".to_string(),
        "u64" => return "constant ulong&".to_string(),
        "i64" => return "constant long&".to_string(),
        _ => {}
    }

    // Structs with PascalCase names (likely have METAL_STRUCT_DEF) → const constant Struct*
    // This detects types like GemvParams, QkvFusedParams, etc.
    if trimmed.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        // It's a struct type - assume it has METAL_STRUCT_DEF
        return format!("const constant {}*", trimmed);
    }

    // Fallback for buffer types
    if is_buffer {
        if is_output {
            "device half*".to_string()
        } else {
            "const device half*".to_string()
        }
    } else {
        "constant uint&".to_string()
    }
}

// Helper to collect ArgInfo from fields
fn collect_arg_infos(fields: &Fields) -> (Vec<ArgInfo>, Vec<proc_macro2::TokenStream>) {
    let mut arg_infos = Vec::new();
    let mut bindings = Vec::new();

    let root = quote::quote! { ::metallic };

    if let Fields::Named(fields) = fields {
        for f in &fields.named {
            let name = &f.ident;
            let ty = &f.ty;
            let mut buffer_index = None;
            let mut is_output = false;
            let mut explicit_bytes = false;
            let mut explicit_skip = false;
            let mut stage_skip = false;
            let mut metal_type: Option<String> = None;

            // Check the type to determine if it's a buffer type
            let type_str = quote::quote!(#ty).to_string();
            let is_buffer_type = type_str.contains("TensorArg")
                || type_str.contains("Tensor <")
                || type_str.starts_with("& Tensor")
                || type_str.starts_with("&Tensor");

            for attr in &f.attrs {
                if attr.path().is_ident("arg") {
                    if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                        for meta in nested.iter() {
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
                                    }
                                }
                                Meta::Path(path) => {
                                    if path.is_ident("output") {
                                        is_output = true;
                                    } else if path.is_ident("stage_skip") {
                                        stage_skip = true;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            // Auto-detect: if it's not a buffer type, treat as bytes
            let is_bytes = explicit_bytes || !is_buffer_type;

            // Collect arg info for signature generation
            if let Some(idx) = buffer_index {
                if !explicit_skip {
                    arg_infos.push(ArgInfo {
                        name: name.as_ref().map(|i| i.to_string()).unwrap_or_default(),
                        name_ident: name.clone(),
                        buffer_index: buffer_index.unwrap_or(0),
                        metal_type: metal_type.clone(),
                        rust_type: type_str.clone(),
                        rust_type_actual: ty.clone(),
                        is_output,
                        is_buffer: is_buffer_type && !explicit_bytes,
                        stage_skip,
                    });
                }
            }

            if !explicit_skip {
                if let Some(idx) = buffer_index {
                    if is_bytes {
                        bindings.push(quote::quote! {
                            let ptr = &self.#name as *const _ as *const core::ffi::c_void;
                            let len = core::mem::size_of_val(&self.#name);
                            unsafe {
                                encoder.setBytes_length_atIndex(core::ptr::NonNull::new(ptr as *mut _).unwrap(), len, #idx as usize);
                            }
                        });
                    } else {
                        // Buffer binding - use KernelArg trait for buf+offset extraction
                        // Auto-flush inputs (non-outputs) before binding
                        let flush_code = if is_output {
                            quote::quote! {}
                        } else {
                            quote::quote! {
                                #root::types::KernelArg::flush(&self.#name);
                            }
                        };
                        bindings.push(quote::quote! {
                            #flush_code
                            unsafe {
                                encoder.setBuffer_offset_atIndex(
                                    Some(&*#root::types::KernelArg::buffer(&self.#name)),
                                    #root::types::KernelArg::offset(&self.#name),
                                    #idx as usize
                                );
                            }
                        });
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
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Get optional struct-level #[metal(name = "...")] override
    let mut metal_struct_name = name.to_string();
    for attr in &input.attrs {
        if attr.path().is_ident("metal") {
            if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                for meta in nested {
                    if let Meta::NameValue(nv) = meta {
                        if nv.path.is_ident("name") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Str(lit) = expr_lit.lit {
                                    metal_struct_name = lit.value();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut metal_fields = Vec::new();

    if let Data::Struct(data) = &input.data {
        if let Fields::Named(fields) = &data.fields {
            for field in &fields.named {
                let field_name = field.ident.as_ref().unwrap();
                let field_type = &field.ty;
                let type_str = quote!(#field_type).to_string();

                // Check for field-level #[metal(name = "...")] or #[metal(skip)]
                let mut metal_field_name = field_name.to_string();
                let mut skip = false;

                for attr in &field.attrs {
                    if attr.path().is_ident("metal") {
                        if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                            for meta in nested {
                                match &meta {
                                    Meta::NameValue(nv) if nv.path.is_ident("name") => {
                                        if let Expr::Lit(expr_lit) = &nv.value {
                                            if let Lit::Str(lit) = &expr_lit.lit {
                                                metal_field_name = lit.value();
                                            }
                                        }
                                    }
                                    Meta::Path(p) if p.is_ident("skip") => {
                                        skip = true;
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                if !skip {
                    let metal_type = rust_type_to_metal(&type_str);
                    metal_fields.push(format!("    {} {};", metal_type, metal_field_name));
                }
            }
        }
    }

    let metal_def = format!("struct {} {{\n{}\n}};", metal_struct_name, metal_fields.join("\n"));

    let expanded = quote! {
        impl #name {
            /// The Metal struct definition for this type.
            pub const METAL_STRUCT_DEF: &'static str = #metal_def;
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro to generate MetalPolicy trait implementation from struct annotations.
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
/// - `header()` → "policies/policy_q8.metal"
/// - `struct_name()` → "PolicyQ8"
/// - `init_params_code()` → "pp.matrix = matrix; pp.scales = scale_bytes; pp.weights_per_block = params->weights_per_block;"
#[proc_macro_derive(MetalPolicy, attributes(policy, param))]
pub fn derive_metal_policy(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Determine crate root path
    let crate_name = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let root = if crate_name == "metallic" {
        quote::quote! { crate }
    } else {
        quote::quote! { ::metallic }
    };

    // Parse #[policy(header = "...", struct_name = "...")]
    let mut header = String::new();
    let mut struct_name = name.to_string();

    for attr in &input.attrs {
        if attr.path().is_ident("policy") {
            if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                for meta in nested {
                    if let Meta::NameValue(nv) = meta {
                        if nv.path.is_ident("header") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Str(lit) = expr_lit.lit {
                                    header = lit.value();
                                }
                            }
                        } else if nv.path.is_ident("struct_name") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Str(lit) = expr_lit.lit {
                                    struct_name = lit.value();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Parse fields for #[param(from = "...")]
    let mut init_statements = Vec::new();

    if let Data::Struct(data) = &input.data {
        if let Fields::Named(fields) = &data.fields {
            for field in &fields.named {
                let field_name = field.ident.as_ref().unwrap();

                for attr in &field.attrs {
                    if attr.path().is_ident("param") {
                        if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                            for meta in nested {
                                if let Meta::NameValue(nv) = meta {
                                    if nv.path.is_ident("from") {
                                        if let Expr::Lit(expr_lit) = nv.value {
                                            if let Lit::Str(lit) = expr_lit.lit {
                                                let source = lit.value();
                                                // Convert "params.field" to "params->field"
                                                let source_code = if source.contains('.') { source.replace('.', "->") } else { source };
                                                init_statements.push(format!("pp.{} = {};", field_name, source_code));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let init_code = init_statements.join(" ");

    let expanded = quote! {
        impl #root::fusion::MetalPolicy for #name {
            fn header(&self) -> &'static str {
                #header
            }

            fn struct_name(&self) -> &'static str {
                #struct_name
            }
        }

        impl #name {
            /// Generated Metal code for initializing policy params.
            pub const INIT_PARAMS_CODE: &'static str = #init_code;
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(KernelArgs, attributes(arg))]
pub fn derive_kernel_args(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    // Determine crate root path
    // Always refer to metallic as ::metallic (requires extern crate self as metallic in lib.rs)
    let root = quote::quote! { ::metallic };

    let (mut arg_infos, bindings) = match input.data {
        Data::Struct(data) => collect_arg_infos(&data.fields),
        _ => panic!("KernelArgs only supports structs"),
    };

    // Sort args by buffer index for signature generation
    arg_infos.sort_by_key(|a| a.buffer_index);

    let binding_code = quote! { #(#bindings)* };

    // Generate METAL_ARGS elements as TokenStreams (all args)
    let metal_args_elements: Vec<_> = arg_infos
        .iter()
        .map(|info| {
            let arg_name = &info.name;
            let idx = info.buffer_index;
            // Use explicit metal_type if specified, otherwise infer from Rust type
            let mtype = info
                .metal_type
                .clone()
                .unwrap_or_else(|| infer_metal_type(&info.rust_type, info.is_buffer, info.is_output));
            quote! { (#arg_name, #idx, #mtype) }
        })
        .collect();

    // Generate STAGE_METAL_ARGS elements (excluding stage_skip buffers)
    let stage_metal_args_elements: Vec<_> = arg_infos
        .iter()
        .filter(|info| !info.stage_skip)
        .map(|info| {
            let arg_name = &info.name;
            let idx = info.buffer_index;
            let mtype = info
                .metal_type
                .clone()
                .unwrap_or_else(|| infer_metal_type(&info.rust_type, info.is_buffer, info.is_output));
            quote! { (#arg_name, #idx, #mtype) }
        })
        .collect();

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            pub fn bind_args(&self, encoder: &#root::types::ComputeCommandEncoder) {
                use objc2_metal::MTLComputeCommandEncoder as _;
                #binding_code
            }
        }

        impl #impl_generics #root::fusion::HasMetalArgs for #name #ty_generics #where_clause {
            /// Metal argument signature parts: (name, buffer_index, metal_type)
            const METAL_ARGS: &'static [(&'static str, u64, &'static str)] = &[
                #(#metal_args_elements),*
            ];
            /// Stage-compatible args (excludes stage_skip buffers like PolicyStage provides)
            const STAGE_METAL_ARGS: &'static [(&'static str, u64, &'static str)] = &[
                #(#stage_metal_args_elements),*
            ];
        }

        impl #impl_generics #root::compound::BindArgs for #name #ty_generics #where_clause {
            fn bind_args(&self, encoder: &#root::types::ComputeCommandEncoder) {
                // Delegate to the inherent method
                self.bind_args(encoder);
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(Kernel, attributes(kernel))]
pub fn derive_kernel(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();

    // Determine crate root path
    let crate_name = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let root = if crate_name == "metallic" {
        quote::quote! { crate }
    } else {
        quote::quote! { ::metallic }
    };

    let mut source = None;
    let mut function = None;
    let mut args_type = None;
    let mut includes = Vec::new();
    let mut stage_function = None;
    let mut threadgroup_decl = None;
    let mut epilogue_emit: Option<String> = None;
    let mut epilogue_out_var: Option<String> = None;

    for attr in &input.attrs {
        if attr.path().is_ident("kernel") {
            if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                for meta in nested {
                    match meta {
                        Meta::NameValue(nv) => {
                            if nv.path.is_ident("source") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        source = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("function") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        function = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("args") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        // Parse the string as a Type
                                        args_type = Some(lit.parse::<syn::Type>().unwrap());
                                    }
                                }
                            } else if nv.path.is_ident("include") {
                                if let Expr::Array(arr) = nv.value {
                                    for elem in arr.elems {
                                        if let Expr::Lit(expr_lit) = elem {
                                            if let Lit::Str(lit) = expr_lit.lit {
                                                includes.push(lit.value());
                                            }
                                        }
                                    }
                                }
                            } else if nv.path.is_ident("stage_function") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        stage_function = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("threadgroup") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        threadgroup_decl = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("epilogue_emit") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        epilogue_emit = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("epilogue_out_var") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        epilogue_out_var = Some(lit.value());
                                    }
                                }
                            }
                        }
                        Meta::List(_) => {}
                        _ => {}
                    }
                }
            }
        }
    }

    let source = source.expect("Kernel must have `source` attribute");
    let function = function.expect("Kernel must have `function` attribute");
    let args_type = args_type.unwrap_or_else(|| syn::parse_str("()").unwrap());

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let id_name = quote::format_ident!("{}KernelId", name);
    let stage_name = quote::format_ident!("{}Stage", name);

    // Generate as_stage() implementation
    let as_stage_impl = if stage_function.is_some() || epilogue_emit.is_some() {
        let (arg_infos, _) = match &input.data {
            Data::Struct(data) => collect_arg_infos(&data.fields),
            _ => (Vec::new(), Vec::new()),
        };
        let field_names: Vec<_> = arg_infos
            .iter()
            .filter(|info| !info.stage_skip)
            .filter_map(|info| info.name_ident.as_ref())
            .collect();

        quote! {
            fn as_stage(&self) -> Box<dyn #root::compound::Stage> {
                Box::new(#stage_name {
                    #( #field_names: self.#field_names.clone() ),*
                })
            }
        }
    } else {
        quote! {
            fn as_stage(&self) -> Box<dyn #root::compound::Stage> {
                panic!("Kernel {} does not support staging (no stage_function or epilogue_emit defined)", stringify!(#name))
            }
        }
    };

    // Generate Stage struct and impl if stage_function or epilogue_emit is specified
    let stage_impl = if stage_function.is_some() || epilogue_emit.is_some() {
        let tg_decl = threadgroup_decl.clone().unwrap_or_default();
        let source_path = source.clone();

        let (arg_infos, _) = match &input.data {
            Data::Struct(data) => collect_arg_infos(&data.fields),
            _ => (Vec::new(), Vec::new()),
        };

        // Filter fields to those that are NOT stage_skip
        let stage_fields: Vec<_> = arg_infos
            .iter()
            .filter(|info| !info.stage_skip)
            .map(|info| {
                let fname = info.name_ident.as_ref().unwrap();
                let ftype = &info.rust_type_actual;
                quote! { pub #fname: #ftype }
            })
            .collect();

        let epilogue_impl = if epilogue_emit.is_some() {
            quote! {
                impl #root::fusion::Epilogue for #stage_name {
                    fn header(&self) -> &'static str { #source_path }
                    fn struct_name(&self) -> &'static str { stringify!(#name) }
                }
            }
        } else {
            quote! {}
        };

        let emit_impl = if let Some(ref emit_tmpl) = epilogue_emit {
            let out_var_expr = if let Some(ref v) = epilogue_out_var {
                quote! { #v.to_string() }
            } else {
                quote! { format!("{}_epilogue", input_var) }
            };
            quote! {
                fn emit(&self, input_var: &str) -> (String, String) {
                    let out_var = #out_var_expr;
                    let mut code = #emit_tmpl.to_string();
                    code = code.replace("{input_var}", input_var);
                    code = code.replace("{out_var}", &out_var);
                    (out_var, code)
                }
            }
        } else if let Some(ref stage_fn) = stage_function {
            quote! {
                fn emit(&self, _input_var: &str) -> (String, String) {
                    let tg = #tg_decl;
                    let fn_name = #stage_fn;
                    let args: Vec<&str> = <#name as #root::fusion::HasMetalArgs>::STAGE_METAL_ARGS
                        .iter()
                        .map(|(name, _, _)| *name)
                        .collect();
                    let args_str = args.join(", ");

                    let tg_vars: Vec<&str> = if tg.is_empty() {
                        vec![]
                    } else {
                        tg.split(';')
                            .filter_map(|decl| {
                                let decl = decl.trim();
                                if decl.is_empty() { return None; }
                                decl.split_whitespace().last()
                                    .map(|s| s.split('[').next().unwrap_or(s))
                            })
                            .collect()
                    };
                    let tg_args_vec: Vec<String> = tg_vars.iter().map(|v| {
                        // Check if the variable was declared as an array in the original tg string
                        // We check for name followed by '[' or if the string ends with name+index info
                        let is_array = tg.contains(&format!("{}[", v)) || tg.ends_with(v); // simplistic check
                        if is_array && tg.contains(&format!("{}[", v)) {
                            v.to_string()
                        } else {
                            format!("&{}", v)
                        }
                    }).collect();
                    let tg_args = if tg_args_vec.is_empty() {
                        String::new()
                    } else {
                        format!(", {}", tg_args_vec.join(", "))
                    };

                    let code = if tg.is_empty() {
                        format!("    {}<Policy>(matrix, {}, scale_bytes, gid, lid);", fn_name, args_str)
                    } else {
                        let tg_decls: Vec<String> = tg.split(';')
                            .filter(|s| !s.trim().is_empty())
                            .map(|decl| format!("threadgroup {}", decl.trim()))
                            .collect();
                        let tg_code = tg_decls.iter().map(|d| format!("    {};", d)).collect::<Vec<_>>().join("\n");
                        format!("{}\n    {}<Policy>(matrix, {}, scale_bytes, gid, lid{});", tg_code, fn_name, args_str, tg_args)
                    };
                    ("void".to_string(), code)
                }
            }
        } else {
            quote! {
                fn emit(&self, _input_var: &str) -> (String, String) {
                    ("void".to_string(), String::new())
                }
            }
        };

        quote! {
            /// Auto-generated Stage for compound kernel fusion.
            #[derive(Clone, Default)]
            pub struct #stage_name {
                #( #stage_fields ),*
            }

            impl #root::compound::Stage for #stage_name {
                fn includes(&self) -> Vec<&'static str> {
                    vec![#source_path]
                }

                fn buffer_args(&self) -> Vec<#root::compound::BufferArg> {
                    <#name as #root::fusion::HasMetalArgs>::STAGE_METAL_ARGS
                        .iter()
                        .map(|(name, idx, metal_type)| #root::compound::BufferArg {
                            name: *name,
                            metal_type: *metal_type,
                            buffer_index: *idx as u32,
                        })
                        .collect()
                }

                fn struct_defs(&self) -> String {
                    <#args_type>::METAL_STRUCT_DEF.to_string()
                }

                #emit_impl
            }

            #epilogue_impl
        }
    } else {
        quote! {}
    };

    let expanded = quote! {
        #[doc(hidden)]
        pub struct #id_name;

        impl #impl_generics #root::foundry::Kernel for #name #ty_generics #where_clause {
            type Args = #args_type;
            type Id = #id_name;

            fn function_name(&self) -> &'static str {
                #function
            }

            fn source(&self) -> #root::foundry::KernelSource {
                #root::foundry::KernelSource::File(#source)
            }

            fn includes(&self) -> #root::foundry::Includes {
                #root::foundry::Includes(vec![#(#includes),*])
            }

            fn bind(&self, encoder: &#root::types::ComputeCommandEncoder) {
                // Call the bind_args method generated by KernelArgs
                self.bind_args(encoder);
            }

            fn dispatch_config(&self) -> #root::types::DispatchConfig {
                Self::dispatch_config(self)
            }

            #[cfg(feature = "built_kernels")]
            fn metallib_bytes(&self) -> Option<&'static [u8]> {
                Some(include_bytes!(concat!(env!("OUT_DIR"), "/", stringify!(#name), ".metallib")))
            }

            fn struct_defs(&self) -> String {
                // Return METAL_STRUCT_DEF from the Args type if it has one
                <#args_type>::METAL_STRUCT_DEF.to_string()
            }

            #as_stage_impl
        }

        #stage_impl
    };

    TokenStream::from(expanded)
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
/// Derive macro for Epilogue stages.
///
/// Implements `Stage` and `Epilogue` traits.
/// Supports `#[epilogue(include = "...", emit = "...", struct = "...")]`.
#[proc_macro_derive(Epilogue, attributes(epilogue, arg))]
pub fn derive_epilogue(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();

    // Determine crate root path
    let crate_name = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let root = if crate_name == "metallic" {
        quote::quote! { crate }
    } else {
        quote::quote! { ::metallic }
    };

    let mut include = None;
    let mut emit = None;
    let mut struct_name_attr = None;
    let mut out_var_name = None;

    for attr in &input.attrs {
        if attr.path().is_ident("epilogue") {
            if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                for meta in nested {
                    match meta {
                        Meta::NameValue(nv) => {
                            if nv.path.is_ident("include") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        include = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("emit") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        emit = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("struct") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        struct_name_attr = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("out_var") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        out_var_name = Some(lit.value());
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let include = include.expect("Epilogue must have `include` attribute");
    let emit_template = emit.expect("Epilogue must have `emit` attribute");
    let struct_name_str = struct_name_attr.unwrap_or_else(|| name.to_string());

    let (arg_infos, _) = match input.data {
        Data::Struct(data) => collect_arg_infos(&data.fields),
        _ => panic!("Epilogue only supports structs"),
    };

    let buffer_args = arg_infos.iter().filter(|info| !info.stage_skip).map(|info| {
        let arg_name = &info.name;
        let idx = info.buffer_index;
        let mtype = info
            .metal_type
            .clone()
            .unwrap_or_else(|| infer_metal_type(&info.rust_type, info.is_buffer, info.is_output));
        quote::quote! {
            #root::compound::BufferArg {
                name: #arg_name,
                metal_type: #mtype,
                buffer_index: #idx as u32,
            }
        }
    });

    let out_var_expr = if let Some(v) = out_var_name {
        quote::quote! { #v.to_string() }
    } else {
        quote::quote! { format!("{}_epilogue", input_var) }
    };

    let expanded = quote::quote! {
        impl #root::fusion::Epilogue for #name {
            fn header(&self) -> &'static str {
                #include
            }

            fn struct_name(&self) -> &'static str {
                #struct_name_str
            }
        }

        impl #root::compound::Stage for #name {
            fn includes(&self) -> Vec<&'static str> {
                vec![#include]
            }

            fn buffer_args(&self) -> Vec<#root::compound::BufferArg> {
                vec![
                    #(#buffer_args),*
                ]
            }

            fn struct_defs(&self) -> String {
                String::new()
            }

            fn emit(&self, input_var: &str) -> (String, String) {
                let out_var = #out_var_expr;
                let mut code = #emit_template.to_string();
                code = code.replace("{input_var}", input_var);
                code = code.replace("{out_var}", &out_var);
                (out_var, code)
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(CompoundKernel, attributes(compound, prologue, main, epilogue))]
pub fn derive_compound_kernel(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Extract #[compound(name = "...")] attribute
    let mut kernel_name = name.to_string().to_lowercase();
    let mut manual_output = false;

    for attr in &input.attrs {
        if attr.path().is_ident("compound") {
            if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                for meta in nested {
                    if let Meta::NameValue(nv) = meta {
                        if nv.path.is_ident("name") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Str(lit) = expr_lit.lit {
                                    kernel_name = lit.value();
                                }
                            }
                        } else if nv.path.is_ident("manual_output") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Bool(lit) = expr_lit.lit {
                                    manual_output = lit.value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Collect stage fields
    let mut prologues = Vec::new();
    let mut main_stage: Option<syn::Ident> = None;
    let mut epilogues = Vec::new();

    if let Data::Struct(data) = &input.data {
        if let Fields::Named(fields) = &data.fields {
            for field in &fields.named {
                let field_name = field.ident.as_ref().unwrap();

                for attr in &field.attrs {
                    if attr.path().is_ident("prologue") {
                        prologues.push(field_name.clone());
                    } else if attr.path().is_ident("main") {
                        main_stage = Some(field_name.clone());
                    } else if attr.path().is_ident("epilogue") {
                        epilogues.push(field_name.clone());
                    }
                }
            }
        }
    }

    // Generate unique ID type
    let id_name = syn::Ident::new(&format!("{}Id", name), name.span());

    // Always refer to metallic as ::metallic (requires extern crate self as metallic in lib.rs)
    let root = quote::quote! { ::metallic };

    let expanded = quote! {
        /// Unique ID for kernel caching.
        pub struct #id_name;

        impl #impl_generics #root::foundry::Kernel for #name #ty_generics #where_clause {
            type Args = Self;
            type Id = #id_name;

            fn function_name(&self) -> &'static str {
                #kernel_name
            }

            fn source(&self) -> #root::foundry::KernelSource {
                // Build compound kernel from stages
                let mut kernel_builder = #root::compound::CompoundKernel::new(#kernel_name)
                    .with_manual_output(#manual_output);

                // Add prologues
                #(
                    kernel_builder = kernel_builder.prologue_dyn(Box::new(self.#prologues.clone()));
                )*

                // Add main stage
                kernel_builder = kernel_builder.main_dyn(Box::new(self.#main_stage.clone()));

                // Add epilogues
                #(
                    kernel_builder = kernel_builder.epilogue_dyn(Box::new(self.#epilogues.clone()));
                )*

                // Build and return source
                let fused = kernel_builder.build();
                #root::foundry::KernelSource::String(fused.source_code().to_string())
            }

            fn includes(&self) -> #root::foundry::Includes {
                // Reconstruct builder to collect includes
                let mut kernel_builder = #root::compound::CompoundKernel::new(#kernel_name)
                    .with_manual_output(#manual_output);

                // Add prologues
                #(
                    kernel_builder = kernel_builder.prologue_dyn(Box::new(self.#prologues.clone()));
                )*

                // Add main stage
                kernel_builder = kernel_builder.main_dyn(Box::new(self.#main_stage.clone()));

                // Add epilogues
                #(
                    kernel_builder = kernel_builder.epilogue_dyn(Box::new(self.#epilogues.clone()));
                )*

                // Delegate to built kernel
                let fused = kernel_builder.build();
                <_ as #root::foundry::Kernel>::includes(&fused)
            }

            fn struct_defs(&self) -> String {
                // Reconstruct builder to collect struct defs
                let mut kernel_builder = #root::compound::CompoundKernel::new(#kernel_name)
                    .with_manual_output(#manual_output);

                // Add prologues
                #(
                    kernel_builder = kernel_builder.prologue_dyn(Box::new(self.#prologues.clone()));
                )*

                // Add main stage
                kernel_builder = kernel_builder.main_dyn(Box::new(self.#main_stage.clone()));

                // Add epilogues
                #(
                    kernel_builder = kernel_builder.epilogue_dyn(Box::new(self.#epilogues.clone()));
                )*

                // Delegate to built kernel
                let fused = kernel_builder.build();
                <_ as #root::foundry::Kernel>::struct_defs(&fused)
            }

            fn bind(&self, encoder: &#root::types::ComputeCommandEncoder) {
                self.bind_args(encoder);
            }

            fn dispatch_config(&self) -> #root::foundry::DispatchConfig {
                Self::dispatch_config(self)
            }
        }
    };

    TokenStream::from(expanded)
}
