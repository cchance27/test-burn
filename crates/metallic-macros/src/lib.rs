#![allow(clippy::all)]
use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Ident, Span};
use quote::quote;
use syn::{
    Data, DeriveInput, Expr, ExprLit, ExprPath, Fields, Lit, Meta, Token, parse_macro_input, punctuated::Punctuated, spanned::Spanned
};

mod conditional;

// --- Shared Helpers for Macros ---

fn validate_metal_template(template: &str, span: Span) -> syn::Result<()> {
    let mut brace_balance = 0;
    let mut paren_balance = 0;
    for c in template.chars() {
        match c {
            '{' => brace_balance += 1,
            '}' => brace_balance -= 1,
            '(' => paren_balance += 1,
            ')' => paren_balance -= 1,
            _ => {}
        }
        if brace_balance < 0 {
            return Err(syn::Error::new(span, "Unbalanced closing brace '}' in Metal template"));
        }
        if paren_balance < 0 {
            return Err(syn::Error::new(span, "Unbalanced closing parenthesis ')' in Metal template"));
        }
    }
    if brace_balance != 0 {
        return Err(syn::Error::new(span, "Unbalanced opening brace '{' in Metal template"));
    }
    if paren_balance != 0 {
        return Err(syn::Error::new(span, "Unbalanced opening parenthesis '(' in Metal template"));
    }
    Ok(())
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
    let msg = format!(
        "Failed to resolve 'metallic-foundry' crate path. CARGO_PKG_NAME={}. Please ensure metallic-foundry is a dependency.",
        pkg
    );
    quote::quote! { compile_error!(#msg); }
}

// Helper to map Rust types to Metal types for MetalStruct
fn rust_type_to_metal(ty_str: &str) -> &'static str {
    // Remove all whitespace for consistent matching
    let trimmed = ty_str.replace(' ', "").replace('\n', "").replace('\r', "").trim().to_string();

    // Check for DynamicValue<T> - extract T and map to Metal
    if let Some(inner) = extract_dynamic_value_inner(&trimmed) {
        return rust_type_to_metal(inner);
    }

    match trimmed.as_str() {
        "u8" => "uchar",
        "i8" => "char",
        "u16" => "ushort",
        "i16" => "short",
        "u32" => "uint",
        "i32" => "int",
        "u64" => "ulong",
        "i64" => "long",
        "f32" => "float",
        // Note: half::f16 shows up as "f16" or "half::f16" in quote! output
        "f16" | "half::f16" | "half::f16::f16" => "half",
        _ => {
            // If it still contains DynamicValue, something is wrong with extraction
            if trimmed.contains("DynamicValue") {
                panic!(
                    "Failed to extract inner type from DynamicValue: '{}'. Please ensure it's a simple type like u32.",
                    trimmed
                );
            }
            unreachable!("Unsupported type: '{}'", trimmed)
        }
    }
}

/// Extract the inner type T from DynamicValue<T> pattern.
/// Returns None if not a DynamicValue.
fn extract_dynamic_value_inner(ty_str: &str) -> Option<&str> {
    // Remove all whitespace and common path prefixes for consistent matching
    let cleaned = ty_str
        .replace(' ', "")
        .replace('\n', "")
        .replace('\r', "")
        .replace("crate::", "")
        .replace("foundry::spec::", "")
        .replace("spec::", "")
        .trim()
        .to_string();

    if cleaned.starts_with("DynamicValue<") && cleaned.ends_with('>') {
        let start = "DynamicValue<".len();
        let end = cleaned.len() - 1;
        if start < end {
            let inner = &cleaned[start..end];
            // We need to return a 'static str, so leak it
            return Some(Box::leak(inner.to_string().into_boxed_str()));
        }
    }

    // Also handle the case where the input already has the format generated by quote! (spaces preserved)
    if ty_str.contains("DynamicValue") && ty_str.contains('<') && ty_str.contains('>') {
        if let Some(start) = ty_str.find('<') {
            if let Some(end) = ty_str.rfind('>') {
                if start < end {
                    let inner = ty_str[start + 1..end].trim();
                    // Recursively clean the inner type string if it contains extra junk
                    let inner_cleaned = inner.replace(' ', "").replace('\n', "").replace('\r', "");
                    return Some(Box::leak(inner_cleaned.into_boxed_str()));
                }
            }
        }
    }

    None
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

                // Collect arg info for signature generation
                arg_infos.push(ArgInfo {
                    name: name.as_ref().map(|i| i.to_string()).unwrap_or_default(),
                    name_ident: name.clone(),
                    buffer_index: idx,
                    metal_type: metal_type.clone(),
                    rust_type: type_str.clone(),
                    rust_type_actual: ty.clone(),
                    is_output,
                    is_buffer: is_buffer_type && !explicit_bytes,
                    stage_skip,
                });

                // Generate binding code
                if is_bytes {
                    bindings.push(quote::quote! {
                        let ptr = &self.#name as *const _ as *const core::ffi::c_void;
                        let len = core::mem::size_of_val(&self.#name);
                        unsafe {
                            encoder.setBytes_length_atIndex(core::ptr::NonNull::new(ptr as *mut _).unwrap(), len, #idx as usize);
                        }
                    });
                } else {
                    bindings.push(quote::quote! {
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

    let root = foundry_crate();

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

    // Track which fields are DynamicValue for Resolvable impl
    let mut has_dynamic_fields = false;
    let mut field_resolve_code = Vec::new();

    if let Data::Struct(data) = &input.data {
        if let Fields::Named(fields) = &data.fields {
            for field in &fields.named {
                let field_name = field.ident.as_ref().unwrap();
                let field_type = &field.ty;
                let type_str = quote!(#field_type).to_string();

                // Check if this field is a DynamicValue
                if extract_dynamic_value_inner(&type_str).is_some() {
                    has_dynamic_fields = true;
                    // For DynamicValue fields, call .resolve(bindings)
                    field_resolve_code.push(quote! {
                        #field_name: self.#field_name.resolve(bindings)
                    });
                } else {
                    // For regular fields, just clone
                    field_resolve_code.push(quote! {
                        #field_name: self.#field_name.clone()
                    });
                }
            }
        }
    }

    let resolvable_impl = if has_dynamic_fields {
        // Create a resolved params type name
        let resolved_name = syn::Ident::new(&format!("{}Resolved", name), name.span());

        // Metal struct name for the resolved type (includes Resolved suffix)
        let resolved_metal_struct_name = format!("{}Resolved", metal_struct_name);

        // Collect field definitions for the resolved struct
        let mut resolved_field_defs = Vec::new();
        let mut resolve_field_assigns = Vec::new();
        let mut resolved_metal_fields = Vec::new();

        if let Data::Struct(data) = &input.data {
            if let Fields::Named(fields) = &data.fields {
                for field in &fields.named {
                    let field_name = field.ident.as_ref().unwrap();
                    let field_type = &field.ty;
                    let type_str = quote!(#field_type).to_string();

                    // Check if this field is a DynamicValue<T>
                    if let Some(inner_type_str) = extract_dynamic_value_inner(&type_str) {
                        // Parse the inner type for the resolved struct
                        let inner_type: syn::Type = syn::parse_str(inner_type_str).expect("Failed to parse inner type of DynamicValue");

                        // Resolved struct has concrete type
                        resolved_field_defs.push(quote! {
                            pub #field_name: #inner_type
                        });

                        // Resolve field by calling .resolve(bindings)
                        resolve_field_assigns.push(quote! {
                            #field_name: self.#field_name.resolve(bindings)
                        });

                        // Metal field uses inner type
                        let inner_metal_type = rust_type_to_metal(&inner_type_str.replace(' ', ""));
                        resolved_metal_fields.push(format!("    {} {};", inner_metal_type, field_name));
                    } else {
                        // Non-dynamic field - same type in resolved struct, clone value
                        resolved_field_defs.push(quote! {
                            pub #field_name: #field_type
                        });

                        resolve_field_assigns.push(quote! {
                            #field_name: self.#field_name.clone()
                        });

                        // Use same Metal type
                        let metal_type = rust_type_to_metal(&type_str.replace(' ', ""));
                        resolved_metal_fields.push(format!("    {} {};", metal_type, field_name));
                    }
                }
            }
        }

        // Generate Metal struct def with Resolved name
        let resolved_metal_def = format!(
            "struct {} {{\n{}\n}};",
            resolved_metal_struct_name,
            resolved_metal_fields.join("\n")
        );

        quote! {
            /// Resolved params type with all DynamicValue fields converted to concrete types.
            /// This struct has the same layout as the Metal shader expects.
            #[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
            #[repr(C)]
            pub struct #resolved_name {
                #(#resolved_field_defs),*
            }

            impl #resolved_name {
                /// The Metal struct definition for this resolved type.
                pub const METAL_STRUCT_DEF: &'static str = #resolved_metal_def;
            }

            impl #root::spec::Resolvable for #name {
                type Resolved = #resolved_name;

                /// Resolve all DynamicValue fields from bindings.
                fn resolve(&self, bindings: &#root::spec::TensorBindings) -> Self::Resolved {
                    #resolved_name {
                        #(#resolve_field_assigns),*
                    }
                }
            }
        }
    } else {
        // No dynamic fields - trivial Resolvable impl (returns clone of self)
        quote! {
            impl #root::spec::Resolvable for #name {
                type Resolved = Self;

                fn resolve(&self, _bindings: &#root::spec::TensorBindings) -> Self::Resolved {
                    self.clone()
                }
            }
        }
    };

    let expanded = quote! {
        impl #name {
            /// The Metal struct definition for this type.
            pub const METAL_STRUCT_DEF: &'static str = #metal_def;
        }

        #resolvable_impl
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

    let root = foundry_crate();

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
    let root = foundry_crate();

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

    let root = foundry_crate();

    let mut source = None;
    let mut function = None;
    let mut args_type = None;
    let mut includes = Vec::new();
    let mut stage_function = None;
    let mut threadgroup_decl = None;
    let mut epilogue_emit: Option<String> = None;
    let mut epilogue_out_var: Option<String> = None;
    let mut enable_step = true; // Default to true
    let mut stage_expr: Option<String> = None;
    let mut stage_emit: Option<(String, Span)> = None;
    let mut stage_out_var: Option<String> = None;
    let mut has_dispatch = true; // Default to true
    let mut dispatch_preset: Option<String> = None;
    let mut dtype_str: Option<String> = None;

    let mut errors = Vec::new();

    for attr in &input.attrs {
        if attr.path().is_ident("kernel") {
            if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                for meta in nested {
                    match meta {
                        Meta::NameValue(nv) => {
                            let ident = if let Some(ident) = nv.path.get_ident() { ident } else { continue };

                            // Helper to get string from Lit or Ident
                            let get_val = |v: &Expr| -> Option<String> {
                                match v {
                                    Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) => Some(s.value()),
                                    Expr::Path(ExprPath { path, .. }) if path.get_ident().is_some() => {
                                        Some(path.get_ident().unwrap().to_string())
                                    }
                                    _ => None,
                                }
                            };

                            let get_bool = |v: &Expr| -> Option<bool> {
                                match v {
                                    Expr::Lit(ExprLit { lit: Lit::Bool(b), .. }) => Some(b.value),
                                    _ => None,
                                }
                            };

                            if ident == "source" {
                                if let Some(v) = get_val(&nv.value) {
                                    source = Some(v);
                                }
                            } else if ident == "function" {
                                if let Some(v) = get_val(&nv.value) {
                                    function = Some(v);
                                }
                            } else if ident == "args" {
                                // Support "Type" string or Type directly
                                match &nv.value {
                                    Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) => match s.parse::<syn::Type>() {
                                        Ok(t) => args_type = Some(t),
                                        Err(e) => errors.push(e),
                                    },
                                    Expr::Path(ExprPath { path, .. }) => {
                                        args_type = Some(syn::Type::Path(syn::TypePath {
                                            qself: None,
                                            path: path.clone(),
                                        }));
                                    }
                                    _ => errors.push(syn::Error::new_spanned(
                                        &nv.value,
                                        "Expected string literal or type path for 'args'",
                                    )),
                                }
                            } else if ident == "include" {
                                if let Expr::Array(arr) = nv.value {
                                    for elem in arr.elems {
                                        if let Some(v) = get_val(&elem) {
                                            includes.push(v);
                                        }
                                    }
                                }
                            } else if ident == "stage_function" {
                                if let Some(v) = get_val(&nv.value) {
                                    stage_function = Some(v);
                                }
                            } else if ident == "threadgroup" {
                                if let Some(v) = get_val(&nv.value) {
                                    threadgroup_decl = Some(v);
                                }
                            } else if ident == "epilogue_emit" {
                                if let Some(v) = get_val(&nv.value) {
                                    epilogue_emit = Some(v.clone());
                                    if let Err(e) = validate_metal_template(&v, nv.value.span()) {
                                        errors.push(e);
                                    }
                                }
                            } else if ident == "epilogue_out_var" {
                                if let Some(v) = get_val(&nv.value) {
                                    epilogue_out_var = Some(v);
                                }
                            } else if ident == "step" {
                                if let Some(b) = get_bool(&nv.value) {
                                    enable_step = b;
                                }
                            } else if ident == "dispatch" {
                                if let Some(b) = get_bool(&nv.value) {
                                    has_dispatch = b;
                                } else if let Some(v) = get_val(&nv.value) {
                                    dispatch_preset = Some(v);
                                    has_dispatch = true;
                                }
                            } else if ident == "stage" {
                                if let Some(v) = get_val(&nv.value) {
                                    stage_expr = Some(v);
                                }
                            } else if ident == "stage_emit" {
                                if let Some(v) = get_val(&nv.value) {
                                    stage_emit = Some((v.clone(), nv.value.span()));
                                    if let Err(e) = validate_metal_template(&v, nv.value.span()) {
                                        errors.push(e);
                                    }
                                }
                            } else if ident == "stage_out_var" {
                                if let Some(v) = get_val(&nv.value) {
                                    stage_out_var = Some(v);
                                }
                            } else if ident == "dtype" {
                                if let Some(v) = get_val(&nv.value) {
                                    dtype_str = Some(v);
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

    if !errors.is_empty() {
        let errs = errors.iter().map(|e| e.to_compile_error());
        return TokenStream::from(quote! { #(#errs)* });
    }

    let source = source.expect("Kernel must have `source` attribute");
    let function = function.expect("Kernel must have `function` attribute");
    let args_type = args_type.unwrap_or_else(|| syn::parse_str("()").unwrap());

    let dtype_expr = if let Some(d) = dtype_str {
        let ident = quote::format_ident!("{}", d);
        quote! { Some(#root::tensor::Dtype::#ident) }
    } else {
        quote! { None }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let id_name = quote::format_ident!("{}KernelId", name);
    let stage_name = quote::format_ident!("{}Stage", name);

    // Generate as_stage() implementation
    let as_stage_impl = if let Some(expr_str) = &stage_expr {
        let expr: syn::Expr = syn::parse_str(expr_str).expect("Failed to parse stage expression");
        quote! {
            fn as_stage(&self) -> Box<dyn #root::compound::Stage> {
                #expr
            }
        }
    } else if stage_function.is_some() || stage_emit.is_some() || epilogue_emit.is_some() {
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
                panic!("Kernel {} does not support staging (no stage_function, stage expression, or epilogue_emit defined)", stringify!(#name))
            }
        }
    };

    // Generate Stage struct and impl if stage_function, stage_emit or epilogue_emit is specified
    let stage_impl = if stage_function.is_some() || stage_emit.is_some() || epilogue_emit.is_some() {
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
        } else if let Some((ref emit_tmpl, _)) = stage_emit {
            let out_var_expr = if let Some(ref v) = stage_out_var {
                quote! { #v.to_string() }
            } else {
                quote! { "void".to_string() }
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

    let dispatch_code = if let Some(preset) = dispatch_preset {
        if preset == "per_element" {
            quote! {
                #root::types::DispatchConfig {
                    grid: #root::types::GridSize::d1((self.params.total_elements as usize + 255) / 256),
                    group: #root::types::ThreadgroupSize::d1(256),
                }
            }
        } else if preset == "per_row" {
            quote! {
                #root::types::DispatchConfig {
                    grid: #root::types::GridSize::d1((self.params.total_elements / self.params.feature_dim) as usize),
                    group: #root::types::ThreadgroupSize::d1(256),
                }
            }
        } else if let Some(n_str) = preset.strip_prefix("vec_") {
            if let Ok(n) = n_str.parse::<usize>() {
                quote! {
                    #root::types::DispatchConfig {
                        grid: #root::types::GridSize::d1(((self.params.total_elements as usize + #n - 1) / #n + 255) / 256),
                        group: #root::types::ThreadgroupSize::d1(256),
                    }
                }
            } else {
                return TokenStream::from(
                    syn::Error::new(Span::call_site(), format!("Invalid vec_N preset: {}", preset)).to_compile_error(),
                );
            }
        } else {
            return TokenStream::from(
                syn::Error::new(Span::call_site(), format!("Unknown dispatch preset: {}", preset)).to_compile_error(),
            );
        }
    } else if has_dispatch {
        quote! { self.dispatch_config() }
    } else {
        quote! {
            panic!(
                "dispatch_config() not implemented for kernel {}. Use dispatch() with explicit grid/group sizes, or set dispatch = true in #[kernel] and implement dispatch_config().",
                stringify!(#name)
            )
        }
    };

    let expanded = quote! {
        #[doc(hidden)]
        pub struct #id_name;

        impl #impl_generics #root::Kernel for #name #ty_generics #where_clause {
            type Args = #args_type;
            type Id = #id_name;

            fn function_name(&self) -> &'static str {
                #function
            }

            fn source(&self) -> #root::KernelSource {
                #root::KernelSource::File(#source)
            }

            fn includes(&self) -> #root::Includes {
                #root::Includes(vec![#(#includes),*])
            }

            fn dtype(&self) -> Option<#root::tensor::Dtype> {
                #dtype_expr
            }

            fn bind(&self, encoder: &#root::types::ComputeCommandEncoder) {
                // Call the bind_args method generated by KernelArgs
                self.bind_args(encoder);
            }

            fn dispatch_config(&self) -> #root::types::DispatchConfig {
                #dispatch_code
            }

            // Foundry kernels use runtime compilation with struct injection (struct_defs),
            // so we don't provide precompiled metallib bytes.
            fn metallib_bytes(&self) -> Option<&'static [u8]> {
                // See build.rs for explanation why we don't do precompiled metallib currently.
                None
            }

            fn struct_defs(&self) -> String {
                // Return METAL_STRUCT_DEF from the Args type if it has one
                <#args_type>::METAL_STRUCT_DEF.to_string()
            }

            #as_stage_impl
        }

        #stage_impl
    };

    // Generate Step impl if step = true
    // This creates a {Kernel}Step struct with Ref fields that deserializes from JSON
    // and resolves refs to TensorArgs at execute time.
    let step_impl = if enable_step {
        let name_str = name.to_string();
        let step_name = quote::format_ident!("{}Step", name);
        let compiled_step_name = quote::format_ident!("Compiled{}Step", name);

        // Collect field info from the kernel struct
        let (arg_infos, _) = match &input.data {
            Data::Struct(data) => collect_arg_infos(&data.fields),
            _ => (Vec::new(), Vec::new()),
        };

        // 1. Generate Step struct definition (JSON serializable)
        let step_fields: Vec<_> = arg_infos
            .iter()
            .map(|info| {
                let fname = info.name_ident.as_ref().unwrap();
                if info.rust_type.contains("TensorArg") {
                    quote! { pub #fname: #root::spec::Ref }
                } else if info.rust_type.contains("Resolved") {
                    // For *Resolved types, use the non-Resolved version with DynamicValue for deserialization
                    // Convention: FooParamsResolved -> FooParams
                    let resolved_type_str = &info.rust_type;
                    let dynamic_type_str = resolved_type_str.replace("Resolved", "");
                    let dynamic_type: syn::Type = syn::parse_str(&dynamic_type_str).expect("Failed to parse dynamic params type");
                    quote! { pub #fname: #dynamic_type }
                } else {
                    let ftype = &info.rust_type_actual;
                    quote! { pub #fname: #ftype }
                }
            })
            .collect();

        // 2. Generate CompiledStep struct definition (Optimized)
        let compiled_fields: Vec<_> = arg_infos
            .iter()
            .map(|info| {
                let fname = info.name_ident.as_ref().unwrap();
                if info.rust_type.contains("TensorArg") {
                    quote! { pub #fname: usize }
                } else if info.rust_type.contains("Resolved") {
                    let resolved_type_str = &info.rust_type;
                    let dynamic_type_str = resolved_type_str.replace("Resolved", "");
                    let dynamic_type: syn::Type = syn::parse_str(&dynamic_type_str).expect("Failed to parse dynamic params type");
                    quote! { pub #fname: #dynamic_type }
                } else {
                    let ftype = &info.rust_type_actual;
                    quote! { pub #fname: #ftype }
                }
            })
            .collect();

        // 3. Generate Resolve Fields for Step::execute (Original)
        let resolve_fields: Vec<_> = arg_infos
            .iter()
            .map(|info| {
                let fname = info.name_ident.as_ref().unwrap();
                if info.rust_type.contains("TensorArg") {
                    quote! { #fname: bindings.resolve(&self.#fname)? }
                } else if info.rust_type.contains("Params") {
                    // Params types need to resolve DynamicValue fields via Resolvable trait
                    quote! { #fname: #root::spec::Resolvable::resolve(&self.#fname, bindings) }
                } else {
                    quote! { #fname: self.#fname.clone() }
                }
            })
            .collect();

        // 4. Generate Compile Fields for Step::compile (New)
        let compile_fields: Vec<_> = arg_infos
            .iter()
            .map(|info| {
                let fname = info.name_ident.as_ref().unwrap();
                if info.rust_type.contains("TensorArg") {
                    quote! {
                        #fname: symbols.get_or_create(resolver.interpolate(self.#fname.0.clone()))
                    }
                } else {
                    quote! { #fname: self.#fname.clone() }
                }
            })
            .collect();

        // 5. Generate Execute Fields for CompiledStep::execute (New)
        let execute_fields: Vec<_> = arg_infos
            .iter()
            .map(|info| {
                let fname = info.name_ident.as_ref().unwrap();
                if info.rust_type.contains("TensorArg") {
                    quote! {
                        #fname: bindings.get(self.#fname).cloned().ok_or_else(|| #root::error::MetalError::InputNotFound("Compiled tensor missing".into()))?
                    }
                } else if info.rust_type.contains("Params") {
                     // Note: using 'globals' (the TensorBindings ref) for resolving vars
                     quote! { #fname: #root::spec::Resolvable::resolve(&self.#fname, globals) }
                } else {
                    quote! { #fname: self.#fname.clone() }
                }
            })
            .collect();

        quote! {
            /// Auto-generated DSL Step for JSON deserialization.
            /// Resolves string refs to TensorArgs at execute time.
            #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
            pub struct #step_name {
                #(#step_fields),*
            }

            /// Auto-generated Compiled Step for fast execution.
            #[derive(Debug, Clone, Default)]
            pub struct #compiled_step_name {
                #(#compiled_fields),*
            }

            #[typetag::serde(name = #name_str)]
            impl #root::spec::Step for #step_name {
                fn execute(&self, foundry: &mut #root::Foundry, bindings: &mut #root::spec::TensorBindings) -> Result<(), #root::error::MetalError> {
                    // Use Default for fields without #[arg] annotations
                    let kernel = #name {
                        #(#resolve_fields,)*
                        ..Default::default()
                    };
                    foundry.run(&kernel)
                }

                fn compile(&self, resolver: &mut #root::spec::TensorBindings, symbols: &mut #root::spec::SymbolTable) -> Vec<Box<dyn #root::spec::CompiledStep>> {
                    let compiled = #compiled_step_name {
                        #(#compile_fields,)*
                        // Using Default is safe because compiled fields cover all struct fields if derived correctly
                        ..Default::default()
                    };
                    vec![Box::new(compiled)]
                }

                fn name(&self) -> &'static str {
                    #name_str
                }
            }

            impl #root::spec::CompiledStep for #compiled_step_name {
                                fn execute(&self, foundry: &mut #root::Foundry, bindings: & #root::spec::FastBindings, globals: & #root::spec::TensorBindings, _symbols: &#root::spec::SymbolTable) -> Result<(), #root::error::MetalError> {
                    let kernel = #name {
                        #(#execute_fields,)*
                        ..Default::default()
                    };
                    foundry.run(&kernel)
                }
            }
        }
    } else {
        quote! {}
    };

    let final_expanded = quote! {
        #expanded
        #step_impl
    };

    TokenStream::from(final_expanded)
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

    let root = foundry_crate();

    let mut include = None;
    let mut emit = None;
    let mut struct_name_attr = None;
    let mut out_var_name = None;
    let mut gemv_struct = None;
    let mut gemv_id = None;
    let mut simd_reduce: Vec<(String, String)> = Vec::new();
    // Flexible SIMD reduce config: from/to level, operation
    let mut _simd_reduce_from: u8 = 16; // default: 32-lane (16 down to 1)
    let mut _simd_reduce_to: u8 = 1;
    let mut _simd_reduce_op: String = "add".to_string();

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
                                        if let Err(e) = validate_metal_template(&lit.value(), lit.span()) {
                                            return TokenStream::from(e.to_compile_error());
                                        }
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
                            } else if nv.path.is_ident("gemv_struct") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        gemv_struct = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("gemv_id") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        gemv_id = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("simd_reduce") {
                                // Format: "gate: acc[0], up: acc[1]"
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        for part in lit.value().split(',') {
                                            let part = part.trim();
                                            if let Some((name, source)) = part.split_once(':') {
                                                simd_reduce.push((name.trim().to_string(), source.trim().to_string()));
                                            }
                                        }
                                    }
                                }
                            } else if nv.path.is_ident("simd_reduce_from") {
                                // Parse from level (u8): simd_reduce_from = "16"
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        _simd_reduce_from = lit.value().parse().unwrap_or(16);
                                    } else if let Lit::Int(lit) = expr_lit.lit {
                                        _simd_reduce_from = lit.base10_parse().unwrap_or(16);
                                    }
                                }
                            } else if nv.path.is_ident("simd_reduce_to") {
                                // Parse to level (u8): simd_reduce_to = "1"
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        _simd_reduce_to = lit.value().parse().unwrap_or(1);
                                    } else if let Lit::Int(lit) = expr_lit.lit {
                                        _simd_reduce_to = lit.base10_parse().unwrap_or(1);
                                    }
                                }
                            } else if nv.path.is_ident("simd_reduce_op") {
                                // Parse op: simd_reduce_op = "add" | "max" | "min"
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        _simd_reduce_op = lit.value().to_lowercase();
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
    let struct_name_str = struct_name_attr.unwrap_or_else(|| name.to_string());
    let has_stage = emit.is_some();
    let has_gemv = gemv_struct.is_some();

    if !has_stage && !has_gemv {
        panic!("Epilogue must provide `emit` (Stage epilogue) and/or `gemv_struct` (GEMV template epilogue)");
    }

    let emit_template = emit.unwrap_or_default();
    let _gemv_struct_str = gemv_struct.unwrap_or_default();
    let _gemv_id_str = gemv_id.unwrap_or_else(|| name.to_string());

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

    let stage_impl = if has_stage {
        quote::quote! {
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
        }
    } else {
        quote::quote! {}
    };

    let gemv_impl = quote::quote! {};

    let expanded = quote::quote! {
        #stage_impl
        #gemv_impl
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

    let root = foundry_crate();

    let expanded = quote! {
        /// Unique ID for kernel caching.
        pub struct #id_name;

        impl #impl_generics #root::Kernel for #name #ty_generics #where_clause {
            type Args = Self;
            type Id = #id_name;

            fn function_name(&self) -> &'static str {
                #kernel_name
            }

            fn source(&self) -> #root::KernelSource {
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
                #root::KernelSource::String(fused.source_code().to_string())
            }

            fn includes(&self) -> #root::Includes {
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
                <_ as #root::Kernel>::includes(&fused)
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
                <_ as #root::Kernel>::struct_defs(&fused)
            }

            fn bind(&self, encoder: &#root::types::ComputeCommandEncoder) {
                self.bind_args(encoder);
            }

            fn dispatch_config(&self) -> #root::DispatchConfig {
                Self::dispatch_config(self)
            }
        }
    };

    TokenStream::from(expanded)
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
    let input = parse_macro_input!(input as DeriveInput);
    conditional::derive_conditional_kernel_impl(input).into()
}

/// Unified GEMV kernel derive that combines Config, Hook, and Epilogue.
///
/// # Example
/// ```ignore
/// #[derive(GemvKernel)]
/// #[gemv_kernel(
///     args = "MyArgs",
///     heads = 2,
///     cols_per_tg = 8,
///     fast_path = true,
///     gemv_n0 = "params->N0",
///     data_ptrs("data_a", "data_b"),
///     result_ptrs("out_a", "out_b"),
///     n_exprs("params->N0", "params->N1"),
///     bias_ptrs("bias_a", "bias_b"),
///     has_bias_flags("params->has_bias0", "params->has_bias1"),
///     struct_defs_type(MyParams),
///     hook = F16CanonicalRmsnormHook,
///     epilogue = SwiGluEpilogue,
/// )]
/// pub struct MyFusedKernel;
/// ```
///
/// Generates:
/// - `impl GemvSimdConfig for MyFusedKernel { ... }`
/// - `type MainStage = GemvSimdMainStage<MyFusedKernel, Hook, Epilogue>`

/// Derive macro for SIMD GEMV Prologue stages.
///
/// # Example
/// ```ignore
/// #[derive(GemvPrologue)]
/// #[gemv_prologue(
///     emit = r#"
///     threadgroup float inv_rms_s;
///     const float inv_rms = gemv_compute_inv_rms(vector_x, params->K, lid, wid, &inv_rms_s, epsilon);
///     "#,
///     includes("matmul_gemv/simd_common.metal")
/// )]
/// pub struct RmsnormPrologue;
/// ```

/// Derive macro for auto-generating CompiledStep boilerplate.
///
/// Use this on Step structs that manually implement `Step` trait.
/// It generates:
/// 1. A `Compiled{Step}` struct with Ref fields converted to `usize` indices
/// 2. `Step::compile()` implementation that maps Ref names to symbol indices
/// 3. `CompiledStep::execute()` implementation that fetches tensors from FastBindings
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
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let compiled_name = quote::format_ident!("Compiled{}", name);

    let root = foundry_crate();

    // Parse attributes
    let mut kernel_type: Option<syn::Type> = None;
    let mut step_name: Option<String> = None;

    for attr in &input.attrs {
        if attr.path().is_ident("compiled_step") {
            if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                for meta in nested {
                    if let Meta::NameValue(nv) = meta {
                        if nv.path.is_ident("kernel") {
                            if let Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) = nv.value {
                                kernel_type = Some(syn::parse_str(&s.value()).expect("Invalid kernel type"));
                            }
                        } else if nv.path.is_ident("name") {
                            if let Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) = nv.value {
                                step_name = Some(s.value());
                            }
                        }
                    }
                }
            }
        }
    }
    // FIXME: step_name is unused
    let _ = step_name;

    // Collect field info
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("CompiledStep only supports named fields"),
        },
        _ => panic!("CompiledStep only supports structs"),
    };

    let mut ref_fields = Vec::new();
    let mut other_fields = Vec::new();

    for field in fields {
        let fname = field.ident.as_ref().unwrap();
        let ftype = &field.ty;
        let is_ref = field.attrs.iter().any(|a| a.path().is_ident("ref_field"));

        if is_ref {
            ref_fields.push((fname.clone(), ftype.clone()));
        } else {
            other_fields.push((fname.clone(), ftype.clone()));
        }
    }

    // Generate compiled struct fields
    let compiled_fields: Vec<_> = ref_fields
        .iter()
        .map(|(fname, _)| {
            let idx_name = quote::format_ident!("{}_idx", fname);
            quote! { pub #idx_name: usize }
        })
        .chain(other_fields.iter().map(|(fname, ftype)| {
            quote! { pub #fname: #ftype }
        }))
        .collect();

    // Generate compile() field mappings
    let compile_mappings: Vec<_> = ref_fields
        .iter()
        .map(|(fname, _)| {
            let idx_name = quote::format_ident!("{}_idx", fname);
            quote! {
                #idx_name: symbols.get_or_create(resolver.interpolate(self.#fname.0.clone()))
            }
        })
        .chain(other_fields.iter().map(|(fname, _)| {
            quote! { #fname: self.#fname.clone() }
        }))
        .collect();

    // Generate execute() tensor fetches
    let tensor_fetches: Vec<_> = ref_fields
        .iter()
        .map(|(fname, _)| {
            let idx_name = quote::format_ident!("{}_idx", fname);
            let err_msg = format!("{} tensor not found at idx {{}}", fname);
            quote! {
                let #fname = fast_bindings
                    .get(self.#idx_name)
                    .ok_or_else(|| #root::error::MetalError::InvalidShape(format!(#err_msg, self.#idx_name)))?;
            }
        })
        .collect();

    // Generate kernel construction for execute
    let kernel_field_args: Vec<_> = ref_fields
        .iter()
        .map(|(fname, _)| quote! { #fname.clone() })
        .chain(other_fields.iter().map(|(fname, _)| quote! { self.#fname.clone() }))
        .collect();

    let kernel_ty = kernel_type.expect("Missing #[compiled_step(kernel = \"...\")]");
    // FIXME: step_name_str is unused
    //let step_name_str = step_name.unwrap_or_else(|| name.to_string().replace("Step", ""));

    let expanded = quote! {
        /// Auto-generated compiled step struct.
        #[derive(Debug)]
        pub struct #compiled_name {
            #(#compiled_fields),*
        }

        impl #name {
            /// Compile this step into an optimized form.
            pub fn do_compile(&self, resolver: &mut #root::spec::TensorBindings, symbols: &mut #root::spec::SymbolTable) -> #compiled_name {
                #compiled_name {
                    #(#compile_mappings),*
                }
            }
        }

        impl #root::spec::CompiledStep for #compiled_name {
            fn execute(
                &self,
                foundry: &mut #root::Foundry,
                fast_bindings: &#root::spec::FastBindings,
                _bindings: &#root::spec::TensorBindings,
                _symbols: &#root::spec::SymbolTable,
            ) -> Result<(), #root::error::MetalError> {
                #(#tensor_fetches)*

                // Note: Kernel construction may need custom logic for some kernels.
                // This is a simplified version that works for simple cases.
                foundry.run(&<#kernel_ty>::new(#(#kernel_field_args),*))
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for Stage trait.
///
/// Generates `Stage` implementation from struct annotations.
/// This is a simpler version of `#[derive(Epilogue)]` without GEMV-specific hooks.
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
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();

    let root = foundry_crate();

    let mut include: Option<String> = None;
    let mut emit: Option<String> = None;
    let mut out_var: Option<String> = None;
    let mut struct_defs_type: Option<String> = None;
    let mut struct_defs_fn: Option<String> = None;

    for attr in &input.attrs {
        if attr.path().is_ident("stage") {
            if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                for meta in nested {
                    if let Meta::NameValue(nv) = meta {
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
                                    if let Err(e) = validate_metal_template(&lit.value(), lit.span()) {
                                        return TokenStream::from(e.to_compile_error());
                                    }
                                }
                            }
                        } else if nv.path.is_ident("out_var") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Str(lit) = expr_lit.lit {
                                    out_var = Some(lit.value());
                                }
                            }
                        } else if nv.path.is_ident("struct_defs") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Str(lit) = expr_lit.lit {
                                    struct_defs_type = Some(lit.value());
                                }
                            }
                        } else if nv.path.is_ident("struct_defs_fn") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Str(lit) = expr_lit.lit {
                                    struct_defs_fn = Some(lit.value());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let emit_template = emit.unwrap_or_default();
    let out_var_expr = if let Some(v) = out_var {
        quote::quote! { #v.to_string() }
    } else {
        quote::quote! { "void".to_string() }
    };

    // Collect buffer args from fields
    let (arg_infos, _) = match input.data {
        Data::Struct(data) => collect_arg_infos(&data.fields),
        _ => panic!("Stage only supports structs"),
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

    // Generate includes vec
    let includes_impl = if let Some(inc) = include {
        quote::quote! { vec![#inc] }
    } else {
        quote::quote! { vec![] }
    };

    // Generate struct_defs impl - supports struct_defs_fn (function) or struct_defs (MetalStruct type)
    let struct_defs_impl = if let Some(fn_name) = struct_defs_fn {
        let fn_ident = Ident::new(&fn_name, Span::call_site());
        quote::quote! {
            Self::#fn_ident()
        }
    } else if let Some(type_name) = struct_defs_type {
        let type_ident = Ident::new(&type_name, Span::call_site());
        quote::quote! {
            <#type_ident as #root::HasMetalStructDef>::METAL_STRUCT_DEF.to_string()
        }
    } else {
        quote::quote! { String::new() }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote::quote! {
        impl #impl_generics #root::compound::Stage for #name #ty_generics #where_clause {
            fn includes(&self) -> Vec<&'static str> {
                #includes_impl
            }

            fn buffer_args(&self) -> Vec<#root::compound::BufferArg> {
                vec![
                    #(#buffer_args),*
                ]
            }

            fn struct_defs(&self) -> String {
                #struct_defs_impl
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
