//! Conditional kernel derive macro implementation.
//!
//! This module implements `#[derive(ConditionalKernel)]` which generates:
//! - A `select(selector)` method for runtime dispatch
//! - `Kernel` trait delegation to the selected variant
//! - Compile-time coverage analysis (gaps and overlaps are errors)

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{Data, DeriveInput, Expr, ExprBinary, Fields, Ident, Lit, Meta, Token, punctuated::Punctuated};

/// Parsed condition from #[when(...)]
#[derive(Debug, Clone)]
pub enum Condition {
    /// `x == value`
    Eq(String, i128),
    /// `x != value`
    Ne(String, i128),
    /// `x == Path::To::Value`
    EqPath(String, String),
    /// `x != Path::To::Value`
    NePath(String, String),
    /// `x < value`
    Lt(String, i128),
    /// `x <= value`
    Le(String, i128),
    /// `x > value`
    Gt(String, i128),
    /// `x >= value`
    Ge(String, i128),
    /// `x in start..end` (exclusive end)
    RangeExclusive(String, i128, i128),
    /// `x in start..=end` (inclusive end)
    RangeInclusive(String, i128, i128),
    /// `x in start..` (unbounded upper)
    RangeFrom(String, i128),
}

impl Condition {
    /// Get the variable name this condition references
    pub fn var_name(&self) -> &str {
        match self {
            Condition::Eq(v, _)
            | Condition::Ne(v, _)
            | Condition::EqPath(v, _)
            | Condition::NePath(v, _)
            | Condition::Lt(v, _)
            | Condition::Le(v, _)
            | Condition::Gt(v, _)
            | Condition::Ge(v, _)
            | Condition::RangeExclusive(v, _, _)
            | Condition::RangeInclusive(v, _, _)
            | Condition::RangeFrom(v, _) => v,
        }
    }

    /// Convert to an interval representation [start, end] (inclusive)
    /// Returns None for conditions that don't map to simple intervals (like != or Paths)
    pub fn to_interval(&self) -> Option<(i128, i128)> {
        match self {
            Condition::Eq(_, v) => Some((*v, *v)),
            Condition::Ne(_, _) => None,
            Condition::EqPath(_, _) => None,
            Condition::NePath(_, _) => None,
            Condition::Lt(_, v) => Some((i128::MIN, *v - 1)),
            Condition::Le(_, v) => Some((i128::MIN, *v)),
            Condition::Gt(_, v) => Some((*v + 1, i128::MAX)),
            Condition::Ge(_, v) => Some((*v, i128::MAX)),
            Condition::RangeExclusive(_, start, end) => Some((*start, *end - 1)),
            Condition::RangeInclusive(_, start, end) => Some((*start, *end)),
            Condition::RangeFrom(_, start) => Some((*start, i128::MAX)),
        }
    }

    /// Generate the condition check as a TokenStream
    pub fn to_condition_tokens(&self) -> TokenStream {
        let var = syn::Ident::new(self.var_name(), proc_macro2::Span::call_site());
        match self {
            Condition::Eq(_, v) => {
                let lit = syn::LitInt::new(&v.to_string(), proc_macro2::Span::call_site());
                quote! { #var == #lit }
            }
            Condition::Ne(_, v) => {
                let lit = syn::LitInt::new(&v.to_string(), proc_macro2::Span::call_site());
                quote! { #var != #lit }
            }
            Condition::EqPath(_, p) => {
                let path: syn::Path = syn::parse_str(p).unwrap();
                quote! { #var == #path }
            }
            Condition::NePath(_, p) => {
                let path: syn::Path = syn::parse_str(p).unwrap();
                quote! { #var != #path }
            }
            Condition::Lt(_, v) => {
                let lit = syn::LitInt::new(&v.to_string(), proc_macro2::Span::call_site());
                quote! { #var < #lit }
            }
            Condition::Le(_, v) => {
                let lit = syn::LitInt::new(&v.to_string(), proc_macro2::Span::call_site());
                quote! { #var <= #lit }
            }
            Condition::Gt(_, v) => {
                let lit = syn::LitInt::new(&v.to_string(), proc_macro2::Span::call_site());
                quote! { #var > #lit }
            }
            Condition::Ge(_, v) => {
                let lit = syn::LitInt::new(&v.to_string(), proc_macro2::Span::call_site());
                quote! { #var >= #lit }
            }
            Condition::RangeExclusive(_, start, end) => {
                let start_lit = syn::LitInt::new(&start.to_string(), proc_macro2::Span::call_site());
                let end_lit = syn::LitInt::new(&end.to_string(), proc_macro2::Span::call_site());
                quote! { (#var >= #start_lit && #var < #end_lit) }
            }
            Condition::RangeInclusive(_, start, end) => {
                let start_lit = syn::LitInt::new(&start.to_string(), proc_macro2::Span::call_site());
                let end_lit = syn::LitInt::new(&end.to_string(), proc_macro2::Span::call_site());
                quote! { (#var >= #start_lit && #var <= #end_lit) }
            }
            Condition::RangeFrom(_, start) => {
                let start_lit = syn::LitInt::new(&start.to_string(), proc_macro2::Span::call_site());
                quote! { #var >= #start_lit }
            }
        }
    }
}

/// Parse a condition expression from the #[when(...)] attribute
pub fn parse_condition(expr: &syn::Expr) -> Result<Condition, String> {
    match expr {
        // Binary expressions: x == 1, x > 5, etc.
        Expr::Binary(bin) => parse_binary_condition(bin),

        // MethodCall: x.in_(0..=10) - alternative syntax if needed
        Expr::MethodCall(call) => parse_method_call_condition(call),

        // Range expressions: Rust doesn't have `in` keyword, so we use method call or custom syntax
        _ => Err(format!("Unsupported condition expression: {}", expr.to_token_stream())),
    }
}

fn parse_binary_condition(bin: &ExprBinary) -> Result<Condition, String> {
    let var_name = extract_ident(&bin.left)?;

    // Try to parse RHS as a literal for simple comparisons
    if let Ok(value) = extract_literal(&bin.right) {
        return match bin.op {
            syn::BinOp::Eq(_) => Ok(Condition::Eq(var_name, value)),
            syn::BinOp::Ne(_) => Ok(Condition::Ne(var_name, value)),
            syn::BinOp::Lt(_) => Ok(Condition::Lt(var_name, value)),
            syn::BinOp::Le(_) => Ok(Condition::Le(var_name, value)),
            syn::BinOp::Gt(_) => Ok(Condition::Gt(var_name, value)),
            syn::BinOp::Ge(_) => Ok(Condition::Ge(var_name, value)),
            _ => Err(format!("Unsupported binary operator: {:?}", bin.op)),
        };
    }

    // Try to parse RHS as a path (e.g. Dtype::F16)
    if let Ok(path) = extract_path(&bin.right) {
        return match bin.op {
            syn::BinOp::Eq(_) => Ok(Condition::EqPath(var_name, path)),
            syn::BinOp::Ne(_) => Ok(Condition::NePath(var_name, path)),
            _ => Err(format!("Unsupported binary operator for path: {:?}", bin.op)),
        };
    }

    // Try to parse RHS as a range (for future && combinations)
    Err(format!("Unsupported comparison RHS: {}", bin.right.to_token_stream()))
}

/// Parse method call syntax like `x.in_(0..=10)` for range conditions
fn parse_method_call_condition(call: &syn::ExprMethodCall) -> Result<Condition, String> {
    // Get the receiver (should be the variable name)
    let var_name = extract_ident(&call.receiver)?;

    // Check method name - we use `in_` because `in` is a keyword
    let method = call.method.to_string();
    if method != "in_" {
        return Err(format!("Unsupported method: {}. Use x.in_(range) for range conditions.", method));
    }

    // Get the range argument
    if call.args.len() != 1 {
        return Err("in_() expects exactly one range argument".to_string());
    }

    let range_arg = &call.args[0];
    parse_range_expr(&var_name, range_arg)
}

/// Parse a range expression like `0..10`, `0..=10`, or `10..`
fn parse_range_expr(var_name: &str, expr: &syn::Expr) -> Result<Condition, String> {
    match expr {
        // Range: 0..10
        Expr::Range(range) => {
            let start = range.start.as_ref().map(|e| extract_literal(e)).transpose()?.unwrap_or(i128::MIN);

            match (&range.end, &range.limits) {
                (Some(end_expr), syn::RangeLimits::HalfOpen(_)) => {
                    let end = extract_literal(end_expr)?;
                    Ok(Condition::RangeExclusive(var_name.to_string(), start, end))
                }
                (Some(end_expr), syn::RangeLimits::Closed(_)) => {
                    let end = extract_literal(end_expr)?;
                    Ok(Condition::RangeInclusive(var_name.to_string(), start, end))
                }
                (None, _) => Ok(Condition::RangeFrom(var_name.to_string(), start)),
            }
        }
        _ => Err(format!("Expected range expression, got: {}", expr.to_token_stream())),
    }
}

fn extract_ident(expr: &Expr) -> Result<String, String> {
    match expr {
        Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                Ok(ident.to_string())
            } else {
                Err("Expected simple identifier".to_string())
            }
        }
        _ => Err(format!("Expected identifier, got: {}", expr.to_token_stream())),
    }
}

fn extract_path(expr: &Expr) -> Result<String, String> {
    match expr {
        Expr::Path(path) => Ok(path.path.to_token_stream().to_string()),
        _ => Err(format!("Expected path, got: {}", expr.to_token_stream())),
    }
}

fn extract_literal(expr: &Expr) -> Result<i128, String> {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            Lit::Int(int_lit) => int_lit
                .base10_parse::<i128>()
                .map_err(|e| format!("Failed to parse integer: {}", e)),
            _ => Err("Expected integer literal".to_string()),
        },
        _ => Err(format!("Expected literal, got: {}", expr.to_token_stream())),
    }
}

/// Parsed variant with its condition
#[derive(Debug)]
pub struct VariantInfo {
    pub name: Ident,
    pub _inner_type: syn::Type,
    pub condition: Condition,
}

/// Interval for coverage analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Interval {
    pub start: i128,
    pub end: i128, // inclusive
}

impl Interval {
    pub fn new(start: i128, end: i128) -> Self {
        Self { start, end }
    }

    /// Check if two intervals overlap
    pub fn overlaps(&self, other: &Interval) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Merge two overlapping intervals
    pub fn merge(&self, other: &Interval) -> Option<Interval> {
        if self.overlaps(other) || self.end + 1 == other.start || other.end + 1 == self.start {
            Some(Interval::new(self.start.min(other.start), self.end.max(other.end)))
        } else {
            None
        }
    }
}

/// Analyze coverage and detect gaps/overlaps
/// Returns (overlaps, gaps) as error messages
pub fn analyze_coverage(variants: &[VariantInfo], selector_type: &str) -> (Vec<String>, Vec<String>) {
    let mut overlaps = Vec::new();
    let mut gaps = Vec::new();

    // Collect all intervals
    let mut intervals: Vec<(Interval, &VariantInfo)> = Vec::new();

    for variant in variants {
        if let Some((start, end)) = variant.condition.to_interval() {
            intervals.push((Interval::new(start, end), variant));
        }
    }

    // Check for overlaps (pairwise)
    for i in 0..intervals.len() {
        for j in (i + 1)..intervals.len() {
            let (int_i, var_i) = &intervals[i];
            let (int_j, var_j) = &intervals[j];
            if int_i.overlaps(int_j) {
                overlaps.push(format!(
                    "Overlapping conditions: {} ({:?}) and {} ({:?})",
                    var_i.name, var_i.condition, var_j.name, var_j.condition
                ));
            }
        }
    }

    // Compute coverage and find gaps
    // Sort intervals by start
    let mut sorted: Vec<_> = intervals.iter().map(|(i, _)| *i).collect();
    sorted.sort_by_key(|i| i.start);

    // Merge overlapping/adjacent intervals
    let mut merged: Vec<Interval> = Vec::new();
    for interval in sorted {
        if let Some(last) = merged.last_mut() {
            if let Some(m) = last.merge(&interval) {
                *last = m;
                continue;
            }
        }
        merged.push(interval);
    }

    // Determine expected range based on selector type
    let (type_min, type_max): (i128, i128) = match selector_type {
        "u8" => (0, 255),
        "u16" => (0, 65535),
        "u32" => (0, u32::MAX as i128),
        "u64" | "usize" => (0, i64::MAX as i128), // Approximate
        "i8" => (-128, 127),
        "i16" => (-32768, 32767),
        "i32" => (i32::MIN as i128, i32::MAX as i128),
        "i64" | "isize" => (i64::MIN as i128, i64::MAX as i128),
        _ => (0, i64::MAX as i128), // Default to unsigned-like
    };

    // Find gaps between merged intervals within the type range
    let mut cursor = type_min;
    for interval in &merged {
        // Clamp interval to type range
        let clamped_start = interval.start.max(type_min);
        let clamped_end = interval.end.min(type_max);

        if clamped_start > cursor {
            gaps.push(format!(
                "Coverage gap: {} in {}..{}",
                variants[0].condition.var_name(),
                cursor,
                clamped_start
            ));
        }
        cursor = clamped_end.saturating_add(1);
    }

    // Check for gap at the end
    if cursor <= type_max && !merged.iter().any(|i| i.end >= type_max) {
        gaps.push(format!(
            "Coverage gap: {} in {}..={}",
            variants[0].condition.var_name(),
            cursor,
            type_max
        ));
    }

    (overlaps, gaps)
}

/// Main derive implementation
pub fn derive_conditional_kernel_impl(input: DeriveInput) -> TokenStream {
    let name = &input.ident;

    // Determine crate root
    let crate_name = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let root = if crate_name == "metallic" {
        quote! { crate }
    } else {
        quote! { ::metallic }
    };

    // Parse #[conditional(selector = "...")] attribute
    let mut selector_params: Vec<(String, String)> = Vec::new(); // (name, type)

    for attr in &input.attrs {
        if attr.path().is_ident("conditional") {
            if let Ok(nested) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                for meta in nested {
                    if let Meta::NameValue(nv) = meta {
                        if nv.path.is_ident("selector") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Str(lit) = expr_lit.lit {
                                    // Parse "batch: u32, seq_k: usize"
                                    for part in lit.value().split(',') {
                                        let part = part.trim();
                                        if let Some((name, ty)) = part.split_once(':') {
                                            selector_params.push((name.trim().to_string(), ty.trim().to_string()));
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

    if selector_params.is_empty() {
        return quote! {
            compile_error!("ConditionalKernel requires #[conditional(selector = \"name: Type, ...\")] attribute");
        };
    }

    // Parse enum variants with #[when(...)] conditions
    let mut variants: Vec<VariantInfo> = Vec::new();

    let data_enum = match &input.data {
        Data::Enum(e) => e,
        _ => {
            return quote! {
                compile_error!("ConditionalKernel can only be derived for enums");
            };
        }
    };

    for variant in &data_enum.variants {
        let variant_name = &variant.ident;

        // Extract inner type from tuple variant
        let inner_type = match &variant.fields {
            Fields::Unnamed(fields) if fields.unnamed.len() == 1 => fields.unnamed.first().unwrap().ty.clone(),
            _ => {
                return quote! {
                    compile_error!("ConditionalKernel variants must be tuple variants with exactly one field, e.g., Variant(InnerType)");
                };
            }
        };

        // Find #[when(...)] attribute
        let mut condition: Option<Condition> = None;
        for attr in &variant.attrs {
            if attr.path().is_ident("when") {
                if let Ok(expr) = attr.parse_args::<syn::Expr>() {
                    match parse_condition(&expr) {
                        Ok(c) => condition = Some(c),
                        Err(e) => {
                            let msg = format!("Failed to parse condition: {}", e);
                            return quote! { compile_error!(#msg); };
                        }
                    }
                }
            }
        }

        let condition = match condition {
            Some(c) => c,
            None => {
                let msg = format!("Variant {} missing #[when(...)] condition", variant_name);
                return quote! { compile_error!(#msg); };
            }
        };

        variants.push(VariantInfo {
            name: variant_name.clone(),
            _inner_type: inner_type,
            condition,
        });
    }

    // Analyze coverage (currently only supports single-variable conditions)
    if !selector_params.is_empty() && variants.iter().all(|v| v.condition.var_name() == selector_params[0].0) {
        let (overlaps, gaps) = analyze_coverage(&variants, &selector_params[0].1);

        if !overlaps.is_empty() {
            let msg = overlaps.join("; ");
            return quote! { compile_error!(#msg); };
        }

        if !gaps.is_empty() {
            let msg = gaps.join("; ");
            return quote! { compile_error!(#msg); };
        }
    }

    // Generate select() method
    let selector_args: Vec<_> = selector_params
        .iter()
        .map(|(name, ty)| {
            let name_ident = syn::Ident::new(name, proc_macro2::Span::call_site());
            let ty_path: syn::Type = syn::parse_str(ty).unwrap();
            quote! { #name_ident: #ty_path }
        })
        .collect();

    // Generate a Variant enum for dispatch without requiring Default
    let variant_enum_name = syn::Ident::new(&format!("{}Variant", name), name.span());
    let variant_enum_variants: Vec<_> = variants
        .iter()
        .map(|v| {
            let variant_name = &v.name;
            quote! { #variant_name }
        })
        .collect();

    let match_arms: Vec<_> = variants
        .iter()
        .map(|v| {
            let variant_name = &v.name;
            let condition = v.condition.to_condition_tokens();
            quote! {
                if #condition {
                    return #variant_enum_name::#variant_name;
                }
            }
        })
        .collect();

    // Generate Kernel trait delegation (mirrors kernel_enum! logic)
    let variant_names: Vec<_> = variants.iter().map(|v| &v.name).collect();

    let expanded = quote! {
        /// Variant discriminant for conditional dispatch.
        /// Use `select()` to get the variant, then construct the kernel.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum #variant_enum_name {
            #( #variant_enum_variants, )*
        }

        impl #name {
            /// Select the appropriate variant based on runtime conditions.
            /// Returns a discriminant that can be pattern-matched.
            pub fn select(#(#selector_args),*) -> #variant_enum_name {
                #(#match_arms)*
                unreachable!("All conditions verified at compile-time")
            }
        }

        impl #root::foundry::Kernel for #name {
            type Args = ();

            fn function_name(&self) -> &str {
                match self {
                    #( Self::#variant_names(k) => k.function_name(), )*
                }
            }

            fn source(&self) -> #root::foundry::KernelSource {
                match self {
                    #( Self::#variant_names(k) => k.source(), )*
                }
            }

            fn includes(&self) -> #root::foundry::Includes {
                match self {
                    #( Self::#variant_names(k) => k.includes(), )*
                }
            }

            fn dtype(&self) -> Option<#root::tensor::Dtype> {
                match self {
                    #( Self::#variant_names(k) => k.dtype(), )*
                }
            }

            fn struct_defs(&self) -> String {
                match self {
                    #( Self::#variant_names(k) => k.struct_defs(), )*
                }
            }

            fn bind(&self, encoder: &#root::types::ComputeCommandEncoder) {
                match self {
                    #( Self::#variant_names(k) => k.bind(encoder), )*
                }
            }

            fn dispatch_config(&self) -> #root::types::DispatchConfig {
                match self {
                    #( Self::#variant_names(k) => k.dispatch_config(), )*
                }
            }

            fn to_stage(&self) -> Box<dyn #root::compound::Stage> {
                match self {
                    #( Self::#variant_names(k) => k.to_stage(), )*
                }
            }
        }
    };

    expanded
}
