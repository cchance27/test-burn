use super::*;

pub(crate) fn derive_metal_policy(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let root = foundry_crate();

    // Parse #[policy(header = "...", struct_name = "...", short_name = "...", element_size = N, ...)]
    let mut header = String::new();
    let mut struct_name = name.to_string();
    let mut short_name: Option<String> = None;
    let mut element_size: Option<usize> = None;
    // Optimization hints
    let mut block_size: Option<usize> = None;
    let mut vector_load_size: Option<usize> = None;
    let mut unroll_factor: Option<usize> = None;
    let mut active_thread_count: Option<usize> = None;
    let mut has_scale: Option<bool> = None;
    let mut block_size_bytes: Option<usize> = None;
    let mut weights_per_block: Option<usize> = None;

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
                        } else if nv.path.is_ident("short_name") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Str(lit) = expr_lit.lit {
                                    short_name = Some(lit.value());
                                }
                            }
                        } else if nv.path.is_ident("element_size") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Int(lit) = expr_lit.lit {
                                    element_size = Some(lit.base10_parse::<usize>().unwrap());
                                }
                            }
                        } else if nv.path.is_ident("block_size") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Int(lit) = expr_lit.lit {
                                    block_size = Some(lit.base10_parse::<usize>().unwrap());
                                }
                            }
                        } else if nv.path.is_ident("vector_load_size") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Int(lit) = expr_lit.lit {
                                    vector_load_size = Some(lit.base10_parse::<usize>().unwrap());
                                }
                            }
                        } else if nv.path.is_ident("unroll_factor") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Int(lit) = expr_lit.lit {
                                    unroll_factor = Some(lit.base10_parse::<usize>().unwrap());
                                }
                            }
                        } else if nv.path.is_ident("active_thread_count") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Int(lit) = expr_lit.lit {
                                    active_thread_count = Some(lit.base10_parse::<usize>().unwrap());
                                }
                            }
                        } else if nv.path.is_ident("has_scale") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Bool(lit) = expr_lit.lit {
                                    has_scale = Some(lit.value);
                                }
                            }
                        } else if nv.path.is_ident("block_size_bytes") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Int(lit) = expr_lit.lit {
                                    block_size_bytes = Some(lit.base10_parse::<usize>().unwrap());
                                }
                            }
                        } else if nv.path.is_ident("weights_per_block") {
                            if let Expr::Lit(expr_lit) = nv.value {
                                if let Lit::Int(lit) = expr_lit.lit {
                                    weights_per_block = Some(lit.base10_parse::<usize>().unwrap());
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
                                                init_statements.push(format!("pp.{field_name} = {source_code};"));
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

    // Generate optimization metadata (stored in PolicyMeta).
    let has_any_hint = block_size.is_some() || vector_load_size.is_some() || unroll_factor.is_some() || active_thread_count.is_some();

    let meta_header = header.clone();
    let meta_struct_name = struct_name.clone();
    let meta_short = short_name.clone().unwrap_or_else(|| "unknown".to_string());
    let meta_address_unit_bytes = element_size.unwrap_or(2);
    let meta_has_scale = has_scale.unwrap_or(true);
    let meta_block_size_bytes = block_size_bytes.unwrap_or(meta_address_unit_bytes);
    let meta_weights_per_block = weights_per_block.unwrap_or(1);
    let meta_opt = if has_any_hint {
        let bs = block_size.unwrap_or(1);
        let vls = vector_load_size.unwrap_or(2);
        let uf = unroll_factor.unwrap_or(1);
        let atc = active_thread_count.unwrap_or(32);
        quote! {
            #root::policy::OptimizationMetadata {
                block_size: #bs,
                vector_load_size: #vls,
                unroll_factor: #uf,
                active_thread_count: #atc,
            }
        }
    } else {
        quote! { #root::policy::OptimizationMetadata::default() }
    };

    let meta_ident = quote::format_ident!("__METALLIC_POLICY_META_{}", name);

    let expanded = quote! {
        #[allow(non_upper_case_globals)]
        const #meta_ident: #root::fusion::PolicyMeta = #root::fusion::PolicyMeta {
            header: #meta_header,
            struct_name: #meta_struct_name,
            short_name: #meta_short,
            address_unit_bytes: #meta_address_unit_bytes,
            has_scale: #meta_has_scale,
            block_size_bytes: #meta_block_size_bytes,
            weights_per_block: #meta_weights_per_block,
            optimization: #meta_opt,
        };

        impl #root::fusion::MetalPolicy for #name {
            fn header(&self) -> &'static str {
                #header
            }

            fn struct_name(&self) -> &'static str {
                #struct_name
            }

            fn short_name(&self) -> &'static str {
                #meta_short
            }

            fn element_size(&self) -> usize {
                #meta_address_unit_bytes
            }

            fn optimization_hints(&self) -> #root::policy::OptimizationMetadata {
                #meta_opt
            }

            fn has_scale(&self) -> bool {
                #meta_has_scale
            }

            fn block_size_bytes(&self) -> usize {
                #meta_block_size_bytes
            }

            fn weights_per_block(&self) -> usize {
                #meta_weights_per_block
            }

            fn meta(&self) -> #root::fusion::PolicyMeta {
                #meta_ident
            }

            fn init_params_code(&self) -> &'static str {
                #init_code
            }
        }

        impl #root::compound::Stage for #name {
            fn includes(&self) -> Vec<&'static str> {
                vec![]
            }

            fn buffer_args(&self) -> Vec<#root::compound::BufferArg> {
                vec![]
            }

            fn emit(&self, _input_var: &str) -> (String, String) {
                (String::new(), String::new())
            }
        }
    };

    TokenStream::from(expanded)
}
