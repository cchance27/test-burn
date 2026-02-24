use super::*;

pub(crate) fn derive_stage(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();

    let root = foundry_crate();

    let mut include: Option<String> = None;
    let mut includes: Vec<String> = Vec::new();
    let mut include_exprs: Vec<Expr> = Vec::new();
    let mut emit: Option<String> = None;
    let mut emit_span: Option<Span> = None;
    let mut out_var: Option<String> = None;
    let mut struct_defs_type: Option<String> = None;
    let mut struct_defs_types: Vec<String> = Vec::new();
    let mut struct_defs_fn: Option<String> = None;
    let mut struct_defs_method: Option<String> = None;
    let mut buffer_args_fn: Option<String> = None;
    let mut activation_field: Option<String> = None;
    let mut policy_field: Option<String> = None;
    let mut template_bindings: Vec<(String, Expr)> = Vec::new();

    for attr in &input.attrs {
        if attr.path().is_ident("stage") {
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
                                        emit_span = Some(lit.span());
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
                                match nv.value {
                                    Expr::Lit(expr_lit) => {
                                        if let Lit::Str(lit) = expr_lit.lit {
                                            struct_defs_type = Some(lit.value());
                                        }
                                    }
                                    Expr::Array(arr) => {
                                        for elem in arr.elems {
                                            let Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) = elem else {
                                                return TokenStream::from(
                                                    syn::Error::new(elem.span(), "struct_defs = [...] expects string literals")
                                                        .to_compile_error(),
                                                );
                                            };
                                            struct_defs_types.push(s.value());
                                        }
                                    }
                                    other => {
                                        return TokenStream::from(
                                            syn::Error::new(other.span(), "struct_defs expects a string literal or [\"TypeA\", \"TypeB\"]")
                                                .to_compile_error(),
                                        );
                                    }
                                }
                            } else if nv.path.is_ident("struct_defs_fn") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        struct_defs_fn = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("struct_defs_method") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        struct_defs_method = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("buffer_args_fn") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        buffer_args_fn = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("activation_field") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        activation_field = Some(lit.value());
                                    }
                                }
                            } else if nv.path.is_ident("policy_field") {
                                if let Expr::Lit(expr_lit) = nv.value {
                                    if let Lit::Str(lit) = expr_lit.lit {
                                        policy_field = Some(lit.value());
                                    }
                                }
                            }
                        }
                        Meta::List(list) if list.path.is_ident("includes") => {
                            let vals = match list.parse_args_with(Punctuated::<Lit, Token![,]>::parse_terminated) {
                                Ok(v) => v,
                                Err(e) => return TokenStream::from(e.to_compile_error()),
                            };
                            for lit in vals {
                                match lit {
                                    Lit::Str(s) => includes.push(s.value()),
                                    _ => {
                                        return TokenStream::from(
                                            syn::Error::new(lit.span(), "includes(...) expects string literals").to_compile_error(),
                                        );
                                    }
                                }
                            }
                        }
                        Meta::List(list) if list.path.is_ident("struct_defs") => {
                            let vals = match list.parse_args_with(Punctuated::<Lit, Token![,]>::parse_terminated) {
                                Ok(v) => v,
                                Err(e) => return TokenStream::from(e.to_compile_error()),
                            };
                            for lit in vals {
                                match lit {
                                    Lit::Str(s) => struct_defs_types.push(s.value()),
                                    _ => {
                                        return TokenStream::from(
                                            syn::Error::new(lit.span(), "struct_defs(...) expects string literals").to_compile_error(),
                                        );
                                    }
                                }
                            }
                        }
                        Meta::List(list) if list.path.is_ident("include_exprs") => {
                            let vals = match list.parse_args_with(Punctuated::<Lit, Token![,]>::parse_terminated) {
                                Ok(v) => v,
                                Err(e) => return TokenStream::from(e.to_compile_error()),
                            };
                            for lit in vals {
                                let Lit::Str(s) = lit else {
                                    return TokenStream::from(
                                        syn::Error::new(lit.span(), "include_exprs(...) expects string literals").to_compile_error(),
                                    );
                                };
                                match syn::parse_str::<Expr>(&s.value()) {
                                    Ok(expr) => include_exprs.push(expr),
                                    Err(e) => return TokenStream::from(e.to_compile_error()),
                                }
                            }
                        }
                        Meta::List(list) if list.path.is_ident("template_bindings") => {
                            let vals = match list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                                Ok(v) => v,
                                Err(e) => return TokenStream::from(e.to_compile_error()),
                            };
                            for meta in vals {
                                let Meta::NameValue(nv) = meta else {
                                    return TokenStream::from(
                                        syn::Error::new(list.span(), "template_bindings(...) expects name = expression pairs")
                                            .to_compile_error(),
                                    );
                                };
                                let Some(id) = nv.path.get_ident() else {
                                    return TokenStream::from(
                                        syn::Error::new(nv.path.span(), "template binding key must be a single identifier")
                                            .to_compile_error(),
                                    );
                                };
                                let value = nv.value;
                                let expr = if let Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) = value {
                                    match syn::parse_str::<Expr>(&s.value()) {
                                        Ok(e) => e,
                                        Err(e) => return TokenStream::from(e.to_compile_error()),
                                    }
                                } else {
                                    value
                                };
                                template_bindings.push((id.to_string(), expr));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let emit_template = emit.unwrap_or_default();
    let emit_span = emit_span.unwrap_or_else(Span::call_site);
    let mut binding_names = std::collections::HashSet::new();
    for (name, _) in &template_bindings {
        if !binding_names.insert(name.clone()) {
            return TokenStream::from(syn::Error::new(emit_span, format!("duplicate template binding `{name}`")).to_compile_error());
        }
    }

    if activation_field.is_some() {
        binding_names.insert("activation_header".to_string());
        binding_names.insert("activation_struct".to_string());
    }
    if policy_field.is_some() {
        binding_names.insert("policy_header".to_string());
        binding_names.insert("policy_struct".to_string());
        binding_names.insert("policy_short".to_string());
    }

    let mut allowed = std::collections::HashSet::new();
    allowed.insert("input_var".to_string());
    allowed.insert("out_var".to_string());
    allowed.extend(binding_names);

    for placeholder in template_placeholder_names(&emit_template) {
        if !allowed.contains(&placeholder) {
            return TokenStream::from(
                syn::Error::new(
                    emit_span,
                    format!("unknown template placeholder `{{{placeholder}}}` in stage emit template"),
                )
                .to_compile_error(),
            );
        }
    }

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

    let require_explicit_metal_type = |info: &ArgInfo| -> Result<String, syn::Error> {
        if let Some(mtype) = info.metal_type.clone() {
            return Ok(mtype);
        }
        if info.is_buffer {
            let field_name = info
                .name_ident
                .as_ref()
                .map(|id| id.to_string())
                .unwrap_or_else(|| info.name.clone());
            return Err(syn::Error::new_spanned(
                &info.rust_type_actual,
                format!(
                    "Stage `{}` field `{}` is missing #[arg(metal_type = \"...\")]. Explicit metal_type is required for buffer/tensor args.",
                    name, field_name
                ),
            ));
        }
        Ok(infer_metal_type(&info.rust_type_actual, info.is_buffer, info.is_output))
    };

    let mut buffer_args = Vec::new();
    for info in arg_infos.iter().filter(|info| !info.stage_skip) {
        let arg_name = &info.name;
        let idx = info.buffer_index;
        let mtype = match require_explicit_metal_type(info) {
            Ok(v) => v,
            Err(err) => return TokenStream::from(err.to_compile_error()),
        };
        buffer_args.push(quote::quote! {
            #root::compound::BufferArg {
                name: #arg_name,
                metal_type: #mtype,
                buffer_index: #idx as u32,
            }
        });
    }

    // Generate includes vec
    let mut static_includes: Vec<String> = Vec::new();
    if let Some(inc) = include {
        static_includes.push(inc);
    }
    static_includes.extend(includes);
    let static_includes_tokens = static_includes.iter().map(|inc| quote::quote! { #inc });
    let include_expr_tokens = include_exprs.iter();
    let activation_include_expr = activation_field.as_ref().map(|f| {
        let ident = Ident::new(f, Span::call_site());
        quote::quote! { includes.push(self.#ident.header()); }
    });
    let policy_include_expr = policy_field.as_ref().map(|f| {
        let ident = Ident::new(f, Span::call_site());
        quote::quote! { includes.push(self.#ident.header()); }
    });
    let includes_impl = quote::quote! {
        let mut includes = vec![#(#static_includes_tokens),*];
        #(includes.push(#include_expr_tokens);)*
        #activation_include_expr
        #policy_include_expr
        includes
    };

    // Generate struct_defs impl - supports struct_defs_method, struct_defs_fn or struct_defs (MetalStruct type)
    let struct_defs_impl = if let Some(method_name) = struct_defs_method {
        let method_ident = Ident::new(&method_name, Span::call_site());
        quote::quote! {
            self.#method_ident()
        }
    } else if let Some(fn_name) = struct_defs_fn {
        let fn_ident = Ident::new(&fn_name, Span::call_site());
        quote::quote! {
            Self::#fn_ident()
        }
    } else if !struct_defs_types.is_empty() {
        let struct_type_idents: Vec<Ident> = struct_defs_types.iter().map(|t| Ident::new(t, Span::call_site())).collect();
        quote::quote! {
            {
                let mut __defs = ::std::string::String::new();
                #( __defs.push_str(#struct_type_idents::METAL_STRUCT_DEF); __defs.push('\n'); )*
                __defs
            }
        }
    } else if let Some(type_name) = struct_defs_type {
        let type_ident = Ident::new(&type_name, Span::call_site());
        quote::quote! {
            #type_ident::METAL_STRUCT_DEF.to_string()
        }
    } else {
        quote::quote! { String::new() }
    };

    let buffer_args_impl = if let Some(fn_name) = buffer_args_fn {
        let fn_ident = Ident::new(&fn_name, Span::call_site());
        quote::quote! {
            self.#fn_ident()
        }
    } else {
        quote::quote! {
            vec![
                #(#buffer_args),*
            ]
        }
    };

    let binding_replacements = template_bindings.iter().map(|(name, expr)| {
        let var = quote::format_ident!("__stage_bind_{}", name);
        let key = format!("{{{name}}}");
        quote::quote! {
            let #var = ::std::string::ToString::to_string(&(#expr));
            code = code.replace(#key, &#var);
        }
    });

    let activation_replacements = activation_field.as_ref().map(|f| {
        let ident = Ident::new(f, Span::call_site());
        quote::quote! {
            let __stage_activation_header = ::std::string::ToString::to_string(&self.#ident.header());
            let __stage_activation_struct = ::std::string::ToString::to_string(&self.#ident.struct_name());
            code = code.replace("{activation_header}", &__stage_activation_header);
            code = code.replace("{activation_struct}", &__stage_activation_struct);
        }
    });

    let policy_replacements = policy_field.as_ref().map(|f| {
        let ident = Ident::new(f, Span::call_site());
        quote::quote! {
            let __stage_policy_header = ::std::string::ToString::to_string(&self.#ident.header());
            let __stage_policy_struct = ::std::string::ToString::to_string(&self.#ident.struct_name());
            let __stage_policy_short = ::std::string::ToString::to_string(&self.#ident.short_name());
            code = code.replace("{policy_header}", &__stage_policy_header);
            code = code.replace("{policy_struct}", &__stage_policy_struct);
            code = code.replace("{policy_short}", &__stage_policy_short);
        }
    });

    let activation_meta_impl = activation_field.as_ref().map(|f| {
        let ident = Ident::new(f, Span::call_site());
        quote::quote! {
            fn activation_meta(&self) -> Option<#root::policy::activation::Activation> {
                Some(self.#ident)
            }
        }
    });

    let policy_meta_impl = policy_field.as_ref().map(|f| {
        let ident = Ident::new(f, Span::call_site());
        quote::quote! {
            fn policy_meta(&self) -> Option<#root::fusion::PolicyMeta> {
                Some(self.#ident.meta())
            }
        }
    });

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote::quote! {
        impl #impl_generics #root::compound::Stage for #name #ty_generics #where_clause {
            fn includes(&self) -> Vec<&'static str> {
                #includes_impl
            }

            fn buffer_args(&self) -> Vec<#root::compound::BufferArg> {
                #buffer_args_impl
            }

            fn struct_defs(&self) -> String {
                let __stage_struct_defs_raw = #struct_defs_impl;
                if __stage_struct_defs_raw.trim().is_empty() {
                    __stage_struct_defs_raw
                } else {
                    let mut __stage_struct_defs_hasher = ::std::collections::hash_map::DefaultHasher::new();
                    ::std::hash::Hash::hash(&__stage_struct_defs_raw, &mut __stage_struct_defs_hasher);
                    let __stage_struct_defs_guard = ::std::format!(
                        "METALLIC_STAGE_STRUCT_DEFS_{:016X}",
                        ::std::hash::Hasher::finish(&__stage_struct_defs_hasher)
                    );
                    ::std::format!(
                        "#ifndef {guard}\n#define {guard}\n{defs}\n#endif\n",
                        guard = __stage_struct_defs_guard,
                        defs = __stage_struct_defs_raw
                    )
                }
            }

            fn emit(&self, input_var: &str) -> (String, String) {
                let out_var = #out_var_expr;
                let mut code = #emit_template.to_string();
                code = code.replace("{input_var}", input_var);
                code = code.replace("{out_var}", &out_var);
                #(#binding_replacements)*
                #activation_replacements
                #policy_replacements
                if !code.ends_with('\n') {
                    code.push('\n');
                }
                (out_var, code)
            }

            #activation_meta_impl
            #policy_meta_impl
        }
    };

    TokenStream::from(expanded)
}
