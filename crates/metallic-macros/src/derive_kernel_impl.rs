use super::*;

pub(crate) fn derive_kernel(input: TokenStream) -> TokenStream {
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
    let mut enable_step = false; // Default to false
    let mut enable_execute = true; // Default to true if step is enabled
    let mut stage_expr: Option<String> = None;
    let mut stage_emit: Option<(String, Span)> = None;
    let mut stage_out_var: Option<String> = None;
    let mut has_dispatch = true; // Default to true
    let mut dispatch_preset: Option<String> = None;
    let mut dtype_str: Option<String> = None;
    let mut struct_defs_fn: Option<String> = None;
    let mut struct_defs_method: Option<String> = None;
    let mut include_exprs: Vec<Expr> = Vec::new();

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
                            } else if ident == "execute" {
                                if let Some(b) = get_bool(&nv.value) {
                                    enable_execute = b;
                                }
                            } else if ident == "dispatch" {
                                // Check for #[kernel(dispatch = true/false)] OR #[kernel(dispatch = "preset")]
                                if let Some(b) = get_bool(&nv.value) {
                                    has_dispatch = b;
                                } else if let Some(v) = get_val(&nv.value) {
                                    dispatch_preset = Some(v);
                                    has_dispatch = true; // Preset implies having dispatch logic
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
                            } else if ident == "struct_defs_fn" {
                                if let Some(v) = get_val(&nv.value) {
                                    struct_defs_fn = Some(v);
                                }
                            } else if ident == "struct_defs_method" {
                                if let Some(v) = get_val(&nv.value) {
                                    struct_defs_method = Some(v);
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
                        Meta::List(_) => {}
                        _ => {}
                    }
                }
            }
        }
    }

    if !errors.is_empty() {
        let errs = errors.iter().map(syn::Error::to_compile_error);
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
    } else {
        quote::quote! {
            <#args_type>::METAL_STRUCT_DEF.to_string()
        }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let stage_name = quote::format_ident!("{}Stage", name);

    // Generate to_stage() implementation
    let to_stage_impl = if let Some(expr_str) = &stage_expr {
        let expr: syn::Expr = syn::parse_str(expr_str).expect("Failed to parse stage expression");
        quote! {
            fn to_stage(&self) -> Box<dyn #root::compound::Stage> {
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
            fn to_stage(&self) -> Box<dyn #root::compound::Stage> {
                Box::new(#stage_name {
                    #( #field_names: self.#field_names.clone() ),*
                })
            }
        }
    } else {
        quote! {
            fn to_stage(&self) -> Box<dyn #root::compound::Stage> {
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
                    if !code.ends_with('\n') {
                        code.push('\n');
                    }
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
                    if !code.ends_with('\n') {
                        code.push('\n');
                    }
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

    let dispatch_code = if let Some(preset) = dispatch_preset.as_ref().map(|s| s.trim()) {
        if preset == "per_element" {
            quote! {
                #root::types::DispatchConfig {
                    grid: #root::types::GridSize::d1((self.params.total_elements as usize + 255) / 256),
                    group: #root::types::ThreadgroupSize::d1(256),
                }
            }
        } else if preset == "per_element_vec" {
            quote! {
                {
                    let vector_width = std::cmp::max(self.params.vector_width as usize, 1);
                    let base_threads = 256;
                    let threads_per_group_width = std::cmp::max(base_threads / vector_width, 1);
                    let total_threads = if self.params.vector_width > 1 {
                        let vectorized = self.params.total_elements / self.params.vector_width;
                        let remainder = self.params.total_elements % self.params.vector_width;
                        (vectorized + remainder) as usize
                    } else {
                        self.params.total_elements as usize
                    };
                    let num_groups = total_threads.div_ceil(threads_per_group_width);
                    #root::types::DispatchConfig {
                        grid: #root::types::GridSize::d1(num_groups),
                        group: #root::types::ThreadgroupSize::d1(threads_per_group_width),
                    }
                }
            }
        } else if preset == "per_row" {
            quote! {
                #root::types::DispatchConfig {
                    grid: #root::types::GridSize::d1((self.params.total_elements / self.params.feature_dim) as usize),
                    group: #root::types::ThreadgroupSize::d1(256),
                }
            }
        } else if preset == "warp_per_row" {
            quote! {
                {
                    let num_tgs = (self.params.n_dim as usize).div_ceil(8);
                    #root::types::DispatchConfig {
                        grid: #root::types::GridSize::new(num_tgs, 1, 1),
                        group: #root::types::ThreadgroupSize::new(256, 1, 1),
                    }
                }
            }
        } else if preset == "warp_per_row_2d" {
            quote! {
                 {
                    let num_tgs = (self.params.n_dim as usize).div_ceil(8);
                    let batch = self.params.batch.max(1) as usize;
                    #root::types::DispatchConfig {
                        grid: #root::types::GridSize::new(num_tgs, batch, 1),
                        group: #root::types::ThreadgroupSize::new(256, 1, 1),
                    }
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
                return TokenStream::from(syn::Error::new(Span::call_site(), format!("Invalid vec_N preset: {preset}")).to_compile_error());
            }
        } else {
            return TokenStream::from(syn::Error::new(Span::call_site(), format!("Unknown dispatch preset: {preset}")).to_compile_error());
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
        impl #impl_generics #root::Kernel for #name #ty_generics #where_clause {
            type Args = #args_type;

            fn function_name(&self) -> &str {
                #function
            }

            fn source(&self) -> #root::KernelSource {
                #root::KernelSource::File(#source)
            }

            fn includes(&self) -> #root::Includes {
                let mut includes = vec![#(#includes),*];
                #(includes.push(#include_exprs);)*
                #root::Includes(includes)
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
                #struct_defs_impl
            }

            #to_stage_impl
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
        // Skip fields that are implicitly derived (scale_for)
        let step_fields: Vec<_> = arg_infos
            .iter()
            .filter(|info| info.scale_for.is_none())
            .map(|info| {
                let fname = info.name_ident.as_ref().unwrap();
                let attrs = &info.attrs;
                let serde_attrs = &info.serde_attrs;
                let is_opt = info.is_option;
                if is_tensor_arg(&info.rust_type_actual) {
                    if is_opt {
                        quote! { #(#attrs)* #(#serde_attrs)* pub #fname: Option<#root::spec::Ref> }
                    } else {
                        quote! { #(#attrs)* #(#serde_attrs)* pub #fname: #root::spec::Ref }
                    }
                } else if info.rust_type.contains("Resolved") {
                    // For *Resolved types, use the non-Resolved version with DynamicValue for deserialization
                    // Convention: FooParamsResolved -> FooParams
                    let resolved_type_str = &info.rust_type;
                    let dynamic_type_str = resolved_type_str.replace("Resolved", "");
                    let dynamic_type: syn::Type = syn::parse_str(&dynamic_type_str).expect("Failed to parse dynamic params type");
                    quote! { #(#attrs)* #(#serde_attrs)* pub #fname: #dynamic_type }
                } else {
                    let ftype = &info.rust_type_actual;
                    quote! { #(#attrs)* #(#serde_attrs)* pub #fname: #ftype }
                }
            })
            .collect();

        // 2. Generate CompiledStep struct definition (Optimized)
        let compiled_fields: Vec<_> = arg_infos
            .iter()
            .map(|info| {
                let fname = info.name_ident.as_ref().unwrap();
                let is_opt = info.is_option;
                if is_tensor_arg(&info.rust_type_actual) {
                    if is_opt {
                        quote! { pub #fname: Option<usize> }
                    } else {
                        quote! { pub #fname: usize }
                    }
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
        // Skip scale_for fields as they aren't in Step
        let _resolve_fields: Vec<_> = arg_infos
            .iter()
            .filter(|info| info.scale_for.is_none())
            .map(|info| {
                let fname = info.name_ident.as_ref().unwrap();
                let is_opt = info.is_option;
                if is_tensor_arg(&info.rust_type_actual) {
                    if is_opt {
                        quote! { #fname: self.#fname.as_ref().map(|r| bindings.resolve(r)).transpose()? }
                    } else {
                        quote! { #fname: bindings.resolve(&self.#fname)? }
                    }
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
                let is_opt = info.is_option;
                if let Some(target) = &info.scale_for {
                    // It's a derived scale arg - look up the target field in Step and append _scales
                    // ASSUMPTION: The target field is a mandatory Ref (not Option).
                    // If target is Option, this will fail. Current usage (Embedding/Gemv) targets mandatory weights.
                    let target_ident = Ident::new(target, Span::call_site());
                    if is_opt {
                        quote! {
                            #fname: Some(symbols.get_or_create(format!("{}_scales", resolver.interpolate(self.#target_ident.0.clone()))))
                        }
                    } else {
                        quote! {
                            #fname: symbols.get_or_create(format!("{}_scales", resolver.interpolate(self.#target_ident.0.clone())))
                        }
                    }
                } else if is_tensor_arg(&info.rust_type_actual) {
                    if is_opt {
                        quote! {
                            #fname: self.#fname.as_ref().map(|r| symbols.get_or_create(resolver.interpolate(r.0.clone())))
                        }
                    } else {
                        quote! {
                            #fname: symbols.get_or_create(resolver.interpolate(self.#fname.0.clone()))
                        }
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
                let is_opt = info.is_option;
                if is_tensor_arg(&info.rust_type_actual) {
                    if is_opt {
                        quote! {
                            #fname: self.#fname.and_then(|idx| bindings.get(idx).cloned())
                        }
                    } else {
                        quote! {
                            #fname: bindings.get(self.#fname).cloned().ok_or_else(|| #root::error::MetalError::InputNotFound("Compiled tensor missing".into()))?
                        }
                    }
                } else if info.rust_type.contains("Params") {
                     // Note: using 'globals' (the TensorBindings ref) for resolving vars
                     quote! { #fname: #root::spec::Resolvable::resolve(&self.#fname, globals) }
                } else {
                    quote! { #fname: self.#fname.clone() }
                }
            })
            .collect();

        let compiled_step_impl = if enable_execute {
            quote! {
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
                    let mut symbols = #root::spec::SymbolTable::new();
                    let compiled = self.compile(bindings, &mut symbols);
                    let mut fast_bindings = #root::spec::FastBindings::new(symbols.len());

                    // Bind all symbols found in the table
                    for (name, symbol_id) in symbols.iter() {
                        if let Ok(tensor) = bindings.get(name) {
                            fast_bindings.set(*symbol_id, tensor);
                        }
                    }

                    for step in compiled {
                        step.execute(foundry, &fast_bindings, bindings, &symbols)?;
                    }

                    Ok(())
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

            #compiled_step_impl
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
