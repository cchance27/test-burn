use super::*;

pub(crate) fn derive_compiled_step(input: TokenStream) -> TokenStream {
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
            let err_msg = format!("{fname} tensor not found at idx {{}}");
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
