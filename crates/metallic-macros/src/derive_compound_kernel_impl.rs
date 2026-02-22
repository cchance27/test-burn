use super::*;

pub(crate) fn derive_compound_kernel(input: TokenStream) -> TokenStream {
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

    let root = foundry_crate();

    let expanded = quote! {
        impl #impl_generics #root::Kernel for #name #ty_generics #where_clause {
            type Args = Self;

            fn function_name(&self) -> &str {
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
                #root::KernelSource::String(fused.source().to_string())
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
