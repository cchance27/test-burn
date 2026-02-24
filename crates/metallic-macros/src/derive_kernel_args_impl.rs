use super::*;

pub(crate) fn derive_kernel_args(input: TokenStream) -> TokenStream {
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
                    "KernelArgs `{}` field `{}` is missing #[arg(metal_type = \"...\")]. Explicit metal_type is required for buffer/tensor args.",
                    name, field_name
                ),
            ));
        }
        Ok(infer_metal_type(&info.rust_type_actual, info.is_buffer, info.is_output))
    };

    // Generate METAL_ARGS elements as TokenStreams (all args EXCEPT meta)
    let mut metal_args_elements: Vec<proc_macro2::TokenStream> = Vec::new();
    for info in arg_infos.iter().filter(|info| !info.is_meta) {
        let arg_name = &info.name;
        let idx = info.buffer_index;
        let mtype = match require_explicit_metal_type(info) {
            Ok(v) => v,
            Err(err) => return TokenStream::from(err.to_compile_error()),
        };
        metal_args_elements.push(quote! { (#arg_name, #idx, #mtype) });
    }

    // Generate STAGE_METAL_ARGS elements (excluding stage_skip AND meta buffers)
    let mut stage_metal_args_elements: Vec<proc_macro2::TokenStream> = Vec::new();
    for info in arg_infos.iter().filter(|info| !info.stage_skip && !info.is_meta) {
        let arg_name = &info.name;
        let idx = info.buffer_index;
        let mtype = match require_explicit_metal_type(info) {
            Ok(v) => v,
            Err(err) => return TokenStream::from(err.to_compile_error()),
        };
        stage_metal_args_elements.push(quote! { (#arg_name, #idx, #mtype) });
    }

    let debug_binding_elements: Vec<_> = arg_infos
        .iter()
        .filter(|info| !info.is_meta)
        .map(|info| {
            let arg_name = info.name.clone();
            let rust_type = info.rust_type.clone();
            let idx = info.buffer_index;
            let mtype = match require_explicit_metal_type(info) {
                Ok(v) => v,
                Err(err) => return err.to_compile_error(),
            };
            let field = info.name_ident.as_ref().expect("named field");
            if is_tensor_arg(&info.rust_type_actual) {
                if info.is_option {
                    quote! {
                        __debug.push(#root::compound::BindingDebugArg {
                            name: #arg_name.to_string(),
                            buffer_index: #idx,
                            metal_type: #mtype.to_string(),
                            rust_type: #rust_type.to_string(),
                            tensor_dtype: self.#field.as_ref().map(|v| #root::types::KernelArg::dtype(v)),
                            max_linear_index: self.#field.as_ref().and_then(|v| #root::types::kernel_arg_max_linear_index(v)),
                        });
                    }
                } else {
                    quote! {
                        __debug.push(#root::compound::BindingDebugArg {
                            name: #arg_name.to_string(),
                            buffer_index: #idx,
                            metal_type: #mtype.to_string(),
                            rust_type: #rust_type.to_string(),
                            tensor_dtype: Some(#root::types::KernelArg::dtype(&self.#field)),
                            max_linear_index: #root::types::kernel_arg_max_linear_index(&self.#field),
                        });
                    }
                }
            } else {
                quote! {
                    __debug.push(#root::compound::BindingDebugArg {
                        name: #arg_name.to_string(),
                        buffer_index: #idx,
                        metal_type: #mtype.to_string(),
                        rust_type: #rust_type.to_string(),
                        tensor_dtype: None,
                        max_linear_index: None,
                    });
                }
            }
        })
        .collect();

    let runtime_dtype_hash_elements: Vec<_> = arg_infos
        .iter()
        .filter(|info| !info.is_meta && is_tensor_arg(&info.rust_type_actual))
        .map(|info| {
            let idx = info.buffer_index;
            let field = info.name_ident.as_ref().expect("named field");
            if info.is_option {
                quote! {
                    #idx.hash(&mut __hasher);
                    match &self.#field {
                        Some(v) => {
                            true.hash(&mut __hasher);
                            #root::types::KernelArg::dtype(v).hash(&mut __hasher);
                            #root::types::kernel_arg_max_linear_index(v).hash(&mut __hasher);
                        }
                        None => {
                            false.hash(&mut __hasher);
                        }
                    }
                }
            } else {
                quote! {
                    #idx.hash(&mut __hasher);
                    true.hash(&mut __hasher);
                    #root::types::KernelArg::dtype(&self.#field).hash(&mut __hasher);
                    #root::types::kernel_arg_max_linear_index(&self.#field).hash(&mut __hasher);
                }
            }
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

            fn debug_bindings(&self) -> Vec<#root::compound::BindingDebugArg> {
                let mut __debug = Vec::new();
                #(#debug_binding_elements)*
                __debug
            }

            fn runtime_dtype_hash(&self) -> u64 {
                use std::hash::{Hash, Hasher};
                let mut __hasher = rustc_hash::FxHasher::default();
                #(#runtime_dtype_hash_elements)*
                __hasher.finish()
            }
        }
    };

    TokenStream::from(expanded)
}
