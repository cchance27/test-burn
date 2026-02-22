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

    // Generate METAL_ARGS elements as TokenStreams (all args EXCEPT meta)
    let metal_args_elements: Vec<_> = arg_infos
        .iter()
        .filter(|info| !info.is_meta)
        .map(|info| {
            let arg_name = &info.name;
            let idx = info.buffer_index;
            // Use explicit metal_type if specified, otherwise infer from Rust type
            let mtype = info
                .metal_type
                .clone()
                .unwrap_or_else(|| infer_metal_type(&info.rust_type_actual, info.is_buffer, info.is_output));
            quote! { (#arg_name, #idx, #mtype) }
        })
        .collect();

    // Generate STAGE_METAL_ARGS elements (excluding stage_skip AND meta buffers)
    let stage_metal_args_elements: Vec<_> = arg_infos
        .iter()
        .filter(|info| !info.stage_skip && !info.is_meta)
        .map(|info| {
            let arg_name = &info.name;
            let idx = info.buffer_index;
            let mtype = info
                .metal_type
                .clone()
                .unwrap_or_else(|| infer_metal_type(&info.rust_type_actual, info.is_buffer, info.is_output));
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
