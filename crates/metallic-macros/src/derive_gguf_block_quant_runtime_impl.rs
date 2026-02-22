use super::*;

pub(crate) fn derive_gguf_block_quant_runtime(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let root = foundry_crate();
    let loader = loader_crate();

    let mut source_dtype_expr: Option<Expr> = None;
    let mut scales_dtype_expr: Option<Expr> = None;
    let mut spec_expr: Option<Expr> = None;
    let mut weights_per_block_expr: Option<Expr> = None;
    let mut block_bytes_expr: Option<Expr> = None;
    let mut scale_bytes_expr: Option<Expr> = None;
    let mut data_bytes_expr: Option<Expr> = None;
    let mut packed_axis_expr: Option<Expr> = None;
    let mut scale_model_expr: Option<Expr> = None;
    let mut write_block_expr: Option<Expr> = None;

    for attr in &input.attrs {
        if !attr.path().is_ident("gguf_runtime") {
            continue;
        }
        let nested = match attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
            Ok(v) => v,
            Err(e) => return e.to_compile_error().into(),
        };

        for meta in nested {
            if let Meta::NameValue(nv) = meta {
                if nv.path.is_ident("source_dtype") {
                    source_dtype_expr = Some(nv.value);
                } else if nv.path.is_ident("scales_dtype") {
                    scales_dtype_expr = Some(nv.value);
                } else if nv.path.is_ident("spec") {
                    spec_expr = Some(nv.value);
                } else if nv.path.is_ident("weights_per_block") {
                    weights_per_block_expr = Some(nv.value);
                } else if nv.path.is_ident("block_bytes") {
                    block_bytes_expr = Some(nv.value);
                } else if nv.path.is_ident("scale_bytes") {
                    scale_bytes_expr = Some(nv.value);
                } else if nv.path.is_ident("data_bytes") {
                    data_bytes_expr = Some(nv.value);
                } else if nv.path.is_ident("packed_axis") {
                    packed_axis_expr = Some(nv.value);
                } else if nv.path.is_ident("scale_model") {
                    scale_model_expr = Some(nv.value);
                } else if nv.path.is_ident("write_block") {
                    write_block_expr = Some(nv.value);
                }
            }
        }
    }

    let source_dtype = match source_dtype_expr {
        Some(v) => v,
        None => {
            return syn::Error::new(input.span(), "missing gguf_runtime(source_dtype = ...)")
                .to_compile_error()
                .into();
        }
    };
    let scales_dtype = match scales_dtype_expr {
        Some(v) => v,
        None => {
            return syn::Error::new(input.span(), "missing gguf_runtime(scales_dtype = ...)")
                .to_compile_error()
                .into();
        }
    };
    let scale_bytes = match scale_bytes_expr {
        Some(v) => v,
        None => {
            return syn::Error::new(input.span(), "missing gguf_runtime(scale_bytes = ...)")
                .to_compile_error()
                .into();
        }
    };
    let data_bytes = match data_bytes_expr {
        Some(v) => v,
        None => {
            return syn::Error::new(input.span(), "missing gguf_runtime(data_bytes = ...)")
                .to_compile_error()
                .into();
        }
    };
    let write_block = match write_block_expr {
        Some(v) => v,
        None => {
            return syn::Error::new(input.span(), "missing gguf_runtime(write_block = path::to::fn)")
                .to_compile_error()
                .into();
        }
    };

    let (weights_per_block_tokens, block_bytes_tokens) = if let Some(spec) = spec_expr {
        (quote! { (#spec).weights_per_block }, quote! { (#spec).block_bytes })
    } else {
        let wpb = match weights_per_block_expr {
            Some(v) => v,
            None => {
                return syn::Error::new(
                    input.span(),
                    "missing gguf_runtime(weights_per_block = ...) or gguf_runtime(spec = ...)",
                )
                .to_compile_error()
                .into();
            }
        };
        let bbytes = match block_bytes_expr {
            Some(v) => v,
            None => {
                return syn::Error::new(input.span(), "missing gguf_runtime(block_bytes = ...) or gguf_runtime(spec = ...)")
                    .to_compile_error()
                    .into();
            }
        };
        (quote! { #wpb }, quote! { #bbytes })
    };

    let packed_axis = packed_axis_expr.unwrap_or_else(|| syn::parse_quote! { #root::policy::block_quant::PackedAxis::Dim1 });
    let scale_model = scale_model_expr.unwrap_or_else(|| syn::parse_quote! { #root::policy::block_quant::ScaleModel::ScaleOnly });

    let expanded = quote! {
        const _: fn(&[u8], &mut [u8]) = #write_block;

        impl #root::policy::block_quant::BlockQuantCodec for #name {
            const SOURCE_DTYPE: #root::tensor::Dtype = #source_dtype;
            const SCALES_DTYPE: #root::tensor::Dtype = #scales_dtype;
            const WEIGHTS_PER_BLOCK: usize = #weights_per_block_tokens;
            const BLOCK_BYTES: usize = #block_bytes_tokens;
            const SCALE_BYTES: usize = #scale_bytes;
            const DATA_BYTES: usize = #data_bytes;
            const LAYOUT: #root::policy::block_quant::BlockQuantLayout = #root::policy::block_quant::BlockQuantLayout {
                packed_axis: #packed_axis,
                scale_model: #scale_model,
            };

            #[inline]
            fn write_block(qs: &[u8], out: &mut [u8]) {
                #write_block(qs, out);
            }
        }

        impl #root::policy::LoaderStage for #name {
            fn params_struct(&self) -> String {
                "".to_string()
            }

            fn bind(
                &self,
                fast_bindings: &#root::spec::FastBindings,
                resolved: &#root::spec::ResolvedSymbols,
            ) -> ::smallvec::SmallVec<[#root::types::TensorArg; 4]> {
                use ::smallvec::smallvec;
                let weight = fast_bindings.get(resolved.weights).expect("quant weight bound");
                let scales_idx = resolved.scales.expect("quant scales index missing");
                let scales = fast_bindings.get(scales_idx).expect("quant scales bound");
                smallvec![weight.clone(), scales.clone()]
            }

            fn quantization_type(&self) -> std::sync::Arc<dyn #root::policy::MetalPolicyRuntime> {
                std::sync::Arc::new(<#name>::default())
            }
        }

        impl #root::policy::MetalPolicyRuntime for #name {
            fn loader_stage(&self) -> Box<dyn #root::policy::LoaderStage> {
                Box::new(<#name>::default())
            }

            fn load_weights(
                &self,
                foundry: &mut #root::Foundry,
                model: &dyn #loader::LoadedModel,
                source_tensor_name: &str,
                logical_name: &str,
                layout: #root::compound::Layout,
            ) -> ::anyhow::Result<Vec<(String, #root::types::TensorArg)>> {
                let loaded = #root::policy::block_quant::load_block_quant_2d_with_codec::<#name>(foundry, model, source_tensor_name, layout)?;
                Ok(vec![
                    (logical_name.to_string(), loaded.weights),
                    (format!("{}_scales", logical_name), loaded.scales),
                ])
            }
        }
    };

    expanded.into()
}
