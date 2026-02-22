use super::*;

pub(crate) fn derive_metal_struct(input: TokenStream) -> TokenStream {
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
                    let metal_type = rust_type_to_metal(field_type);
                    metal_fields.push(format!("    {metal_type} {metal_field_name};"));
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

                // Check if this field is a DynamicValue
                if extract_dynamic_value_inner(field_type).is_some() {
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
        let resolved_name = syn::Ident::new(&format!("{name}Resolved"), name.span());

        // Metal struct name for the resolved type (includes Resolved suffix)
        let resolved_metal_struct_name = format!("{metal_struct_name}Resolved");

        // Collect field definitions for the resolved struct
        let mut resolved_field_defs = Vec::new();
        let mut resolve_field_assigns = Vec::new();
        let mut resolved_metal_fields = Vec::new();

        if let Data::Struct(data) = &input.data {
            if let Fields::Named(fields) = &data.fields {
                for field in &fields.named {
                    let field_name = field.ident.as_ref().unwrap();
                    let field_type = &field.ty;

                    // Check if this field is a DynamicValue<T>
                    if let Some(inner_type) = extract_dynamic_value_inner(field_type) {
                        // Resolved struct has concrete type
                        resolved_field_defs.push(quote! {
                            pub #field_name: #inner_type
                        });

                        // Resolve field by calling .resolve(bindings)
                        resolve_field_assigns.push(quote! {
                            #field_name: self.#field_name.resolve(bindings)
                        });

                        // Metal field uses inner type
                        let inner_metal_type = rust_type_to_metal(&inner_type);
                        resolved_metal_fields.push(format!("    {inner_metal_type} {field_name};"));
                    } else {
                        // Non-dynamic field - same type in resolved struct, clone value
                        resolved_field_defs.push(quote! {
                            pub #field_name: #field_type
                        });

                        resolve_field_assigns.push(quote! {
                            #field_name: self.#field_name.clone()
                        });

                        // Use same Metal type
                        let metal_type = rust_type_to_metal(field_type);
                        resolved_metal_fields.push(format!("    {metal_type} {field_name};"));
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
