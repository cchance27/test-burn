use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseBuffer, ParseStream};
use syn::parse_macro_input;
use syn::spanned::Spanned;
use syn::{braced, Expr, ExprClosure, Ident, LitStr, Result, Token, Type};

mod kw {
    syn::custom_keyword!(library);
}

struct KernelMacroInput {
    library_ident: Ident,
    source: Expr,
    functions: Vec<FunctionSpec>,
    operations: Vec<OperationSpec>,
}

struct FunctionSpec {
    ident: Ident,
    mappings: Vec<DtypeMapping>,
}

struct DtypeMapping {
    dtype: Ident,
    name: LitStr,
}

struct OperationSpec {
    ident: Ident,
    function_expr: Expr,
    args_ty: Type,
    pipeline: PipelineKind,
    state_fields: Vec<StateField>,
    new_closure: ExprClosure,
    encode_closure: ExprClosure,
}

struct StateField {
    ident: Ident,
    ty: Type,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineKind {
    Required,
    Optional,
}

impl Parse for KernelMacroInput {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse::<kw::library>()?;
        let library_ident: Ident = input.parse()?;
        let content;
        braced!(content in input);

        let mut source: Option<Expr> = None;
        let mut functions: Option<Vec<FunctionSpec>> = None;
        let mut operations: Vec<OperationSpec> = Vec::new();

        while !content.is_empty() {
            let key: Ident = content.parse()?;
            content.parse::<Token![:]>()?;
            match key.to_string().as_str() {
                "source" => {
                    if source.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `source` entry"));
                    }
                    source = Some(content.parse()?);
                }
                "functions" => {
                    if functions.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `functions` entry"));
                    }
                    let block;
                    braced!(block in content);
                    functions = Some(parse_functions(&block)?);
                }
                "operations" => {
                    let block;
                    braced!(block in content);
                    operations = parse_operations(&block)?;
                }
                other => {
                    return Err(syn::Error::new(Span::call_site(), format!("unknown key `{}`", other)));
                }
            }

            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }

        let source = source.ok_or_else(|| syn::Error::new(Span::call_site(), "missing `source` entry"))?;
        let functions = functions.ok_or_else(|| syn::Error::new(Span::call_site(), "missing `functions` entry"))?;

        Ok(KernelMacroInput {
            library_ident,
            source,
            functions,
            operations,
        })
    }
}

fn parse_functions(input: &ParseBuffer) -> Result<Vec<FunctionSpec>> {
    let mut functions = Vec::new();
    while !input.is_empty() {
        let ident: Ident = input.parse()?;
        input.parse::<Token![=>]>()?;
        let block;
        braced!(block in input);
        let mappings = parse_dtype_mappings(&block)?;
        functions.push(FunctionSpec { ident, mappings });

        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
    }
    Ok(functions)
}

fn parse_dtype_mappings(input: &ParseBuffer) -> Result<Vec<DtypeMapping>> {
    let mut mappings = Vec::new();
    while !input.is_empty() {
        let dtype: Ident = input.parse()?;
        input.parse::<Token![=>]>()?;
        let name: LitStr = input.parse()?;
        mappings.push(DtypeMapping { dtype, name });
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
    }
    Ok(mappings)
}

fn parse_operations(input: &ParseBuffer) -> Result<Vec<OperationSpec>> {
    let mut operations = Vec::new();
    while !input.is_empty() {
        let ident: Ident = input.parse()?;
        input.parse::<Token![=>]>()?;
        let block;
        braced!(block in input);
        operations.push(parse_operation_body(ident, &block)?);
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
    }
    Ok(operations)
}

fn parse_operation_body(ident: Ident, input: &ParseBuffer) -> Result<OperationSpec> {
    let mut function_expr: Option<Expr> = None;
    let mut args_ty: Option<Type> = None;
    let mut pipeline: Option<PipelineKind> = None;
    let mut state_fields: Option<Vec<StateField>> = None;
    let mut new_closure: Option<ExprClosure> = None;
    let mut encode_closure: Option<ExprClosure> = None;

    while !input.is_empty() {
        let key: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        match key.to_string().as_str() {
            "function" => {
                if function_expr.is_some() {
                    return Err(syn::Error::new(key.span(), "duplicate `function` entry"));
                }
                function_expr = Some(input.parse()?);
            }
            "args" => {
                if args_ty.is_some() {
                    return Err(syn::Error::new(key.span(), "duplicate `args` entry"));
                }
                args_ty = Some(input.parse()?);
            }
            "pipeline" => {
                if pipeline.is_some() {
                    return Err(syn::Error::new(key.span(), "duplicate `pipeline` entry"));
                }
                let mode: Ident = input.parse()?;
                pipeline = match mode.to_string().as_str() {
                    "required" => Some(PipelineKind::Required),
                    "optional" => Some(PipelineKind::Optional),
                    other => {
                        return Err(syn::Error::new(
                            mode.span(),
                            format!("expected `required` or `optional`, found `{}`", other),
                        ));
                    }
                };
            }
            "state" => {
                if state_fields.is_some() {
                    return Err(syn::Error::new(key.span(), "duplicate `state` entry"));
                }
                let block;
                braced!(block in input);
                state_fields = Some(parse_state_fields(&block)?);
            }
            "new" => {
                if new_closure.is_some() {
                    return Err(syn::Error::new(key.span(), "duplicate `new` entry"));
                }
                new_closure = Some(input.parse()?);
            }
            "encode" => {
                if encode_closure.is_some() {
                    return Err(syn::Error::new(key.span(), "duplicate `encode` entry"));
                }
                encode_closure = Some(input.parse()?);
            }
            other => {
                return Err(syn::Error::new(Span::call_site(), format!("unknown key `{}`", other)));
            }
        }

        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
    }

    let function_expr = function_expr.ok_or_else(|| syn::Error::new(ident.span(), "operation is missing a `function` entry"))?;
    let args_ty = args_ty.ok_or_else(|| syn::Error::new(ident.span(), "operation is missing an `args` entry"))?;
    let pipeline = pipeline.ok_or_else(|| syn::Error::new(ident.span(), "operation is missing a `pipeline` entry"))?;
    let state_fields = state_fields.unwrap_or_default();
    let new_closure = new_closure.ok_or_else(|| syn::Error::new(ident.span(), "operation is missing a `new` entry"))?;
    let encode_closure = encode_closure.ok_or_else(|| syn::Error::new(ident.span(), "operation is missing an `encode` entry"))?;

    Ok(OperationSpec {
        ident,
        function_expr,
        args_ty,
        pipeline,
        state_fields,
        new_closure,
        encode_closure,
    })
}

fn parse_state_fields(input: &ParseBuffer) -> Result<Vec<StateField>> {
    let mut fields = Vec::new();
    while !input.is_empty() {
        let ident: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty: Type = input.parse()?;
        fields.push(StateField { ident, ty });
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
    }
    Ok(fields)
}

#[proc_macro]
pub fn metal_kernel(tokens: TokenStream) -> TokenStream {
    let input = parse_macro_input!(tokens as KernelMacroInput);
    match expand_kernel(input) {
        Ok(stream) => stream.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_kernel(input: KernelMacroInput) -> Result<TokenStream2> {
    let library_ident = input.library_ident;
    let source_expr = input.source;

    let library_descriptor_ident = format_ident!("{}Descriptor", library_ident);

    let function_defs: Vec<TokenStream2> = input
        .functions
        .iter()
        .map(|spec| expand_function_spec(&library_descriptor_ident, spec))
        .collect::<Result<Vec<_>>>()?;

    let operation_defs: Vec<TokenStream2> = input.operations.iter().map(expand_operation_spec).collect::<Result<Vec<_>>>()?;

    let expanded = quote! {
        #[allow(non_upper_case_globals)]
        pub static #library_descriptor_ident: crate::metallic::kernels::KernelDescriptor = crate::metallic::kernels::KernelDescriptor {
            id: stringify!(#library_ident),
            source: #source_expr,
        };

        #(#function_defs)*
        #(#operation_defs)*
    };

    Ok(expanded)
}

fn expand_function_spec(library_descriptor_ident: &Ident, spec: &FunctionSpec) -> Result<TokenStream2> {
    let helper_fn_ident = format_ident!("__{}_name_for_dtype", spec.ident);
    let descriptor_ident = format_ident!("{}Descriptor", spec.ident);
    let function_ident = &spec.ident;

    let mut match_arms = Vec::new();
    for mapping in &spec.mappings {
        let dtype_ident = &mapping.dtype;
        let kernel_name = &mapping.name;
        match_arms.push(quote! {
            crate::metallic::Dtype::#dtype_ident => Some(#kernel_name),
        });
    }
    match_arms.push(quote! {
        _ => None,
    });

    Ok(quote! {
        fn #helper_fn_ident(dtype: crate::metallic::Dtype) -> Option<&'static str> {
            match dtype {
                #(#match_arms)*
            }
        }

        #[allow(non_upper_case_globals)]
        pub static #descriptor_ident: crate::metallic::kernels::KernelFunctionDescriptor = crate::metallic::kernels::KernelFunctionDescriptor {
            id: stringify!(#function_ident),
            library: &#library_descriptor_ident,
            name_for_dtype: #helper_fn_ident,
        };
    })
}

fn expand_operation_spec(spec: &OperationSpec) -> Result<TokenStream2> {
    let state_ident = format_ident!("{}State", spec.ident);
    let op_ident = &spec.ident;
    let function_expr = &spec.function_expr;
    let args_ty = &spec.args_ty;
    let new_closure = &spec.new_closure;
    let encode_closure = &spec.encode_closure;

    if new_closure.inputs.len() != 4 {
        return Err(syn::Error::new(
            new_closure.span(),
            "`new` closure must take exactly four arguments: (ctx, args, pipeline, cache)",
        ));
    }
    if encode_closure.inputs.len() != 3 {
        return Err(syn::Error::new(
            encode_closure.span(),
            "`encode` closure must take exactly three arguments: (command_buffer, cache, state)",
        ));
    }

    let mut new_inputs = new_closure.inputs.iter();
    let ctx_pat = new_inputs.next().unwrap();
    let args_pat = new_inputs.next().unwrap();
    let pipeline_pat = new_inputs.next().unwrap();
    let cache_pat = new_inputs.next().unwrap();

    let mut encode_inputs = encode_closure.inputs.iter();
    let command_buffer_pat = encode_inputs.next().unwrap();
    let encode_cache_pat = encode_inputs.next().unwrap();
    let state_pat = encode_inputs.next().unwrap();

    let pipeline_setup = match spec.pipeline {
        PipelineKind::Required => {
            quote! {
                let pipeline = pipeline.ok_or_else(|| crate::metallic::MetalError::OperationNotSupported(
                    concat!(stringify!(#op_ident), " requires a Metal pipeline").to_string(),
                ))?;
                let pipeline_value = pipeline;
            }
        }
        PipelineKind::Optional => {
            quote! {
                let pipeline_value = pipeline;
            }
        }
    };

    let state_fields = spec.state_fields.iter().map(|field| {
        let ident = &field.ident;
        let ty = &field.ty;
        quote! { pub #ident: #ty }
    });

    let new_body = &new_closure.body;
    let encode_body = &encode_closure.body;

    Ok(quote! {
        pub struct #op_ident;

        pub struct #state_ident<T: crate::metallic::TensorElement> {
            #(#state_fields,)*
        }

        impl crate::metallic::kernels::KernelInvocable for #op_ident {
            type Args<'a, T: crate::metallic::TensorElement> = #args_ty;

            fn function_id() -> Option<crate::metallic::kernels::KernelFunction> {
                #function_expr
            }

            fn new<'a, T: crate::metallic::TensorElement>(
                ctx: &mut crate::metallic::Context<T>,
                args: Self::Args<'a, T>,
                pipeline: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>>,
                cache: Option<&mut crate::metallic::resource_cache::ResourceCache>,
            ) -> Result<(Box<dyn crate::metallic::Operation>, crate::metallic::Tensor<T>), crate::metallic::MetalError> {
                #pipeline_setup
                let result: Result<(#state_ident<T>, crate::metallic::Tensor<T>), crate::metallic::MetalError> = {
                    let #ctx_pat = ctx;
                    let #args_pat = args;
                    let #pipeline_pat = pipeline_value;
                    let #cache_pat = cache;
                    #new_body
                };
                let (state, output) = result?;
                Ok((Box::new(state), output))
            }
        }

        impl<T: crate::metallic::TensorElement> crate::metallic::Operation for #state_ident<T> {
            fn encode(
                &self,
                command_buffer: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
                cache: &mut crate::metallic::resource_cache::ResourceCache,
            ) -> Result<(), crate::metallic::MetalError> {
                let #command_buffer_pat = command_buffer;
                let #encode_cache_pat = cache;
                let #state_pat = self;
                #encode_body
            }
        }
    })
}
