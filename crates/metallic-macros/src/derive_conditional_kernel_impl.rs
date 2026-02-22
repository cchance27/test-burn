use super::*;

pub(crate) fn derive_conditional_kernel(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    conditional::derive_conditional_kernel_impl(input).into()
}
