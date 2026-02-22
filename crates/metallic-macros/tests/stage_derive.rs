#[test]
fn stage_derive_dynamic_bindings_compile() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/stage/pass_dynamic_bindings.rs");
}
