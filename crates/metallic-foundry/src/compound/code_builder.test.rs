#![cfg(test)]

use super::*;

#[test]
fn test_metal_type_display() {
    assert_eq!(MetalType::Half.as_str(), "half");
    assert_eq!(MetalType::Float4.as_str(), "float4");
    assert_eq!(MetalType::Custom("MyStruct").as_str(), "MyStruct");
}

#[test]
fn test_metal_type_pointer_declaration() {
    let ptr = MetalType::DevicePtr(MetalScalar::Half);
    assert_eq!(ptr.declaration_str(), "device half*");

    let const_ptr = MetalType::ConstantPtr(MetalScalar::Float);
    assert_eq!(const_ptr.declaration_str(), "constant float*");
}

#[test]
fn test_metal_var_declaration() {
    let var = MetalVar {
        name: "test_var".to_string(),
        metal_type: MetalType::Float,
    };
    assert_eq!(var.declaration(), "float test_var;");
    assert_eq!(var.declaration_with_init("1.0f"), "float test_var = 1.0f;");
}

#[test]
fn test_code_builder_basic() {
    let mut builder = CodeBuilder::new("rmsnorm");
    let inv_rms = builder.declare_var("inv_rms", MetalType::Float);
    assert_eq!(inv_rms.name(), "rmsnorm_inv_rms1");
    assert_eq!(inv_rms.metal_type(), MetalType::Float);

    builder.set_output_var(&inv_rms);
    builder.emit_comment("test comment");
    builder.emit_decl_init(&inv_rms, "1.0f");

    let (out, code) = builder.finish();
    assert!(code.contains("// test comment"));
    assert!(code.contains("float rmsnorm_inv_rms1 = 1.0f;"));
    assert_eq!(out, inv_rms.name());
}

#[test]
fn test_simd_reduce_config_levels() {
    let config = SimdReduceConfig::default();
    let levels: Vec<u8> = config.levels().collect();
    assert_eq!(levels, vec![16, 8, 4, 2, 1]);

    let config_16 = SimdReduceConfig::lane_16(ReduceOp::Add);
    let levels_16: Vec<u8> = config_16.levels().collect();
    assert_eq!(levels_16, vec![8, 4, 2, 1]);

    let config_single = SimdReduceConfig::new(4, 4, ReduceOp::Max);
    let levels_single: Vec<u8> = config_single.levels().collect();
    assert_eq!(levels_single, vec![4]);
}

#[test]
fn test_simd_reduce_default() {
    let mut builder = CodeBuilder::new("test");
    builder.emit_simd_reduce("acc");
    let code = builder.code();
    assert!(code.contains("acc += simd_shuffle_xor(acc, 16u)"));
    assert!(code.contains("acc += simd_shuffle_xor(acc, 8u)"));
    assert!(code.contains("acc += simd_shuffle_xor(acc, 1u)"));
}

#[test]
fn test_simd_reduce_max() {
    let mut builder = CodeBuilder::new("test");
    let config = SimdReduceConfig::full_32_lane(ReduceOp::Max);
    builder.emit_simd_reduce_with_config("val", config);
    let code = builder.code();
    assert!(code.contains("val = max(val, simd_shuffle_xor(val, 16u))"));
    assert!(code.contains("val = max(val, simd_shuffle_xor(val, 1u))"));
}

#[test]
fn test_simd_reduce_multi() {
    let mut builder = CodeBuilder::new("test");
    let config = SimdReduceConfig::new(4, 1, ReduceOp::Add);
    builder.emit_simd_reduce_multi(&["a", "b"], config);
    let code = builder.code();
    // Should interleave reductions for a and b at each level
    assert!(code.contains("a += simd_shuffle_xor(a, 4u)"));
    assert!(code.contains("b += simd_shuffle_xor(b, 4u)"));
    assert!(code.contains("a += simd_shuffle_xor(a, 1u)"));
    assert!(code.contains("b += simd_shuffle_xor(b, 1u)"));
}

#[test]
fn test_external_var() {
    let mut builder = CodeBuilder::new("stage");
    let params = builder.external_var("params", MetalType::ConstantPtr(MetalScalar::Float));
    assert_eq!(params.name(), "params");
    assert!(builder.has_var("params"));
}

#[test]
fn test_reduce_op_combine() {
    assert_eq!(ReduceOp::Add.combine_expr("x", "y"), "x += y");
    assert_eq!(ReduceOp::Max.combine_expr("x", "y"), "x = max(x, y)");
    assert_eq!(ReduceOp::Min.combine_expr("x", "y"), "x = min(x, y)");
}
