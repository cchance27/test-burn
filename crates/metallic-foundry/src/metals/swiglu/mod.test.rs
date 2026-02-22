#![cfg(test)]

use super::*;

#[test]
fn test_swiglu_params_metal_struct() {
    let def = SwigluParams::METAL_STRUCT_DEF;
    assert!(def.contains("struct SwigluParams"));
    assert!(def.contains("total_elements"));
    assert!(def.contains("bias_len"));
    assert!(def.contains("vector_width"));
}
