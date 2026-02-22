#![cfg(test)]

use super::*;

#[test]
fn test_elemwise_add_params_metal_struct() {
    let def = ElemwiseAddParams::METAL_STRUCT_DEF;
    assert!(def.contains("struct ElemwiseAddParams"));
    assert!(def.contains("total_elements"));
}
