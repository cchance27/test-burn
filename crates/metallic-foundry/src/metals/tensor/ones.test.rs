#![cfg(test)]

use super::*;

#[test]
fn test_ones_params_metal_struct() {
    let def = OnesParams::METAL_STRUCT_DEF;
    assert!(def.contains("struct OnesParams"));
    assert!(def.contains("total_elements"));
}
