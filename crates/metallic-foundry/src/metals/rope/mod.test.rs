#![cfg(test)]

use super::*;

#[test]
fn test_rope_params_metal_struct() {
    let def = RopeParams::METAL_STRUCT_DEF;
    assert!(def.contains("struct RopeParams"));
    assert!(def.contains("dim"));
    assert!(def.contains("seq_len"));
    assert!(def.contains("position_offset"));
    assert!(def.contains("total_elements"));
}
