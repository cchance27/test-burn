#![cfg(test)]

use super::*;

#[test]
fn test_repeat_kv_heads_params_metal_struct() {
    let def = RepeatKvHeadsParams::METAL_STRUCT_DEF;
    assert!(def.contains("struct RepeatKvHeadsParams"));
    assert!(def.contains("group_size"));
    assert!(def.contains("n_heads"));
    assert!(def.contains("cache_stride"));
}
