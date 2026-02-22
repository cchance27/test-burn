#![cfg(test)]

use super::*;

#[test]
fn test_kv_rearrange_params_metal_struct() {
    let def = KvRearrangeParams::METAL_STRUCT_DEF;
    assert!(def.contains("struct KvRearrangeParams"));
    assert!(def.contains("kv_dim"));
    assert!(def.contains("n_heads"));
    assert!(def.contains("n_kv_heads"));
}
