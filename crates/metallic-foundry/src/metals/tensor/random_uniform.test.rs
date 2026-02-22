#![cfg(test)]

use super::*;

#[test]
fn test_random_uniform_params_metal_struct() {
    let def = RandomUniformParams::METAL_STRUCT_DEF;
    assert!(def.contains("struct RandomUniformParams"));
    assert!(def.contains("seed"));
    assert!(def.contains("min_val"));
}
