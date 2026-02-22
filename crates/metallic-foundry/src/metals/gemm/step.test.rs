#![cfg(test)]

use super::*;

#[test]
fn test_gemm_params_simple() {
    let params = GemmParams::simple(128, 256, 64, false, false, TileConfig::Default);
    assert_eq!(params.m, 128);
    assert_eq!(params.n, 256);
    assert_eq!(params.k, 64);
    assert_eq!(params.tiles_m, 4); // 128 / 32
    assert_eq!(params.tiles_n, 8); // 256 / 32
    assert_eq!(params.gemm_k_iterations, 4); // 64 / 16
    assert_eq!(params.gemm_k_remainder, 0);
}

#[test]
fn test_gemm_params_unaligned() {
    let params = GemmParams::simple(100, 200, 50, false, false, TileConfig::Default);
    assert_eq!(params.tiles_m, 4); // ceil(100 / 32)
    assert_eq!(params.tiles_n, 7); // ceil(200 / 32)
    assert_eq!(params.gemm_k_iterations, 3); // 50 / 16
    assert_eq!(params.gemm_k_remainder, 2); // 50 % 16
}
