#![cfg(test)]
use crate::kernels::matmul_dispatcher::prefs::load_prefs_from_env;
use crate::kernels::matmul_dispatcher::types::{MatmulBackend, Prefs};
use std::env;

#[test]
fn env_backend_parsing() {
    env::set_var("METALLIC_MATMUL_BACKEND", "mlx");
    let p = load_prefs_from_env();
    assert!(matches!(p.backend, MatmulBackend::Mlx));
    env::set_var("METALLIC_MATMUL_BACKEND", "mps");
    let p = load_prefs_from_env();
    assert!(matches!(p.backend, MatmulBackend::Mps));
    env::set_var("METALLIC_MATMUL_BACKEND", "auto");
    let p = load_prefs_from_env();
    assert!(matches!(p.backend, MatmulBackend::Auto));
}

#[test]
fn env_force_smalln_parsing() {
    env::remove_var("METALLIC_MATMUL_FORCE_SMALLN");
    let p = load_prefs_from_env();
    assert!(!p.force_smalln);
    env::set_var("METALLIC_MATMUL_FORCE_SMALLN", "1");
    let p = load_prefs_from_env();
    assert!(p.force_smalln);
    env::set_var("METALLIC_MATMUL_FORCE_SMALLN", "true");
    let p = load_prefs_from_env();
    assert!(p.force_smalln);
}
