#![cfg(test)]
use crate::kernels::matmul_dispatcher::prefs::load_prefs_from_env;
use crate::kernels::matmul_dispatcher::types::MatmulBackend;
use std::env;

#[test]
fn env_backend_parsing() {
    unsafe { env::set_var("METALLIC_MATMUL_BACKEND", "mlx"); }
    let p = load_prefs_from_env();
    assert!(matches!(p.backend, MatmulBackend::Mlx));
    unsafe { env::set_var("METALLIC_MATMUL_BACKEND", "mps"); }  
    let p = load_prefs_from_env();
    assert!(matches!(p.backend, MatmulBackend::Mps));
    unsafe { env::set_var("METALLIC_MATMUL_BACKEND", "auto"); }
    let p = load_prefs_from_env();
    assert!(matches!(p.backend, MatmulBackend::Auto));
}

#[test]
fn env_force_smalln_parsing() {
    unsafe { env::remove_var("METALLIC_MATMUL_FORCE_SMALLN"); }
    let p = load_prefs_from_env();
    assert!(!p.force_smalln);
    unsafe { env::set_var("METALLIC_MATMUL_FORCE_SMALLN", "1"); }
    let p = load_prefs_from_env();
    assert!(p.force_smalln);
    unsafe { env::set_var("METALLIC_MATMUL_FORCE_SMALLN", "true"); }
    let p = load_prefs_from_env();
    assert!(p.force_smalln);
}
