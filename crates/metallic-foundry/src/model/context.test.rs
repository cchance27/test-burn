#![cfg(test)]

use metallic_env::{EnvVarGuard, FoundryEnvVar};

use super::*;
use crate::spec::Architecture;

fn mock_arch(max_seq_len: usize) -> Architecture {
    use crate::spec::ArchValue;
    let mut params = rustc_hash::FxHashMap::default();
    params.insert("d_model".to_string(), ArchValue::USize(512));
    params.insert("n_heads".to_string(), ArchValue::USize(8));
    params.insert("n_kv_heads".to_string(), ArchValue::USize(2));
    params.insert("n_layers".to_string(), ArchValue::USize(4));
    params.insert("ff_dim".to_string(), ArchValue::USize(2048));
    params.insert("vocab_size".to_string(), ArchValue::USize(1000));
    params.insert("max_seq_len".to_string(), ArchValue::USize(max_seq_len));
    params.insert("rope_base".to_string(), ArchValue::F32(10000.0));
    params.insert("rms_eps".to_string(), ArchValue::F32(1e-6));

    Architecture {
        params,
        tensor_names: Default::default(),
        metadata_keys: Default::default(),
        prepare: Default::default(),
        weight_bindings: Vec::new(),
        forward: Vec::new(),
    }
}

#[test]
#[serial_test::serial]
fn test_context_config_priority() {
    let arch = mock_arch(4096);

    // 1. Default to model max
    let config = ContextConfig::from_architecture(&arch, None);
    assert_eq!(config.max_context_len, 4096);

    // 2. DSL/runtime override acts as a cap
    let config2 = ContextConfig::from_architecture(&arch, Some(2048));
    assert_eq!(config2.max_context_len, 2048);

    // 3. Env var acts as a cap
    let _max_context_guard = EnvVarGuard::set(FoundryEnvVar::MaxContextLen, "1024");
    let config3 = ContextConfig::from_architecture(&arch, None);
    assert_eq!(config3.max_context_len, 1024);
}

#[test]
fn test_alignment() {
    assert_eq!(ContextConfig::align_capacity(0), 128);
    assert_eq!(ContextConfig::align_capacity(1), 128);
    assert_eq!(ContextConfig::align_capacity(127), 128);
    assert_eq!(ContextConfig::align_capacity(128), 128);
    assert_eq!(ContextConfig::align_capacity(129), 256);
    assert_eq!(ContextConfig::align_capacity(200), 256);
}

#[test]
fn test_memory_estimation() {
    let arch = mock_arch(2048);
    let est = ContextConfig::estimate_kv_memory(&arch, 2048);
    // Compact GQA (n_kv_heads=2): 4 * 2 * 2 * 2048 * 64 * 2 = 4,194,304 bytes
    assert_eq!(est.kv_cache_bytes, 4194304);
}
