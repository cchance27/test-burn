#![cfg(test)]

use metallic_env::{EnvVarGuard, FoundryEnvVar};

use super::*;
use crate::spec::{Architecture, ModelSpec};

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

#[test]
fn test_memory_estimation_uses_prepare_tensor_dtypes_f32() {
    let spec = ModelSpec::from_json(
        r#"
        {
          "name": "ctx-f32-kv-estimate",
          "architecture": {
            "d_model": 512,
            "n_heads": 8,
            "n_kv_heads": 2,
            "n_layers": 4,
            "ff_dim": 2048,
            "vocab_size": 1000,
            "max_seq_len": 4096,
            "rope_base": 10000.0,
            "rms_eps": 0.000001,
            "prepare": {
              "tensors": [
                {
                  "name": "k_cache_{i}",
                  "repeat": { "count": "n_layers", "var": "i" },
                  "dtype": "F32",
                  "storage": "kv_cache",
                  "dims": ["n_kv_heads", "max_seq_len", "d_model / n_heads"],
                  "grow_with_kv": true
                },
                {
                  "name": "v_cache_{i}",
                  "repeat": { "count": "n_layers", "var": "i" },
                  "dtype": "F32",
                  "storage": "kv_cache",
                  "dims": ["n_kv_heads", "max_seq_len", "d_model / n_heads"],
                  "grow_with_kv": true
                }
              ]
            },
            "forward": []
          }
        }
        "#,
    )
    .expect("parse spec");

    let est = ContextConfig::estimate_kv_memory(&spec.architecture, 1024);
    assert_eq!(est.per_layer_bytes, 1_048_576);
    assert_eq!(est.kv_cache_bytes, 4_194_304);
}

#[test]
fn test_memory_estimation_uses_prepare_tensor_dtypes_i8_with_scales() {
    let spec = ModelSpec::from_json(
        r#"
        {
          "name": "ctx-int8-scale-kv-estimate",
          "architecture": {
            "d_model": 512,
            "n_heads": 8,
            "n_kv_heads": 2,
            "n_layers": 2,
            "ff_dim": 2048,
            "vocab_size": 1000,
            "max_seq_len": 4096,
            "rope_base": 10000.0,
            "rms_eps": 0.000001,
            "prepare": {
              "tensors": [
                {
                  "name": "k_cache_data_{i}",
                  "repeat": { "count": "n_layers", "var": "i" },
                  "dtype": "I8",
                  "storage": "kv_cache",
                  "dims": ["n_kv_heads", "max_seq_len", "d_model / n_heads"],
                  "grow_with_kv": true
                },
                {
                  "name": "v_cache_data_{i}",
                  "repeat": { "count": "n_layers", "var": "i" },
                  "dtype": "I8",
                  "storage": "kv_cache",
                  "dims": ["n_kv_heads", "max_seq_len", "d_model / n_heads"],
                  "grow_with_kv": true
                },
                {
                  "name": "k_cache_scale_{i}",
                  "repeat": { "count": "n_layers", "var": "i" },
                  "dtype": "F16",
                  "storage": "kv_cache",
                  "dims": ["n_kv_heads", "max_seq_len", "(d_model / n_heads) / 32"],
                  "grow_with_kv": true
                },
                {
                  "name": "v_cache_scale_{i}",
                  "repeat": { "count": "n_layers", "var": "i" },
                  "dtype": "F16",
                  "storage": "kv_cache",
                  "dims": ["n_kv_heads", "max_seq_len", "(d_model / n_heads) / 32"],
                  "grow_with_kv": true
                }
              ]
            },
            "forward": []
          }
        }
        "#,
    )
    .expect("parse spec");

    let est = ContextConfig::estimate_kv_memory(&spec.architecture, 256);
    assert_eq!(est.per_layer_bytes, 69_632);
    assert_eq!(est.kv_cache_bytes, 139_264);
}
