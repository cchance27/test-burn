#![cfg(test)]

use super::*;
use crate::spec::{ArchValue, Architecture};

fn mock_arch() -> Architecture {
    let mut params = rustc_hash::FxHashMap::default();
    params.insert("d_model".to_string(), ArchValue::USize(3072));
    params.insert("n_heads".to_string(), ArchValue::USize(24));
    params.insert("n_kv_heads".to_string(), ArchValue::USize(8));
    params.insert("n_layers".to_string(), ArchValue::USize(28));
    params.insert("ff_dim".to_string(), ArchValue::USize(8192));
    params.insert("vocab_size".to_string(), ArchValue::USize(128256));
    params.insert("max_seq_len".to_string(), ArchValue::USize(8192));
    params.insert("rope_base".to_string(), ArchValue::F32(500000.0));
    params.insert("rms_eps".to_string(), ArchValue::F32(1e-5));

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
fn kv_geometry_compact_defaults_for_gqa() {
    let g = KvGeometry::from_architecture(&mock_arch());
    assert_eq!(g.layout, KvCacheLayout::CompactKvHeads);
    assert_eq!(g.n_heads, 24);
    assert_eq!(g.n_kv_heads, 8);
    assert_eq!(g.group_size, 3);
    assert_eq!(g.head_dim, 128);
    assert_eq!(g.cache_heads(), 8);
}
