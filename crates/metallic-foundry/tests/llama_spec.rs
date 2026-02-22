use std::path::Path;

use metallic_foundry::spec::ModelSpec;

fn spec_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/llama.json")
}

fn key_list_contains(keys: &[String], expected: &str) -> bool {
    keys.iter().any(|k| k == expected)
}

#[test]
fn llama_spec_parses_and_maps_required_metadata_keys() {
    let spec = ModelSpec::from_file(spec_path()).expect("llama spec should parse");
    let keys = &spec.architecture.metadata_keys.keys;

    assert!(key_list_contains(
        keys.get("d_model").expect("d_model mapping"),
        "llama.embedding_length"
    ));
    assert!(key_list_contains(
        keys.get("n_heads").expect("n_heads mapping"),
        "llama.attention.head_count"
    ));
    assert!(key_list_contains(
        keys.get("n_kv_heads").expect("n_kv_heads mapping"),
        "llama.attention.head_count_kv"
    ));
    assert!(key_list_contains(
        keys.get("n_layers").expect("n_layers mapping"),
        "llama.block_count"
    ));
    assert!(key_list_contains(
        keys.get("ff_dim").expect("ff_dim mapping"),
        "llama.feed_forward_length"
    ));
    assert!(key_list_contains(
        keys.get("max_seq_len").expect("max_seq_len mapping"),
        "llama.context_length"
    ));
    assert!(key_list_contains(
        keys.get("vocab_size").expect("vocab_size mapping"),
        "llama.vocab_size"
    ));
    assert!(key_list_contains(
        keys.get("rope_base").expect("rope_base mapping"),
        "llama.rope.freq_base"
    ));
    assert!(key_list_contains(
        keys.get("rms_eps").expect("rms_eps mapping"),
        "llama.attention.layer_norm_rms_epsilon"
    ));
}

#[test]
fn llama_spec_output_weight_keeps_tied_embedding_fallback() {
    let spec = ModelSpec::from_file(spec_path()).expect("llama spec should parse");
    let output_weight = &spec.architecture.tensor_names.output_weight;
    assert!(
        key_list_contains(output_weight, "token_embd.weight"),
        "llama output_weight should fallback to token_embd.weight for tied heads"
    );
}
