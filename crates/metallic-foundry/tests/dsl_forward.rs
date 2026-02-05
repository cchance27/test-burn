//! Integration tests for DSL-based forward pass execution.
//!
//! Tests the complete DSL pipeline: load JSON spec → resolve tensors → execute steps.

use metallic_foundry::{
    Foundry, model::ModelBuilder, spec::{ModelSpec, TensorBindings}
};
use metallic_loader::ModelLoader;

/// Test that ModelSpec can parse a minimal JSON spec.
#[test]
fn test_model_spec_parse() {
    let json = r#"{
        "name": "test-model",
        "architecture": {
            "d_model": 64,
            "n_heads": 4,
            "n_kv_heads": 2,
            "n_layers": 2,
            "ff_dim": 128,
            "vocab_size": 1000,
            "max_seq_len": 512,
            "forward": []
        }
    }"#;

    let spec = ModelSpec::from_json(json).expect("Failed to parse JSON");
    assert_eq!(spec.name, "test-model");
    assert_eq!(spec.architecture.d_model(), 64);
    assert_eq!(spec.architecture.n_layers(), 2);
}

/// Test TensorBindings variable interpolation.
#[test]
fn test_tensor_bindings_interpolation() {
    let mut bindings = TensorBindings::new();

    // Set a global variable
    bindings.set_global("n_layers", "24".to_string());

    // Push a scope with a loop variable
    bindings.push_scope();
    bindings.set_var("i", "5".to_string());

    // Test interpolation
    assert_eq!(bindings.interpolate("blk.{i}.attn_q.weight".to_string()), "blk.5.attn_q.weight");
    assert_eq!(bindings.interpolate("layer_{i}_norm".to_string()), "layer_5_norm");

    // Pop scope and verify variable is gone
    bindings.pop_scope();
    assert_eq!(bindings.interpolate("blk.{i}.attn_q.weight".to_string()), "blk.{i}.attn_q.weight"); // No substitution
}

/// Test TensorBindings global variable access.
#[test]
fn test_tensor_bindings_globals() {
    let mut bindings = TensorBindings::new();

    // Set globals
    bindings.set_global("d_model", "512".to_string());
    bindings.set_global("n_layers", "12".to_string());

    // Verify we can retrieve them
    assert_eq!(bindings.get_var("d_model"), Some(&"512".to_string()));
    assert_eq!(bindings.get_var("n_layers"), Some(&"12".to_string()));
}

/// Test parsing a model spec with tensor_names section.
#[test]
fn test_model_spec_with_tensor_names() {
    let json = r#"{
        "name": "test-model-names",
        "architecture": {
            "d_model": 64,
            "n_heads": 4,
            "n_kv_heads": 2,
            "n_layers": 2,
            "ff_dim": 128,
            "vocab_size": 1000,
            "max_seq_len": 512,
            "tensor_names": {
                "embedding": ["token_embd.weight", "model.embed_tokens.weight"],
                "output_weight": ["output.weight", "lm_head.weight"],
                "layer": {
                    "attn_q": ["blk.{i}.attn_q.weight"]
                }
            },
            "forward": []
        }
    }"#;

    let spec = ModelSpec::from_json(json).expect("Failed to parse JSON with tensor_names");
    assert_eq!(spec.architecture.tensor_names.embedding.len(), 2);
    assert_eq!(spec.architecture.tensor_names.output_weight.len(), 2);
    assert_eq!(spec.architecture.tensor_names.layer.attn_q.len(), 1);
}

/// End-to-end forward pass test with real Qwen2.5 model.
///
/// This test is ignored by default since it requires:
/// - A GPU (macOS Metal)
/// - The model file at the expected path
///
/// To run: `cargo test -p metallic --test dsl_forward test_e2e_forward_pass -- --ignored`
#[test]
// #[ignore]
fn test_e2e_forward_pass() {
    use std::path::Path;

    // Model path - adjust if your model is elsewhere
    let model_path = "../../models/qwen2.5-coder-0.5b-instruct-fp16.gguf";
    let spec_path = "../../models/qwen25.json";

    // Check if model exists
    if !Path::new(model_path).exists() {
        eprintln!("Skipping e2e test: model not found at {}", model_path);
        eprintln!("Set METALLIC_TEST_MODEL=/path/to/model.gguf to use a different path");
        return;
    }

    // Check if spec exists
    if !Path::new(spec_path).exists() {
        panic!("Model spec not found at {}", spec_path);
    }

    // Load model spec
    let spec = ModelSpec::from_file(spec_path).expect("Failed to load model spec");
    eprintln!(
        "Loaded spec: {} (d_model={}, n_layers={})",
        spec.name,
        spec.architecture.d_model(),
        spec.architecture.n_layers()
    );

    // Create foundry
    let mut foundry = Foundry::new().expect("Failed to create Foundry");
    eprintln!("Created Foundry with device: {:?}", foundry.device);

    // Load model via builder
    let loaded_model = ModelLoader::from_file(model_path).expect("Failed to load GGUF");
    let model = ModelBuilder::new()
        .with_spec(spec)
        .with_model(loaded_model)
        .build(&mut foundry)
        .expect("Failed to build model");

    eprintln!("Loaded model: {}", model.name());

    // Prepare bindings
    let (mut bindings, mut fast_bindings) = model.prepare_bindings(&mut foundry).expect("Failed to prepare bindings");
    eprintln!("Prepared {} bindings", bindings.len());

    // Prepare input_ids from a coherent prompt
    let tokenizer = model.tokenizer().expect("Failed to create tokenizer");
    let prompt = "write a helloworld JavaScript program";
    let prompt_tokens = tokenizer
        .encode_single_turn_chat_prompt(prompt)
        .expect("Failed to encode prompt with chat template");
    eprintln!("Prompt: '{}'", prompt);
    eprintln!("Encoded prompt: {:?}", prompt_tokens);

    let input_buffer = {
        use objc2_metal::MTLDevice;
        let byte_size = prompt_tokens.len() * 4;
        let buf = foundry
            .device
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate input buffer");
        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(prompt_tokens.as_ptr(), ptr, prompt_tokens.len());
        }
        metallic_foundry::types::MetalBuffer::from_retained(buf)
    };
    let input_tensor = metallic_foundry::types::TensorArg::from_buffer(
        input_buffer,
        metallic_foundry::tensor::Dtype::U32,
        vec![prompt_tokens.len()],
        vec![1],
    );
    bindings.insert("input_ids".to_string(), input_tensor.clone());
    if let Some(id) = model.symbol_id("input_ids") {
        fast_bindings.set(id, input_tensor);
    }
    eprintln!("Bound input_ids with {} token(s)", prompt_tokens.len());

    // Get architecture config for globals
    let arch = model.architecture();

    // CRITICAL: Set dynamic globals for kernel dispatch
    let seq_len = prompt_tokens.len();
    let d_model = arch.d_model();
    let n_heads = arch.n_heads();
    let n_kv_heads = arch.n_kv_heads();
    let ff_dim = arch.ff_dim();
    let head_dim = d_model / n_heads;
    let position_offset = 0usize;
    let kv_seq_len = position_offset + seq_len;

    bindings.set_global("seq_len", seq_len.to_string());
    bindings.set_global("position_offset", position_offset.to_string());
    bindings.set_global("kv_seq_len", kv_seq_len.to_string());
    bindings.set_global("total_elements_q", (n_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_k", (n_kv_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_hidden", (seq_len * d_model).to_string());
    bindings.set_global("total_elements_ffn", (seq_len * ff_dim).to_string());
    bindings.set_global("total_elements_slice", (n_kv_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_repeat", (n_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_write", (n_kv_heads * seq_len * head_dim).to_string());
    eprintln!(
        "Set dynamic globals: seq_len={}, total_elements_hidden={}",
        seq_len,
        seq_len * d_model
    );

    // For a smoke test, we verify the forward pass executes
    if !arch.forward.is_empty() {
        eprintln!("Running forward pass ({} steps)...", arch.forward.len());

        match model.forward(&mut foundry, &mut bindings, &fast_bindings) {
            Ok(()) => eprintln!("Forward pass completed successfully!"),
            Err(e) => panic!("Forward pass failed: {:?}", e),
        }
    } else {
        eprintln!("Note: spec has no forward steps defined yet");
    }

    // Verify we can at least access key tensors
    assert!(!bindings.is_empty(), "Should have some bindings");
    eprintln!("E2E test completed!");

    // Now test actual token generation using the generate() method
    eprintln!("\n=== Testing Token Generation ===");

    let stop_tokens: Vec<u32> = vec![tokenizer.special_tokens().eos_token_id.unwrap_or(151645)];
    let max_new_tokens = 50;

    match model.generate(
        &mut foundry,
        &prompt_tokens,
        max_new_tokens,
        &stop_tokens,
        0.7, // temperature
        50,  // top_k
        0.9, // top_p
    ) {
        Ok(generated) => {
            eprintln!("Generated {} tokens: {:?}", generated.len(), generated);
            let decoded = tokenizer.decode(&generated).expect("Failed to decode generated tokens");
            eprintln!("Generated text:\n---\n{}\n---", decoded);
            assert!(!generated.is_empty(), "Should generate at least one token");
            eprintln!("Generation test passed!");
        }
        Err(e) => panic!("Generation failed: {:?}", e),
    }
}
