use metallic_context::{Context, F16Element, generation::generate_autoregressive_with_kv_cache, models::Qwen25};
use metallic_foundry::{Foundry, MetalError, model::ModelBuilder};
use metallic_loader::ModelLoader;
use serial_test::serial;

const GGUF_PATH: &str = "../../models/qwen2.5-coder-0.5b-instruct-fp16.gguf";
const MODEL_SPEC_PATH: &str = "../../models/qwen25.json";

#[test]
#[serial]
#[ignore = "requires qwen2.5 model file"]
fn test_dsl_vs_context_chat_prefill_greedy_next_token_parity() -> Result<(), MetalError> {
    let gguf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(GGUF_PATH);
    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);

    if !gguf_path.exists() {
        return Err(MetalError::InvalidOperation(format!("Missing GGUF at {}", gguf_path.display())));
    }

    // Load Foundry model.
    let mut foundry = Foundry::new()?;
    let loaded_model = ModelLoader::from_file(&gguf_path).unwrap();
    let dsl_model = ModelBuilder::new()
        .with_spec_file(&spec_path)?
        .with_model(loaded_model)
        .build(&mut foundry)?;
    let dsl_tokenizer = dsl_model.tokenizer()?;

    // Load legacy model + tokenizer.
    let mut ctx = Context::<F16Element>::new().map_err(|e| MetalError::OperationFailed(format!("Context init failed: {e:?}")))?;
    let gguf_file = metallic_context::gguf::GGUFFile::load_mmap_and_get_metadata(&gguf_path)
        .map_err(|e| MetalError::InvalidShape(format!("GGUF load failed: {e}")))?;
    let legacy_tokenizer = metallic_context::tokenizer::Tokenizer::from_gguf_metadata(&gguf_file.metadata)
        .map_err(|e| MetalError::OperationFailed(format!("Legacy tokenizer load failed: {e:?}")))?;
    let loader = metallic_context::gguf::model_loader::GGUFModelLoader::new(gguf_file);
    let gguf_model = loader
        .load_model()
        .map_err(|e| MetalError::InvalidShape(format!("GGUF model failed: {e}")))?;
    let mut legacy_model: Qwen25<F16Element> = gguf_model
        .instantiate(&mut ctx)
        .map_err(|e| MetalError::InvalidShape(format!("Legacy instantiate failed: {e}")))?;

    let prompt = "Hello";
    let dsl_tokens = dsl_tokenizer.encode_single_turn_chat_prompt(prompt)?;
    let legacy_tokens = legacy_tokenizer
        .encode_single_turn_chat_prompt(prompt)
        .map_err(|e| MetalError::OperationFailed(format!("Legacy tokenizer encode failed: {e:?}")))?;
    assert_eq!(dsl_tokens, legacy_tokens, "Chat prompt tokenization mismatch");

    // --- Next-token logits parity via greedy generation of 1 token ---
    let eos = dsl_tokenizer
        .special_tokens()
        .eos_token_id
        .ok_or_else(|| MetalError::InvalidOperation("Tokenizer metadata missing required 'eos_token_id'".into()))?;

    // DSL: greedy next token
    let dsl_next = dsl_model.generate_with_seed(&mut foundry, &dsl_tokens, 1, &[eos], 0.0, 1, 0.0, 1337)?;
    let dsl_tok = dsl_next.first().copied().unwrap_or(0);

    // Legacy: greedy next token
    let legacy_cfg = metallic_context::generation::GenerationConfig {
        max_tokens: 1,
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        kv_initial_headroom_tokens: 1,
        seed: Some(1337),
    };
    let legacy_out = generate_autoregressive_with_kv_cache(&mut legacy_model, &legacy_tokenizer, &mut ctx, &legacy_tokens, &legacy_cfg)
        .map_err(|e| MetalError::OperationFailed(format!("Legacy generation failed: {e:?}")))?;
    let legacy_tok = legacy_out.first().copied().unwrap_or(0);

    assert_eq!(dsl_tok, legacy_tok, "Greedy next token mismatch after chat prefill");
    Ok(())
}
