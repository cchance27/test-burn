use std::sync::Arc;

use metallic_foundry::{
    Foundry, MetalError, generation::default_text_generation_workflow, model::{CompiledModel, ModelBuilder}, workflow::{Value, WorkflowRunner}
};
use metallic_loader::ModelLoader;
use rustc_hash::FxHashMap;
use serial_test::serial;

const GGUF_PATH_DEFAULT: &str = "../../models/qwen2.5-coder-0.5b-instruct-fp16.gguf";

fn get_gguf_path() -> String {
    std::env::var("GGUF_PATH").unwrap_or_else(|_| GGUF_PATH_DEFAULT.to_string())
}
const MODEL_SPEC_PATH: &str = "../../models/qwen25.json";

#[allow(clippy::too_many_arguments)]
fn run_default_workflow_generate(
    foundry: &mut Foundry,
    model: Arc<CompiledModel>,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    repeat_last_n: usize,
    seed: u32,
) -> Result<Vec<u32>, MetalError> {
    let mut models: FxHashMap<String, Arc<CompiledModel>> = FxHashMap::default();
    models.insert("llm".to_string(), model);
    let workflow = default_text_generation_workflow();
    let mut runner = WorkflowRunner::new(models);
    let mut inputs: FxHashMap<String, Value> = FxHashMap::default();
    inputs.insert("prompt_tokens".to_string(), Value::TokensU32(prompt_tokens.to_vec()));
    inputs.insert("max_tokens".to_string(), Value::Usize(max_new_tokens));
    inputs.insert("temperature".to_string(), Value::F32(temperature));
    inputs.insert("top_k".to_string(), Value::U32(top_k));
    inputs.insert("top_p".to_string(), Value::F32(top_p));
    inputs.insert("min_p".to_string(), Value::F32(min_p));
    inputs.insert("repeat_penalty".to_string(), Value::F32(repeat_penalty));
    inputs.insert("repeat_last_n".to_string(), Value::Usize(repeat_last_n));
    inputs.insert("presence_penalty".to_string(), Value::F32(0.0));
    inputs.insert("frequency_penalty".to_string(), Value::F32(0.0));
    inputs.insert("seed".to_string(), Value::U32(seed));
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let _ = runner.run_streaming(foundry, &workflow, inputs, |tok, _prefill, _setup, _iter| {
        generated.push(tok);
        Ok(true)
    })?;
    Ok(generated)
}

#[test]
#[serial]
#[ignore]
fn test_dsl_qwen25_generation() -> Result<(), MetalError> {
    eprintln!("\n=== Generation Quality Test ===\n");
    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let mut foundry = Foundry::new()?;

    // Build DSL model
    eprintln!("Building model from spec: {:?}", spec_path);
    let loaded_model = ModelLoader::from_file(get_gguf_path()).unwrap();
    let dsl_model = Arc::new(
        ModelBuilder::new()
            .with_spec_file(&spec_path)?
            .with_model(loaded_model)
            .build(&mut foundry)?,
    );

    let tokenizer = dsl_model.tokenizer()?;
    let prompt = "create a short js fibonacci function";
    let prompt_tokens = tokenizer.encode_single_turn_chat_prompt(prompt)?;
    if prompt_tokens.is_empty() {
        return Err(MetalError::InvalidShape("Tokenizer returned empty prompt encoding".into()));
    }

    eprintln!("Prompt: '{}'", prompt);
    let max_new_tokens = 100;
    // Validate workflow-driven autoregressive generation using the chat template.
    // Use sampling defaults similar to the CLI to avoid greedy repetition.
    let generated = run_default_workflow_generate(
        &mut foundry,
        dsl_model.clone(),
        &prompt_tokens,
        max_new_tokens,
        0.7,
        40,
        0.95,
        0.05,
        1.1,
        64,
        42,
    )?;

    eprintln!("Generated {} new tokens (prompt_len={})", generated.len(), prompt_tokens.len());

    println!("\n\n=== Generated Text ===");
    if let Ok(decoded) = tokenizer.decode(&generated) {
        println!("{}", decoded);
    }
    println!("======================");
    println!("\nGeneration Complete.");
    Ok(())
}
