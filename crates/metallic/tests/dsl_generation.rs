use metallic::{
    MetalError, foundry::{Foundry, model::ModelBuilder}
};
use serial_test::serial;

const GGUF_PATH_DEFAULT: &str = "/Volumes/2TB/test-burn/models/qwen2.5-coder-0.5b-instruct-fp16.gguf";

fn get_gguf_path() -> String {
    std::env::var("GGUF_PATH").unwrap_or_else(|_| GGUF_PATH_DEFAULT.to_string())
}
const MODEL_SPEC_PATH: &str = "src/foundry/spec/qwen25.json";

#[test]
#[serial]
#[ignore]
fn test_dsl_qwen25_generation() -> Result<(), MetalError> {
    eprintln!("\n=== Generation Quality Test ===\n");
    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let mut foundry = Foundry::new()?;

    // Build DSL model
    eprintln!("Building model from spec: {:?}", spec_path);
    let dsl_model = ModelBuilder::new()
        .with_spec_file(&spec_path)?
        .with_gguf(&get_gguf_path())?
        .build(&mut foundry)?;

    let tokenizer = dsl_model.tokenizer()?;
    let prompt = "create a short js fibonacci function";
    let prompt_tokens = tokenizer.encode_single_turn_chat_prompt(prompt)?;
    if prompt_tokens.is_empty() {
        return Err(MetalError::InvalidShape("Tokenizer returned empty prompt encoding".into()));
    }

    eprintln!("Prompt: '{}'", prompt);
    let max_new_tokens = 100;
    let eos = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);

    // Validate the model's built-in autoregressive generate() implementation using the chat template.
    // Use sampling defaults similar to the CLI to avoid greedy repetition.
    let generated = dsl_model.generate(&mut foundry, &prompt_tokens, max_new_tokens, &[eos], 0.7, 40, 0.95)?;

    eprintln!("Generated {} new tokens (prompt_len={})", generated.len(), prompt_tokens.len());

    println!("\n\n=== Generated Text ===");
    if let Ok(decoded) = tokenizer.decode(&generated) {
        println!("{}", decoded);
    }
    println!("======================");
    println!("\nGeneration Complete.");
    Ok(())
}
