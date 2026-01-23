use std::{path::PathBuf, sync::Once};

use metallic_foundry::{Foundry, MetalError, model::ModelBuilder};

const MODEL_SPEC_PATH: &str = "../../models/qwen25.json";
const GGUF_PATH: &str = "../../models/qwen2.5-coder-0.5b-instruct-fp16.gguf";

static INIT: Once = Once::new();

fn init_tracing() {
    INIT.call_once(|| {
        tracing_subscriber::fmt().init();
    });
}

fn get_model_paths() -> (PathBuf, PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    (root.join(MODEL_SPEC_PATH), root.join(GGUF_PATH))
}

fn load_tokenizer(path: &PathBuf) -> Result<metallic_foundry::Tokenizer, MetalError> {
    let file = metallic_foundry::gguf::file::GGUFFile::load_mmap_and_get_metadata(path)
        .map_err(|e| MetalError::OperationFailed(format!("{:?}", e)))?;
    metallic_foundry::Tokenizer::from_gguf_metadata(&file.metadata)
}

#[test]
fn test_batched_prefill_multiturn_consistency() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let (spec_path, gguf_path) = get_model_paths();
    if !spec_path.exists() || !gguf_path.exists() {
        eprintln!("Skipping test: models not found at {:?} / {:?}", spec_path, gguf_path);
        return Ok(());
    }

    let mut foundry = Foundry::new()?;

    let tokenizer = load_tokenizer(&gguf_path)?;

    // 1. Run Baseline (Sequential / Trusted)
    println!("Running Baseline (Sequential)...");

    // Long system prompt to push context > 64
    let system_prompt = "You are a helpful assistant. ".repeat(20); // ~100 tokens
    let text1 = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        system_prompt
    );
    let prompt1 = tokenizer.encode(&text1)?;

    // Turn 2 with length > 32
    let long_user_input = "Please explain the theory of relativity in detail and how it relates to time dilation. ".repeat(3); // ~40-50 tokens

    // We need to construct prompt2 carefully to be the continuation
    // Since we don't have a chat template engine here, we just append tokens?
    // Wait, the test uses `generate` with proper session state.
    // So `prompt2` should just be the new user turn tokens.
    let mut prompt2 = tokenizer.encode(&format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", long_user_input))?;

    // Remove BOS
    if let Some(bos) = tokenizer.special_tokens().bos_token_id {
        if prompt2.first() == Some(&bos) {
            prompt2.remove(0);
        }
    }

    println!("Prompt1 len: {}", prompt1.len());
    println!("Prompt2 len: {}", prompt2.len());

    let mut baseline_tokens = Vec::new();
    {
        // Reset session for fresh start
        let model_baseline = ModelBuilder::new()
            .with_spec_file(&spec_path)?
            .with_gguf(&gguf_path)?
            .build(&mut foundry)?;

        unsafe {
            std::env::set_var("METALLIC_DISABLE_BATCHED_PREFILL", "1");
            std::env::remove_var("METALLIC_FORCE_BATCHED_PREFILL");
        }

        // Turn 1
        let out1 = model_baseline.generate(&mut foundry, &prompt1, 20, &[], 0.0, 1, 1.0)?;
        baseline_tokens.extend(out1);

        // Turn 2
        let out2 = model_baseline.generate(&mut foundry, &prompt2, 20, &[], 0.0, 1, 1.0)?;
        baseline_tokens.extend(out2);
    }

    // 2. Run Test (Batched / Suspect)
    println!("Running Test (Batched)...");
    let mut test_tokens = Vec::new();
    {
        let model_test = ModelBuilder::new()
            .with_spec_file(&spec_path)?
            .with_gguf(&gguf_path)?
            .build(&mut foundry)?;

        unsafe {
            std::env::remove_var("METALLIC_FORCE_BATCHED_PREFILL");
            std::env::remove_var("METALLIC_DISABLE_BATCHED_PREFILL");
        }

        // Turn 1
        let out1 = model_test.generate(&mut foundry, &prompt1, 20, &[], 0.0, 1, 1.0)?;
        test_tokens.extend(out1);

        // Turn 2 -- This is where we expect divergence if bug exists
        let out2 = model_test.generate(&mut foundry, &prompt2, 20, &[], 0.0, 1, 1.0)?;
        test_tokens.extend(out2);
    }

    let baseline_text = tokenizer.decode(&baseline_tokens)?;
    let test_text = tokenizer.decode(&test_tokens)?;

    println!("Baseline text: {:?}", baseline_text);
    println!("Test text:     {:?}", test_text);

    assert_eq!(baseline_text, test_text, "Mismatch between sequential and batched execution!");

    Ok(())
}
