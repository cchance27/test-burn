use std::{
    path::PathBuf, sync::{Arc, Once}
};

use metallic_env::{EnvVarGuard, FoundryEnvVar};
use metallic_foundry::{
    BPETokenizer, Foundry, MetalError, generation::default_text_generation_workflow, model::{CompiledModel, ModelBuilder}, workflow::{Value, WorkflowRunner}
};
use metallic_loader::ModelLoader;
use rustc_hash::FxHashMap;

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

fn load_tokenizer(path: &std::path::PathBuf) -> Result<BPETokenizer, MetalError> {
    let model = ModelLoader::from_file(path).map_err(|e| MetalError::OperationFailed(format!("{:?}", e)))?;
    BPETokenizer::from_metadata(model.metadata())
}

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
    if let Some(bos) = tokenizer.special_tokens().bos_token_id
        && prompt2.first() == Some(&bos)
    {
        prompt2.remove(0);
    }

    println!("Prompt1 len: {}", prompt1.len());
    println!("Prompt2 len: {}", prompt2.len());

    let mut baseline_tokens = Vec::new();
    {
        // Reset session for fresh start
        let loaded_1 = ModelLoader::from_file(&gguf_path).unwrap();
        let model_baseline = Arc::new(
            ModelBuilder::new()
                .with_spec_file(&spec_path)?
                .with_model(loaded_1)
                .build(&mut foundry)?,
        );

        let _disable_batched_prefill_guard = EnvVarGuard::set(FoundryEnvVar::DisableBatchedPrefill, "1");

        // Turn 1
        let out1 = run_default_workflow_generate(&mut foundry, model_baseline.clone(), &prompt1, 20, 0.0, 1, 1.0, 0.0, 1.0, 64, 42)?;
        baseline_tokens.extend(out1);

        // Turn 2
        let out2 = run_default_workflow_generate(&mut foundry, model_baseline.clone(), &prompt2, 20, 0.0, 1, 1.0, 0.0, 1.0, 64, 42)?;
        baseline_tokens.extend(out2);
    }

    // 2. Run Test (Batched / Suspect)
    println!("Running Test (Batched)...");
    let mut test_tokens = Vec::new();
    {
        let loaded_2 = ModelLoader::from_file(&gguf_path).unwrap();
        let model_test = Arc::new(
            ModelBuilder::new()
                .with_spec_file(&spec_path)?
                .with_model(loaded_2)
                .build(&mut foundry)?,
        );

        let _disable_batched_prefill_guard = EnvVarGuard::unset(FoundryEnvVar::DisableBatchedPrefill);

        // Turn 1
        let out1 = run_default_workflow_generate(&mut foundry, model_test.clone(), &prompt1, 20, 0.0, 1, 1.0, 0.0, 1.0, 64, 42)?;
        test_tokens.extend(out1);

        // Turn 2 -- This is where we expect divergence if bug exists
        let out2 = run_default_workflow_generate(&mut foundry, model_test.clone(), &prompt2, 20, 0.0, 1, 1.0, 0.0, 1.0, 64, 42)?;
        test_tokens.extend(out2);
    }

    let baseline_text = tokenizer.decode(&baseline_tokens)?;
    let test_text = tokenizer.decode(&test_tokens)?;

    println!("Baseline text: {:?}", baseline_text);
    println!("Test text:     {:?}", test_text);

    assert_eq!(baseline_text, test_text, "Mismatch between sequential and batched execution!");

    Ok(())
}
