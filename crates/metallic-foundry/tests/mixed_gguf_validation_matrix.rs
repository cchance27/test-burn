use std::{path::PathBuf, sync::Arc, time::Instant};

use metallic_env::{MAX_PREFILL_CHUNK, PREFILL_CHUNK_SIZE};
use metallic_foundry::{
    Foundry, MetalError, generation::default_text_generation_workflow, model::{CompiledModel, ModelBuilder}, workflow::{Value, WorkflowRunner}
};
use metallic_loader::ModelLoader;
use rustc_hash::{FxHashMap, FxHashSet};
use serial_test::serial;

#[derive(Clone)]
struct MatrixCase {
    id: &'static str,
    spec_rel: &'static str,
    gguf_rel: &'static str,
}

#[derive(Debug)]
struct MatrixResult {
    id: String,
    n_heads: usize,
    n_kv_heads: usize,
    group_size: usize,
    prompt1_tokens: usize,
    prompt2_tokens: usize,
    turn1_generated: usize,
    turn2_generated: usize,
    turn1_elapsed_ms: u128,
    turn2_elapsed_ms: u128,
}

fn parse_env_usize(key: &'static str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn resolve_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../").join(rel)
}

fn selected_case_ids() -> Option<FxHashSet<String>> {
    let raw = std::env::var("METALLIC_MIXED_GGUF_MATRIX_CASES").ok()?;
    let ids = raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<FxHashSet<_>>();
    if ids.is_empty() { None } else { Some(ids) }
}

#[allow(clippy::too_many_arguments)]
fn run_generate(
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

    let mut generated = Vec::with_capacity(max_new_tokens);
    let _ = runner.run_streaming(foundry, &workflow, inputs, |tok, _prefill, _setup, _iter| {
        generated.push(tok);
        Ok(true)
    })?;
    Ok(generated)
}

fn run_case(
    case: &MatrixCase,
    max_new_tokens: usize,
    long_prompt_repeats: usize,
) -> Result<Option<MatrixResult>, Box<dyn std::error::Error>> {
    let spec = resolve_path(case.spec_rel);
    let gguf = resolve_path(case.gguf_rel);
    if !spec.exists() || !gguf.exists() {
        eprintln!(
            "Skipping {} (missing spec/model): spec={} gguf={}",
            case.id,
            spec.display(),
            gguf.display()
        );
        return Ok(None);
    }

    let mut foundry = Foundry::new()?;
    let loaded = ModelLoader::from_file(&gguf)?;
    let model = Arc::new(ModelBuilder::new().with_spec_file(&spec)?.with_model(loaded).build(&mut foundry)?);
    let tokenizer = model.tokenizer()?;

    let arch = model.architecture();
    let n_heads = arch.n_heads();
    let n_kv_heads = arch.n_kv_heads().max(1);
    let group_size = (n_heads / n_kv_heads).max(1);
    assert!(n_heads > 0, "case={} invalid n_heads=0", case.id);
    assert!(n_heads >= n_kv_heads, "case={} invalid n_heads < n_kv_heads", case.id);

    let long_prompt = format!(
        "{}\n{}",
        "Summarize this request in one short Rust comment:",
        "Keep the output short and deterministic. ".repeat(long_prompt_repeats)
    );
    let followup_prompt = "Now output only a one-line function signature in Rust.";

    let prompt1 = tokenizer.encode_single_turn_chat_prompt(&long_prompt)?;
    let prompt2 = tokenizer.encode_single_turn_chat_prompt(followup_prompt)?;
    assert!(
        prompt1.len() > 64,
        "case={} long prefill prompt did not exceed chunking threshold",
        case.id
    );
    assert!(!prompt2.is_empty(), "case={} follow-up prompt tokenized to empty input", case.id);

    let start1 = Instant::now();
    let gen1 = run_generate(&mut foundry, model.clone(), &prompt1, max_new_tokens, 0.0, 1, 1.0, 0.0, 1.0, 64, 42)?;
    let elapsed1 = start1.elapsed().as_millis();

    let start2 = Instant::now();
    let gen2 = run_generate(&mut foundry, model.clone(), &prompt2, max_new_tokens, 0.0, 1, 1.0, 0.0, 1.0, 64, 43)?;
    let elapsed2 = start2.elapsed().as_millis();

    Ok(Some(MatrixResult {
        id: case.id.to_string(),
        n_heads,
        n_kv_heads,
        group_size,
        prompt1_tokens: prompt1.len(),
        prompt2_tokens: prompt2.len(),
        turn1_generated: gen1.len(),
        turn2_generated: gen2.len(),
        turn1_elapsed_ms: elapsed1,
        turn2_elapsed_ms: elapsed2,
    }))
}

#[test]
#[serial]
#[ignore = "requires local GGUF models + Metal device; run manually as mixed-GGUF phase gate"]
fn mixed_gguf_validation_matrix_prefill_decode_long_context_gqa() -> Result<(), Box<dyn std::error::Error>> {
    let _prefill_chunk_guard = PREFILL_CHUNK_SIZE.set_guard(64)?;
    let _max_prefill_guard = MAX_PREFILL_CHUNK.set_guard(64)?;

    let max_new_tokens = parse_env_usize("METALLIC_MIXED_GGUF_MATRIX_MAX_NEW_TOKENS", 8);
    let long_prompt_repeats = parse_env_usize("METALLIC_MIXED_GGUF_MATRIX_LONG_PROMPT_REPEATS", 96);
    let selected = selected_case_ids();

    let cases = [
        MatrixCase {
            id: "qwen_fp16_dense",
            spec_rel: "models/qwen25.json",
            gguf_rel: "models/qwen2.5-coder-0.5b-instruct-fp16.gguf",
        },
        MatrixCase {
            id: "qwen_q6k_mixed",
            spec_rel: "models/qwen25.json",
            gguf_rel: "models/qwen2.5-0.5b-instruct-q6_k.gguf",
        },
        MatrixCase {
            id: "llama_q5k_mixed",
            spec_rel: "models/llama.json",
            gguf_rel: "models/llama-3.3-8b-instruct-q5_k_m.gguf",
        },
        MatrixCase {
            id: "smollm_q8",
            spec_rel: "models/smollm3.json",
            gguf_rel: "models/SmolLM3-Q8_0.gguf",
        },
    ];

    let mut results = Vec::new();
    for case in &cases {
        if let Some(filter) = selected.as_ref()
            && !filter.contains(case.id)
        {
            continue;
        }
        if let Some(result) = run_case(case, max_new_tokens, long_prompt_repeats)? {
            println!(
                "matrix case={} heads={} kv_heads={} group={} prompt1={} prompt2={} gen1={} gen2={} t1={}ms t2={}ms",
                result.id,
                result.n_heads,
                result.n_kv_heads,
                result.group_size,
                result.prompt1_tokens,
                result.prompt2_tokens,
                result.turn1_generated,
                result.turn2_generated,
                result.turn1_elapsed_ms,
                result.turn2_elapsed_ms
            );
            results.push(result);
        }
    }

    assert!(
        !results.is_empty(),
        "mixed-GGUF matrix ran zero cases; provide local model files or set METALLIC_MIXED_GGUF_MATRIX_CASES to existing cases"
    );

    let gqa_groups = results.iter().map(|r| r.group_size).collect::<FxHashSet<_>>();
    assert!(
        gqa_groups.len() >= 2,
        "expected at least 2 GQA group-size variants across matrix, got {:?}",
        gqa_groups
    );

    Ok(())
}
