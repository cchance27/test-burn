use half::f16;
use metallic::{
    Context, F16Element, MetalError, foundry::{Foundry, model::ModelBuilder}, generation::{GenerationConfig, generate_autoregressive_with_kv_cache}, models::Qwen25
};
use serial_test::serial;

const GGUF_PATH_DEFAULT: &str = "../../models/qwen2.5-coder-0.5b-instruct-fp16.gguf";
const MODEL_SPEC_PATH: &str = "../../models/qwen25.json";

fn get_gguf_path() -> String {
    std::env::var("GGUF_PATH").unwrap_or_else(|_| GGUF_PATH_DEFAULT.to_string())
}

fn trim_trailing_token(mut tokens: Vec<u32>, token: u32) -> Vec<u32> {
    while tokens.last().copied() == Some(token) {
        tokens.pop();
    }
    tokens
}

#[test]
#[serial]
#[ignore = "requires qwen2.5 model file"]
fn test_dsl_vs_context_generation_seed_parity() -> Result<(), MetalError> {
    run_seed_parity()
}

fn run_seed_parity() -> Result<(), MetalError> {
    // Ensure ambient sampling tuning env vars don't invalidate parity expectations.
    // Both implementations should observe the same env, but Foundry historically ignored these.
    unsafe {
        std::env::set_var("METALLIC_SDPA_BACKEND", "unroll");
        std::env::remove_var("METALLIC_SAMPLE_TPTG");
        std::env::remove_var("METALLIC_SAMPLE_PER_THREAD_M");
    }

    let gguf_path = get_gguf_path();
    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);

    let prompt = "create a short js fibonacci function";
    let max_new_tokens = 64usize;
    let base_seed = 1337u32;
    let temperature = 0.7f32;
    let top_k = 40u32;
    let top_p = 0.95f32;

    // --- DSL / Foundry model ---
    let mut foundry = Foundry::new()?;
    let dsl_model = ModelBuilder::new()
        .with_spec_file(&spec_path)?
        .with_gguf(&gguf_path)?
        .build(&mut foundry)?;
    let tokenizer = dsl_model.tokenizer()?;
    let prompt_tokens = tokenizer.encode_single_turn_chat_prompt(prompt)?;
    if prompt_tokens.is_empty() {
        return Err(MetalError::InvalidShape("Tokenizer returned empty prompt encoding".into()));
    }
    let eos = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);

    // Sanity check: greedy next-token parity on this (templated) prompt.
    // If this fails, the divergence is in logits/forward, not sampling.
    let _dsl_greedy = dsl_model.generate_with_seed(&mut foundry, &prompt_tokens, 1, &[eos], 0.0, 0, 0.0, base_seed)?;

    // --- Legacy / Context model ---
    let mut ctx = Context::<F16Element>::new()?;
    let gguf_file = metallic::gguf::GGUFFile::load_mmap_and_get_metadata(&gguf_path)
        .map_err(|e| MetalError::InvalidShape(format!("GGUF load failed: {e}")))?;
    let loader = metallic::gguf::model_loader::GGUFModelLoader::new(gguf_file);
    let gguf_model = loader
        .load_model()
        .map_err(|e| MetalError::InvalidShape(format!("GGUF model failed: {e}")))?;
    let mut legacy_model: Qwen25<F16Element> = gguf_model
        .instantiate(&mut ctx)
        .map_err(|e| MetalError::InvalidShape(format!("Model instantiate failed: {e}")))?;

    let greedy_cfg = GenerationConfig {
        max_tokens: 1,
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        kv_initial_headroom_tokens: 1,
        seed: Some(base_seed),
    };
    let legacy_greedy = generate_autoregressive_with_kv_cache(&mut legacy_model, &tokenizer, &mut ctx, &prompt_tokens, &greedy_cfg)?;

    // Manual DSL step to inspect tensors
    {
        use objc2_metal::{MTLBuffer, MTLDevice as _};
        let (mut bindings, mut fast_bindings) = dsl_model.prepare_bindings(&mut foundry)?;
        // allocate input_ids
        let input_buffer = foundry
            .device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate input_ids");
        let input_tensor = metallic::types::TensorArg::from_buffer(
            metallic::types::MetalBuffer::from_retained(input_buffer.clone()),
            metallic::tensor::Dtype::U32,
            vec![1],
            vec![1],
        );

        // Manual binding - logic matching executor.rs
        bindings.insert("input_ids".to_string(), input_tensor.clone());
        if let Some(id) = dsl_model.symbol_id("input_ids") {
            fast_bindings.set(id, input_tensor.clone());
        }

        let _arch = dsl_model.architecture();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            unsafe {
                *(input_buffer.contents().as_ptr() as *mut u32) = tok;
            }
            let kv_seq_len = pos + 1;
            bindings.set_global("seq_len", "1".to_string());
            bindings.set_global("position_offset", pos.to_string());
            bindings.set_global("kv_seq_len", kv_seq_len.to_string());

            // Set missing globals that caused kernel to exit early
            let d_model = 896;
            let n_heads = 14;
            let n_kv_heads = 2;
            let head_dim = 64;
            let total_q = 1 * 1 * n_heads * head_dim;
            let total_k = 1 * 1 * n_kv_heads * head_dim;
            bindings.set_global("total_elements_q", total_q.to_string());
            bindings.set_global("total_elements_k", total_k.to_string());
            bindings.set_global("total_elements_hidden", d_model.to_string());
            bindings.set_global("total_elements_write", total_k.to_string());
            // total_elements_repeat for repeat_kv?
            // It uses cache_stride * ... ?
            // Actually RepeatKvHeads parameters in JSON are:
            // "total_elements": "{total_elements_repeat}"
            // RepeatKvHeads kernel (if used) might need it.
            // But for seq=1, it copies 1 seq.
            // Let's set it to total_k ??
            // RepeatKvHeads outputs k_expanded: [batch, n_heads, seq, head_dim].
            // So total elements = batch * n_heads * seq * head_dim = total_q.
            bindings.set_global("total_elements_repeat", total_q.to_string());

            dsl_model.forward(&mut foundry, &mut bindings, &fast_bindings)?;
        }

        // Inspect last token tensors
        let hidden = bindings.get("hidden")?;
        let q = bindings.get("q")?;
        let q_heads = bindings.get("q_heads")?;
        let rope_cos = bindings.get("rope_cos")?;
        let q_rot = bindings.get("q_rot")?;
        let attn_out = bindings.get("attn_out")?;

        fn dump_tensor(name: &str, arg: &metallic::types::TensorArg) {
            use metallic::types::KernelArg;
            let buf = arg.buffer(); // returns &MetalBuffer
            {
                let count = 10;
                // MetalBuffer derefs to MTLBuffer?
                // Or we need .as_raw() or similar?
                // Usually it derefs to ProtocolObject<MTLBuffer>.
                // But let's check content.
                // If it derefs to MTLBuffer, it has contents().
                let ptr = buf.contents().as_ptr() as *const f16;
                let mut vals = vec![];
                for i in 0..count {
                    vals.push(unsafe { *ptr.add(i) }.to_f32());
                }
                println!("DEBUG: {} first 10: {:?}", name, vals);
            }
        }

        dump_tensor("hidden", &hidden);
        dump_tensor("q", &q);
        dump_tensor("q_heads", &q_heads);
        dump_tensor("rope_cos", &rope_cos);
        let k_expanded = bindings.get("k_expanded")?;
        dump_tensor("k_expanded", &k_expanded);
        dump_tensor("q_rot", &q_rot);
        dump_tensor("attn_out", &attn_out);
    }

    // Generate greedy token with DSL
    let dsl_tokens = dsl_model.generate_with_seed(
        &mut foundry,
        &prompt_tokens,
        1, // max_tokens
        &[eos],
        0.0, // temp
        0,   // top_k
        0.0, // top_p
        base_seed,
    )?;

    println!("EOS Token: {}", eos);
    println!("Legacy Raw: {:?}", legacy_greedy);
    println!("DSL Raw: {:?}", dsl_tokens);
    println!("Legacy Decoded: {:?}", tokenizer.decode(&legacy_greedy));
    println!("DSL Decoded: {:?}", tokenizer.decode(&dsl_tokens));
    let dsl_greedy = dsl_tokens.iter().map(|&t| t).collect::<Vec<_>>();

    // Print logits
    // We need to fetch logits from the last step?
    // The helper functions don't return logits.
    // We can't easily get logits without modifying the helper or running a manual step.

    // ... manual run last token ...
    // Let's rely on the mismatch for now.

    let dsl_greedy_trimmed = trim_trailing_token(dsl_greedy.clone(), eos);
    let legacy_greedy_trimmed = trim_trailing_token(legacy_greedy.clone(), eos);

    assert_eq!(
        legacy_greedy_trimmed, dsl_greedy_trimmed,
        "Greedy next-token parity failed for templated prompt; fix logits parity before seeded sampling parity"
    );

    let dsl_tokens = dsl_model.generate_with_seed(
        &mut foundry,
        &prompt_tokens,
        max_new_tokens,
        &[eos],
        temperature,
        top_k,
        top_p,
        base_seed,
    )?;

    let cfg = GenerationConfig {
        max_tokens: max_new_tokens,
        temperature,
        top_p,
        top_k: top_k as usize,
        kv_initial_headroom_tokens: max_new_tokens,
        seed: Some(base_seed),
    };

    let legacy_tokens = generate_autoregressive_with_kv_cache(&mut legacy_model, &tokenizer, &mut ctx, &prompt_tokens, &cfg)?;

    // Legacy path includes EOS in the collected token stream; Foundry's generate() returns only newly generated
    // tokens and stops before pushing stop tokens. Normalize by trimming EOS from both sides.
    let legacy_trimmed = trim_trailing_token(legacy_tokens, eos);
    let dsl_trimmed = trim_trailing_token(dsl_tokens, eos);

    assert_eq!(legacy_trimmed, dsl_trimmed, "Seeded generation token parity mismatch");

    Ok(())
}
