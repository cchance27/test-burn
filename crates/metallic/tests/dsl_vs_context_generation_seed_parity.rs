use half::f16;
use metallic::{
    Context, F16Element, MetalError, foundry::{Foundry, model::ModelBuilder}, generation::{GenerationConfig, generate_autoregressive_with_kv_cache}, models::Qwen25, types::KernelArg as _
};
use serial_test::serial;

const GGUF_PATH_DEFAULT: &str = "/Volumes/2TB/test-burn/models/qwen2.5-coder-0.5b-instruct-fp16.gguf";
const MODEL_SPEC_PATH: &str = "src/foundry/spec/qwen25.json";

fn get_gguf_path() -> String {
    std::env::var("GGUF_PATH").unwrap_or_else(|_| GGUF_PATH_DEFAULT.to_string())
}

fn trim_trailing_token(mut tokens: Vec<u32>, token: u32) -> Vec<u32> {
    while tokens.last().copied() == Some(token) {
        tokens.pop();
    }
    tokens
}

fn read_f16_tensorarg(arg: &metallic::types::TensorArg) -> Vec<f16> {
    let buffer = arg.buffer();
    let len = arg.dims().iter().product::<usize>();
    unsafe {
        use objc2_metal::MTLBuffer;
        let ptr = buffer.contents().as_ptr() as *const f16;
        std::slice::from_raw_parts(ptr, len).to_vec()
    }
}

fn sample_topk_topp_cpu(logits_f16: &[f16], top_k: usize, top_p: f32, temperature: f32, seed: u32) -> u32 {
    if logits_f16.is_empty() {
        return 0;
    }

    if temperature <= 0.0 || !temperature.is_finite() || top_k == 0 {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        let mut found = false;
        for (i, &raw) in logits_f16.iter().enumerate() {
            let v = raw.to_f32();
            if !v.is_finite() {
                continue;
            }
            if !found || v > best_val || (v == best_val && i > best_idx) {
                found = true;
                best_val = v;
                best_idx = i;
            }
        }
        return best_idx as u32;
    }

    let inv_t = 1.0f32 / temperature.max(1e-6);
    let mut pairs: Vec<(f32, u32)> = logits_f16
        .iter()
        .enumerate()
        .map(|(i, &raw)| (raw.to_f32() * inv_t, i as u32))
        .collect();

    // Sort by logit desc, then index asc (matches deterministic tie-break intent).
    pairs.sort_by(|(av, ai), (bv, bi)| bv.partial_cmp(av).unwrap_or(std::cmp::Ordering::Equal).then_with(|| ai.cmp(bi)));

    let k = top_k.min(pairs.len());
    pairs.truncate(k);

    // Softmax over top-k
    let maxv = pairs[0].0;
    let mut sum = 0.0f32;
    let mut probs: Vec<(f32, u32)> = Vec::with_capacity(pairs.len());
    for (v, idx) in pairs {
        let e = (v - maxv).exp();
        sum += e;
        probs.push((e, idx));
    }
    if !(sum > 0.0) || !sum.is_finite() {
        return probs[0].1;
    }
    for (p, _idx) in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p cutoff on the sorted probs
    let mut cumulative = 0.0f32;
    let mut cutoff = probs.len().saturating_sub(1);
    let mut cutoff_sum = 0.0f32;
    for (i, (p, _idx)) in probs.iter().enumerate() {
        cumulative += *p;
        if cumulative >= top_p {
            cutoff = i;
            cutoff_sum = cumulative;
            break;
        }
    }
    if cutoff == probs.len().saturating_sub(1) && cutoff_sum == 0.0f32 {
        cutoff_sum = cumulative;
    }
    let renorm = if cutoff_sum > 0.0 && cutoff_sum.is_finite() {
        1.0f32 / cutoff_sum
    } else {
        1.0f32
    };
    for i in 0..=cutoff {
        probs[i].0 *= renorm;
    }

    // RNG matches Metal kernel (LCG + float-from-bits trick).
    let mut state = if seed == 0 { 1u32 } else { seed };
    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let r = f32::from_bits((state >> 9) | 0x3f800000) - 1.0f32;

    let mut acc = 0.0f32;
    let mut chosen = probs[0].1;
    for i in 0..=cutoff {
        acc += probs[i].0;
        if r <= acc {
            chosen = probs[i].1;
            break;
        }
    }
    chosen
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
    let dsl_greedy = dsl_model.generate_with_seed(&mut foundry, &prompt_tokens, 1, &[eos], 0.0, 0, 0.0, base_seed)?;

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

    let dsl_greedy_trimmed = trim_trailing_token(dsl_greedy, eos);
    let legacy_greedy_trimmed = trim_trailing_token(legacy_greedy, eos);
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

    if legacy_trimmed != dsl_trimmed {
        // Diagnostics: compare the logits used for the very first sample (post-prompt).
        // This helps distinguish "sampling mismatch" from "logits mismatch" (sampling is extremely sensitive).
        let (legacy_hidden_f16, legacy_logits_f16, legacy_v_hist_f16, legacy_n_heads, legacy_n_kv_heads, legacy_head_dim) = {
            let mut fresh_ctx = Context::<F16Element>::new()?;
            let gguf_file = metallic::gguf::GGUFFile::load_mmap_and_get_metadata(&gguf_path)
                .map_err(|e| MetalError::InvalidShape(format!("GGUF load failed (diag): {e}")))?;
            let loader = metallic::gguf::model_loader::GGUFModelLoader::new(gguf_file);
            let gguf_model = loader
                .load_model()
                .map_err(|e| MetalError::InvalidShape(format!("GGUF model failed (diag): {e}")))?;
            let legacy_model_diag: Qwen25<F16Element> = gguf_model
                .instantiate(&mut fresh_ctx)
                .map_err(|e| MetalError::InvalidShape(format!("Model instantiate failed (diag): {e}")))?;

            // forward_step requires KV caches to be allocated.
            // Match Foundry's fixed KV cache capacity (2048) to avoid parity being affected by
            // accidental out-of-bounds reads beyond the active sequence length.
            let kv_capacity = legacy_model_diag.config.seq_len;
            let legacy_n_heads = legacy_model_diag.config.n_heads;
            let legacy_n_kv_heads = legacy_model_diag.config.n_kv_heads;
            let legacy_d_model = legacy_model_diag.config.d_model;
            let kv_dim = legacy_d_model * legacy_n_kv_heads / legacy_n_heads;
            let kv_head_dim = kv_dim / legacy_n_kv_heads;
            for layer_idx in 0..legacy_model_diag.config.n_layers {
                fresh_ctx.alloc_kv_cache(layer_idx, kv_capacity, legacy_n_heads, kv_head_dim)?;
            }

            let mut logits: Option<metallic::Tensor<F16Element>> = None;
            let mut last_hidden: Option<metallic::Tensor<F16Element>> = None;
            for (pos, &tok) in prompt_tokens.iter().enumerate() {
                let embedded = legacy_model_diag.embed(&[tok], &mut fresh_ctx)?;
                let (hidden, _) = legacy_model_diag.forward_step(&embedded, pos, &mut fresh_ctx)?;
                last_hidden = Some(hidden.clone());
                logits = Some(legacy_model_diag.output(&hidden, &mut fresh_ctx)?);
            }
            fresh_ctx.synchronize();
            let t = logits.ok_or_else(|| MetalError::InvalidShape("Missing legacy logits after prompt (diag)".into()))?;
            let h = last_hidden.ok_or_else(|| MetalError::InvalidShape("Missing legacy hidden after prompt (diag)".into()))?;
            let hidden_f16: Vec<f16> = h.as_slice().iter().map(|v| f16::from_f32(v.to_f32())).collect();
            let logits_f16: Vec<f16> = t.as_slice().iter().map(|v| f16::from_f32(v.to_f32())).collect();

            let v_hist_f16 = {
                let v_cache = {
                    let layer0 = fresh_ctx
                        .kv_caches()
                        .get(&0)
                        .ok_or_else(|| MetalError::InvalidOperation("Missing legacy v_cache layer 0 (diag)".into()))?;
                    layer0.v.clone()
                };
                let v_view = fresh_ctx.kv_cache_history_view(&v_cache, prompt_tokens.len())?.0;
                let v_data = v_view.try_to_vec()?;
                v_data.into_iter().map(|v| f16::from_f32(v.to_f32())).collect::<Vec<f16>>()
            };

            (hidden_f16, logits_f16, v_hist_f16, legacy_n_heads, legacy_n_kv_heads, kv_head_dim)
        };

        let (dsl_hidden_f16, dsl_logits_f16, dsl_v_cache_0_f16) = {
            use objc2_metal::{MTLBuffer, MTLDevice as _};

            let mut bindings = dsl_model.prepare_bindings(&mut foundry)?;

            let input_buffer = {
                let buf = foundry
                    .device
                    .newBufferWithLength_options(std::mem::size_of::<u32>(), objc2_metal::MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| MetalError::BufferCreationFailed(std::mem::size_of::<u32>()))?;
                metallic::types::MetalBuffer::from_retained(buf)
            };
            let input_tensor =
                metallic::types::TensorArg::from_buffer(input_buffer.clone(), metallic::tensor::Dtype::U32, vec![1], vec![1]);
            bindings.insert("input_ids".to_string(), input_tensor);

            let arch = dsl_model.architecture();
            let d_model = arch.d_model;
            let n_heads = arch.n_heads;
            let n_kv_heads = arch.n_kv_heads;
            let head_dim = d_model / n_heads;
            let ff_dim = arch.ff_dim;

            for (pos, &tok) in prompt_tokens.iter().enumerate() {
                unsafe {
                    let ptr = input_buffer.contents().as_ptr() as *mut u32;
                    *ptr = tok;
                }

                let kv_seq_len = pos + 1;
                bindings.set_global("seq_len", "1".to_string());
                bindings.set_global("position_offset", pos.to_string());
                bindings.set_global("kv_seq_len", kv_seq_len.to_string());
                bindings.set_global("total_elements_q", (n_heads * head_dim).to_string());
                bindings.set_global("total_elements_k", (n_kv_heads * head_dim).to_string());
                bindings.set_global("total_elements_hidden", d_model.to_string());
                bindings.set_global("total_elements_ffn", ff_dim.to_string());
                bindings.set_global("total_elements_slice", (n_kv_heads * kv_seq_len * head_dim).to_string());
                bindings.set_global("total_elements_repeat", (n_heads * kv_seq_len * head_dim).to_string());
                bindings.set_global("total_elements_write", (n_kv_heads * head_dim).to_string());

                dsl_model.forward(&mut foundry, &mut bindings)?;
            }

            let logits = bindings.get("logits")?;
            let hidden = bindings.get("final_norm_out").or_else(|_| bindings.get("hidden"))?;
            let v_cache_0 = bindings.get("v_cache_0")?;
            (
                read_f16_tensorarg(&hidden),
                read_f16_tensorarg(&logits),
                read_f16_tensorarg(&v_cache_0),
            )
        };

        let cpu_legacy_first = sample_topk_topp_cpu(&legacy_logits_f16, top_k as usize, top_p, temperature, base_seed);
        let cpu_dsl_first = sample_topk_topp_cpu(&dsl_logits_f16, top_k as usize, top_p, temperature, base_seed);

        let len = legacy_logits_f16.len().min(dsl_logits_f16.len());
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f32;
        let mut n = 0usize;
        let mut non_finite = 0usize;
        for i in 0..len {
            let a = legacy_logits_f16[i].to_f32();
            let b = dsl_logits_f16[i].to_f32();
            if !a.is_finite() || !b.is_finite() {
                non_finite += 1;
                continue;
            }
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            sum_abs += d;
            n += 1;
        }
        let avg_abs = if n == 0 { 0.0 } else { sum_abs / n as f32 };
        eprintln!(
            "First-sample logits compare: len={} max_abs_diff={:.6} avg_abs_diff={:.8} non_finite={}",
            len, max_abs, avg_abs, non_finite
        );
        let hidden_len = legacy_hidden_f16.len().min(dsl_hidden_f16.len());
        let mut hidden_max = 0.0f32;
        let mut hidden_sum = 0.0f32;
        let mut hidden_n = 0usize;
        for i in 0..hidden_len {
            let a = legacy_hidden_f16[i].to_f32();
            let b = dsl_hidden_f16[i].to_f32();
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            let d = (a - b).abs();
            hidden_max = hidden_max.max(d);
            hidden_sum += d;
            hidden_n += 1;
        }
        let hidden_avg = if hidden_n == 0 { 0.0 } else { hidden_sum / hidden_n as f32 };
        eprintln!(
            "First-sample hidden compare: len={} max_abs_diff={:.6} avg_abs_diff={:.8}",
            hidden_len, hidden_max, hidden_avg
        );

        // Compare KV cache history for layer 0 (V only), mapping legacy head-major caches to DSL KV-head caches.
        let active_seq = prompt_tokens.len();
        let group_size = legacy_n_heads / legacy_n_kv_heads;
        let mut v_max = 0.0f32;
        let mut v_sum = 0.0f32;
        let mut v_n = 0usize;
        for kv_head in 0..legacy_n_kv_heads {
            let legacy_head = kv_head * group_size;
            for seq_idx in 0..active_seq {
                let legacy_base = (legacy_head * active_seq + seq_idx) * legacy_head_dim;
                let dsl_base = (kv_head * 2048 + seq_idx) * legacy_head_dim;
                for d in 0..legacy_head_dim {
                    if legacy_base + d >= legacy_v_hist_f16.len() || dsl_base + d >= dsl_v_cache_0_f16.len() {
                        continue;
                    }
                    let a = legacy_v_hist_f16[legacy_base + d].to_f32();
                    let b = dsl_v_cache_0_f16[dsl_base + d].to_f32();
                    if !a.is_finite() || !b.is_finite() {
                        continue;
                    }
                    let diff = (a - b).abs();
                    v_max = v_max.max(diff);
                    v_sum += diff;
                    v_n += 1;
                }
            }
        }
        let v_avg = if v_n == 0 { 0.0 } else { v_sum / v_n as f32 };
        eprintln!(
            "First-sample v_cache(layer0) compare: max_abs_diff={:.6} avg_abs_diff={:.8} (n={})",
            v_max, v_avg, v_n
        );
        eprintln!(
            "CPU sampler (seed={}) first_token: legacy={} dsl={}",
            base_seed, cpu_legacy_first, cpu_dsl_first
        );

        let first_mismatch = legacy_trimmed.iter().zip(dsl_trimmed.iter()).position(|(a, b)| a != b);

        let legacy_text = tokenizer
            .decode(&legacy_trimmed)
            .unwrap_or_else(|_| "<legacy decode failed>".to_string());
        let dsl_text = tokenizer.decode(&dsl_trimmed).unwrap_or_else(|_| "<dsl decode failed>".to_string());

        eprintln!("Prompt: {prompt:?}");
        eprintln!("seed={base_seed} temp={temperature} top_k={top_k} top_p={top_p} max_new_tokens={max_new_tokens}");
        eprintln!("First mismatch at: {first_mismatch:?}");
        eprintln!("Legacy tokens (len={}): {:?}", legacy_trimmed.len(), legacy_trimmed);
        eprintln!("DSL tokens (len={}): {:?}", dsl_trimmed.len(), dsl_trimmed);
        eprintln!("\n=== Legacy decoded ===\n{legacy_text}");
        eprintln!("\n=== DSL decoded ===\n{dsl_text}");
    }

    assert_eq!(legacy_trimmed, dsl_trimmed, "Seeded generation token parity mismatch");

    Ok(())
}
