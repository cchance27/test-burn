use crate::metallic::generation::{GenerationConfig, aggregate_matmul_totals, generate};
use crate::metallic::kernels::matmul::{MatMulBackend, MatMulSample};
use crate::metallic::kernels::sampling::MAX_TOP_K;
use crate::metallic::models::{Qwen25, Qwen25Config};
use crate::metallic::sampling::{sample_top_k_top_p, sample_top_k_top_p_with_random_value};
use crate::metallic::{
    Context, F16Element, F32Element, MetalError, SamplerBuffers, Tensor, TensorElement, TensorInit, TensorStorage, Tokenizer,
};
use half::f16;
use rustc_hash::FxHashMap;

#[test]
fn test_generation_pipeline() {
    let mut ctx = Context::<F32Element>::new().unwrap();
    let cfg = Qwen25Config {
        n_layers: 1,
        d_model: 8,
        ff_dim: 16,
        n_heads: 2,
        n_kv_heads: 1,
        seq_len: 256,
        vocab_size: 32,
        rope_freq_base: 1e6,
        rms_eps: 1e-6,
    };
    let mut model = Qwen25::new(cfg, &mut ctx).unwrap();

    // Create a simple tokenizer for testing
    use rustc_hash::FxHashMap;
    let mut vocab = FxHashMap::default();
    vocab.insert(0, "<unk>".to_string());
    vocab.insert(1, "hello".to_string());
    vocab.insert(2, "world".to_string());
    vocab.insert(3, "test".to_string());

    // Add more tokens to match the model's vocab_size
    for i in 4..32 {
        vocab.insert(i, format!("token_{}", i));
    }

    let merges = vec![];
    let tokenizer = Tokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    // Test generation with a simple prompt
    let prompt = "hello";
    let gen_config = GenerationConfig {
        max_tokens: 10,
        temperature: 1.0,
        top_p: 0.95,
        top_k: 40,
    };

    // This test just verifies that the generation pipeline can be called
    // without crashing. Actual generation quality would require a trained model.
    let result = generate(&mut model, &tokenizer, &mut ctx, prompt, &gen_config);

    // Print the error if there is one
    match &result {
        Ok(output) => println!("Generation successful: {}", output),
        Err(e) => println!("Generation failed with error: {:?}", e),
    }

    // For this test, we're just checking that the pipeline runs without panicking
    // The actual output will be random since we're using an untrained model
    // We'll consider the test passed if it doesn't panic
    assert!(result.is_ok(), "Generation should succeed or at least not panic");

    println!("Generation test completed successfully");
}

#[test]
fn test_full_generation_correctness() -> Result<(), crate::metallic::MetalError> {
    let mut ctx = Context::<F32Element>::new()?;
    let cfg = Qwen25Config {
        n_layers: 2,
        d_model: 32,
        ff_dim: 64,
        n_heads: 4,
        n_kv_heads: 2,
        seq_len: 256,
        vocab_size: 100,
        rope_freq_base: 1e6,
        rms_eps: 1e-6,
    };
    let mut model = Qwen25::new(cfg, &mut ctx)?;

    let mut vocab = FxHashMap::default();
    for i in 0..100 {
        vocab.insert(i, format!("token_{}", i));
    }
    let tokenizer = Tokenizer::from_vocab_and_merges(vocab, vec![]).unwrap();

    let prompt = "hello world";
    let input_ids = tokenizer.encode(prompt)?;
    let gen_cfg = GenerationConfig {
        max_tokens: 5,
        temperature: 0.0, // Use 0 for deterministic greedy sampling
        top_p: 1.0,
        top_k: 1,
    };

    // --- Run 1: No-Cache Reference Implementation ---
    fn generate_reference<T: TensorElement>(
        model: &mut Qwen25<T>,
        ctx: &mut Context<T>,
        input_ids: &[u32],
        gen_cfg: &GenerationConfig,
    ) -> Result<Vec<u32>, MetalError> {
        let mut generated = input_ids.to_vec();
        let vocab_size = model.config.vocab_size;

        for _ in 0..gen_cfg.max_tokens {
            ctx.reset_pool();
            ctx.clear_cache(); // Clear cache to ensure no caching between steps

            let input_tensor = model.embed(&generated, ctx)?;
            let hidden_states = model.forward(&input_tensor, ctx)?;
            let logits_tensor = model.output(&hidden_states, ctx)?;

            let logits = logits_tensor.to_vec();
            let seq_len = generated.len();
            let start_idx = (seq_len - 1) * vocab_size;
            let end_idx = start_idx + vocab_size;
            let vocab_slice = &logits[start_idx..end_idx];

            // Greedy sampling (argmax)
            let next_token = vocab_slice
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            println!(
                "[Ref] Step {}: token={}, logits={:?}",
                generated.len() - input_ids.len(),
                next_token,
                &vocab_slice[..10]
            );

            generated.push(next_token);
        }
        Ok(generated)
    }

    let reference_ids = generate_reference(&mut model, &mut ctx, &input_ids, &gen_cfg)?;

    // --- Run 2: KV Cache Implementation ---
    // Reset context to ensure a clean run
    ctx.clear_kv_caches();
    let kv_cache_new_ids =
        crate::metallic::generation::generate_autoregressive_with_kv_cache(&mut model, &tokenizer, &mut ctx, &input_ids, &gen_cfg, &[])?;
    let mut kv_cache_ids = input_ids.clone();
    kv_cache_ids.extend(kv_cache_new_ids);

    // --- Compare ---
    if reference_ids != kv_cache_ids {
        let mismatch_index = reference_ids.iter().zip(kv_cache_ids.iter()).position(|(a, b)| a != b);

        if let Some(idx) = mismatch_index {
            println!(
                "❌ Divergence detected at position {} (reference={}, kv={})",
                idx, reference_ids[idx], kv_cache_ids[idx]
            );
        } else if reference_ids.len() != kv_cache_ids.len() {
            println!(
                "❌ Sequence length mismatch: reference {} tokens vs KV {} tokens",
                reference_ids.len(),
                kv_cache_ids.len()
            );
        }

        let reference_text = tokenizer.decode(&reference_ids).unwrap_or_default();
        let kv_text = tokenizer.decode(&kv_cache_ids).unwrap_or_default();
        println!("Reference text: {}", reference_text);
        println!("KV-cache text: {}", kv_text);

        panic!(
            "Mismatch between reference and KV cache generation!\nReference: {:?}\nKV Cache:  {:?}",
            reference_ids, kv_cache_ids
        );
    }

    println!("✅ KV cache implementation passed full generation correctness test.");

    Ok(())
}

#[test]
fn test_sample_top_k_top_p_top_k_reduction_matches_reference() {
    let logits = vec![0.1f32, 0.9f32, 0.8f32, 0.7f32, 0.6f32];
    let top_k = 2usize;
    let top_p = 0.3f32; // Ensures only the maximum-probability token remains after top-p filtering.
    let temperature = 1.0f32;

    let result = run_sampler(&logits, top_k, top_p, temperature);

    assert_eq!(result, 1, "Top-k reduction should preserve the highest-probability index");
}

#[test]
fn test_sample_top_k_top_p_zero_temperature_prefers_highest_index() {
    let logits = vec![0.0f32, 5.0f32, 5.0f32, 4.0f32];
    let top_k = 4usize;
    let top_p = 0.95f32;
    let temperature = 0.0f32;

    let result = run_sampler(&logits, top_k, top_p, temperature);

    assert_eq!(
        result, 2,
        "Zero temperature should greedily select the highest logit, preferring the last index on ties"
    );
}

#[test]
fn matmul_sample_aggregation_sums_backend_totals() {
    use std::time::Duration;

    let totals = aggregate_matmul_totals(vec![
        MatMulSample {
            backend: MatMulBackend::Mps,
            duration: Duration::from_millis(8),
            dims: None,
            handle: None,
        },
        MatMulSample {
            backend: MatMulBackend::Mps,
            duration: Duration::from_millis(4),
            dims: None,
            handle: None,
        },
        MatMulSample {
            backend: MatMulBackend::Mps,
            duration: Duration::from_millis(0),
            dims: None,
            handle: None,
        },
    ]);

    let total = totals
        .get(&MatMulBackend::Mps)
        .copied()
        .expect("aggregated totals should include the MPS backend");

    assert_eq!(total, Duration::from_millis(12));
    assert_eq!(totals.len(), 1);
}

#[test]
fn test_sample_top_k_top_p_ignores_nan_logits_in_probabilities() {
    let logits = vec![f32::NAN, 0.2f32, f32::NAN, 0.4f32];
    let top_k = 4usize;
    let top_p = 0.5f32;
    let temperature = 1.0f32;

    let result = run_sampler(&logits, top_k, top_p, temperature);

    assert_eq!(result, 3, "NaN logits should be ignored during sampling");
}

#[test]
fn test_device_sampling_matches_cpu_with_fixed_seed() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;
    let logits: Vec<f32> = vec![0.25, -0.15, 1.2, 0.7, 0.05, -1.0, 0.9, 0.33, 0.61, -0.42, 1.75, 0.0];
    let tensor = {
        let ctx_ref: &Context<F32Element> = &ctx;
        Tensor::new(vec![logits.len()], TensorStorage::Dedicated(ctx_ref), TensorInit::CopyFrom(&logits))?
    };

    let top_k = 5usize;
    let top_p = 0.85f32;
    let temperature = 0.95f32;
    ctx.reseed_sampler(1234);
    let mut buffers = SamplerBuffers::default();

    for _ in 0..4 {
        let random = ctx.next_sampler_random();
        let gpu_token = ctx
            .sample_top_k_top_p_device(&tensor, logits.len(), top_k, top_p, temperature, random)?
            .expect("GPU kernel should support f32 logits within limit");
        let cpu_idx = sample_top_k_top_p_with_random_value::<F32Element>(&logits, top_k, top_p, temperature, random, &mut buffers);
        assert_eq!(gpu_token as usize, cpu_idx);
    }

    Ok(())
}

#[test]
fn test_device_sampling_matches_cpu_with_fixed_seed_f16() -> Result<(), MetalError> {
    let mut ctx = Context::<F16Element>::new()?;
    let logits: Vec<f32> = vec![0.25, -0.15, 1.2, 0.7, 0.05, -1.0, 0.9, 0.33, 0.61, -0.42, 1.75, 0.0];
    let logits_f16: Vec<f16> = logits.iter().copied().map(f16::from_f32).collect();
    let tensor = {
        let ctx_ref: &Context<F16Element> = &ctx;
        Tensor::new(
            vec![logits_f16.len()],
            TensorStorage::Dedicated(ctx_ref),
            TensorInit::CopyFrom(&logits_f16),
        )?
    };

    let top_k = 5usize;
    let top_p = 0.85f32;
    let temperature = 0.95f32;
    ctx.reseed_sampler(1234);
    let mut buffers = SamplerBuffers::default();

    for _ in 0..4 {
        let random = ctx.next_sampler_random();
        let gpu_token = ctx
            .sample_top_k_top_p_device(&tensor, logits_f16.len(), top_k, top_p, temperature, random)?
            .expect("GPU kernel should support f16 logits within limit");
        let cpu_idx = sample_top_k_top_p_with_random_value::<F16Element>(&logits_f16, top_k, top_p, temperature, random, &mut buffers);
        assert_eq!(gpu_token as usize, cpu_idx);
    }

    Ok(())
}

#[test]
fn test_device_sampling_falls_back_for_large_top_k() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;
    let vocab_size = MAX_TOP_K + 16;
    let logits: Vec<f32> = (0..vocab_size).map(|i| ((i % 19) as f32) * 0.1 - 0.8).collect();
    let tensor = {
        let ctx_ref: &Context<F32Element> = &ctx;
        Tensor::new(vec![logits.len()], TensorStorage::Dedicated(ctx_ref), TensorInit::CopyFrom(&logits))?
    };

    let top_k = MAX_TOP_K + 8;
    let top_p = 0.9f32;
    let temperature = 1.1f32;
    ctx.reseed_sampler(42);
    let random = ctx.next_sampler_random();

    let device_result = ctx.sample_top_k_top_p_device(&tensor, vocab_size, top_k, top_p, temperature, random)?;
    assert!(device_result.is_none(), "device sampler should defer when top-k exceeds limit");

    let mut buffers = SamplerBuffers::default();
    let cpu_idx = ctx.sample_top_k_top_p_host(&tensor, vocab_size, top_k, top_p, temperature, random)?;
    let expected = sample_top_k_top_p_with_random_value::<F32Element>(&logits, top_k, top_p, temperature, random, &mut buffers);
    assert_eq!(cpu_idx as usize, expected);

    Ok(())
}

fn run_sampler(logits: &[f32], top_k: usize, top_p: f32, temperature: f32) -> usize {
    let mut buffers = SamplerBuffers::default();
    sample_top_k_top_p::<F32Element>(logits, top_k, top_p, temperature, &mut buffers)
}
