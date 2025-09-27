use crate::metallic::generation::{GenerationConfig, generate};
use crate::metallic::models::{Qwen25, Qwen25Config};
use crate::metallic::{Context, MetalError, Tensor, Tokenizer};
use rustc_hash::FxHashMap;

#[test]
fn test_generation_pipeline() {
    let mut ctx = Context::new().unwrap();
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
    let mut ctx = Context::new()?;
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
    fn generate_reference(
        model: &mut Qwen25,
        ctx: &mut Context,
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
            let vocab_logits = &logits[start_idx..end_idx];

            // Greedy sampling (argmax)
            let next_token = vocab_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            println!(
                "[Ref] Step {}: token={}, logits={:?}",
                generated.len() - input_ids.len(),
                next_token,
                &vocab_logits[..10]
            );

            generated.push(next_token);
        }
        Ok(generated)
    }

    let reference_ids = generate_reference(&mut model, &mut ctx, &input_ids, &gen_cfg)?;

    // --- Run 2: KV Cache Implementation ---
    // Reset context to ensure a clean run
    ctx.kv_caches.clear();
    let kv_cache_new_ids =
        crate::metallic::generation::generate_autoregressive_with_kv_cache(&mut model, &tokenizer, &mut ctx, &input_ids, &gen_cfg)?;
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
