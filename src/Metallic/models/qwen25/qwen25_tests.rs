#![cfg(test)]
use crate::metallic::models::{Qwen25, Qwen25Config};

use super::*;

#[test]
fn test_qwen25_basic_construct_and_forward() -> Result<(), MetalError> {
    let mut ctx = Context::new()?;
    let cfg = Qwen25Config {
        n_layers: 2,
        d_model: 8,
        ff_dim: 32, // small FF for tests
        n_heads: 2,
        n_kv_heads: 1,
        seq_len: 4,
        vocab_size: 32,
        rope_freq_base: 1e6,
        rms_eps: 1e-6,
    };
    let model = Qwen25::new(cfg, &mut ctx)?;

    let input_data: Vec<f32> = vec![0.5; model.config.seq_len * model.config.d_model];
    let input = Tensor::create_tensor_from_slice(&input_data, vec![1, model.config.seq_len, model.config.d_model], &ctx)?;

    let out = model.forward(&input, &mut ctx)?;
    assert_eq!(out.dims(), input.dims());
    Ok(())
}

#[test]
fn test_qwen25_embed() -> Result<(), MetalError> {
    let mut ctx = Context::new()?;
    let cfg = Qwen25Config {
        n_layers: 1,
        d_model: 8,
        ff_dim: 16,
        n_heads: 2,
        n_kv_heads: 1,
        seq_len: 4,
        vocab_size: 16,
        rope_freq_base: 1e6,
        rms_eps: 1e-6,
    };
    let model = Qwen25::new(cfg, &mut ctx)?;

    // Test embedding a few tokens
    let tokens = vec![1, 2, 3];
    let embedded = model.embed(&tokens, &mut ctx)?;

    // Check the output shape
    assert_eq!(embedded.dims(), &[1, 3, 8]);

    Ok(())
}

#[test]
fn test_qwen25_forward_tokens() -> Result<(), MetalError> {
    let mut ctx = Context::new()?;
    let cfg = Qwen25Config {
        n_layers: 1,
        d_model: 8,
        ff_dim: 16,
        n_heads: 2,
        n_kv_heads: 1,
        seq_len: 4,
        vocab_size: 16,
        rope_freq_base: 1e6,
        rms_eps: 1e-6,
    };
    let model = Qwen25::new(cfg, &mut ctx)?;

    // Test forward pass with tokens
    let tokens = vec![1, 2, 3];
    let logits = model.forward_tokens(&tokens, &mut ctx)?;

    // Check the output shape - should be [batch, seq, vocab_size]
    assert_eq!(logits.dims(), &[1, 3, 16]);

    Ok(())
}

#[test]
fn test_kv_cache_correctness() -> Result<(), MetalError> {
    let mut ctx = Context::new()?;
    let cfg = Qwen25Config {
        n_layers: 1,
        d_model: 16,
        ff_dim: 32,
        n_heads: 2,
        n_kv_heads: 1,
        seq_len: 16,
        vocab_size: 16,
        rope_freq_base: 1e6,
        rms_eps: 1e-6,
    };
    let model = Qwen25::new(cfg, &mut ctx)?;

    let prompt_tokens = [1, 2, 3, 4, 5];
    let vocab_size = model.config.vocab_size;

    // Pre-allocate KV cache
    let n_kv_heads = model.config.n_kv_heads;
    let d_model = model.config.d_model;
    let n_heads = model.config.n_heads;
    let kv_dim = d_model * n_kv_heads / n_heads;
    let kv_head_dim = kv_dim / n_kv_heads;
    for i in 0..model.config.n_layers {
        ctx.alloc_kv_cache(i, model.config.seq_len, n_kv_heads, kv_head_dim)?;
    }

    // --- Multi-step correctness check ---
    let mut kv_cache_logits_history = Vec::new();

    // 1. Run incremental `forward_step` for each token
    ctx.reset_pool();
    ctx.clear_cache();
    let tmp = 0..prompt_tokens.len();
    for i in tmp {
        let token_embedding = model.embed(&[prompt_tokens[i]], &mut ctx)?;
        let hidden_state = model.forward_step(&token_embedding, i, &mut ctx)?;
        let logits_tensor = model.output(&hidden_state, &mut ctx)?;
        kv_cache_logits_history.push(logits_tensor.to_vec());
    }

    // 2. Run full `forward` for each sequence length and compare
    for i in 0..prompt_tokens.len() {
        ctx.reset_pool();
        ctx.clear_cache();
        let current_sequence = &prompt_tokens[0..=i];
        let sequence_embedding = model.embed(current_sequence, &mut ctx)?;
        let hidden_state = model.forward(&sequence_embedding, &mut ctx)?;
        let logits_tensor = model.output(&hidden_state, &mut ctx)?;
        let full_forward_logits = logits_tensor.to_vec();

        // Get the logits for the last token from the full forward pass
        let last_token_logits = full_forward_logits[i * vocab_size..].to_vec();

        // Get the corresponding logits from the KV cache run
        let kv_cache_logits = &kv_cache_logits_history[i];

        // Compare
        assert_eq!(last_token_logits.len(), kv_cache_logits.len());
        let mut diff_sum = 0.0;
        for j in 0..last_token_logits.len() {
            diff_sum += (last_token_logits[j] - kv_cache_logits[j]).abs();
        }
        let avg_diff = diff_sum / last_token_logits.len() as f32;

        assert!(avg_diff < 1e-5, "Logits mismatch at step {}. Avg diff: {}", i, avg_diff);
        println!("✅ Logits match at step {}.", i);
    }

    Ok(())
}
