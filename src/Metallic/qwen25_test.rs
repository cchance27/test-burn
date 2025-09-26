#![cfg(test)]
use crate::metallic::qwen25::{Qwen25, Qwen25Config};

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
