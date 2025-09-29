use crate::metallic::instrumentation::new_latency_collector;
use crate::metallic::models::{Qwen25, Qwen25Config};
use crate::metallic::{TensorInit, TensorStorage};

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
    let input = Tensor::new(
        vec![1, model.config.seq_len, model.config.d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )?;

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
    let batch = 1;
    let group_size = model.config.n_heads / model.config.n_kv_heads;

    // Pre-allocate KV cache
    let n_kv_heads = model.config.n_kv_heads;
    let d_model = model.config.d_model;
    let n_heads = model.config.n_heads;
    let kv_dim = d_model * n_kv_heads / n_heads;
    let kv_head_dim = kv_dim / n_kv_heads;
    let batch_size = 1;
    let kv_capacity = prompt_tokens.len().max(1);
    for i in 0..model.config.n_layers {
        ctx.alloc_kv_cache(i, kv_capacity, batch_size * n_kv_heads, batch_size * n_heads, kv_head_dim)?;
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

        // Validate that the repeated cache matches the legacy repeat kernel for both K and V.
        let cache_snapshots: Vec<_> = ctx.kv_caches.iter().map(|(&layer_idx, entry)| (layer_idx, entry.clone())).collect();

        for (layer_idx, entry) in cache_snapshots {
            let canonical_k_history = Qwen25::gather_cache_history(&entry.k, i + 1, &mut ctx)?;
            let canonical_v_history = Qwen25::gather_cache_history(&entry.v, i + 1, &mut ctx)?;
            let repeated_k_history = Qwen25::gather_cache_history(&entry.repeated_k, i + 1, &mut ctx)?;
            let repeated_v_history = Qwen25::gather_cache_history(&entry.repeated_v, i + 1, &mut ctx)?;

            let expected_k = Qwen25::repeat_kv_heads(&canonical_k_history, group_size, batch, n_kv_heads, n_heads, kv_head_dim, &mut ctx)?;
            let expected_v = Qwen25::repeat_kv_heads(&canonical_v_history, group_size, batch, n_kv_heads, n_heads, kv_head_dim, &mut ctx)?;

            ctx.synchronize();

            let actual_k = repeated_k_history.tensor.as_slice();
            let actual_v = repeated_v_history.tensor.as_slice();
            let expected_k_slice = expected_k.as_slice();
            let expected_v_slice = expected_v.as_slice();

            assert_eq!(
                actual_k.len(),
                expected_k_slice.len(),
                "K repeated length mismatch for layer {} at step {}",
                layer_idx,
                i
            );
            assert_eq!(
                actual_v.len(),
                expected_v_slice.len(),
                "V repeated length mismatch for layer {} at step {}",
                layer_idx,
                i
            );

            for (idx, (actual, expected)) in actual_k.iter().zip(expected_k_slice.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "K repeat mismatch at layer {}, step {}, element {}: got {} expected {}",
                    layer_idx,
                    i,
                    idx,
                    actual,
                    expected
                );
            }

            for (idx, (actual, expected)) in actual_v.iter().zip(expected_v_slice.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "V repeat mismatch at layer {}, step {}, element {}: got {} expected {}",
                    layer_idx,
                    i,
                    idx,
                    actual,
                    expected
                );
            }
        }
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

#[test]
fn test_repeat_kv_heads_gpu_matches_cpu() -> Result<(), MetalError> {
    let mut ctx = Context::new()?;

    let batch = 2usize;
    let n_kv_heads = 2usize;
    let group_size = 3usize;
    let n_heads = n_kv_heads * group_size;
    let seq = 4usize;
    let head_dim = 6usize;

    let element_count = batch * n_kv_heads * seq * head_dim;
    let input_data: Vec<f32> = (0..element_count).map(|v| v as f32).collect();
    let input = Tensor::new(
        vec![batch * n_kv_heads, seq, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )?;
    let history = CacheHistory {
        tensor: input.clone(),
        active_seq: seq,
        cache_capacity: seq,
    };

    let expected = {
        let mut out = vec![0.0f32; batch * n_heads * seq * head_dim];
        for b in 0..batch {
            for h_kv in 0..n_kv_heads {
                let input_offset_base = ((b * n_kv_heads + h_kv) * seq) * head_dim;
                for g in 0..group_size {
                    let h = h_kv * group_size + g;
                    let output_offset_base = ((b * n_heads + h) * seq) * head_dim;
                    for s in 0..seq {
                        let input_offset = input_offset_base + s * head_dim;
                        let output_offset = output_offset_base + s * head_dim;
                        out[output_offset..output_offset + head_dim].copy_from_slice(&input_data[input_offset..input_offset + head_dim]);
                    }
                }
            }
        }
        out
    };

    let output = Qwen25::repeat_kv_heads(&history, group_size, batch, n_kv_heads, n_heads, head_dim, &mut ctx)?;
    ctx.synchronize();

    assert_eq!(output.dims(), &[batch * n_heads, seq, head_dim]);
    let gpu_values = output.as_slice();
    assert_eq!(gpu_values.len(), expected.len());
    for (idx, (gpu, cpu)) in gpu_values.iter().zip(expected.iter()).enumerate() {
        assert!((gpu - cpu).abs() < 1e-5, "Mismatch at index {}: gpu={} expected={}", idx, gpu, cpu);
    }

    Ok(())
}

#[test]
fn test_forward_step_records_kv_repeat_phase() -> Result<(), MetalError> {
    let mut ctx = Context::new()?;
    let cfg = Qwen25Config {
        n_layers: 1,
        d_model: 16,
        ff_dim: 32,
        n_heads: 2,
        n_kv_heads: 1,
        seq_len: 8,
        vocab_size: 32,
        rope_freq_base: 1e6,
        rms_eps: 1e-6,
    };
    let model = Qwen25::new(cfg, &mut ctx)?;

    let block = &model.blocks[0];
    let kv_dim = block.kv_dim;
    let kv_head_dim = kv_dim / model.config.n_kv_heads;
    let batch_size = 1;
    let canonical_heads = batch_size * model.config.n_kv_heads;
    let repeated_heads = batch_size * model.config.n_heads;
    let kv_capacity = 1usize;
    for layer_idx in 0..model.config.n_layers {
        ctx.alloc_kv_cache(layer_idx, kv_capacity, canonical_heads, repeated_heads, kv_head_dim)?;
    }

    let collector = new_latency_collector(model.config.n_layers);
    ctx.set_latency_collector(Some(collector.clone()));

    let input = model.embed(&[0], &mut ctx)?;
    let _ = model.forward_step(&input, 0, &mut ctx)?;
    ctx.synchronize();
    ctx.set_latency_collector(None);

    let snapshot = collector.borrow().snapshot();
    assert_eq!(snapshot.blocks.len(), model.config.n_layers);
    let has_kv_repeat = snapshot.blocks[0].phases.iter().any(|phase| phase.label == "kv_repeat");
    assert!(has_kv_repeat, "kv_repeat phase was not recorded");

    Ok(())
}

#[test]
fn test_gather_cache_history_gpu_path() -> Result<(), MetalError> {
    let mut ctx = Context::new()?;

    let seq = 4;
    let batch_heads = 3;
    let head_dim = 5;
    let mut data = Vec::with_capacity(seq * batch_heads * head_dim);
    for bh in 0..batch_heads {
        for s in 0..seq {
            for d in 0..head_dim {
                data.push((bh * 100 + s * 10 + d) as f32);
            }
        }
    }

    let cache = Tensor::new(
        vec![batch_heads, seq, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&data),
    )?;

    for steps in 1..=seq {
        let history = Qwen25::gather_cache_history(&cache, steps, &mut ctx)?;
        assert_eq!(history.tensor.dims(), &[batch_heads, steps, head_dim]);
        assert_eq!(history.active_seq, steps);
        assert_eq!(history.cache_capacity, seq);

        let mut gpu_values = Vec::with_capacity(steps * batch_heads * head_dim);
        for bh in 0..batch_heads {
            let batch_view = history.tensor.get_batch(bh)?;
            gpu_values.extend_from_slice(batch_view.as_slice());
        }
        let mut expected = Vec::with_capacity(steps * batch_heads * head_dim);
        for bh in 0..batch_heads {
            for s in 0..steps {
                let src_idx = (bh * seq + s) * head_dim;
                expected.extend_from_slice(&data[src_idx..src_idx + head_dim]);
            }
        }

        assert_eq!(gpu_values.len(), expected.len());
        for (idx, (gpu, exp)) in gpu_values.iter().zip(expected.iter()).enumerate() {
            assert!(
                (gpu - exp).abs() < 1e-5,
                "Mismatch at steps={}, element {}: gpu={} expected={}",
                steps,
                idx,
                gpu,
                exp
            );
        }
    }

    Ok(())
}
