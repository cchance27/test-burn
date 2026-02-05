//! Test suite for SampleTopKTopP kernel.

use half::f16;
use metallic_foundry::{
    Foundry, metals::sampling::{ApplyRepetitionPenalty, RepetitionStateInit, SampleTopK}, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit, dtypes::U32}, types::TensorArg
};
use serial_test::serial;

#[test]
#[serial]
fn test_sample_topk_foundry_argmax() {
    let mut foundry = Foundry::new().unwrap();

    let vocab_size = 32000;
    let k = 1;
    let top_p = 1.0;
    let temperature = 1.0;
    let seed = 42;

    // Argmax check: 100.0 vs 0.0
    let mut logits_data: Vec<f16> = vec![f16::from_f32(0.0); vocab_size];
    let expected_token = 12345;
    logits_data[expected_token] = f16::from_f32(100.0);

    let logits_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![vocab_size], TensorInit::CopyFrom(&logits_data)).unwrap();

    let output_foundry = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![1], TensorInit::Uninitialized).unwrap();

    let logits_arg = TensorArg::from_tensor(&logits_foundry);
    let output_arg = TensorArg::from_tensor(&output_foundry);

    let min_p = 0.0;
    let kernel = SampleTopK::new(&logits_arg, &output_arg, vocab_size as u32, k, top_p, min_p, temperature, seed);

    foundry.run(&kernel).unwrap();

    let foundry_token = FoundryTensor::to_vec(&output_foundry, &foundry)[0];
    println!("Foundry Token: {}, Expected: {}", foundry_token, expected_token);
    assert_eq!(foundry_token, expected_token as u32, "Foundry failed Argmax");
}

#[test]
#[serial]
fn test_sample_topk_fused_determinism() {
    let mut foundry = Foundry::new().unwrap();

    let vocab_size = 10000;
    let k = 40;
    let top_p = 0.95;
    let min_p = 0.0;
    let temperature = 1.0;
    let seed = 12345;

    let logits_data: Vec<f16> = (0..vocab_size).map(|i| f16::from_f32((i % 100) as f32)).collect();

    let logits_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![vocab_size], TensorInit::CopyFrom(&logits_data)).unwrap();

    let output_foundry_1 = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![1], TensorInit::Uninitialized).unwrap();

    let output_foundry_2 = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![1], TensorInit::Uninitialized).unwrap();

    let kernel1 = SampleTopK::new(
        &TensorArg::from_tensor(&logits_foundry),
        &TensorArg::from_tensor(&output_foundry_1),
        vocab_size as u32,
        k,
        top_p,
        min_p,
        temperature,
        seed,
    );
    foundry.run(&kernel1).unwrap();

    let kernel2 = SampleTopK::new(
        &TensorArg::from_tensor(&logits_foundry),
        &TensorArg::from_tensor(&output_foundry_2),
        vocab_size as u32,
        k,
        top_p,
        min_p,
        temperature,
        seed,
    );
    foundry.run(&kernel2).unwrap();

    let token1 = FoundryTensor::to_vec(&output_foundry_1, &foundry)[0];
    let token2 = FoundryTensor::to_vec(&output_foundry_2, &foundry)[0];

    assert_eq!(token1, token2, "Determinism failed");
}

#[test]
#[serial]
fn test_repetition_state_penalty_changes_argmax() {
    let mut foundry = Foundry::new().unwrap();

    let vocab_size = 4096u32;
    let repeat_tok = 1234u32;
    let other_tok = 2345u32;

    let mut logits: Vec<f16> = vec![f16::from_f32(0.0); vocab_size as usize];
    logits[repeat_tok as usize] = f16::from_f32(10.0);
    logits[other_tok as usize] = f16::from_f32(9.0);

    let logits_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![vocab_size as usize], TensorInit::CopyFrom(&logits)).unwrap();
    let output_foundry = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![1], TensorInit::Uninitialized).unwrap();

    let logits_arg = TensorArg::from_tensor(&logits_foundry);
    let output_arg = TensorArg::from_tensor(&output_foundry);

    // Create repetition state buffers for a small window.
    let window_len = 8usize;
    let ring = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![window_len], TensorInit::Uninitialized).unwrap();
    let pairs = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![window_len * 2], TensorInit::Uninitialized).unwrap();
    let meta = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![4], TensorInit::Uninitialized).unwrap();

    let ring_arg = TensorArg::from_tensor(&ring);
    let pairs_arg = TensorArg::from_tensor(&pairs);
    let meta_arg = TensorArg::from_tensor(&meta);

    // Seed state with the repeated token.
    let seed_tokens = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(&[repeat_tok])).unwrap();
    let seed_arg = TensorArg::from_tensor(&seed_tokens);

    foundry
        .run(&RepetitionStateInit::new(
            &ring_arg,
            &pairs_arg,
            &meta_arg,
            &seed_arg,
            1,
            window_len as u32,
        ))
        .unwrap();

    // Apply a strong repeat penalty so repeat_tok loses.
    foundry
        .run(&ApplyRepetitionPenalty::new(
            &logits_arg,
            &pairs_arg,
            vocab_size,
            window_len as u32,
            10.0,
            0.0,
            0.0,
        ))
        .unwrap();

    // Now argmax sampling should pick other_tok.
    foundry
        .run(&SampleTopK::new(&logits_arg, &output_arg, vocab_size, 1, 1.0, 0.0, 1.0, 42))
        .unwrap();

    let sampled = FoundryTensor::to_vec(&output_foundry, &foundry)[0];
    assert_eq!(sampled, other_tok);
}

#[derive(Clone, Copy)]
struct CpuRng {
    state: u32,
}

impl CpuRng {
    fn mix(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^ (x >> 16)
    }

    fn seeded(seed: u32) -> Self {
        let v = if seed == 0 { 1 } else { seed };
        let mut state = Self::mix(v);
        if state == 0 {
            state = 1;
        }
        Self { state }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        let bits = (x >> 9) | 0x3f80_0000;
        f32::from_bits(bits) - 1.0
    }
}

fn cpu_sample_topk_topp(logits: &[f32], top_k: u32, top_p: f32, min_p: f32, temperature: f32, seed: u32) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    let mut fallback_idx = 0usize;
    let mut fallback_found = false;
    let mut fallback_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && (!fallback_found || v > fallback_val || (v == fallback_val && i < fallback_idx)) {
            fallback_idx = i;
            fallback_val = v;
            fallback_found = true;
        }
    }
    let fallback = if fallback_found { fallback_idx as u32 } else { 0 };

    let denom = temperature.max(1.0e-6);
    let inv_t = 1.0f32 / denom;
    let k = if top_k == 0 { 1usize } else { (top_k as usize).min(logits.len()) };
    if k == 0 {
        return fallback;
    }

    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            if !v.is_finite() {
                return None;
            }
            Some((i as u32, v * inv_t))
        })
        .collect();

    if candidates.is_empty() {
        return fallback;
    }

    candidates.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    candidates.truncate(k);

    let maxv = candidates[0].1;
    let mut probs: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
    let mut sum = 0.0f32;
    for (idx, v) in candidates {
        let e = (v - maxv).exp();
        if e.is_finite() && e > 0.0 {
            probs.push((idx, e));
            sum += e;
        }
    }

    if probs.is_empty() || !sum.is_finite() || sum <= 0.0 {
        return fallback;
    }

    let top_p = if top_p.is_finite() { top_p.clamp(0.0, 1.0) } else { 1.0 };
    let min_p = if min_p.is_finite() { min_p.clamp(0.0, 1.0) } else { 0.0 };
    let max_prob = probs[0].1 / sum;
    let min_p_cut = if min_p > 0.0 { min_p * max_prob } else { 0.0 };

    let mut filtered: Vec<(u32, f32)> = Vec::with_capacity(probs.len());
    let mut cumulative = 0.0f32;
    for (idx, e) in probs {
        let p = e / sum;
        if min_p_cut > 0.0 && p < min_p_cut {
            continue;
        }
        filtered.push((idx, p));
        cumulative += p;
        if cumulative >= top_p {
            break;
        }
    }

    if filtered.is_empty() {
        return fallback;
    }

    let renorm = if cumulative.is_finite() && cumulative > 0.0 {
        1.0f32 / cumulative
    } else {
        1.0
    };
    for (_, p) in &mut filtered {
        *p *= renorm;
    }

    let mut rng = CpuRng::seeded(seed);
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    let mut chosen = filtered[0].0;
    for (idx, p) in filtered {
        acc += p;
        if r <= acc {
            chosen = idx;
            break;
        }
    }
    chosen
}

fn next_u32(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

fn build_logits(seed: u32, vocab_size: usize) -> Vec<f16> {
    let mut state = seed;
    let mut logits = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        let raw = next_u32(&mut state);
        let base = ((raw >> 8) % 20000) as f32 * 0.001 - 10.0;
        let slope = (i as f32) * 1.0e-6;
        logits.push(f16::from_f32(base + slope));
    }

    // Inject a sparse set of stronger candidates to emulate real decode logits tails.
    for j in 0..64u32 {
        let idx = (next_u32(&mut state) as usize) % vocab_size;
        let bonus = 6.0 + (j as f32) * 0.03125;
        logits[idx] = f16::from_f32(logits[idx].to_f32() + bonus);
    }
    logits
}

#[test]
#[serial]
fn test_sample_topk_gpu_cpu_parity_stress() {
    let mut foundry = Foundry::new().unwrap();

    let vocab_size = 151_936usize;
    let k = 40u32;
    let top_p = 0.95f32;
    let min_p = 0.05f32;
    let temperature = 0.8f32;

    let mut mismatches = Vec::new();
    for case in 0..24u32 {
        let logits_data = build_logits(0xC0FFEE00u32.wrapping_add(case.wrapping_mul(97)), vocab_size);
        let logits_f32: Vec<f32> = logits_data.iter().map(|v| v.to_f32()).collect();

        let logits_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![vocab_size], TensorInit::CopyFrom(&logits_data)).unwrap();
        let output_foundry = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![1], TensorInit::Uninitialized).unwrap();

        let seed = 42u32.wrapping_add(case.wrapping_mul(9973));
        let kernel = SampleTopK::new(
            &TensorArg::from_tensor(&logits_foundry),
            &TensorArg::from_tensor(&output_foundry),
            vocab_size as u32,
            k,
            top_p,
            min_p,
            temperature,
            seed,
        );
        foundry.run(&kernel).unwrap();

        let gpu_token = FoundryTensor::to_vec(&output_foundry, &foundry)[0];
        let cpu_token = cpu_sample_topk_topp(&logits_f32, k, top_p, min_p, temperature, seed);

        if gpu_token != cpu_token {
            mismatches.push((case, gpu_token, cpu_token));
            if mismatches.len() >= 8 {
                break;
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "SampleTopK GPU/CPU parity mismatch count={} examples={:?}",
        mismatches.len(),
        mismatches
    );
}

#[test]
#[serial]
fn test_sample_topk_gpu_cpu_parity_concentrated_lane() {
    let mut foundry = Foundry::new().unwrap();

    let vocab_size = 1024usize * 192usize;
    let k = 40u32;
    let top_p = 1.0f32;
    let min_p = 0.0f32;
    let temperature = 1.0f32;

    let mut logits_data = vec![f16::from_f32(-20.0); vocab_size];
    let base_lane = 7usize;
    // Deliberately concentrate >top_k strong candidates into one lane bucket
    // (same index modulo threads-per-threadgroup) to stress per-thread truncation.
    for n in 0..80usize {
        let idx = base_lane + n * 1024usize;
        logits_data[idx] = f16::from_f32(8.0);
    }

    let logits_f32: Vec<f32> = logits_data.iter().map(|v| v.to_f32()).collect();
    let logits_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![vocab_size], TensorInit::CopyFrom(&logits_data)).unwrap();

    for case in 0..8u32 {
        let output_foundry = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![1], TensorInit::Uninitialized).unwrap();
        let seed = 1337u32.wrapping_add(case.wrapping_mul(17));
        let kernel = SampleTopK::new(
            &TensorArg::from_tensor(&logits_foundry),
            &TensorArg::from_tensor(&output_foundry),
            vocab_size as u32,
            k,
            top_p,
            min_p,
            temperature,
            seed,
        );
        foundry.run(&kernel).unwrap();

        let gpu_token = FoundryTensor::to_vec(&output_foundry, &foundry)[0];
        let cpu_token = cpu_sample_topk_topp(&logits_f32, k, top_p, min_p, temperature, seed);
        assert_eq!(
            gpu_token, cpu_token,
            "Concentrated-lane parity mismatch seed={} gpu={} cpu={}",
            seed, gpu_token, cpu_token
        );
    }
}
