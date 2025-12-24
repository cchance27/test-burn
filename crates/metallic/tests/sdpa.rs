use half::f16;
use metallic::{
    Context, foundry::{Foundry, storage::Pooled, tensor::Tensor as FoundryTensor}, metals::sdpa::scaled_dot_product_attention, tensor::{Tensor as LegacyTensor, TensorInit, dtypes::F16 as F16Dtype}
};
use rand::{Rng, SeedableRng};

fn get_gpu() -> Foundry {
    Foundry::new().expect("Failed to create Foundry")
}

fn random_f16_vec(rng: &mut impl Rng, len: usize) -> Vec<f16> {
    (0..len).map(|_| f16::from_f32(rng.random::<f32>() - 0.5)).collect()
}

/// Test Legacy SDPA vs Foundry SDPA parity.
/// Both should use the same underlying Metal kernels and produce identical results.
#[test]
fn test_sdpa_legacy_vs_foundry_parity() {
    // Create both contexts
    let mut foundry = get_gpu();
    let mut ctx: Context<F16Dtype> = Context::new().expect("Failed to create legacy Context");

    // Params - use decode-like setup (seq_q=1) which is the primary use case
    let batch = 2;
    let seq_q = 1;
    let seq_k = 32;
    let dim = 64;
    let causal = false;
    let query_offset = 0u32;

    let total_q = batch * seq_q * dim;
    let total_k = batch * seq_k * dim;

    // Same random data for both
    let mut rng = rand::rngs::StdRng::seed_from_u64(99);
    let q_data = random_f16_vec(&mut rng, total_q);
    let k_data = random_f16_vec(&mut rng, total_k);
    let v_data = random_f16_vec(&mut rng, total_k);

    // === Legacy Path ===
    let q_legacy = LegacyTensor::<F16Dtype>::new(
        vec![batch, seq_q, dim],
        metallic::TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&q_data),
    )
    .expect("Failed to create legacy Q");

    let k_legacy = LegacyTensor::<F16Dtype>::new(
        vec![batch, seq_k, dim],
        metallic::TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&k_data),
    )
    .expect("Failed to create legacy K");

    let v_legacy = LegacyTensor::<F16Dtype>::new(
        vec![batch, seq_k, dim],
        metallic::TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&v_data),
    )
    .expect("Failed to create legacy V");

    let legacy_out = ctx
        .scaled_dot_product_attention_with_offset(&q_legacy, &k_legacy, &v_legacy, causal, query_offset as usize)
        .expect("Legacy SDPA failed");

    let legacy_result: Vec<f16> = legacy_out.as_slice().to_vec();

    // === Foundry Path ===
    let q_foundry = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![batch, seq_q, dim], TensorInit::CopyFrom(&q_data)).unwrap();

    let k_foundry = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![batch, seq_k, dim], TensorInit::CopyFrom(&k_data)).unwrap();

    let v_foundry = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![batch, seq_k, dim], TensorInit::CopyFrom(&v_data)).unwrap();

    let foundry_out =
        scaled_dot_product_attention(&mut foundry, &q_foundry, &k_foundry, &v_foundry, causal, query_offset).expect("Foundry SDPA failed");

    let foundry_result = foundry_out.to_vec(&foundry);

    // === Compare ===
    assert_eq!(legacy_result.len(), foundry_result.len(), "Output length mismatch");

    let mut max_diff = 0.0f32;
    for (i, (&l, &f)) in legacy_result.iter().zip(foundry_result.iter()).enumerate() {
        let diff = (l.to_f32() - f.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        // Should be near-identical (same kernels) - use tight tolerance
        if diff > 1e-4 {
            panic!(
                "Legacy/Foundry mismatch at index {}: Legacy={}, Foundry={}, Diff={}",
                i,
                l.to_f32(),
                f.to_f32(),
                diff
            );
        }
    }
    println!("Legacy vs Foundry SDPA Max Diff: {} (should be ~0)", max_diff);
}

/// Test Foundry SDPA against CPU reference for correctness.
#[test]
fn test_sdpa_cpu_parity() {
    let mut foundry = get_gpu();

    let batch = 2;
    let seq_q = 1; // Decode case
    let seq_k = 16;
    let dim = 32;
    let causal = false;
    let query_offset = 0;

    let total_q = batch * seq_q * dim;
    let total_k = batch * seq_k * dim;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let q_data = random_f16_vec(&mut rng, total_q);
    let k_data = random_f16_vec(&mut rng, total_k);
    let v_data = random_f16_vec(&mut rng, total_k);

    // CPU Reference
    let cpu_out = sdpa_cpu_ref(&q_data, &k_data, &v_data, batch, seq_q, seq_k, dim, causal, query_offset);

    // Foundry
    let q_tensor = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![batch, seq_q, dim], TensorInit::CopyFrom(&q_data)).unwrap();
    let k_tensor = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![batch, seq_k, dim], TensorInit::CopyFrom(&k_data)).unwrap();
    let v_tensor = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![batch, seq_k, dim], TensorInit::CopyFrom(&v_data)).unwrap();

    let metal_out_tensor =
        scaled_dot_product_attention(&mut foundry, &q_tensor, &k_tensor, &v_tensor, causal, query_offset).expect("SDPA failed");

    let metal_out = metal_out_tensor.to_vec(&foundry);

    assert_eq!(metal_out.len(), cpu_out.len());
    let mut max_diff = 0.0f32;
    for (i, (&c, &m)) in cpu_out.iter().zip(metal_out.iter()).enumerate() {
        let diff = (c.to_f32() - m.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.05 {
            panic!("CPU/Metal mismatch at {}: CPU={}, Metal={}", i, c.to_f32(), m.to_f32());
        }
    }
    println!("SDPA CPU Parity Max Diff: {}", max_diff);
}

// CPU reference implementation for SDPA
fn sdpa_cpu_ref(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    batch: usize,
    seq_q: usize,
    seq_k: usize,
    dim: usize,
    causal: bool,
    query_offset: u32,
) -> Vec<f16> {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut output = vec![f16::from_f32(0.0); batch * seq_q * dim];

    for b in 0..batch {
        for i in 0..seq_q {
            let mut scores = vec![0.0f32; seq_k];
            for j in 0..seq_k {
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += q[b * seq_q * dim + i * dim + d].to_f32() * k[b * seq_k * dim + j * dim + d].to_f32();
                }
                scores[j] = dot * scale;
            }

            if causal {
                let qi = query_offset as usize + i;
                for j in 0..seq_k {
                    if j > qi {
                        scores[j] = f32::NEG_INFINITY;
                    }
                }
            }

            let max_s = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut probs: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum: f32 = probs.iter().sum();
            probs.iter_mut().for_each(|p| *p /= sum);

            for d in 0..dim {
                let mut val = 0.0f32;
                for j in 0..seq_k {
                    val += probs[j] * v[b * seq_k * dim + j * dim + d].to_f32();
                }
                output[b * seq_q * dim + i * dim + d] = f16::from_f32(val);
            }
        }
    }
    output
}

#[test]
fn test_sdpa_causal_masking() {
    let mut foundry = get_gpu();

    let batch = 1;
    let seq_q = 2;
    let seq_k = 2;
    let dim = 8;

    let total_q = batch * seq_q * dim;
    let total_k = batch * seq_k * dim;

    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let q_data = random_f16_vec(&mut rng, total_q);
    let k_data = random_f16_vec(&mut rng, total_k);
    let v_data = random_f16_vec(&mut rng, total_k);

    let cpu_out = sdpa_cpu_ref(&q_data, &k_data, &v_data, batch, seq_q, seq_k, dim, true, 0);

    let q_tensor = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![batch, seq_q, dim], TensorInit::CopyFrom(&q_data)).unwrap();
    let k_tensor = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![batch, seq_k, dim], TensorInit::CopyFrom(&k_data)).unwrap();
    let v_tensor = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![batch, seq_k, dim], TensorInit::CopyFrom(&v_data)).unwrap();

    let metal_out_tensor = scaled_dot_product_attention(&mut foundry, &q_tensor, &k_tensor, &v_tensor, true, 0).expect("SDPA failed");

    let metal_out = metal_out_tensor.to_vec(&foundry);

    let mut max_diff = 0.0f32;
    for (&c, &m) in cpu_out.iter().zip(metal_out.iter()) {
        let diff = (c.to_f32() - m.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("SDPA Causal Parity Max Diff: {}", max_diff);
    assert!(max_diff < 0.05, "Causal masking mismatch: max_diff={}", max_diff);
}
