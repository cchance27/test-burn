use crate::metallic::{Context, Tensor};
use crate::sdpa_burn::scaled_dot_product_attention_burn;
use burn::prelude::*;
use burn::tensor::{Distribution, Tensor as BurnTensor};

type MyBackend = burn::backend::Metal;

use super::*;

/// Helper function to compare Metallic SDPA against Burn with tolerance
#[allow(clippy::too_many_arguments)]
fn compare_sdpa_implementations(
    batch: usize,
    seq_q: usize,
    seq_k: usize,
    dim: usize,
    causal: bool,
    q_data: Vec<f32>,
    k_data: Vec<f32>,
    v_data: Vec<f32>,
) {
    // Burn implementation
    let device = <MyBackend as Backend>::Device::default();
    let q_burn = BurnTensor::<MyBackend, 3>::from_data(
        burn::tensor::TensorData::new(q_data.clone(), vec![batch, seq_q, dim]),
        &device,
    );
    let k_burn = BurnTensor::<MyBackend, 3>::from_data(
        burn::tensor::TensorData::new(k_data.clone(), vec![batch, seq_k, dim]),
        &device,
    );
    let v_burn = BurnTensor::<MyBackend, 3>::from_data(
        burn::tensor::TensorData::new(v_data.clone(), vec![batch, seq_k, dim]),
        &device,
    );

    let burn_out = scaled_dot_product_attention_burn(q_burn, k_burn, v_burn, None, causal);
    let burn_data = burn_out.to_data();
    let burn_slice = burn_data.as_slice::<f32>().unwrap();

    // Metallic implementation
    let mut ctx = Context::new().unwrap();
    let q_tensor =
        Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &ctx).unwrap();
    let k_tensor =
        Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &ctx).unwrap();
    let v_tensor =
        Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &ctx).unwrap();

    let metal_out = ctx
        .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, causal)
        .unwrap();
    let metal_slice = metal_out.as_slice();

    // Validate with tolerance
    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for (i, (metal_val, burn_val)) in metal_slice.iter().zip(burn_slice.iter()).enumerate() {
        let diff = ((*metal_val) as f64 - (*burn_val) as f64).abs();
        let rel_err = if burn_val.abs() > 1e-8 {
            diff / ((*burn_val).abs() as f64)
        } else {
            diff
        };

        // Check for NaN or Infinity
        assert!(
            (*metal_val).is_finite(),
            "Metallic output contains non-finite value at index {}: {}",
            i,
            metal_val
        );
        assert!(
            (*burn_val).is_finite(),
            "Burn output contains non-finite value at index {}: {}",
            i,
            burn_val
        );

        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, burn={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            burn_val,
            diff,
            rel_err
        );
    }
}

#[test]
fn sdpa_numerical_stability_large_magnitudes() {
    let batch = 1;
    let seq_q = 4;
    let seq_k = 4;
    let dim = 4;

    // Create base data
    let mut q_data: Vec<f32> = (0..(batch * seq_q * dim))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.1)
        .collect();

    // Add large offset to Q to create large logits
    for val in q_data.iter_mut() {
        *val += 1000.0;
    }

    compare_sdpa_implementations(batch, seq_q, seq_k, dim, false, q_data, k_data, v_data);
}

#[test]
fn sdpa_numerical_stability_large_magnitudes_causal() {
    let batch = 1;
    let seq_q = 4;
    let seq_k = 4;
    let dim = 4;

    // Create base data
    let mut q_data: Vec<f32> = (0..(batch * seq_q * dim))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.1)
        .collect();

    // Add large offset to Q to create large logits
    for val in q_data.iter_mut() {
        *val += 1000.0;
    }

    compare_sdpa_implementations(batch, seq_q, seq_k, dim, true, q_data, k_data, v_data);
}

#[test]
fn sdpa_numerical_stability_mixed_extremes() {
    let batch = 1;
    let seq_q = 3;
    let seq_k = 3;
    let dim = 2;

    // Create data with extreme values
    let q_data = vec![
        1e10, 1e-10, // Very large and very small
        -1e10, -1e-10, // Very large negative and very small negative
        0.0, 0.0, // Zeros
    ];

    let k_data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    let v_data = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];

    compare_sdpa_implementations(batch, seq_q, seq_k, dim, false, q_data, k_data, v_data);
}

#[test]
fn sdpa_numerical_stability_random_large() {
    let batch = 2;
    let seq_q = 8;
    let seq_k = 16;
    let dim = 8;

    // Use Burn to generate random data with large values
    let device = <MyBackend as Backend>::Device::default();
    let q_burn = BurnTensor::<MyBackend, 3>::random(
        [batch, seq_q, dim],
        Distribution::Uniform(-1000.0, 1000.0),
        &device,
    );
    let k_burn = BurnTensor::<MyBackend, 3>::random(
        [batch, seq_k, dim],
        Distribution::Uniform(-1000.0, 1000.0),
        &device,
    );
    let v_burn = BurnTensor::<MyBackend, 3>::random(
        [batch, seq_k, dim],
        Distribution::Uniform(-1000.0, 1000.0),
        &device,
    );

    let q_data = q_burn
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
    let k_data = k_burn
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
    let v_data = v_burn
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    compare_sdpa_implementations(batch, seq_q, seq_k, dim, false, q_data, k_data, v_data);
}

#[test]
fn sdpa_numerical_stability_random_large_causal() {
    let batch = 2;
    let seq_q = 8;
    let seq_k = 16;
    let dim = 8;

    // Use Burn to generate random data with large values
    let device = <MyBackend as Backend>::Device::default();
    let q_burn = BurnTensor::<MyBackend, 3>::random(
        [batch, seq_q, dim],
        Distribution::Uniform(-1000.0, 1000.0),
        &device,
    );
    let k_burn = BurnTensor::<MyBackend, 3>::random(
        [batch, seq_k, dim],
        Distribution::Uniform(-1000.0, 1000.0),
        &device,
    );
    let v_burn = BurnTensor::<MyBackend, 3>::random(
        [batch, seq_k, dim],
        Distribution::Uniform(-1000.0, 1000.0),
        &device,
    );

    let q_data = q_burn
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
    let k_data = k_burn
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
    let v_data = v_burn
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    compare_sdpa_implementations(batch, seq_q, seq_k, dim, true, q_data, k_data, v_data);
}
