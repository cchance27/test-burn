// Test SdpaMaterializedStep GEMM prefill path (M>1).
// Validates that Q is interpreted as head-major ([n_heads, M, head_dim]) and output is token-major ([M, d_model]).

use half::f16;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    MetalError, foundry::{
        Foundry, spec::{Step, TensorBindings}, storage::Pooled, tensor::Tensor as FoundryTensor
    }, metals::sdpa::step::SdpaMaterializedStep, tensor::{F16, TensorInit}, types::TensorArg
};

#[test]
fn test_sdpa_materialized_prefill_parity_m4() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;
    let mut bindings = TensorBindings::new();
    let mut rng = StdRng::seed_from_u64(123);

    // Small config for correctness.
    let m: usize = 4;
    let kv_seq_len: usize = 8;
    let head_dim: usize = 64;
    let n_heads: usize = 2;
    let d_model: usize = n_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Q/K/V are head-major:
    // Q: [n_heads, m, head_dim]
    // K/V: [n_heads, kv_seq_len, head_dim]
    let q_data: Vec<f16> = (0..n_heads * m * head_dim)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let k_data: Vec<f16> = (0..n_heads * kv_seq_len * head_dim)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let v_data: Vec<f16> = (0..n_heads * kv_seq_len * head_dim)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();

    let q_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * m * head_dim], TensorInit::CopyFrom(&q_data))?;
    let k_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * kv_seq_len * head_dim], TensorInit::CopyFrom(&k_data))?;
    let v_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * kv_seq_len * head_dim], TensorInit::CopyFrom(&v_data))?;

    // SdpaMaterializedStep writes token-major: [m, d_model].
    let out_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m * d_model], TensorInit::Uninitialized)?;

    bindings.insert("q".to_string(), TensorArg::from_tensor(&q_tensor));
    bindings.insert("k".to_string(), TensorArg::from_tensor(&k_tensor));
    bindings.insert("v".to_string(), TensorArg::from_tensor(&v_tensor));
    bindings.insert("out".to_string(), TensorArg::from_tensor(&out_tensor));

    // Non-causal to keep reference simple.
    let sdpa_step = SdpaMaterializedStep {
        q: "q".into(),
        k: "k".into(),
        v: "v".into(),
        output: "out".into(),
        causal: false,
        query_offset: crate::foundry::spec::DynamicValue::Literal(0),
        n_heads: crate::foundry::spec::DynamicValue::Literal(n_heads as u32),
        head_dim: crate::foundry::spec::DynamicValue::Literal(head_dim as u32),
        kv_seq_len: crate::foundry::spec::DynamicValue::Literal(kv_seq_len as u32),
        m: crate::foundry::spec::DynamicValue::Literal(m as u32),
        kv_head_major: true,
    };

    sdpa_step.execute(&mut foundry, &mut bindings)?;
    foundry.synchronize()?;

    let gpu_out: Vec<f16> = out_tensor.to_vec(&foundry);
    let gpu_out_f32: Vec<f32> = gpu_out.iter().map(|x| x.to_f32()).collect();

    // CPU reference.
    let q_f32: Vec<f32> = q_data.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k_data.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v_data.iter().map(|x| x.to_f32()).collect();

    let mut cpu_out = vec![0.0f32; m * d_model];
    for h in 0..n_heads {
        let q_off = h * m * head_dim;
        let k_off = h * kv_seq_len * head_dim;
        let v_off = h * kv_seq_len * head_dim;

        for t in 0..m {
            // Scores[t, s] = Q[t] @ K[s]^T * scale
            let mut scores = vec![0.0f32; kv_seq_len];
            for s in 0..kv_seq_len {
                let mut sum = 0.0f32;
                for d in 0..head_dim {
                    sum += q_f32[q_off + t * head_dim + d] * k_f32[k_off + s * head_dim + d];
                }
                scores[s] = sum * scale;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let probs: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

            // Output[t, d] = sum_s probs[s] * V[s, d]
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for s in 0..kv_seq_len {
                    sum += probs[s] * v_f32[v_off + s * head_dim + d];
                }
                cpu_out[t * d_model + h * head_dim + d] = sum;
            }
        }
    }

    // Compare: allow FP16/GEMM drift; this should mainly catch layout/stride issues.
    let mut max_diff = 0.0f32;
    for (g, c) in gpu_out_f32.iter().zip(cpu_out.iter()) {
        max_diff = max_diff.max((g - c).abs());
    }
    assert!(max_diff < 0.15, "SDPA prefill parity failed (max diff {})", max_diff);

    Ok(())
}
