use half::f16;
use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    foundry::{
        Foundry, spec::{Step, TensorBindings}, storage::Pooled, tensor::Tensor as FoundryTensor
    }, metals::swiglu::step::FusedFfnSwiGluRmsNormStep, tensor::{F16, TensorInit}
};

fn cpu_rmsnorm_rows(x: &Array2<f32>, gamma: &[f32], eps: f32) -> Array2<f32> {
    let (m, k) = x.dim();
    let mut out = Array2::<f32>::zeros((m, k));
    for row in 0..m {
        let mut sum_sq = 0.0f32;
        for col in 0..k {
            let v = x[(row, col)];
            sum_sq += v * v;
        }
        let inv_rms = 1.0f32 / ((sum_sq / (k as f32) + eps).sqrt());
        for col in 0..k {
            out[(row, col)] = x[(row, col)] * inv_rms * gamma[col];
        }
    }
    out
}

fn cpu_matmul_xt(x: &Array2<f32>, w: &Array2<f32>) -> Array2<f32> {
    // x: [M, K], w: [N, K] => out: [M, N]
    let (m, k) = x.dim();
    let (n, wk) = w.dim();
    assert_eq!(k, wk);
    let mut out = Array2::<f32>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[(i, kk)] * w[(j, kk)];
            }
            out[(i, j)] = acc;
        }
    }
    out
}

fn cpu_swiglu(gate: &Array2<f32>, up: &Array2<f32>, b_gate: &[f32], b_up: &[f32]) -> Array2<f32> {
    let (m, n) = gate.dim();
    let mut out = Array2::<f32>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let g = gate[(i, j)] + b_gate[j];
            let u = up[(i, j)] + b_up[j];
            let silu = g / (1.0f32 + (-g).exp());
            out[(i, j)] = silu * u;
        }
    }
    out
}

#[test]
fn test_fused_ffn_swiglu_rmsnorm_parity_m2() -> Result<(), Box<dyn std::error::Error>> {
    let mut foundry = Foundry::new()?;
    let mut bindings = TensorBindings::new();
    let mut rng = StdRng::seed_from_u64(42);

    let m: usize = 2;
    let k_dim: usize = 128;
    let n_dim: usize = 256;

    bindings.set_int_global("m", m);

    let input_cpu = Array2::from_shape_fn((m, k_dim), |_| rng.random_range(-1.0f32..1.0f32));
    let gamma_cpu: Vec<f32> = (0..k_dim).map(|_| rng.random_range(0.9f32..1.1f32)).collect();

    let w_gate_cpu = Array2::from_shape_fn((n_dim, k_dim), |_| rng.random_range(-0.1f32..0.1f32));
    let w_up_cpu = Array2::from_shape_fn((n_dim, k_dim), |_| rng.random_range(-0.1f32..0.1f32));

    let b_gate_cpu: Vec<f32> = (0..n_dim).map(|_| rng.random_range(-0.01f32..0.01f32)).collect();
    let b_up_cpu: Vec<f32> = (0..n_dim).map(|_| rng.random_range(-0.01f32..0.01f32)).collect();

    let input_f16: Vec<f16> = input_cpu.iter().map(|&v| f16::from_f32(v)).collect();
    let gamma_f16: Vec<f16> = gamma_cpu.iter().map(|&v| f16::from_f32(v)).collect();
    let w_gate_f16: Vec<f16> = w_gate_cpu.iter().map(|&v| f16::from_f32(v)).collect();
    let w_up_f16: Vec<f16> = w_up_cpu.iter().map(|&v| f16::from_f32(v)).collect();
    let b_gate_f16: Vec<f16> = b_gate_cpu.iter().map(|&v| f16::from_f32(v)).collect();
    let b_up_f16: Vec<f16> = b_up_cpu.iter().map(|&v| f16::from_f32(v)).collect();

    let input = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, k_dim], TensorInit::CopyFrom(&input_f16))?;
    let gamma = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k_dim], TensorInit::CopyFrom(&gamma_f16))?;
    let w_gate = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_dim, k_dim], TensorInit::CopyFrom(&w_gate_f16))?;
    let w_up = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_dim, k_dim], TensorInit::CopyFrom(&w_up_f16))?;
    let b_gate = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_dim], TensorInit::CopyFrom(&b_gate_f16))?;
    let b_up = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_dim], TensorInit::CopyFrom(&b_up_f16))?;
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, n_dim], TensorInit::Uninitialized)?;

    bindings.insert("input".to_string(), crate::types::TensorArg::from_tensor(&input));
    bindings.insert("gamma".to_string(), crate::types::TensorArg::from_tensor(&gamma));
    bindings.insert("w_gate".to_string(), crate::types::TensorArg::from_tensor(&w_gate));
    bindings.insert("w_up".to_string(), crate::types::TensorArg::from_tensor(&w_up));
    bindings.insert("b_gate".to_string(), crate::types::TensorArg::from_tensor(&b_gate));
    bindings.insert("b_up".to_string(), crate::types::TensorArg::from_tensor(&b_up));
    bindings.insert("output".to_string(), crate::types::TensorArg::from_tensor(&output));

    let fused = FusedFfnSwiGluRmsNormStep {
        input: "input".into(),
        gamma: "gamma".into(),
        w_gate: "w_gate".into(),
        w_up: "w_up".into(),
        b_gate: Some("b_gate".into()),
        b_up: Some("b_up".into()),
        output: "output".into(),
        weights_per_block: 32,
    };

    fused.execute(&mut foundry, &mut bindings)?;
    foundry.synchronize()?;

    let input_norm = cpu_rmsnorm_rows(&input_cpu, &gamma_cpu, 1e-6);
    let gate_ref = cpu_matmul_xt(&input_norm, &w_gate_cpu);
    let up_ref = cpu_matmul_xt(&input_norm, &w_up_cpu);
    let out_ref = cpu_swiglu(&gate_ref, &up_ref, &b_gate_cpu, &b_up_cpu);

    let out_gpu: Vec<f16> = output.to_vec(&foundry);
    let out_gpu_f32 = out_gpu.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();

    let mut max_diff = 0.0f32;
    for (g, c) in out_gpu_f32.iter().zip(out_ref.iter()) {
        max_diff = max_diff.max((g - c).abs());
    }

    assert!(max_diff < 0.2, "max diff too high: {}", max_diff);

    Ok(())
}
