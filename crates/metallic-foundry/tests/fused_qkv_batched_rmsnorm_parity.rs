use half::f16;
use metallic_foundry::{
    Foundry, metals::qkv::FusedQkvStep, spec::{DynamicValue, Step, TensorBindings}, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}
};
use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn pack_canonical_f16(weights_nk: &Array2<f32>, weights_per_block: usize) -> Vec<f16> {
    let (n, k) = weights_nk.dim();
    let blocks_per_k = k.div_ceil(weights_per_block);
    let mut out = vec![f16::from_f32(0.0); blocks_per_k * n * weights_per_block];
    for row in 0..n {
        for kk in 0..k {
            let blk = kk / weights_per_block;
            let in_blk = kk % weights_per_block;
            let dst = in_blk + weights_per_block * (row + blk * n);
            out[dst] = f16::from_f32(weights_nk[(row, kk)]);
        }
    }
    out
}

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

#[test]
fn test_fused_qkv_batched_rmsnorm_parity_m2() -> Result<(), Box<dyn std::error::Error>> {
    let mut foundry = Foundry::new()?;
    let mut bindings = TensorBindings::new();
    let mut rng = StdRng::seed_from_u64(42);

    let m: usize = 2;
    let k_dim: usize = 128;
    let n_dim: usize = 128;
    let n_kv: usize = 64;
    let weights_per_block: usize = 32;

    bindings.set_int_global("m", m);
    bindings.set_int_global("seq_len", m);

    // Inputs
    let hidden_cpu = Array2::from_shape_fn((m, k_dim), |_| rng.random_range(-1.0f32..1.0f32));
    let gamma_cpu: Vec<f32> = (0..k_dim).map(|_| 1.0f32).collect();

    // Weights (row-major [N, K] CPU), uploaded as F16 (Foundry test path)
    let w_q_cpu = Array2::from_shape_fn((n_dim, k_dim), |_| rng.random_range(-0.1f32..0.1f32));
    let w_k_cpu = Array2::from_shape_fn((n_kv, k_dim), |_| rng.random_range(-0.1f32..0.1f32));
    let w_v_cpu = Array2::from_shape_fn((n_kv, k_dim), |_| rng.random_range(-0.1f32..0.1f32));

    let hidden_f16: Vec<f16> = hidden_cpu.iter().map(|&v| f16::from_f32(v)).collect();
    let gamma_f16: Vec<f16> = gamma_cpu.iter().map(|&v| f16::from_f32(v)).collect();
    let w_q_canon = pack_canonical_f16(&w_q_cpu, weights_per_block);
    let w_k_canon = pack_canonical_f16(&w_k_cpu, weights_per_block);
    let w_v_canon = pack_canonical_f16(&w_v_cpu, weights_per_block);

    let hidden = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, k_dim], TensorInit::CopyFrom(&hidden_f16))?;
    let gamma = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k_dim], TensorInit::CopyFrom(&gamma_f16))?;
    let w_q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![w_q_canon.len()], TensorInit::CopyFrom(&w_q_canon))?;
    let w_k = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![w_k_canon.len()], TensorInit::CopyFrom(&w_k_canon))?;
    let w_v = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![w_v_canon.len()], TensorInit::CopyFrom(&w_v_canon))?;

    // Outputs
    let q_out = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, n_dim], TensorInit::Uninitialized)?;
    let k_out = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, n_kv], TensorInit::Uninitialized)?;
    let v_out = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, n_kv], TensorInit::Uninitialized)?;

    bindings.insert("hidden".to_string(), metallic_foundry::types::TensorArg::from_tensor(&hidden));
    bindings.insert("gamma".to_string(), metallic_foundry::types::TensorArg::from_tensor(&gamma));
    bindings.insert("w_q".to_string(), metallic_foundry::types::TensorArg::from_tensor(&w_q));
    bindings.insert("w_k".to_string(), metallic_foundry::types::TensorArg::from_tensor(&w_k));
    bindings.insert("w_v".to_string(), metallic_foundry::types::TensorArg::from_tensor(&w_v));
    bindings.insert("q".to_string(), metallic_foundry::types::TensorArg::from_tensor(&q_out));
    bindings.insert("k".to_string(), metallic_foundry::types::TensorArg::from_tensor(&k_out));
    bindings.insert("v".to_string(), metallic_foundry::types::TensorArg::from_tensor(&v_out));

    let fused = FusedQkvStep {
        input: "hidden".into(),
        gamma: Some("gamma".into()),
        w_q: "w_q".into(),
        w_k: "w_k".into(),
        w_v: "w_v".into(),
        bias_q: None,
        bias_k: None,
        bias_v: None,
        s_q: None,
        s_k: None,
        s_v: None,
        out_q: "q".into(),
        out_k: "k".into(),
        out_v: "v".into(),
        k_dim: DynamicValue::Literal(k_dim as u32),
        n_dim: DynamicValue::Literal(n_dim as u32),
        n_kv: DynamicValue::Literal(n_kv as u32),
        m: DynamicValue::Variable("m".into()),
        weights_per_block: DynamicValue::Literal(weights_per_block as u32),
        strategy: metallic_foundry::metals::gemv::step::GemvStrategy::Vectorized,
    };

    fused.execute(&mut foundry, &mut bindings)?;
    foundry.synchronize()?;

    // CPU ref
    let hidden_norm = cpu_rmsnorm_rows(&hidden_cpu, &gamma_cpu, 1e-6);
    let q_ref = cpu_matmul_xt(&hidden_norm, &w_q_cpu);
    let k_ref = cpu_matmul_xt(&hidden_norm, &w_k_cpu);
    let v_ref = cpu_matmul_xt(&hidden_norm, &w_v_cpu);

    let q_gpu: Vec<f16> = q_out.to_vec(&foundry);
    let k_gpu: Vec<f16> = k_out.to_vec(&foundry);
    let v_gpu: Vec<f16> = v_out.to_vec(&foundry);

    let to_f32 = |v: &[f16]| v.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
    let q_gpu_f32 = to_f32(&q_gpu);
    let k_gpu_f32 = to_f32(&k_gpu);
    let v_gpu_f32 = to_f32(&v_gpu);

    let mut max_diff = 0.0f32;
    for (g, c) in q_gpu_f32.iter().zip(q_ref.iter()) {
        max_diff = max_diff.max((g - c).abs());
    }
    for (g, c) in k_gpu_f32.iter().zip(k_ref.iter()) {
        max_diff = max_diff.max((g - c).abs());
    }
    for (g, c) in v_gpu_f32.iter().zip(v_ref.iter()) {
        max_diff = max_diff.max((g - c).abs());
    }

    assert!(max_diff < 0.2, "max diff too high: {}", max_diff);

    Ok(())
}
