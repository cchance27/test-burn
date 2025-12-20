use crate::{
    Context, Tensor, context::QkvWeights, kernels::silu::SiluOp, tensor::{TensorType, canonical::CanonicalF16Tensor}
};

#[test]
fn test_f16_canonical_qkv_parity() {
    let mut ctx = Context::<crate::tensor::F16>::new().unwrap();
    let k = 64;
    let nq = 48;
    let nk = 16;
    let nv = 16;

    // Create random inputs
    let x = Tensor::random_uniform(vec![1, k], &mut ctx).unwrap();

    // Create weights in NK layout (K, N)
    let wq_nk = Tensor::random_uniform(vec![k, nq], &mut ctx).unwrap();
    let wk_nk = Tensor::random_uniform(vec![k, nk], &mut ctx).unwrap();
    let wv_nk = Tensor::random_uniform(vec![k, nv], &mut ctx).unwrap();

    // Create Canonical weights
    let mut wq_canon = CanonicalF16Tensor::new(vec![k, nq], &mut ctx).unwrap();
    wq_canon.write_from_nk_tensor(&wq_nk, 0).unwrap();

    let mut wk_canon = CanonicalF16Tensor::new(vec![k, nk], &mut ctx).unwrap();
    wk_canon.write_from_nk_tensor(&wk_nk, 0).unwrap();

    let mut wv_canon = CanonicalF16Tensor::new(vec![k, nv], &mut ctx).unwrap();
    wv_canon.write_from_nk_tensor(&wv_nk, 0).unwrap();

    let q_bias = Tensor::random_uniform(vec![nq], &mut ctx).unwrap();
    let k_bias = Tensor::random_uniform(vec![nk], &mut ctx).unwrap();
    let v_bias = Tensor::random_uniform(vec![nv], &mut ctx).unwrap();

    // 1. Run Fused Kernel via qkv()
    let (yq, yk, yv) = ctx
        .qkv(
            &x,
            QkvWeights::DenseCanonical {
                wq: &wq_canon,
                wk: &wk_canon,
                wv: &wv_canon,
                q_bias: &q_bias,
                k_bias: &k_bias,
                v_bias: &v_bias,
            },
        )
        .unwrap();

    ctx.ensure_active_cmd_buffer().unwrap();

    // 2. Run Reference (3 separate matmuls using Canonical weights)
    let yq_ref = ctx
        .matmul(&x, &TensorType::DenseCanonical(&wq_canon), false, false, Some(&q_bias), None, None)
        .unwrap();
    let yk_ref = ctx
        .matmul(&x, &TensorType::DenseCanonical(&wk_canon), false, false, Some(&k_bias), None, None)
        .unwrap();
    let yv_ref = ctx
        .matmul(&x, &TensorType::DenseCanonical(&wv_canon), false, false, Some(&v_bias), None, None)
        .unwrap();

    ctx.ensure_active_cmd_buffer().unwrap();
    ctx.synchronize();

    // Compare
    let yq_vec = yq.to_f32_vec();
    let yq_ref_vec = yq_ref.to_f32_vec();

    let max_diff_q = yq_vec
        .iter()
        .zip(yq_ref_vec.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let yk_vec = yk.to_f32_vec();
    let yk_ref_vec = yk_ref.to_f32_vec();
    let max_diff_k = yk_vec
        .iter()
        .zip(yk_ref_vec.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let yv_vec = yv.to_f32_vec();
    let yv_ref_vec = yv_ref.to_f32_vec();
    let max_diff_v = yv_vec
        .iter()
        .zip(yv_ref_vec.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // print first few
    println!("Q[0]: {} vs {}", yq_vec[0], yq_ref_vec[0]);
    println!("Max Diff Q: {}", max_diff_q);
    println!("Max Diff K: {}", max_diff_k);
    println!("Max Diff V: {}", max_diff_v);

    // Relaxed tolerance for FP16
    assert!(max_diff_q < 0.2, "Q output mismatch");
    assert!(max_diff_k < 0.2, "K output mismatch");
    assert!(max_diff_v < 0.2, "V output mismatch");
}

#[test]
fn test_f16_canonical_swiglu_parity() {
    let mut ctx = Context::<crate::tensor::F16>::new().unwrap();
    let k = 64;
    let n0 = 48; // gate
    let n1 = 48; // up

    let x = Tensor::random_uniform(vec![1, k], &mut ctx).unwrap();

    let wg_nk = Tensor::random_uniform(vec![k, n0], &mut ctx).unwrap();
    let wu_nk = Tensor::random_uniform(vec![k, n1], &mut ctx).unwrap();

    let mut wg_canon = CanonicalF16Tensor::new(vec![k, n0], &mut ctx).unwrap();
    wg_canon.write_from_nk_tensor(&wg_nk, 0).unwrap();

    let mut wu_canon = CanonicalF16Tensor::new(vec![k, n1], &mut ctx).unwrap();
    wu_canon.write_from_nk_tensor(&wu_nk, 0).unwrap();

    let bias_g = Tensor::random_uniform(vec![n0], &mut ctx).unwrap();
    let bias_u = Tensor::random_uniform(vec![n1], &mut ctx).unwrap();

    // 1. Run Fused SwiGLU
    let y = ctx
        .swiglu(
            &x,
            &TensorType::DenseCanonical(&wg_canon),
            &TensorType::DenseCanonical(&wu_canon),
            Some(&bias_g),
            Some(&bias_u),
        )
        .unwrap();

    ctx.ensure_active_cmd_buffer().unwrap();

    // 2. Run Reference: Silu(Gate) * Up
    let gate_out = ctx
        .matmul(&x, &TensorType::DenseCanonical(&wg_canon), false, false, Some(&bias_g), None, None)
        .unwrap();
    let up_out = ctx
        .matmul(&x, &TensorType::DenseCanonical(&wu_canon), false, false, Some(&bias_u), None, None)
        .unwrap();

    // Silu
    let gate_act = ctx.call::<SiluOp>(gate_out.clone(), None).unwrap(); // call returns Tensor

    let y_ref = gate_act.mul_elem(&up_out, &mut ctx).unwrap();

    ctx.ensure_active_cmd_buffer().unwrap();
    ctx.synchronize();

    let y_vec = y.to_f32_vec();
    let y_ref_vec = y_ref.to_f32_vec();

    let max_diff = y_vec
        .iter()
        .zip(y_ref_vec.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("SwiGLU[0]: {} vs {}", y_vec[0], y_ref_vec[0]);
    println!("Max Diff SwiGLU: {}", max_diff);

    assert!(max_diff < 0.5, "SwiGLU output mismatch");
}
