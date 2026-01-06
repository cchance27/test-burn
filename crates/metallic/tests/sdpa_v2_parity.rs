use half::f16;
use metallic::{
    foundry::{Foundry, storage::Pooled, tensor::Tensor}, metals::{
        rope::{Rope, RopeParamsResolved}, sdpa::sdpa::scaled_dot_product_attention, v2::attention::{stages::SdpaParamsResolved, step::FusedMhaStep}
    }, tensor::{TensorInit, dtypes::F16}, types::TensorArg
};
use rand::Rng;

fn generate_random_f16(size: usize) -> Vec<f16> {
    let mut rng = rand::rng();
    (0..size).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect()
}

fn compare_tensors(a: &[f16], b: &[f16], name: &str, tolerance: f32) {
    assert_eq!(a.len(), b.len(), "Tensor {} size mismatch", name);
    let mut max_diff = 0.0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x.to_f32() - y.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tolerance {
            panic!(
                "Mismatch in {} at index {}: v2={} vs legacy={} (diff={})",
                name,
                i,
                x.to_f32(),
                y.to_f32(),
                diff
            );
        }
    }
    println!("Max diff for {}: {}", name, max_diff);
}

fn run_parity_test_case(batch: usize, heads: usize, kv_len: usize, head_dim: usize) {
    println!(
        "Testing Parity: Batch={}, Heads={}, KV_Len={}, HeadDim={}",
        batch, heads, kv_len, head_dim
    );
    let mut foundry = Foundry::new().unwrap();
    let q_len = 1; // Decode case

    let total_batch = batch * heads;
    let q_dims_3d = vec![total_batch, q_len, head_dim];
    let k_dims_3d = vec![total_batch, kv_len, head_dim];
    let v_dims_3d = vec![total_batch, kv_len, head_dim];
    let rope_cache_dims = vec![kv_len, head_dim / 2];

    // Data
    let q_data = generate_random_f16(total_batch * q_len * head_dim);
    let k_data = generate_random_f16(total_batch * kv_len * head_dim);
    let v_data = generate_random_f16(total_batch * kv_len * head_dim);
    let cos_data = generate_random_f16(kv_len * head_dim / 2);
    let sin_data = generate_random_f16(kv_len * head_dim / 2);

    // Initialize Tensors
    let q = Tensor::<F16, Pooled>::new(&mut foundry, q_dims_3d.clone(), TensorInit::CopyFrom(&q_data)).unwrap();
    let k = Tensor::<F16, Pooled>::new(&mut foundry, k_dims_3d.clone(), TensorInit::CopyFrom(&k_data)).unwrap();
    let v = Tensor::<F16, Pooled>::new(&mut foundry, v_dims_3d.clone(), TensorInit::CopyFrom(&v_data)).unwrap();

    let cos = Tensor::<F16, Pooled>::new(&mut foundry, rope_cache_dims.clone(), TensorInit::CopyFrom(&cos_data)).unwrap();
    let sin = Tensor::<F16, Pooled>::new(&mut foundry, rope_cache_dims.clone(), TensorInit::CopyFrom(&sin_data)).unwrap();

    let output_v2 = Tensor::<F16, Pooled>::new(&mut foundry, q_dims_3d.clone(), TensorInit::Uninitialized).unwrap();

    // ----------------------------------------------------------------
    // Pre-calculate Roped K
    // ----------------------------------------------------------------
    let k_roped = Tensor::<F16, Pooled>::new(&mut foundry, k_dims_3d.clone(), TensorInit::Uninitialized).unwrap();
    let rope_k_params = RopeParamsResolved {
        dim: head_dim as u32,
        seq_len: kv_len as u32,
        position_offset: 0,
        total_elements: 0,
    };
    let rope_k = Rope::new(
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&k_roped),
        &TensorArg::from_tensor(&cos),
        &TensorArg::from_tensor(&sin),
        rope_k_params,
    );
    foundry.run(&rope_k).unwrap();

    // ----------------------------------------------------------------
    // V2 Execution (Fused)
    // ----------------------------------------------------------------

    let rope_params = RopeParamsResolved {
        dim: head_dim as u32,
        seq_len: kv_len as u32,
        position_offset: (kv_len - 1) as u32,
        total_elements: (total_batch * head_dim) as u32,
    };

    let sdpa_params = SdpaParamsResolved {
        kv_len: kv_len as u32,
        head_dim: head_dim as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
        stride_k_s: k_roped.strides()[1] as u32,
        stride_v_s: v.strides()[1] as u32,
    };

    let q_strides = (q.strides()[0] as u32, q.strides()[1] as u32);
    let k_strides = (k_roped.strides()[0] as u32, k_roped.strides()[1] as u32);
    let v_strides = (v.strides()[0] as u32, v.strides()[1] as u32);
    let out_strides = (output_v2.strides()[0] as u32, output_v2.strides()[1] as u32);

    let v2_step = FusedMhaStep::compile(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k_roped),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&cos),
        &TensorArg::from_tensor(&sin),
        &TensorArg::from_tensor(&output_v2),
        rope_params,
        sdpa_params,
        total_batch as u32,
        1, // kernel sees 1 head per dispatch item (because we flatten batch*heads)
        head_dim as u32,
        q_strides,
        k_strides,
        v_strides,
        out_strides,
    )
    .unwrap();

    use metallic::foundry::spec::{CompiledStep, FastBindings, TensorBindings};
    v2_step
        .execute(&mut foundry, &FastBindings::default(), &TensorBindings::default())
        .unwrap();

    // ----------------------------------------------------------------
    // Legacy Execution via Separate Kernels
    // ----------------------------------------------------------------

    // 1. Rope Q
    let q_roped = Tensor::<F16, Pooled>::new(&mut foundry, q_dims_3d.clone(), TensorInit::Uninitialized).unwrap();
    let rope_q_params = RopeParamsResolved {
        dim: head_dim as u32,
        seq_len: 1,
        position_offset: (kv_len - 1) as u32,
        total_elements: 0,
    };
    let rope_q = Rope::new(
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&q_roped),
        &TensorArg::from_tensor(&cos),
        &TensorArg::from_tensor(&sin),
        rope_q_params,
    );
    foundry.run(&rope_q).unwrap();

    // 3. SDPA
    let output_legacy = scaled_dot_product_attention(
        &mut foundry,
        &q_roped,
        &k_roped,
        &v,
        true,                // causal
        (kv_len - 1) as u32, // query_offset
    )
    .unwrap();

    // ----------------------------------------------------------------
    // Comparison
    // ----------------------------------------------------------------

    let res_v2 = output_v2.to_vec(&foundry);
    let res_legacy = output_legacy.to_vec(&foundry);

    compare_tensors(&res_v2, &res_legacy, "SDPA Output", 1e-3);
}

#[test]
fn test_sdpa_v2_parity_decode() {
    // Fuzz different shapes
    let batches = [1, 2];
    let heads_list = [1, 4];
    let kv_lens = [32, 128];
    let head_dims = [64, 128];

    for &b in &batches {
        for &h in &heads_list {
            for &kv in &kv_lens {
                for &d in &head_dims {
                    run_parity_test_case(b, h, kv, d);
                }
            }
        }
    }
}
