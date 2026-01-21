use half::f16;
use metallic_foundry::{
    Foundry, metals::{
        rope::RopeParamsResolved, sdpa::{stages::SdpaParamsResolved, step::FusedMhaStep}
    }, storage::Pooled, tensor::{Tensor, TensorInit, dtypes::F16}, types::TensorArg
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

#[allow(clippy::too_many_arguments)]
fn cpu_sdpa_attention(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    cos: &[f16],
    sin: &[f16],
    batch: usize,
    heads: usize,
    kv_len: usize,
    head_dim: usize,
) -> Vec<f16> {
    let total_batch = batch * heads;
    let dim_half = head_dim / 2;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // 1. RoPE Q (Single position: kv_len - 1)
    let q_pos = kv_len - 1;
    let mut q_roped = q.to_vec();
    for b in 0..total_batch {
        let q_offset = b * head_dim;
        for i in 0..dim_half {
            let c = cos[q_pos * dim_half + i].to_f32();
            let s = sin[q_pos * dim_half + i].to_f32();
            let val0 = q[q_offset + i].to_f32();
            let val1 = q[q_offset + dim_half + i].to_f32();

            q_roped[q_offset + i] = f16::from_f32(val0 * c - val1 * s);
            q_roped[q_offset + dim_half + i] = f16::from_f32(val0 * s + val1 * c);
        }
    }

    // 2. RoPE K (All positions 0..kv_len)
    let mut k_roped = k.to_vec();
    for b in 0..total_batch {
        for t in 0..kv_len {
            let k_offset = b * kv_len * head_dim + t * head_dim;
            for i in 0..dim_half {
                let c = cos[t * dim_half + i].to_f32();
                let s = sin[t * dim_half + i].to_f32();
                let val0 = k[k_offset + i].to_f32();
                let val1 = k[k_offset + dim_half + i].to_f32();

                k_roped[k_offset + i] = f16::from_f32(val0 * c - val1 * s);
                k_roped[k_offset + dim_half + i] = f16::from_f32(val0 * s + val1 * c);
            }
        }
    }

    // 3. Online Softmax Loop (Matching GPU execution order)
    let mut output = vec![f16::ZERO; total_batch * head_dim];

    for b in 0..total_batch {
        let q_offset = b * head_dim;

        let mut max_score = -1e30f32;
        let mut sum_exp = 0.0f32;
        let mut out_acc = vec![0.0f32; head_dim];

        for t in 0..kv_len {
            let k_offset = b * kv_len * head_dim + t * head_dim;
            let v_offset = b * kv_len * head_dim + t * head_dim;

            // Dot product Q * K_t
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_roped[q_offset + d].to_f32() * k_roped[k_offset + d].to_f32();
            }
            let score = dot * scale;

            // Online Softmax Update
            let m_prev = max_score;
            let m_new = m_prev.max(score);
            let exp_score = (score - m_new).exp();
            let exp_prev = (m_prev - m_new).exp();

            let scale_v = exp_score;
            let scale_prev = exp_prev;

            sum_exp = sum_exp * scale_prev + scale_v;
            max_score = m_new;

            // Accumulate V
            for d in 0..head_dim {
                out_acc[d] = out_acc[d] * scale_prev + v[v_offset + d].to_f32() * scale_v;
            }
        }

        // Final Normalize
        let out_offset = b * head_dim;
        for d in 0..head_dim {
            let val = if sum_exp.abs() > 1e-6 { out_acc[d] / sum_exp } else { out_acc[d] };
            output[out_offset + d] = f16::from_f32(val);
        }
    }

    output
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
    // Identity RoPE so this test isolates SDPA numerical behavior.
    let cos_data = vec![f16::ONE; kv_len * head_dim / 2];
    let sin_data = vec![f16::ZERO; kv_len * head_dim / 2];

    // Initialize Tensors
    let q = Tensor::<F16, Pooled>::new(&mut foundry, q_dims_3d.clone(), TensorInit::CopyFrom(&q_data)).unwrap();
    let k = Tensor::<F16, Pooled>::new(&mut foundry, k_dims_3d.clone(), TensorInit::CopyFrom(&k_data)).unwrap();
    let v = Tensor::<F16, Pooled>::new(&mut foundry, v_dims_3d.clone(), TensorInit::CopyFrom(&v_data)).unwrap();

    let cos = Tensor::<F16, Pooled>::new(&mut foundry, rope_cache_dims.clone(), TensorInit::CopyFrom(&cos_data)).unwrap();
    let sin = Tensor::<F16, Pooled>::new(&mut foundry, rope_cache_dims.clone(), TensorInit::CopyFrom(&sin_data)).unwrap();

    let output_v2 = Tensor::<F16, Pooled>::new(&mut foundry, q_dims_3d.clone(), TensorInit::Uninitialized).unwrap();

    // K is already "roped" under identity RoPE.

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
        stride_k_s: k.strides()[1] as u32,
        stride_v_s: v.strides()[1] as u32,
    };

    let q_strides = (q.strides()[0] as u32, q.strides()[1] as u32);
    let k_strides = (k.strides()[0] as u32, k.strides()[1] as u32);
    let v_strides = (v.strides()[0] as u32, v.strides()[1] as u32);
    let out_strides = (output_v2.strides()[0] as u32, output_v2.strides()[1] as u32);

    let v2_step = FusedMhaStep::compile(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&cos),
        &TensorArg::from_tensor(&sin),
        &TensorArg::from_tensor(&output_v2),
        rope_params,
        sdpa_params,
        total_batch as u32,
        1,
        head_dim as u32,
        q_strides,
        k_strides,
        v_strides,
        out_strides,
    )
    .unwrap();

    use metallic_foundry::spec::{CompiledStep, FastBindings, SymbolTable, TensorBindings};
    v2_step
        .execute(
            &mut foundry,
            &FastBindings::default(),
            &TensorBindings::default(),
            &SymbolTable::new(),
        )
        .unwrap();

    // ----------------------------------------------------------------
    // CPU Verification
    // ----------------------------------------------------------------

    let res_v2 = output_v2.to_vec(&foundry);
    let res_cpu = cpu_sdpa_attention(&q_data, &k_data, &v_data, &cos_data, &sin_data, batch, heads, kv_len, head_dim);

    compare_tensors(&res_v2, &res_cpu, "SDPA Output (Fused vs CPU)", 0.05); // Verified deviation due to F16 storage + Tree Reduction
}

#[test]
fn test_sdpa_v2_parity_decode() {
    // Single case for debugging/verification first
    run_parity_test_case(1, 1, 128, 64);
    // Add more if stable
    run_parity_test_case(2, 4, 32, 128);
}
