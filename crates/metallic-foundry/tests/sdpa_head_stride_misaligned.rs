use half::f16;
use metallic_foundry::{
    Foundry, MetalError, metals::sdpa::step::SdpaMaterializedStep, spec::{DynamicValue, Ref, Step, TensorBindings}, storage::Pooled, tensor::{Tensor, TensorInit, dtypes::F16}, types::TensorArg
};

#[test]
fn test_sdpa_head_stride_misaligned_m() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads = 2;
    let head_dim = 64;
    let m = 17; // intentionally not aligned to GEMM tile M=32
    let seq_k = 15;

    // Q: [n_heads, m, head_dim] (head-major)
    let mut q_f32 = vec![0.0f32; n_heads * m * head_dim];
    for h in 0..n_heads {
        for t in 0..m {
            for d in 0..head_dim {
                let idx = (h * m * head_dim) + (t * head_dim) + d;
                q_f32[idx] = (h as f32 + 1.0) * (t as f32 + 1.0) * (d as f32 + 1.0) * 0.001;
            }
        }
    }
    let q_data: Vec<f16> = q_f32.iter().copied().map(f16::from_f32).collect();

    // K/V: [n_heads, seq_k, head_dim]
    let mut k_f32 = vec![0.0f32; n_heads * seq_k * head_dim];
    let mut v_f32 = vec![0.0f32; n_heads * seq_k * head_dim];
    for h in 0..n_heads {
        for s in 0..seq_k {
            for d in 0..head_dim {
                let idx = (h * seq_k * head_dim) + (s * head_dim) + d;
                k_f32[idx] = (h as f32 + 1.0) * (s as f32 + 1.0) * (d as f32 + 1.0) * 0.00001;
                v_f32[idx] = (h as f32 + 1.0) * (s as f32 + 1.0) * ((d % 7) as f32 + 1.0) * 0.0001;
            }
        }
    }
    let k_data: Vec<f16> = k_f32.iter().copied().map(f16::from_f32).collect();
    let v_data: Vec<f16> = v_f32.iter().copied().map(f16::from_f32).collect();

    let q = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_heads, m, head_dim], TensorInit::CopyFrom(&q_data))?;
    let k = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_heads, seq_k, head_dim], TensorInit::CopyFrom(&k_data))?;
    let v = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_heads, seq_k, head_dim], TensorInit::CopyFrom(&v_data))?;
    let output = Tensor::<F16, Pooled>::new(&mut foundry, vec![m, n_heads * head_dim], TensorInit::Uninitialized)?;

    let mut bindings = TensorBindings::new();
    bindings.insert("Q", TensorArg::from_tensor(&q));
    bindings.insert("K", TensorArg::from_tensor(&k));
    bindings.insert("V", TensorArg::from_tensor(&v));
    bindings.insert("Out", TensorArg::from_tensor(&output));

    let step = SdpaMaterializedStep {
        q: Ref("Q".into()),
        k: Ref("K".into()),
        v: Ref("V".into()),
        output: Ref("Out".into()),
        causal: false,
        query_offset: DynamicValue::Literal(0),
        n_heads: DynamicValue::Literal(n_heads as u32),
        head_dim: DynamicValue::Literal(head_dim as u32),
        kv_seq_len: DynamicValue::Literal(seq_k as u32),
        m: DynamicValue::Literal(m as u32),
        kv_head_major: true,
    };

    step.execute(&mut foundry, &mut bindings)?;
    foundry.synchronize()?;

    let out_data = output.to_vec(&foundry);

    // CPU reference: output is token-major [m, n_heads * head_dim]
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut expected = vec![0.0f32; m * n_heads * head_dim];
    for t in 0..m {
        for h in 0..n_heads {
            let q_off = (h * m * head_dim) + (t * head_dim);
            let q_vec = &q_f32[q_off..q_off + head_dim];

            let mut scores = vec![0.0f32; seq_k];
            for (s, score) in scores.iter_mut().enumerate().take(seq_k) {
                let k_off = (h * seq_k * head_dim) + (s * head_dim);
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_vec[d] * k_f32[k_off + d];
                }
                *score = dot * scale;
            }

            let max = scores.iter().copied().fold(f32::NEG_INFINITY, |a, b| if a > b { a } else { b });
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max).exp();
                sum += *s;
            }

            for (s, score) in scores.iter().enumerate().take(seq_k) {
                let p = *score / sum;
                let v_off = (h * seq_k * head_dim) + (s * head_dim);
                for d in 0..head_dim {
                    let out_idx = (t * n_heads * head_dim) + (h * head_dim) + d;
                    expected[out_idx] += p * v_f32[v_off + d];
                }
            }
        }
    }

    for (i, got) in out_data.iter().enumerate() {
        let got = got.to_f32();
        let exp = expected[i];
        assert!((got - exp).abs() < 0.05, "Mismatch at index {}: got {}, exp {}", i, got, exp);
    }

    Ok(())
}
