use std::time::Instant;

use metallic_foundry::{
    Foundry, FoundryConfig, MetalError, metals::{
        flashattention::{stages::SdpaParams, step::RopeFlashDecodeStep}, rope::{Rope, RopeParamsResolved}, sdpa::step::SdpaMaterializedStep
    }, spec::{CompiledStep, DynamicValue, FastBindings, Step, SymbolTable, TensorBindings}, storage::Pooled, tensor::{F32, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};

fn parse_env_usize(key: &'static str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_env_f32(key: &'static str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(default)
}

fn lcg_next_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let mantissa = *state >> 9;
    (mantissa as f32) * (1.0 / ((1u32 << 23) as f32))
}

fn fill_f32(state: &mut u32, len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|_| {
            let r01 = lcg_next_f32(state);
            (r01 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn build_rope_cache(kv_len: usize, half_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0.0f32; kv_len * half_dim];
    let mut sin = vec![0.0f32; kv_len * half_dim];
    for pos in 0..kv_len {
        for i in 0..half_dim {
            let theta = (pos as f32) / base.powf((2.0 * i as f32) / (2.0 * half_dim as f32));
            let idx = pos * half_dim + i;
            cos[idx] = theta.cos();
            sin[idx] = theta.sin();
        }
    }
    (cos, sin)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn measure_us(
    foundry: &mut Foundry,
    warmup: usize,
    trials: usize,
    iters: usize,
    mut f: impl FnMut(&mut Foundry) -> Result<(), MetalError>,
) -> Result<f64, MetalError> {
    for _ in 0..warmup {
        f(foundry)?;
    }
    foundry.synchronize()?;

    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        foundry.start_capture()?;
        for _ in 0..iters {
            f(foundry)?;
        }
        let cb = foundry.end_capture()?;
        cb.wait_until_completed();
        samples.push(start.elapsed().as_micros() as f64 / iters as f64);
    }
    Ok(mean(&samples))
}

#[test]
#[ignore]
fn flashattention_rope_decode_f32_parity_and_perf_sweep() -> Result<(), Box<dyn std::error::Error>> {
    let config = FoundryConfig::default()
        .with_compute_dtype(metallic_foundry::tensor::Dtype::F32)
        .with_accum_dtype(metallic_foundry::tensor::Dtype::F32);
    let mut foundry = match Foundry::new_with_config(config) {
        Ok(v) => v,
        Err(MetalError::DeviceNotFound) => {
            eprintln!("Skipping flashattention_rope_decode_f32_parity_and_perf_sweep: Metal device unavailable");
            return Ok(());
        }
        Err(e) => return Err(Box::new(e)),
    };

    let warmup = parse_env_usize("METALLIC_FA_F32_SWEEP_WARMUP", 3);
    let trials = parse_env_usize("METALLIC_FA_F32_SWEEP_TRIALS", 5);
    let iters = parse_env_usize("METALLIC_FA_F32_SWEEP_ITERS", 20);
    let parity_tol = parse_env_f32("METALLIC_FA_F32_SWEEP_PARITY_TOL", 1.5e-2);
    let max_ratio = parse_env_f32("METALLIC_FA_F32_SWEEP_MAX_RATIO", 1.25);
    let long_kv = parse_env_usize("METALLIC_FA_F32_SWEEP_LONG_KV", 4096) as u32;
    let short_kv = parse_env_usize("METALLIC_FA_F32_SWEEP_SHORT_KV", 128) as u32;

    println!(
        "FA F32 RoPE->Flash sweep: warmup={warmup}, trials={trials}, iters={iters}, parity_tol={parity_tol:.4}, max_ratio={max_ratio:.3}, short_kv={short_kv}, long_kv={long_kv}"
    );

    let cases = [
        (14u32, 64u32, short_kv),
        (14u32, 64u32, long_kv),
        (8u32, 128u32, short_kv),
        (8u32, 128u32, long_kv),
    ];

    for (n_heads, head_dim, kv_len) in cases {
        let kv_len_usize = kv_len as usize;
        let half_dim = (head_dim as usize) / 2;
        let mut rng = 17u32 ^ n_heads ^ head_dim ^ kv_len;
        let q_host = fill_f32(&mut rng, n_heads as usize * head_dim as usize, 0.125);
        let k_host = fill_f32(&mut rng, n_heads as usize * kv_len_usize * head_dim as usize, 0.125);
        let v_host = fill_f32(&mut rng, n_heads as usize * kv_len_usize * head_dim as usize, 0.125);
        let (cos_host, sin_host) = build_rope_cache(kv_len_usize, half_dim, 10000.0);

        let q = FoundryTensor::<F32, Pooled>::new(
            &mut foundry,
            vec![n_heads as usize, 1, head_dim as usize],
            TensorInit::CopyFrom(&q_host),
        )?;
        let k = FoundryTensor::<F32, Pooled>::new(
            &mut foundry,
            vec![n_heads as usize, kv_len_usize, head_dim as usize],
            TensorInit::CopyFrom(&k_host),
        )?;
        let v = FoundryTensor::<F32, Pooled>::new(
            &mut foundry,
            vec![n_heads as usize, kv_len_usize, head_dim as usize],
            TensorInit::CopyFrom(&v_host),
        )?;
        let cos = FoundryTensor::<F32, Pooled>::new(&mut foundry, vec![kv_len_usize, half_dim], TensorInit::CopyFrom(&cos_host))?;
        let sin = FoundryTensor::<F32, Pooled>::new(&mut foundry, vec![kv_len_usize, half_dim], TensorInit::CopyFrom(&sin_host))?;

        let q_roped = FoundryTensor::<F32, Pooled>::new(
            &mut foundry,
            vec![n_heads as usize, 1, head_dim as usize],
            TensorInit::Uninitialized,
        )?;
        let k_roped = FoundryTensor::<F32, Pooled>::new(
            &mut foundry,
            vec![n_heads as usize, kv_len_usize, head_dim as usize],
            TensorInit::Uninitialized,
        )?;
        let out_fused = FoundryTensor::<F32, Pooled>::new(
            &mut foundry,
            vec![n_heads as usize, 1, head_dim as usize],
            TensorInit::Uninitialized,
        )?;
        let out_ref = FoundryTensor::<F32, Pooled>::new(
            &mut foundry,
            vec![n_heads as usize, 1, head_dim as usize],
            TensorInit::Uninitialized,
        )?;

        let rope_q_params = RopeParamsResolved {
            dim: head_dim,
            seq_len: kv_len,
            position_offset: kv_len.saturating_sub(1),
            total_elements: n_heads * head_dim,
        };
        let rope_k_params = RopeParamsResolved {
            dim: head_dim,
            seq_len: kv_len,
            position_offset: 0,
            total_elements: n_heads * kv_len * head_dim,
        };
        let rope_q = Rope::new(
            &TensorArg::from_tensor(&q),
            &TensorArg::from_tensor(&q_roped),
            &TensorArg::from_tensor(&cos),
            &TensorArg::from_tensor(&sin),
            rope_q_params,
        );
        let rope_k = Rope::new(
            &TensorArg::from_tensor(&k),
            &TensorArg::from_tensor(&k_roped),
            &TensorArg::from_tensor(&cos),
            &TensorArg::from_tensor(&sin),
            rope_k_params,
        );
        foundry.run(&rope_k)?;

        let sdpa_params = SdpaParams {
            kv_len,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            stride_k_s: k_roped.strides()[1] as u32,
            stride_v_s: v.strides()[1] as u32,
        };
        let q_strides = (q.strides()[0] as u32, q.strides()[1] as u32);
        let k_strides = (k_roped.strides()[0] as u32, k_roped.strides()[1] as u32);
        let v_strides = (v.strides()[0] as u32, v.strides()[1] as u32);
        let out_strides = (out_fused.strides()[0] as u32, out_fused.strides()[1] as u32);
        let fused = RopeFlashDecodeStep::compile(
            &mut foundry,
            &TensorArg::from_tensor(&q),
            &TensorArg::from_tensor(&k_roped),
            &TensorArg::from_tensor(&v),
            &TensorArg::from_tensor(&cos),
            &TensorArg::from_tensor(&sin),
            &TensorArg::from_tensor(&out_fused),
            rope_q_params,
            sdpa_params,
            1,
            n_heads,
            head_dim,
            q_strides,
            k_strides,
            v_strides,
            out_strides,
        )?;

        // Parity: fused RoPE->Flash versus unfused Rope(Q) + SdpaMaterialized with pre-roped K.
        foundry.run(&rope_q)?;
        let mut bindings = TensorBindings::new();
        bindings.insert("q".to_string(), TensorArg::from_tensor(&q_roped));
        bindings.insert("k".to_string(), TensorArg::from_tensor(&k_roped));
        bindings.insert("v".to_string(), TensorArg::from_tensor(&v));
        bindings.insert("o".to_string(), TensorArg::from_tensor(&out_ref));
        let reference = SdpaMaterializedStep {
            q: "q".into(),
            k: "k".into(),
            v: "v".into(),
            output: "o".into(),
            causal: true,
            query_offset: DynamicValue::Literal(kv_len - 1),
            n_heads: DynamicValue::Literal(n_heads),
            head_dim: DynamicValue::Literal(head_dim),
            kv_seq_len: DynamicValue::Literal(kv_len),
            m: DynamicValue::Literal(1),
            kv_head_major: true,
        };
        reference.execute(&mut foundry, &mut bindings)?;
        fused.execute(
            &mut foundry,
            &FastBindings::default(),
            &TensorBindings::default(),
            &SymbolTable::new(),
        )?;
        foundry.synchronize()?;

        let fused_out = out_fused.to_vec(&foundry);
        let ref_out = out_ref.to_vec(&foundry);
        let diff = max_abs_diff(&fused_out, &ref_out);

        println!("case heads={n_heads} d={head_dim} kv={kv_len}: parity max_abs_diff={diff:.6}");
        assert!(
            diff <= parity_tol,
            "F32 fused RoPE->Flash parity failed for heads={n_heads}, d={head_dim}, kv={kv_len}: diff={diff:.6} > tol={parity_tol:.6}"
        );

        let fused_us = measure_us(&mut foundry, warmup, trials, iters, |f| {
            fused.execute(f, &FastBindings::default(), &TensorBindings::default(), &SymbolTable::new())
        })?;

        let ref_us = measure_us(&mut foundry, warmup, trials, iters, |f| {
            f.run(&rope_q)?;
            let mut b = TensorBindings::new();
            b.insert("q".to_string(), TensorArg::from_tensor(&q_roped));
            b.insert("k".to_string(), TensorArg::from_tensor(&k_roped));
            b.insert("v".to_string(), TensorArg::from_tensor(&v));
            b.insert("o".to_string(), TensorArg::from_tensor(&out_ref));
            reference.execute(f, &mut b)?;
            Ok(())
        })?;

        let ratio = (fused_us / ref_us) as f32;
        println!("case heads={n_heads} d={head_dim} kv={kv_len}: fused={fused_us:.2} us, unfused={ref_us:.2} us, ratio={ratio:.3}x");
        assert!(
            ratio <= max_ratio,
            "F32 fused RoPE->Flash perf regression for heads={n_heads}, d={head_dim}, kv={kv_len}: ratio={ratio:.3}x > max_ratio={max_ratio:.3}x"
        );
    }

    Ok(())
}
