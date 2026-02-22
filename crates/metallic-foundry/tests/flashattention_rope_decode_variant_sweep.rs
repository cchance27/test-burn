use std::time::Instant;

use half::f16;
use metallic_foundry::{
    Foundry, MetalError, metals::{
        flashattention::{FlashDecodeScalar, FlashDecodeTgOut, FlashDecodeVariant, stages::SdpaParams, step::RopeFlashDecodeStep}, rope::RopeParamsResolved
    }, spec::{CompiledStep, FastBindings, SymbolTable, TensorBindings}, storage::Pooled, tensor::{Tensor, TensorInit, dtypes::F16}, types::TensorArg
};

fn parse_env_usize(key: &'static str) -> Option<usize> {
    std::env::var(key).ok().and_then(|s| s.trim().parse::<usize>().ok())
}

fn median_us(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    if n == 0 {
        return f64::NAN;
    }
    if (n & 1) == 1 {
        samples[n / 2]
    } else {
        0.5 * (samples[(n / 2) - 1] + samples[n / 2])
    }
}

fn mean_us(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    samples.iter().sum::<f64>() / samples.len() as f64
}

fn stddev_us(samples: &[f64], mean: f64) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let var = samples
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (samples.len() as f64 - 1.0);
    var.sqrt()
}

fn bench(foundry: &mut Foundry, label: &str, warmup: usize, trials: usize, iters_per_trial: usize, f: impl Fn(&mut Foundry)) {
    for _ in 0..warmup {
        f(foundry);
    }
    foundry.synchronize().unwrap();

    let mut avgs = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        foundry.start_capture().unwrap();
        for _ in 0..iters_per_trial {
            f(foundry);
        }
        let buf = foundry.end_capture().unwrap();
        buf.wait_until_completed();
        let elapsed = start.elapsed();
        avgs.push(elapsed.as_micros() as f64 / iters_per_trial as f64);
    }

    let min = avgs.iter().copied().fold(f64::INFINITY, f64::min);
    let med = median_us(avgs.clone());
    let mean = mean_us(&avgs);
    let sd = stddev_us(&avgs, mean);
    println!("  -> {label}: min={min:.2} us med={med:.2} us mean={mean:.2} us sd={sd:.2} us (trials={trials} iters={iters_per_trial})");
}

fn variants_for_head_dim(head_dim: u32) -> Vec<FlashDecodeVariant> {
    match head_dim {
        64 => vec![
            // Baselines (tg_out float vs half for A/B)
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 32,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Half,
            },
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 32,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Half,
            },
            // Explore: 384-thread TG and non-power-of-two key blocks (float tg_out only)
            FlashDecodeVariant {
                warps: 12,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 12,
                keys_per_warp: 24,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 24,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Float,
            },
            // Half4 explorations (float tg_out only, plus a baseline A/B)
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 32,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Half,
            },
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 32,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Half,
            },
            FlashDecodeVariant {
                warps: 12,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 12,
                keys_per_warp: 24,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 24,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
        ],
        128 => vec![
            // Baselines (tg_out float vs half for A/B)
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 8,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 8,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Half,
            },
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Half,
            },
            // Comparisons
            FlashDecodeVariant {
                warps: 12,
                keys_per_warp: 8,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 12,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 8,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
        ],
        _ => vec![],
    }
}

fn kv_lengths() -> Vec<u32> {
    let mut kvs = vec![32u32, 64, 128, 256, 512, 1024, 2048, 4096];
    if let Some(max) = std::env::var("METALLIC_FA_SWEEP_KV_MAX")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
    {
        kvs.retain(|&k| k <= max);
    }
    kvs
}

#[test]
#[ignore]
fn flashattention_rope_decode_variant_sweep() -> Result<(), MetalError> {
    let mut foundry = match Foundry::new() {
        Ok(foundry) => foundry,
        Err(MetalError::DeviceNotFound) => {
            eprintln!("Skipping: Metal device not available (DeviceNotFound)");
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    // This sweep measures the *fused* RoPE→FlashAttention decode compound kernel,
    // which is closer to what `run_throughput.sh` exercises than the non-fused core sweep.
    let cases = [
        (14u32, 64u32), // Qwen/Llama-ish decode
        (8u32, 128u32), // larger head-dim decode
    ];

    let warmup = parse_env_usize("METALLIC_FA_SWEEP_WARMUP").unwrap_or(25);
    let trials = parse_env_usize("METALLIC_FA_SWEEP_TRIALS").unwrap_or(10);
    let iters_per_trial = parse_env_usize("METALLIC_FA_SWEEP_ITERS").unwrap_or(100);

    for (n_heads, head_dim) in cases {
        println!("\n=== Fused RoPE→Flash Decode Sweep: heads={n_heads} head_dim={head_dim} ===");

        // Q/out are [H, 1, D] and K/V are [H, KV, D] (head-major).
        let q_len = 1usize;
        let q = Tensor::<F16, Pooled>::new(
            &mut foundry,
            vec![n_heads as usize, q_len, head_dim as usize],
            TensorInit::Uninitialized,
        )?;
        let output = Tensor::<F16, Pooled>::new(
            &mut foundry,
            vec![n_heads as usize, q_len, head_dim as usize],
            TensorInit::Uninitialized,
        )?;

        for kv_len in kv_lengths() {
            let kv_len_usize = kv_len as usize;
            let k = Tensor::<F16, Pooled>::new(
                &mut foundry,
                vec![n_heads as usize, kv_len_usize, head_dim as usize],
                TensorInit::Uninitialized,
            )?;
            let v = Tensor::<F16, Pooled>::new(
                &mut foundry,
                vec![n_heads as usize, kv_len_usize, head_dim as usize],
                TensorInit::Uninitialized,
            )?;

            // Identity RoPE caches (reduces NaN risk; perf impact still includes the RoPE math + memory reads).
            let half_dim = (head_dim as usize) / 2;
            let cos_data = vec![f16::ONE; kv_len_usize * half_dim];
            let sin_data = vec![f16::ZERO; kv_len_usize * half_dim];
            let cos = Tensor::<F16, Pooled>::new(&mut foundry, vec![kv_len_usize, half_dim], TensorInit::CopyFrom(&cos_data))?;
            let sin = Tensor::<F16, Pooled>::new(&mut foundry, vec![kv_len_usize, half_dim], TensorInit::CopyFrom(&sin_data))?;

            // Strides are in elements.
            let q_strides = (q.strides()[0] as u32, q.strides()[1] as u32);
            let k_strides = (k.strides()[0] as u32, k.strides()[1] as u32);
            let v_strides = (v.strides()[0] as u32, v.strides()[1] as u32);
            let out_strides = (output.strides()[0] as u32, output.strides()[1] as u32);

            // RoPEStage only uses `dim` + `position_offset`, but keep the struct populated defensively.
            let rope_params = RopeParamsResolved {
                dim: head_dim,
                seq_len: kv_len,
                position_offset: kv_len.saturating_sub(1),
                total_elements: n_heads * head_dim,
            };
            let sdpa_params = SdpaParams {
                kv_len,
                head_dim,
                scale: 1.0 / (head_dim as f32).sqrt(),
                stride_k_s: k.strides()[1] as u32,
                stride_v_s: v.strides()[1] as u32,
            };

            println!("\nKV={kv_len}");
            for variant in variants_for_head_dim(head_dim) {
                variant.validate_for_head_dim(head_dim)?;
                let compiled = RopeFlashDecodeStep::compile_with_variant(
                    &mut foundry,
                    &TensorArg::from_tensor(&q),
                    &TensorArg::from_tensor(&k),
                    &TensorArg::from_tensor(&v),
                    &TensorArg::from_tensor(&cos),
                    &TensorArg::from_tensor(&sin),
                    &TensorArg::from_tensor(&output),
                    rope_params,
                    sdpa_params,
                    1,
                    n_heads,
                    head_dim,
                    variant,
                    q_strides,
                    k_strides,
                    v_strides,
                    out_strides,
                )?;

                let run = |f: &mut Foundry| {
                    compiled
                        .execute(f, &FastBindings::default(), &TensorBindings::default(), &SymbolTable::new())
                        .unwrap();
                };
                bench(&mut foundry, &format!("{variant:?}"), warmup, trials, iters_per_trial, run);
            }
        }
    }

    Ok(())
}
