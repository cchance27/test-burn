use std::time::Instant;

use metallic_foundry::{
    Foundry, MetalError, metals::flashattention::{FlashDecodeScalar, FlashDecodeTgOut, FlashDecodeVariant, step::run_flash_decode_with_variant}, storage::Pooled, tensor::{Tensor, TensorInit, dtypes::F16}, types::TensorArg
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
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 32,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 32,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 4,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Float,
            },
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
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 32,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 32,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 4,
                keys_per_warp: 16,
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
            FlashDecodeVariant {
                warps: 8,
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
                warps: 12,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 12,
                keys_per_warp: 8,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 8,
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
fn flashattention_decode_variant_sweep() -> Result<(), MetalError> {
    let mut foundry = match Foundry::new() {
        Ok(foundry) => foundry,
        Err(MetalError::DeviceNotFound) => {
            eprintln!("Skipping: Metal device not available (DeviceNotFound)");
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    let cases = [
        (14u32, 64u32), // Qwen/Llama-ish decode
        (8u32, 128u32), // larger head-dim decode
    ];

    let warmup = parse_env_usize("METALLIC_FA_SWEEP_WARMUP").unwrap_or(25);
    let trials = parse_env_usize("METALLIC_FA_SWEEP_TRIALS").unwrap_or(10);
    let iters_per_trial = parse_env_usize("METALLIC_FA_SWEEP_ITERS").unwrap_or(100);

    for (n_heads, head_dim) in cases {
        let d_model = (n_heads * head_dim) as usize;
        let q = Tensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;

        println!("\n=== FlashAttention Decode Sweep: heads={n_heads} head_dim={head_dim} ===");
        for kv_len in kv_lengths() {
            // K/V cache head-major: [H, capacity, D].
            let capacity = kv_len as usize;
            let k = Tensor::<F16, Pooled>::new(
                &mut foundry,
                vec![n_heads as usize, capacity, head_dim as usize],
                TensorInit::Uninitialized,
            )?;
            let v = Tensor::<F16, Pooled>::new(
                &mut foundry,
                vec![n_heads as usize, capacity, head_dim as usize],
                TensorInit::Uninitialized,
            )?;
            let out = Tensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;

            println!("\nKV={kv_len}");
            for variant in variants_for_head_dim(head_dim) {
                let label = format!("{variant:?}");
                let run = |f: &mut Foundry| {
                    run_flash_decode_with_variant(
                        f,
                        &TensorArg::from_tensor(&q),
                        &TensorArg::from_tensor(&k),
                        &TensorArg::from_tensor(&v),
                        &TensorArg::from_tensor(&out),
                        n_heads,
                        head_dim,
                        kv_len,
                        1,
                        true,
                        variant,
                    )
                    .unwrap();
                };
                bench(&mut foundry, &label, warmup, trials, iters_per_trial, run);
            }
        }
    }

    Ok(())
}
