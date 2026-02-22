use std::{
    sync::{Mutex, OnceLock}, time::Instant
};

use half::f16;
use metallic_foundry::{
    Foundry, MetalError, metals::flashattention::step::run_flash_decode, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
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

fn bench(
    foundry: &mut Foundry,
    label: &str,
    warmup: usize,
    trials: usize,
    iters_per_trial: usize,
    f: impl Fn(&mut Foundry) -> Result<(), MetalError>,
) -> Result<(), MetalError> {
    for _ in 0..warmup {
        f(foundry)?;
    }
    foundry.synchronize()?;

    let mut avgs = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        foundry.start_capture()?;
        for _ in 0..iters_per_trial {
            f(foundry)?;
        }
        let buf = foundry.end_capture()?;
        buf.wait_until_completed();
        let elapsed = start.elapsed();
        avgs.push(elapsed.as_micros() as f64 / iters_per_trial as f64);
    }

    let min = avgs.iter().copied().fold(f64::INFINITY, f64::min);
    let med = median_us(avgs.clone());
    let mean = mean_us(&avgs);
    let sd = stddev_us(&avgs, mean);
    println!("  -> {label}: min={min:.2} us med={med:.2} us mean={mean:.2} us sd={sd:.2} us (trials={trials} iters={iters_per_trial})");
    Ok(())
}

fn with_env_lock<R>(f: impl FnOnce() -> R) -> R {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().unwrap();
    f()
}

fn with_env_var<R>(key: &str, value: &str, f: impl FnOnce() -> R) -> R {
    with_env_lock(|| {
        let old = std::env::var(key).ok();
        // `set_var`/`remove_var` are `unsafe` on newer Rust because the process environment is
        // globally shared and mutation is not data-race-safe. We guard with a global mutex.
        unsafe { std::env::set_var(key, value) };
        let out = f();
        match old {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        out
    })
}

fn lcg_next_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let mantissa = *state >> 9; // 23 bits
    (mantissa as f32) * (1.0 / ((1u32 << 23) as f32))
}

fn fill_f16(state: &mut u32, len: usize, scale: f32) -> Vec<f16> {
    (0..len)
        .map(|_| {
            let r01 = lcg_next_f32(state);
            let r = (r01 * 2.0 - 1.0) * scale;
            f16::from_f32(r)
        })
        .collect()
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

fn m_lengths() -> Vec<u32> {
    let mut ms = vec![2u32, 4, 8, 16, 32, 64];
    if let Some(max) = std::env::var("METALLIC_FA_SWEEP_M_MAX")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
    {
        ms.retain(|&m| m <= max);
    }
    ms
}

#[test]
#[ignore]
fn flashattention_prefill_variant_sweep() -> Result<(), MetalError> {
    let mut foundry = match Foundry::new() {
        Ok(foundry) => foundry,
        Err(MetalError::DeviceNotFound) => {
            eprintln!("Skipping: Metal device not available (DeviceNotFound)");
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    let warmup = parse_env_usize("METALLIC_FA_SWEEP_WARMUP").unwrap_or(25);
    let trials = parse_env_usize("METALLIC_FA_SWEEP_TRIALS").unwrap_or(10);
    let iters_per_trial = parse_env_usize("METALLIC_FA_SWEEP_ITERS").unwrap_or(100);

    // Prefill (M>1) only: sweep warps for a few common head dims.
    let cases = [
        (14u32, 64u32), // Qwen/Llama-ish prefill
        (8u32, 128u32), // larger head dim
    ];
    let warps = [4u32, 8u32];

    for (n_heads, head_dim) in cases {
        println!("\n=== FlashAttention Prefill Sweep: heads={n_heads} head_dim={head_dim} ===");
        let d_model = (n_heads * head_dim) as usize;

        for kv_len in kv_lengths() {
            println!("\nKV={kv_len}");
            for m in m_lengths() {
                // Maintain causal invariant for prefill: query_offset + m == kv_len.
                // If m>kv_len, skip (not a realistic causal prefill case).
                if m > kv_len {
                    continue;
                }
                let query_offset = kv_len - m;

                let mut rng = 123u32 ^ (m.wrapping_mul(31) ^ kv_len.wrapping_mul(131));
                let q_host = fill_f16(&mut rng, (m as usize) * d_model, 0.25);
                let k_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.25);
                let v_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.25);

                let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::CopyFrom(&q_host))?;
                let k = FoundryTensor::<F16, Pooled>::new(
                    &mut foundry,
                    vec![n_heads as usize, kv_len as usize, head_dim as usize],
                    TensorInit::CopyFrom(&k_host),
                )?;
                let v = FoundryTensor::<F16, Pooled>::new(
                    &mut foundry,
                    vec![n_heads as usize, kv_len as usize, head_dim as usize],
                    TensorInit::CopyFrom(&v_host),
                )?;
                let out = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::Uninitialized)?;

                for w in warps {
                    let label = format!("m={m} warps={w} (query_offset={query_offset})");
                    let w_s = w.to_string();
                    let bench_result = with_env_var("METALLIC_FA_PREFILL_WARPS", &w_s, || {
                        bench(&mut foundry, &label, warmup, trials, iters_per_trial, |foundry| {
                            run_flash_decode(
                                foundry,
                                &TensorArg::from_tensor(&q),
                                &TensorArg::from_tensor(&k),
                                &TensorArg::from_tensor(&v),
                                &TensorArg::from_tensor(&out),
                                n_heads,
                                head_dim,
                                kv_len,
                                m,
                                true,
                            )
                        })
                    });
                    if let Err(MetalError::OutOfMemory) = bench_result {
                        eprintln!("Skipping remaining sweep due to OutOfMemory at {label}, kv_len={kv_len}, head_dim={head_dim}");
                        return Ok(());
                    }
                    bench_result?;
                }
            }
        }
    }

    Ok(())
}
