use std::time::Instant;

use metallic_env::{EnvVarGuard, FoundryEnvVar};
use metallic_foundry::{
    Foundry, MetalError, metals::flashattention::{FlashDecodeScalar, FlashDecodeTgOut, FlashDecodeVariant, step::run_flash_decode_with_variant}, storage::Pooled, tensor::{Tensor, TensorInit, dtypes::F16}, types::TensorArg
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok().and_then(|s| s.trim().parse::<f64>().ok())
}

fn scalar_name(s: FlashDecodeScalar) -> &'static str {
    match s {
        FlashDecodeScalar::Half2 => "half2",
        FlashDecodeScalar::Half4 => "half4",
    }
}

fn run_decode_bench(
    foundry: &mut Foundry,
    q: &Tensor<F16, Pooled>,
    k: &Tensor<F16, Pooled>,
    v: &Tensor<F16, Pooled>,
    out: &Tensor<F16, Pooled>,
    n_heads: u32,
    head_dim: u32,
    kv_len: u32,
    variant: FlashDecodeVariant,
    engine: &'static str,
    warmup: usize,
    iters: usize,
) -> Result<f64, MetalError> {
    let _engine_guard = EnvVarGuard::set(FoundryEnvVar::FaDecodeEngine, engine);
    let _scalar_guard = EnvVarGuard::set(FoundryEnvVar::FaDecodeScalar, scalar_name(variant.scalar));

    for _ in 0..warmup {
        run_flash_decode_with_variant(
            foundry,
            &TensorArg::from_tensor(q),
            &TensorArg::from_tensor(k),
            &TensorArg::from_tensor(v),
            &TensorArg::from_tensor(out),
            n_heads,
            head_dim,
            kv_len,
            1,
            true,
            variant,
        )?;
    }
    foundry.synchronize()?;

    let start = Instant::now();
    foundry.start_capture()?;
    for _ in 0..iters {
        run_flash_decode_with_variant(
            foundry,
            &TensorArg::from_tensor(q),
            &TensorArg::from_tensor(k),
            &TensorArg::from_tensor(v),
            &TensorArg::from_tensor(out),
            n_heads,
            head_dim,
            kv_len,
            1,
            true,
            variant,
        )?;
    }
    let cmd = foundry.end_capture()?;
    cmd.wait_until_completed();
    let elapsed = start.elapsed();
    Ok(elapsed.as_micros() as f64 / iters as f64)
}

#[test]
#[ignore]
#[serial_test::serial]
fn benchmark_flashattention_decode_scalar_vs_mma_micro() -> Result<(), Box<dyn std::error::Error>> {
    let mut foundry = match Foundry::new() {
        Ok(foundry) => foundry,
        Err(MetalError::DeviceNotFound) => {
            eprintln!("Skipping: Metal device not available (DeviceNotFound)");
            return Ok(());
        }
        Err(e) => return Err(Box::new(e)),
    };

    // Qwen-like defaults; override via env for focused sweeps.
    let n_heads = env_usize("METALLIC_FA_MICRO_N_HEADS", 14) as u32;
    let head_dim = env_usize("METALLIC_FA_MICRO_HEAD_DIM", 64) as u32;
    let kv_len = env_usize("METALLIC_FA_MICRO_KV_LEN", 4096) as u32;
    let warps = env_usize("METALLIC_FA_MICRO_WARPS", 8) as u32;
    let keys_per_warp = env_usize("METALLIC_FA_MICRO_KEYS_PER_WARP", 16) as u32;
    let warmup = env_usize("METALLIC_FA_MICRO_WARMUP", 20);
    let iters = env_usize("METALLIC_FA_MICRO_ITERS", 200);

    let scalar_mode = std::env::var("METALLIC_FA_MICRO_SCALAR")
        .unwrap_or_else(|_| "both".to_string())
        .to_lowercase();

    let mut scalar_variants = Vec::new();
    if scalar_mode == "both" || scalar_mode == "half2" {
        scalar_variants.push(FlashDecodeScalar::Half2);
    }
    if scalar_mode == "both" || scalar_mode == "half4" {
        scalar_variants.push(FlashDecodeScalar::Half4);
    }
    if scalar_variants.is_empty() {
        return Err(format!("METALLIC_FA_MICRO_SCALAR must be one of: both|half2|half4 (got '{scalar_mode}')").into());
    }

    let d_model = (n_heads * head_dim) as usize;
    let q = Tensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;
    let k = Tensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::Uninitialized,
    )?;
    let v = Tensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::Uninitialized,
    )?;
    let out_scalar = Tensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;
    let out_mma = Tensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;

    println!(
        "FA decode micro-bench: heads={n_heads} head_dim={head_dim} kv_len={kv_len} warps={warps} keys_per_warp={keys_per_warp} warmup={warmup} iters={iters}"
    );

    let max_slowdown_pct = env_f64("METALLIC_FA_MICRO_ASSERT_MAX_SLOWDOWN_PCT");

    for scalar in scalar_variants {
        if head_dim == 128 && matches!(scalar, FlashDecodeScalar::Half2) {
            println!("  -> skipping half2 for head_dim=128");
            continue;
        }

        let variant = FlashDecodeVariant {
            warps,
            keys_per_warp,
            scalar,
            tg_out: FlashDecodeTgOut::Float,
        };

        let scalar_us = run_decode_bench(
            &mut foundry,
            &q,
            &k,
            &v,
            &out_scalar,
            n_heads,
            head_dim,
            kv_len,
            variant,
            "scalar",
            warmup,
            iters,
        )?;
        let mma_us = run_decode_bench(
            &mut foundry,
            &q,
            &k,
            &v,
            &out_mma,
            n_heads,
            head_dim,
            kv_len,
            variant,
            "mma",
            warmup,
            iters,
        )?;

        let delta_pct = ((mma_us / scalar_us) - 1.0) * 100.0;
        let speedup = scalar_us / mma_us;
        println!(
            "  -> scalar={} | scalar={:.2} us | mma={:.2} us | delta={:+.2}% | speedup={:.3}x",
            scalar_name(scalar),
            scalar_us,
            mma_us,
            delta_pct,
            speedup
        );

        if let Some(max_pct) = max_slowdown_pct {
            let worst_allowed = scalar_us * (1.0 + max_pct / 100.0);
            assert!(
                mma_us <= worst_allowed,
                "mma regression exceeded threshold: scalar={} scalar_us={:.2} mma_us={:.2} allowed_max={:.2} (max_slowdown_pct={:.2})",
                scalar_name(scalar),
                scalar_us,
                mma_us,
                worst_allowed,
                max_pct
            );
        }
    }

    Ok(())
}
