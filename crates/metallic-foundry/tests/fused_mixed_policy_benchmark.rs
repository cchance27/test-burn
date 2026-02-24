use std::{io::Write, time::Instant};

use half::f16;
use metallic_foundry::{
    Foundry, MetalError, metals::{gemv::step::GemvStrategy, qkv::FusedQkvStep, swiglu::step::FusedSwigluStep}, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::Pooled, tensor::{F16, Q8_0, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};

fn bench_iterations() -> usize {
    std::env::var("METALLIC_FUSED_POLICY_BENCH_ITERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|v| v.clamp(1, 500))
        .unwrap_or(200)
}

fn bench_warmup() -> usize {
    std::env::var("METALLIC_FUSED_POLICY_BENCH_WARMUP")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|v| v.min(100))
        .unwrap_or(20)
}

fn bench_log_every() -> usize {
    std::env::var("METALLIC_FUSED_POLICY_BENCH_LOG_EVERY")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|v| v.max(1))
        .unwrap_or(10)
}

fn env_usize(key: &'static str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn qkv_shape() -> (usize, usize, usize) {
    let k_dim = env_usize("METALLIC_FUSED_QKV_BENCH_K_DIM", 1024);
    let n_dim = env_usize("METALLIC_FUSED_QKV_BENCH_N_DIM", 1024);
    let n_kv = env_usize("METALLIC_FUSED_QKV_BENCH_N_KV", 256);
    (k_dim, n_dim, n_kv)
}

fn swiglu_shape() -> (usize, usize) {
    let k_dim = env_usize("METALLIC_FUSED_SWIGLU_BENCH_K_DIM", 1024);
    let n_dim = env_usize("METALLIC_FUSED_SWIGLU_BENCH_N_DIM", 1024);
    (k_dim, n_dim)
}

fn bind_all_symbols(symbols: &SymbolTable, bindings: &TensorBindings) -> FastBindings {
    let mut fast = FastBindings::new(symbols.len());
    for (name, id) in symbols.iter() {
        if let Ok(arg) = bindings.get(name) {
            fast.set(*id, arg);
        }
    }
    fast
}

fn execute_compiled(
    foundry: &mut Foundry,
    steps: &[Box<dyn CompiledStep>],
    fast: &FastBindings,
    bindings: &TensorBindings,
    symbols: &SymbolTable,
) -> Result<(), MetalError> {
    for step in steps {
        step.execute(foundry, fast, bindings, symbols)?;
    }
    Ok(())
}

fn bench_materialized(
    foundry: &mut Foundry,
    label: &str,
    warmup: usize,
    iterations: usize,
    run: impl Fn(&mut Foundry) -> Result<(), MetalError>,
) -> Result<f64, MetalError> {
    println!("  -> {label}: warmup start ({warmup} iterations)");
    let _ = std::io::stdout().flush();
    let warmup_log_every = bench_log_every().min(warmup.max(1));
    for i in 0..warmup {
        run(foundry)?;
        foundry.synchronize()?;
        if i == 0 || (i + 1) % warmup_log_every == 0 || i + 1 == warmup {
            println!("     [{label}] warmup {}/{}", i + 1, warmup);
            let _ = std::io::stdout().flush();
        }
    }

    println!("  -> {label}: measure start ({iterations} iterations)");
    let _ = std::io::stdout().flush();
    let start = Instant::now();
    let iter_log_every = bench_log_every().min(iterations.max(1));
    for i in 0..iterations {
        run(foundry)?;
        // Force each iteration to fully materialize on GPU for stable mixed-policy comparisons.
        foundry.synchronize()?;
        if i == 0 || (i + 1) % iter_log_every == 0 || i + 1 == iterations {
            println!("     [{label}] iter {}/{}", i + 1, iterations);
            let _ = std::io::stdout().flush();
        }
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    println!("  -> {label}: {avg_us:.2} us (materialized, iters={iterations})");
    Ok(avg_us)
}

fn build_qkv_exec(
    foundry: &mut Foundry,
    q_mixed: bool,
    k_mixed: bool,
    v_mixed: bool,
    k_dim: usize,
    n_dim: usize,
    n_kv: usize,
) -> Result<(Vec<Box<dyn CompiledStep>>, FastBindings, TensorBindings, SymbolTable), MetalError> {
    let mut bindings = TensorBindings::new();
    let blocks_per_k = k_dim.div_ceil(32);

    let hidden = FoundryTensor::<F16, Pooled>::new(foundry, vec![1, k_dim], TensorInit::CopyFrom(&vec![f16::from_f32(0.1); k_dim]))?;
    let gamma = FoundryTensor::<F16, Pooled>::new(foundry, vec![k_dim], TensorInit::CopyFrom(&vec![f16::from_f32(1.0); k_dim]))?;
    let q_out = FoundryTensor::<F16, Pooled>::new(foundry, vec![1, n_dim], TensorInit::Uninitialized)?;
    let k_out = FoundryTensor::<F16, Pooled>::new(foundry, vec![1, n_kv], TensorInit::Uninitialized)?;
    let v_out = FoundryTensor::<F16, Pooled>::new(foundry, vec![1, n_kv], TensorInit::Uninitialized)?;

    bindings.insert("hidden".to_string(), TensorArg::from_tensor(&hidden));
    bindings.insert("gamma".to_string(), TensorArg::from_tensor(&gamma));
    bindings.insert("q".to_string(), TensorArg::from_tensor(&q_out));
    bindings.insert("k".to_string(), TensorArg::from_tensor(&k_out));
    bindings.insert("v".to_string(), TensorArg::from_tensor(&v_out));

    if q_mixed {
        let w_q = FoundryTensor::<Q8_0, Pooled>::new(foundry, vec![n_dim * k_dim], TensorInit::CopyFrom(&vec![0u8; n_dim * k_dim]))?;
        let s_q = FoundryTensor::<Q8_0, Pooled>::new(
            foundry,
            vec![n_dim * blocks_per_k * 2],
            TensorInit::CopyFrom(&vec![0u8; n_dim * blocks_per_k * 2]),
        )?;
        bindings.insert("w_q".to_string(), TensorArg::from_tensor(&w_q));
        bindings.insert("w_q_scales".to_string(), TensorArg::from_tensor(&s_q));
    } else {
        let w_q = FoundryTensor::<F16, Pooled>::new(
            foundry,
            vec![n_dim * k_dim],
            TensorInit::CopyFrom(&vec![f16::from_f32(0.1); n_dim * k_dim]),
        )?;
        bindings.insert("w_q".to_string(), TensorArg::from_tensor(&w_q));
    }

    if k_mixed {
        let w_k = FoundryTensor::<Q8_0, Pooled>::new(foundry, vec![n_kv * k_dim], TensorInit::CopyFrom(&vec![0u8; n_kv * k_dim]))?;
        let s_k = FoundryTensor::<Q8_0, Pooled>::new(
            foundry,
            vec![n_kv * blocks_per_k * 2],
            TensorInit::CopyFrom(&vec![0u8; n_kv * blocks_per_k * 2]),
        )?;
        bindings.insert("w_k".to_string(), TensorArg::from_tensor(&w_k));
        bindings.insert("w_k_scales".to_string(), TensorArg::from_tensor(&s_k));
    } else {
        let w_k = FoundryTensor::<F16, Pooled>::new(
            foundry,
            vec![n_kv * k_dim],
            TensorInit::CopyFrom(&vec![f16::from_f32(0.1); n_kv * k_dim]),
        )?;
        bindings.insert("w_k".to_string(), TensorArg::from_tensor(&w_k));
    }

    if v_mixed {
        let w_v = FoundryTensor::<Q8_0, Pooled>::new(foundry, vec![n_kv * k_dim], TensorInit::CopyFrom(&vec![0u8; n_kv * k_dim]))?;
        let s_v = FoundryTensor::<Q8_0, Pooled>::new(
            foundry,
            vec![n_kv * blocks_per_k * 2],
            TensorInit::CopyFrom(&vec![0u8; n_kv * blocks_per_k * 2]),
        )?;
        bindings.insert("w_v".to_string(), TensorArg::from_tensor(&w_v));
        bindings.insert("w_v_scales".to_string(), TensorArg::from_tensor(&s_v));
    } else {
        let w_v = FoundryTensor::<F16, Pooled>::new(
            foundry,
            vec![n_kv * k_dim],
            TensorInit::CopyFrom(&vec![f16::from_f32(0.1); n_kv * k_dim]),
        )?;
        bindings.insert("w_v".to_string(), TensorArg::from_tensor(&w_v));
    }

    let step = FusedQkvStep {
        input: Ref("hidden".into()),
        gamma: Some(Ref("gamma".into())),
        w_q: Ref("w_q".into()),
        w_k: Ref("w_k".into()),
        w_v: Ref("w_v".into()),
        bias_q: None,
        bias_k: None,
        bias_v: None,
        s_q: None,
        s_k: None,
        s_v: None,
        out_q: Ref("q".into()),
        out_k: Ref("k".into()),
        out_v: Ref("v".into()),
        k_dim: DynamicValue::Literal(k_dim as u32),
        n_dim: DynamicValue::Literal(n_dim as u32),
        n_kv: DynamicValue::Literal(n_kv as u32),
        m: DynamicValue::Literal(1),
        weights_per_block: DynamicValue::Literal(32),
        strategy: GemvStrategy::Canonical,
    };

    let mut symbols = SymbolTable::new();
    let compiled = step.compile(&mut bindings, &mut symbols);
    let fast = bind_all_symbols(&symbols, &bindings);
    Ok((compiled, fast, bindings, symbols))
}

fn build_swiglu_exec(
    foundry: &mut Foundry,
    gate_mixed: bool,
    up_mixed: bool,
    k_dim: usize,
    n_dim: usize,
) -> Result<(Vec<Box<dyn CompiledStep>>, FastBindings, TensorBindings, SymbolTable), MetalError> {
    let mut bindings = TensorBindings::new();
    let blocks_per_k = k_dim.div_ceil(32);

    let input = FoundryTensor::<F16, Pooled>::new(foundry, vec![1, k_dim], TensorInit::CopyFrom(&vec![f16::from_f32(0.2); k_dim]))?;
    let gamma = FoundryTensor::<F16, Pooled>::new(foundry, vec![k_dim], TensorInit::CopyFrom(&vec![f16::from_f32(1.0); k_dim]))?;
    let output = FoundryTensor::<F16, Pooled>::new(foundry, vec![1, n_dim], TensorInit::Uninitialized)?;
    bindings.insert("input".to_string(), TensorArg::from_tensor(&input));
    bindings.insert("gamma".to_string(), TensorArg::from_tensor(&gamma));
    bindings.insert("out".to_string(), TensorArg::from_tensor(&output));

    if gate_mixed {
        let w_gate = FoundryTensor::<Q8_0, Pooled>::new(foundry, vec![n_dim * k_dim], TensorInit::CopyFrom(&vec![0u8; n_dim * k_dim]))?;
        let s_gate = FoundryTensor::<Q8_0, Pooled>::new(
            foundry,
            vec![n_dim * blocks_per_k * 2],
            TensorInit::CopyFrom(&vec![0u8; n_dim * blocks_per_k * 2]),
        )?;
        bindings.insert("wg".to_string(), TensorArg::from_tensor(&w_gate));
        bindings.insert("wg_scales".to_string(), TensorArg::from_tensor(&s_gate));
    } else {
        let w_gate = FoundryTensor::<F16, Pooled>::new(
            foundry,
            vec![n_dim * k_dim],
            TensorInit::CopyFrom(&vec![f16::from_f32(0.1); n_dim * k_dim]),
        )?;
        bindings.insert("wg".to_string(), TensorArg::from_tensor(&w_gate));
    }

    if up_mixed {
        let w_up = FoundryTensor::<Q8_0, Pooled>::new(foundry, vec![n_dim * k_dim], TensorInit::CopyFrom(&vec![0u8; n_dim * k_dim]))?;
        let s_up = FoundryTensor::<Q8_0, Pooled>::new(
            foundry,
            vec![n_dim * blocks_per_k * 2],
            TensorInit::CopyFrom(&vec![0u8; n_dim * blocks_per_k * 2]),
        )?;
        bindings.insert("wu".to_string(), TensorArg::from_tensor(&w_up));
        bindings.insert("wu_scales".to_string(), TensorArg::from_tensor(&s_up));
    } else {
        let w_up = FoundryTensor::<F16, Pooled>::new(
            foundry,
            vec![n_dim * k_dim],
            TensorInit::CopyFrom(&vec![f16::from_f32(0.1); n_dim * k_dim]),
        )?;
        bindings.insert("wu".to_string(), TensorArg::from_tensor(&w_up));
    }

    let step = FusedSwigluStep {
        input: Ref("input".into()),
        gamma: Ref("gamma".into()),
        wg: Ref("wg".into()),
        wu: Ref("wu".into()),
        bg: None,
        bu: None,
        out: Ref("out".into()),
        epsilon: 1e-6,
        weights_per_block: 32,
    };

    let mut symbols = SymbolTable::new();
    let compiled = step.compile(&mut bindings, &mut symbols);
    let fast = bind_all_symbols(&symbols, &bindings);
    Ok((compiled, fast, bindings, symbols))
}

#[test]
#[ignore]
fn benchmark_fused_qkv_mixed_policy_materialized() -> Result<(), MetalError> {
    let mut foundry = match Foundry::new() {
        Ok(foundry) => foundry,
        Err(MetalError::DeviceNotFound) => {
            eprintln!("Skipping benchmark_fused_qkv_mixed_policy_materialized: Metal device not available");
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    let iterations = bench_iterations();
    let warmup = bench_warmup();
    let (k_dim, n_dim, n_kv) = qkv_shape();
    println!(
        "FusedQkv mixed-policy benchmark (materialized): iters={iterations}, warmup={warmup}, k_dim={k_dim}, n_dim={n_dim}, n_kv={n_kv}"
    );
    let _ = std::io::stdout().flush();

    println!("  -> building uniform fused QKV graph");
    let _ = std::io::stdout().flush();
    let (uniform_steps, uniform_fast, uniform_bindings, uniform_symbols) =
        build_qkv_exec(&mut foundry, false, false, false, k_dim, n_dim, n_kv)?;
    println!("  -> building mixed fused QKV graph");
    let _ = std::io::stdout().flush();
    let (mixed_steps, mixed_fast, mixed_bindings, mixed_symbols) = build_qkv_exec(&mut foundry, true, false, false, k_dim, n_dim, n_kv)?;

    let uniform_us = bench_materialized(&mut foundry, "uniform f16/f16/f16", warmup, iterations, |f| {
        execute_compiled(f, &uniform_steps, &uniform_fast, &uniform_bindings, &uniform_symbols)
    })?;
    let mixed_us = bench_materialized(&mut foundry, "mixed q8/f16/f16", warmup, iterations, |f| {
        execute_compiled(f, &mixed_steps, &mixed_fast, &mixed_bindings, &mixed_symbols)
    })?;

    println!(
        "  -> mixed penalty vs uniform: {:+.2}% ({:.2}x)",
        ((mixed_us / uniform_us) - 1.0) * 100.0,
        mixed_us / uniform_us
    );
    Ok(())
}

#[test]
#[ignore]
fn benchmark_fused_swiglu_mixed_policy_materialized() -> Result<(), MetalError> {
    let mut foundry = match Foundry::new() {
        Ok(foundry) => foundry,
        Err(MetalError::DeviceNotFound) => {
            eprintln!("Skipping benchmark_fused_swiglu_mixed_policy_materialized: Metal device not available");
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    let iterations = bench_iterations();
    let warmup = bench_warmup();
    let (k_dim, n_dim) = swiglu_shape();
    println!("FusedSwiGlu mixed-policy benchmark (materialized): iters={iterations}, warmup={warmup}, k_dim={k_dim}, n_dim={n_dim}");
    let _ = std::io::stdout().flush();

    println!("  -> building uniform fused SwiGLU graph");
    let _ = std::io::stdout().flush();
    let (uniform_steps, uniform_fast, uniform_bindings, uniform_symbols) = build_swiglu_exec(&mut foundry, false, false, k_dim, n_dim)?;
    println!("  -> building mixed fused SwiGLU graph");
    let _ = std::io::stdout().flush();
    let (mixed_steps, mixed_fast, mixed_bindings, mixed_symbols) = build_swiglu_exec(&mut foundry, true, false, k_dim, n_dim)?;

    let uniform_us = bench_materialized(&mut foundry, "uniform f16/f16", warmup, iterations, |f| {
        execute_compiled(f, &uniform_steps, &uniform_fast, &uniform_bindings, &uniform_symbols)
    })?;
    let mixed_us = bench_materialized(&mut foundry, "mixed q8/f16", warmup, iterations, |f| {
        execute_compiled(f, &mixed_steps, &mixed_fast, &mixed_bindings, &mixed_symbols)
    })?;

    println!(
        "  -> mixed penalty vs uniform: {:+.2}% ({:.2}x)",
        ((mixed_us / uniform_us) - 1.0) * 100.0,
        mixed_us / uniform_us
    );
    Ok(())
}
