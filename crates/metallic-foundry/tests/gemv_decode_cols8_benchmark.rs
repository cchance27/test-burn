use std::time::Instant;

use metallic_env::{EnvVarGuard, FoundryEnvVar};
use metallic_foundry::{
    Foundry, MetalError, compound::Layout, metals::{
        gemm::{GemmParams, GemmV2Step}, gemv::{GemvStrategy, GemvV2Params, GemvV2UnifiedExecutionStep}, matmul::MatMulStep
    }, policy::activation::Activation, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};

#[derive(Clone, Copy, Debug)]
struct Shape {
    k: usize,
    n: usize,
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_shapes() -> Vec<Shape> {
    let raw = std::env::var("METALLIC_DECODE_GEMV_SHAPES")
        .unwrap_or_else(|_| "896x896,896x4864,896x151936".to_string());
    let mut shapes = Vec::new();
    for item in raw.split(',') {
        let part = item.trim();
        if part.is_empty() {
            continue;
        }
        let mut it = part.split('x');
        let Some(k) = it.next().and_then(|v| v.trim().parse::<usize>().ok()) else {
            continue;
        };
        let Some(n) = it.next().and_then(|v| v.trim().parse::<usize>().ok()) else {
            continue;
        };
        shapes.push(Shape { k, n });
    }
    if shapes.is_empty() {
        vec![
            Shape { k: 896, n: 896 },
            Shape { k: 896, n: 4864 },
            Shape { k: 896, n: 151_936 },
        ]
    } else {
        shapes
    }
}

fn build_fast_bindings(bindings: &TensorBindings, symbols: &SymbolTable) -> FastBindings {
    let mut fast_bindings = FastBindings::new(symbols.len());
    for (name, id) in symbols.iter() {
        if let Ok(arg) = bindings.get(name) {
            fast_bindings.set(*id, arg);
        }
    }
    fast_bindings
}

fn measure_steps(
    foundry: &mut Foundry,
    steps: &[Box<dyn CompiledStep>],
    fast_bindings: &FastBindings,
    bindings: &TensorBindings,
    symbols: &SymbolTable,
    warmup: usize,
    iters: usize,
) -> Result<f64, MetalError> {
    for _ in 0..warmup {
        for step in steps {
            step.execute(foundry, fast_bindings, bindings, symbols)?;
        }
    }
    foundry.synchronize()?;

    let start = Instant::now();
    foundry.start_capture()?;
    for _ in 0..iters {
        for step in steps {
            step.execute(foundry, fast_bindings, bindings, symbols)?;
        }
    }
    let cb = foundry.end_capture()?;
    cb.wait_until_completed();

    Ok(start.elapsed().as_micros() as f64 / iters as f64)
}

fn run_case(cols8_on: bool, shape: Shape, warmup: usize, iters: usize) -> Result<(), Box<dyn std::error::Error>> {
    let _cols8_guard = EnvVarGuard::set(FoundryEnvVar::GemvF16Cols8, if cols8_on { "1" } else { "0" });
    let mut foundry = Foundry::new()?;

    let a = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, shape.k], TensorInit::Uninitialized)?;
    // Row-major [N, K]; transpose_b=true in matmul/gemm keeps math as [K, N].
    let b = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![shape.n, shape.k], TensorInit::Uninitialized)?;
    let out = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, shape.n], TensorInit::Uninitialized)?;

    let mut bindings = TensorBindings::new();
    bindings.insert("a".to_string(), TensorArg::from_tensor(&a));
    bindings.insert("b".to_string(), TensorArg::from_tensor(&b));
    bindings.insert("out".to_string(), TensorArg::from_tensor(&out));

    // GEMV (decode path baseline)
    let gemv = GemvV2UnifiedExecutionStep {
        input: Ref("a".into()),
        weights: Ref("b".into()),
        output: Ref("out".into()),
        bias: None,
        residual: None,
        scale_bytes: None,
        params: GemvV2Params {
            k_dim: DynamicValue::Literal(shape.k as u32),
            n_dim: DynamicValue::Literal(shape.n as u32),
            weights_per_block: 32,
            batch: DynamicValue::Literal(1),
        },
        layout: Layout::RowMajor,
        strategy: Some(GemvStrategy::Vectorized),
        activation: Activation::None,
        alpha: 1.0,
        beta: 0.0,
        has_bias: 0,
        has_residual: 0,
    };
    let mut sym_gemv = SymbolTable::new();
    let compiled_gemv = gemv.compile(&mut bindings, &mut sym_gemv);
    let fast_gemv = build_fast_bindings(&bindings, &sym_gemv);

    let gemv_auto = GemvV2UnifiedExecutionStep {
        strategy: Some(GemvStrategy::Auto),
        ..gemv
    };
    let mut sym_gemv_auto = SymbolTable::new();
    let compiled_gemv_auto = gemv_auto.compile(&mut bindings, &mut sym_gemv_auto);
    let fast_gemv_auto = build_fast_bindings(&bindings, &sym_gemv_auto);

    // Unified matmul (runtime-dispatch path used by graph)
    let unified = MatMulStep {
        a: Ref("a".into()),
        b: Ref("b".into()),
        output: Ref("out".into()),
        bias: None,
        b_scales: None,
        c: None,
        m: DynamicValue::Literal(1),
        n: DynamicValue::Literal(shape.n as u32),
        k: DynamicValue::Literal(shape.k as u32),
        transpose_a: false,
        transpose_b: true,
        alpha: 1.0,
        beta: 0.0,
        weights_per_block: 32,
        activation: Activation::None,
    };
    let mut sym_uni = SymbolTable::new();
    let compiled_uni = unified.compile(&mut bindings, &mut sym_uni);
    let fast_uni = build_fast_bindings(&bindings, &sym_uni);

    // GEMM compare (not decode-default, but useful to verify crossover behavior).
    let gemm = GemmV2Step {
        a: Ref("a".into()),
        b: Ref("b".into()),
        d: Ref("out".into()),
        c: None,
        bias: None,
        b_scales: None,
        weights_per_block: 32,
        alpha: 1.0,
        beta: 0.0,
        b_is_canonical: 0,
        params: GemmParams::default(),
        m_dim: DynamicValue::Literal(1),
        n_dim: DynamicValue::Literal(shape.n as u32),
        k_dim: DynamicValue::Literal(shape.k as u32),
        transpose_a: false,
        transpose_b: true,
        tile_config: None,
        activation: Activation::None,
    };
    let mut sym_gemm = SymbolTable::new();
    let compiled_gemm = gemm.compile(&mut bindings, &mut sym_gemm);
    let fast_gemm = build_fast_bindings(&bindings, &sym_gemm);

    let us_gemv_vec = measure_steps(&mut foundry, &compiled_gemv, &fast_gemv, &bindings, &sym_gemv, warmup, iters)?;
    let us_gemv_auto =
        measure_steps(&mut foundry, &compiled_gemv_auto, &fast_gemv_auto, &bindings, &sym_gemv_auto, warmup, iters)?;
    let us_unified = measure_steps(&mut foundry, &compiled_uni, &fast_uni, &bindings, &sym_uni, warmup, iters)?;
    let us_gemm = measure_steps(&mut foundry, &compiled_gemm, &fast_gemm, &bindings, &sym_gemm, warmup, iters)?;

    println!(
        "decode-gemv cols8={} k={} n={} iters={} | gemv(vec)={:.2} us | gemv(auto)={:.2} us | unified={:.2} us | gemm={:.2} us",
        if cols8_on { "on" } else { "off" },
        shape.k,
        shape.n,
        iters,
        us_gemv_vec,
        us_gemv_auto,
        us_unified,
        us_gemm
    );

    Ok(())
}

#[test]
#[ignore]
#[serial_test::serial]
fn benchmark_decode_gemv_cols8_matrix() -> Result<(), Box<dyn std::error::Error>> {
    let warmup = env_usize("METALLIC_DECODE_GEMV_WARMUP", 5);
    let iters = env_usize("METALLIC_DECODE_GEMV_ITERS", 25);
    let shapes = parse_shapes();
    println!(
        "decode-gemv benchmark: warmup={} iters={} shapes={:?}",
        warmup,
        iters,
        shapes.iter().map(|s| format!("{}x{}", s.k, s.n)).collect::<Vec<_>>()
    );

    for shape in shapes {
        run_case(true, shape, warmup, iters)?;
        run_case(false, shape, warmup, iters)?;
    }

    Ok(())
}
