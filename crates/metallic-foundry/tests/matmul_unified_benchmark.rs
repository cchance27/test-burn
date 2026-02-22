use std::time::Instant;

use metallic_foundry::{
    Foundry, metals::matmul::MatMulStep, policy::activation::Activation, spec::{DynamicValue, FastBindings, Ref, Step, TensorBindings}, storage::Pooled, tensor::{Tensor, TensorInit, dtypes::F16}, types::TensorArg
};

struct BenchmarkConfig {
    m: usize,
    n: usize,
    k: usize,
    iterations: usize,
}

fn run_matmul_benchmark(foundry: &mut Foundry, cfg: BenchmarkConfig) {
    println!("\nBenchmarking Unified MatMul: M={}, N={}, K={}", cfg.m, cfg.n, cfg.k);

    // Allocation
    let dims_a = vec![cfg.m, cfg.k];
    let dims_b = vec![cfg.k, cfg.n];
    let dims_out = vec![cfg.m, cfg.n];

    let a = Tensor::<F16, Pooled>::new(foundry, dims_a, TensorInit::Uninitialized).unwrap();
    let b = Tensor::<F16, Pooled>::new(foundry, dims_b, TensorInit::Uninitialized).unwrap();
    let out = Tensor::<F16, Pooled>::new(foundry, dims_out, TensorInit::Uninitialized).unwrap();

    let mut bindings = TensorBindings::new();
    bindings.insert("a".to_string(), TensorArg::from_tensor(&a));
    bindings.insert("b".to_string(), TensorArg::from_tensor(&b));
    bindings.insert("output".to_string(), TensorArg::from_tensor(&out));

    let step = MatMulStep {
        a: Ref("a".into()),
        b: Ref("b".into()),
        c: None,
        output: Ref("output".into()),
        bias: None,
        b_scales: None,
        m: DynamicValue::Literal(cfg.m as u32),
        n: DynamicValue::Literal(cfg.n as u32),
        k: DynamicValue::Literal(cfg.k as u32),
        transpose_a: false,
        transpose_b: false, // Standard layout -> ColMajor for GEMV (KxN weights)
        alpha: 1.0,
        beta: 0.0,
        activation: Activation::None,
        weights_per_block: 32,
    };

    let mut symbols = metallic_foundry::spec::SymbolTable::new();
    let compiled_steps = step.compile(&mut bindings, &mut symbols);

    let mut fast_bindings = FastBindings::new(symbols.len());
    for (name, id) in symbols.iter() {
        if let Ok(arg) = bindings.get(name) {
            fast_bindings.set(*id, arg);
        }
    }

    let execute = |f: &mut Foundry| {
        for s in &compiled_steps {
            s.execute(f, &fast_bindings, &bindings, &symbols).unwrap();
        }
    };

    // Warmup
    for _ in 0..10 {
        execute(foundry);
    }

    foundry.synchronize().unwrap();

    // Measure
    let start = Instant::now();
    foundry.start_capture().unwrap();
    for _ in 0..cfg.iterations {
        execute(foundry);
    }
    let buf = foundry.end_capture().unwrap();
    buf.wait_until_completed();
    let duration = start.elapsed();

    let micros = duration.as_micros() as f64 / cfg.iterations as f64;
    println!("  -> Latency: {:.2} us", micros);
}

#[test]
fn benchmark_unified_dispatch() {
    let mut foundry = Foundry::new().unwrap();

    // Case 1: M=1 -> Should hit GEMV path (expect ~70-80us)
    run_matmul_benchmark(
        &mut foundry,
        BenchmarkConfig {
            m: 1,
            n: 4864,
            k: 896,
            iterations: 1000,
        },
    );

    // Case 2: M=32 -> Should hit GEMM path (expect ~80us)
    run_matmul_benchmark(
        &mut foundry,
        BenchmarkConfig {
            m: 32,
            n: 4864,
            k: 896,
            iterations: 1000,
        },
    );
}
