use std::time::Instant;

use metallic_foundry::{
    Foundry, compound::Layout, metals::gemv::{GemvStrategy, GemvV2Step}, policy::activation::Activation, spec::{DynamicValue, FastBindings, Ref, Step, TensorBindings}, storage::Pooled, tensor::{Tensor, TensorInit, dtypes::F16}, types::TensorArg
}; // Added GemvStrategy

fn run_benchmark_case(
    foundry: &mut Foundry,
    k: usize,
    n: usize,
    layout: Layout,
    strategy: Option<GemvStrategy>,
    alpha: f32,
    iterations: usize,
) {
    let layout_str = match layout {
        Layout::RowMajor => "NK (RowMajor)",
        Layout::ColMajor => "KN (ColMajor)",
        Layout::Canonical { .. } => "Blocked (Canonical)",
    };
    let strategy_str = match strategy {
        Some(GemvStrategy::Auto) => "Auto",
        Some(GemvStrategy::Scalar) => "Scalar",
        Some(GemvStrategy::Vectorized) => "Vectorized",
        Some(GemvStrategy::Canonical) => "Canonical",
        None => "Default",
    };
    println!(
        "Benchmarking {} [{}]: K={}, N={}, alpha={:.2}",
        layout_str, strategy_str, k, n, alpha
    );

    // Allocation
    let dims_weights = vec![k * n];
    let dims_input = vec![k];
    let dims_output = vec![n];

    let weights = Tensor::<F16, Pooled>::new(foundry, dims_weights, TensorInit::Uninitialized).unwrap();
    let input = Tensor::<F16, Pooled>::new(foundry, dims_input, TensorInit::Uninitialized).unwrap();
    let output_v2 = Tensor::<F16, Pooled>::new(foundry, dims_output.clone(), TensorInit::Uninitialized).unwrap();

    // ----------------------------------------------------------------
    // Setup V2
    // ----------------------------------------------------------------
    let mut bindings = TensorBindings::new();
    bindings.insert("weights".to_string(), TensorArg::from_tensor(&weights));
    bindings.insert("input".to_string(), TensorArg::from_tensor(&input));
    bindings.insert("output".to_string(), TensorArg::from_tensor(&output_v2));

    let step = GemvV2Step {
        weights: Ref("weights".to_string()),
        scale_bytes: None, // F16 benchmark
        input: Ref("input".to_string()),
        output: Ref("output".to_string()),
        bias: None,
        residual: None,
        k_dim: DynamicValue::Literal(k as u32),
        n_dim: DynamicValue::Literal(n as u32),
        weights_per_block: 32,
        layout,
        strategy,
        alpha: 1.0,
        beta: 0.0,
        activation: Activation::None,
    };

    let mut symbols = metallic_foundry::spec::SymbolTable::new();
    let compiled_steps = step.compile(&mut bindings, &mut symbols);

    // Create FastBindings from SymbolTable
    let mut fast_bindings = FastBindings::new(symbols.len());
    for (name, id) in symbols.iter() {
        if let Ok(arg) = bindings.get(name) {
            fast_bindings.set(*id, arg);
        }
    }

    let run_v2 = |f: &mut Foundry| {
        for c_step in &compiled_steps {
            c_step.execute(f, &fast_bindings, &bindings, &symbols).unwrap();
        }
    };

    // Warmup
    for _ in 0..5 {
        run_v2(foundry);
    }

    // Measure V2
    let start = Instant::now();
    for _ in 0..iterations {
        run_v2(foundry);
    }
    let v2_duration = start.elapsed();

    let v2_avg = v2_duration.as_micros() as f64 / iterations as f64;

    println!("  -> V2: {:.2} us", v2_avg);
}

#[test]
fn benchmark_gemv_v2_perf() {
    let mut foundry = Foundry::new().unwrap();
    let iterations = 1000;

    let cases = vec![(128, 128), (512, 128), (128, 512), (4096, 4096), (4096, 128), (128, 4096)];

    println!("\n=== NK Layout (RowMajor) - GemvV2 ===");
    for (k, n) in &cases {
        run_benchmark_case(&mut foundry, *k, *n, Layout::RowMajor, None, 1.0, iterations);
    }
    run_benchmark_case(&mut foundry, 4096, 4096, Layout::RowMajor, None, 0.5, iterations);

    println!("\n=== KN Layout (ColMajor) - GemvV2 ===");
    for (k, n) in &cases {
        run_benchmark_case(&mut foundry, *k, *n, Layout::ColMajor, None, 1.0, iterations);
    }
    run_benchmark_case(&mut foundry, 4096, 4096, Layout::ColMajor, None, 0.5, iterations);

    println!("\n=== Large N Optimization (ColMajor) ===");
    // Test large N case to verify Auto -> Scalar switch
    let large_n_cases = vec![(896, 4096), (896, 16384)];
    for (k, n) in &large_n_cases {
        // Explicit Vectorized (Baseline)
        run_benchmark_case(
            &mut foundry,
            *k,
            *n,
            Layout::ColMajor,
            Some(GemvStrategy::Vectorized),
            1.0,
            iterations,
        );
        // Explicit Scalar (Optimized)
        run_benchmark_case(&mut foundry, *k, *n, Layout::ColMajor, Some(GemvStrategy::Scalar), 1.0, iterations);
        // Auto (Should match Scalar)
        run_benchmark_case(&mut foundry, *k, *n, Layout::ColMajor, Some(GemvStrategy::Auto), 1.0, iterations);
    }

    println!("\n=== Decode Shapes (RowMajor) ===");
    let decode_cases = vec![(896, 896), (896, 151_936)];
    let decode_iterations = 200;
    for (k, n) in &decode_cases {
        run_benchmark_case(
            &mut foundry,
            *k,
            *n,
            Layout::RowMajor,
            Some(GemvStrategy::Vectorized),
            1.0,
            decode_iterations,
        );
        run_benchmark_case(
            &mut foundry,
            *k,
            *n,
            Layout::RowMajor,
            Some(GemvStrategy::Auto),
            1.0,
            decode_iterations,
        );
    }

    println!("\n=== Blocked (Canonical) Layout - GemvV2 ===");
    let canonical_cases = vec![(128, 128), (4096, 128), (4096, 4096)];
    for (k, n) in &canonical_cases {
        run_benchmark_case(
            &mut foundry,
            *k,
            *n,
            Layout::Canonical {
                expected_k: 0,
                expected_n: 0,
            },
            None,
            1.0,
            iterations,
        );
    }
    run_benchmark_case(
        &mut foundry,
        4096,
        4096,
        Layout::Canonical {
            expected_k: 0,
            expected_n: 0,
        },
        None,
        2.0,
        iterations,
    );
}
