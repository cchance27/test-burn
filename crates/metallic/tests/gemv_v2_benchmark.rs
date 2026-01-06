use std::time::Instant;

use metallic::{
    compound::stages::Layout, foundry::{
        Foundry, spec::{DynamicValue, FastBindings, Ref, Step, TensorBindings}, storage::Pooled, tensor::Tensor
    }, metals::{
        gemv::{GemvColMajor, GemvParams, GemvRowMajor}, v2::gemv::step::GemvV2Step
    }, tensor::{TensorInit, dtypes::F16}, types::TensorArg
};

fn run_benchmark_case(foundry: &mut Foundry, k: usize, n: usize, layout: Layout, iterations: usize) {
    let layout_str = match layout {
        Layout::RowMajor => "NK (RowMajor)",
        Layout::ColMajor => "KN (ColMajor)",
    };
    println!("Benchmarking {}: K={}, N={}", layout_str, k, n);

    // Allocation
    let dims_weights = vec![k * n];
    let dims_input = vec![k];
    let dims_output = vec![n];

    let weights = Tensor::<F16, Pooled>::new(foundry, dims_weights, TensorInit::Uninitialized).unwrap();
    let input = Tensor::<F16, Pooled>::new(foundry, dims_input, TensorInit::Uninitialized).unwrap();
    let output_legacy = Tensor::<F16, Pooled>::new(foundry, dims_output.clone(), TensorInit::Uninitialized).unwrap();
    let output_v2 = Tensor::<F16, Pooled>::new(foundry, dims_output.clone(), TensorInit::Uninitialized).unwrap();

    // ----------------------------------------------------------------
    // Setup Legacy
    // ----------------------------------------------------------------
    let params = match layout {
        Layout::RowMajor => GemvParams {
            k: k as u32,
            n: n as u32,
            batch: 1,
            stride_x: 1,
            stride_y: 1,
            stride_a: 0,
            stride_w: k as u32,
            blocks_per_k: (k / 32) as u32,
            weights_per_block: 32,
            stride_scale: 0,
        },
        Layout::ColMajor => GemvParams {
            k: k as u32,
            n: n as u32,
            batch: 1,
            stride_x: 1,
            stride_y: 1,
            stride_a: 0,
            stride_w: n as u32,
            blocks_per_k: 0,
            weights_per_block: 0,
            stride_scale: 0,
        },
    };

    let weights_arg = TensorArg::from_tensor(&weights);
    let input_arg = TensorArg::from_tensor(&input);
    let output_legacy_arg = TensorArg::from_tensor(&output_legacy);

    let run_legacy: Box<dyn Fn(&mut Foundry)> = match layout {
        Layout::RowMajor => {
            let kernel = GemvColMajor::new(&weights_arg, &input_arg, &output_legacy_arg, params);
            Box::new(move |f| f.run(&kernel).unwrap())
        }
        Layout::ColMajor => {
            let kernel = GemvRowMajor::new(&weights_arg, &input_arg, &output_legacy_arg, params);
            Box::new(move |f| f.run(&kernel).unwrap())
        }
    };

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
        k_dim: DynamicValue::Literal(k as u32),
        n_dim: DynamicValue::Literal(n as u32),
        weights_per_block: 32,
        layout: layout,
    };

    let mut symbols = metallic::foundry::spec::SymbolTable::new();
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
            c_step.execute(f, &fast_bindings, &bindings).unwrap();
        }
    };

    // Warmup
    for _ in 0..5 {
        run_legacy(foundry);
        run_v2(foundry);
    }

    // Measure Legacy
    let start = Instant::now();
    for _ in 0..iterations {
        run_legacy(foundry);
    }
    let legacy_duration = start.elapsed();

    // Measure V2
    let start = Instant::now();
    for _ in 0..iterations {
        run_v2(foundry);
    }
    let v2_duration = start.elapsed();

    let legacy_avg = legacy_duration.as_micros() as f64 / iterations as f64;
    let v2_avg = v2_duration.as_micros() as f64 / iterations as f64;
    let speedup = legacy_avg / v2_avg;

    println!(
        "  -> Legacy: {:.2} us | V2: {:.2} us | Speedup: {:.2}x",
        legacy_avg, v2_avg, speedup
    );
}

#[test]
fn benchmark_gemv_v2_perf() {
    let mut foundry = Foundry::new().unwrap();
    let iterations = 1000;

    let cases = vec![(128, 128), (512, 128), (128, 512), (4096, 4096), (4096, 128), (128, 4096)];

    println!("\n=== NK Layout (RowMajor) - GemvV2 ===");
    for (k, n) in &cases {
        run_benchmark_case(&mut foundry, *k, *n, Layout::RowMajor, iterations);
    }

    println!("\n=== KN Layout (ColMajor) - GemvV2 ===");
    for (k, n) in &cases {
        run_benchmark_case(&mut foundry, *k, *n, Layout::ColMajor, iterations);
    }
}
