use std::time::Instant;

use half::f16;
use metallic_foundry::{
    Foundry, metals::gemm::GemmV2Step, policy::activation::Activation, spec::{DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::Pooled, tensor::{Tensor as FoundryTensor, TensorInit, dtypes::F16}, types::TensorArg
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TestQuantization {
    F16,
    Q8,
}

struct BenchmarkConfig {
    m: usize,
    n: usize,
    k: usize,
    transpose_a: bool,
    transpose_b: bool,
    quant_b: TestQuantization,
    iterations: usize,
}

fn run_gemm_benchmark_case(foundry: &mut Foundry, cfg: BenchmarkConfig) {
    let mode_str = match cfg.quant_b {
        TestQuantization::F16 => "F16",
        TestQuantization::Q8 => "Q8",
    };
    println!(
        "\nBenchmarking GEMM {}: M={}, N={}, K={} (ta={}, tb={})",
        mode_str, cfg.m, cfg.n, cfg.k, cfg.transpose_a, cfg.transpose_b
    );

    // 1. Allocation & Data Prep
    let a_rows = if cfg.transpose_a { cfg.k } else { cfg.m };
    let a_cols = if cfg.transpose_a { cfg.m } else { cfg.k };
    let b_rows = if cfg.transpose_b { cfg.n } else { cfg.k };
    let b_cols = if cfg.transpose_b { cfg.k } else { cfg.n };

    let a_data = vec![f16::from_f32(0.1); a_rows * a_cols];
    let b_data = vec![f16::from_f32(0.2); b_rows * b_cols];

    let a = FoundryTensor::<F16, Pooled>::new(foundry, vec![a_rows, a_cols], TensorInit::CopyFrom(&a_data)).unwrap();
    let output = FoundryTensor::<F16, Pooled>::new(foundry, vec![cfg.m, cfg.n], TensorInit::Uninitialized).unwrap();

    let mut bindings = TensorBindings::new();
    bindings.insert("a".to_string(), TensorArg::from_tensor(&a));
    bindings.insert("output".to_string(), TensorArg::from_tensor(&output));

    // Optional quantized B and its scales
    let _b_f16: Option<FoundryTensor<F16, Pooled>>;
    let _b_weights: Option<FoundryTensor<metallic_foundry::tensor::Q8_0, Pooled>>;
    let _b_scales: Option<FoundryTensor<F16, Pooled>>;

    let (b_ref, b_scales_ref) = if cfg.quant_b == TestQuantization::Q8 {
        let blocks_per_k = cfg.k.div_ceil(32);
        let bw = FoundryTensor::<metallic_foundry::tensor::Q8_0, Pooled>::new(
            foundry,
            vec![cfg.n, blocks_per_k * 32],
            TensorInit::Uninitialized,
        )
        .unwrap();
        let bs = FoundryTensor::<F16, Pooled>::new(foundry, vec![cfg.n, blocks_per_k], TensorInit::Uninitialized).unwrap();

        bindings.insert("b_weights".to_string(), TensorArg::from_tensor(&bw));
        bindings.insert("b_scales".to_string(), TensorArg::from_tensor(&bs));

        _b_weights = Some(bw);
        _b_scales = Some(bs);
        _b_f16 = None;
        (Ref("b_weights".to_string()), Some(Ref("b_scales".to_string())))
    } else {
        let b = FoundryTensor::<F16, Pooled>::new(foundry, vec![b_rows, b_cols], TensorInit::CopyFrom(&b_data)).unwrap();
        bindings.insert("b".to_string(), TensorArg::from_tensor(&b));
        _b_f16 = Some(b);
        _b_weights = None;
        _b_scales = None;
        (Ref("b".to_string()), None)
    };

    // 2. Step Compilation
    let step = GemmV2Step {
        a: Ref("a".to_string()),
        b: b_ref,
        d: Ref("output".to_string()),
        b_scales: b_scales_ref,
        m_dim: DynamicValue::Literal(cfg.m as u32),
        n_dim: DynamicValue::Literal(cfg.n as u32),
        k_dim: DynamicValue::Literal(cfg.k as u32),
        transpose_a: cfg.transpose_a,
        transpose_b: cfg.transpose_b,
        alpha: 1.0,
        beta: 0.0,
        b_is_canonical: 0,
        activation: Activation::None,
        bias: None,
        c: None,
        weights_per_block: 32,
        tile_config: None,
        params: Default::default(),
    };

    let mut symbols = SymbolTable::new();
    let compiled_steps = step.compile(&mut bindings, &mut symbols);
    let mut fast_bindings = FastBindings::new(symbols.len());
    for (name, id) in symbols.iter() {
        if let Ok(arg) = bindings.get(name) {
            fast_bindings.set(*id, arg);
        }
    }

    let execute_v2 = |f: &mut Foundry| {
        for c_step in &compiled_steps {
            if let Err(e) = c_step.execute(f, &fast_bindings, &bindings, &symbols) {
                panic!("V2 execution failed: {:?}", e);
            }
        }
    };

    // 3. V2 Warmup & Measurement
    for _ in 0..50 {
        execute_v2(foundry);
    }
    foundry.synchronize().unwrap();

    let start_v2 = Instant::now();
    foundry.start_capture().unwrap();
    for _ in 0..cfg.iterations {
        execute_v2(foundry);
    }
    let cpu_time = start_v2.elapsed();
    let buf = foundry.end_capture().unwrap();
    buf.wait_until_completed();
    let total_time = start_v2.elapsed();

    let v2_micros = total_time.as_micros() as f64 / cfg.iterations as f64;
    let cpu_micros = cpu_time.as_micros() as f64 / cfg.iterations as f64;
    let v2_tflops = (2.0 * cfg.m as f64 * cfg.n as f64 * cfg.k as f64) / (v2_micros * 1e6);

    println!(
        "  -> V2:  {:>8.2} us (Overhead: {:>8.2} us) | {:>6.2} TFLOPS",
        v2_micros, cpu_micros, v2_tflops
    );
}

#[test]
fn benchmark_qwen25_shapes() {
    let mut foundry = Foundry::new().unwrap();
    let iterations = 1000;
    const RUN_Q8: bool = false;

    let hidden = 896;
    let intermediate = 4864;
    let vocab_subset = 16384;

    let batch_sizes = vec![1, 32, 512];

    for m in batch_sizes {
        println!("\n--- Batch Size M={} ---", m);

        // MLP Up
        run_gemm_benchmark_case(
            &mut foundry,
            BenchmarkConfig {
                m,
                n: intermediate,
                k: hidden,
                transpose_a: false,
                transpose_b: true,
                quant_b: TestQuantization::F16,
                iterations,
            },
        );
        if RUN_Q8 {
            run_gemm_benchmark_case(
                &mut foundry,
                BenchmarkConfig {
                    m,
                    n: intermediate,
                    k: hidden,
                    transpose_a: false,
                    transpose_b: true,
                    quant_b: TestQuantization::Q8,
                    iterations,
                },
            );
        }

        // MLP Down
        run_gemm_benchmark_case(
            &mut foundry,
            BenchmarkConfig {
                m,
                n: hidden,
                k: intermediate,
                transpose_a: false,
                transpose_b: true,
                quant_b: TestQuantization::F16,
                iterations,
            },
        );
        if RUN_Q8 {
            run_gemm_benchmark_case(
                &mut foundry,
                BenchmarkConfig {
                    m,
                    n: hidden,
                    k: intermediate,
                    transpose_a: false,
                    transpose_b: true,
                    quant_b: TestQuantization::Q8,
                    iterations,
                },
            );
        }

        // LM Head
        run_gemm_benchmark_case(
            &mut foundry,
            BenchmarkConfig {
                m,
                n: vocab_subset,
                k: hidden,
                transpose_a: false,
                transpose_b: true,
                quant_b: TestQuantization::F16,
                iterations,
            },
        );
        if RUN_Q8 {
            run_gemm_benchmark_case(
                &mut foundry,
                BenchmarkConfig {
                    m,
                    n: vocab_subset,
                    k: hidden,
                    transpose_a: false,
                    transpose_b: true,
                    quant_b: TestQuantization::Q8,
                    iterations,
                },
            );

            // Extra Q8 benchmarks with transpose_b = false
            run_gemm_benchmark_case(
                &mut foundry,
                BenchmarkConfig {
                    m,
                    n: vocab_subset,
                    k: hidden,
                    transpose_a: false,
                    transpose_b: false, 
                    quant_b: TestQuantization::Q8,
                    iterations,
                },
            );
        }
    }
}
