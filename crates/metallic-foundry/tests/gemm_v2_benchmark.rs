use std::time::Instant;

use half::f16;
use metallic_context::{
    Context, QuantizedQ8_0Tensor, kernels::matmul_mlx::MatMulMlxOp, tensor::{
        F16 as LegacyF16, QuantizedTensor as LegacyQuantizedTensor, Tensor as LegacyTensor, TensorInit as LegacyInit, TensorStorage as LegacyStorage, TensorType
    }
};
use metallic_foundry::{
    Foundry, compound::stages::Quantization, metals::gemm::step::GemmV2Step, spec::{DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::Pooled, tensor::{Tensor as FoundryTensor, TensorInit, dtypes::F16}, types::TensorArg
};
use objc2_metal::MTLCommandBuffer as _;

struct BenchmarkConfig {
    m: usize,
    n: usize,
    k: usize,
    transpose_a: bool,
    transpose_b: bool,
    quant_b: Quantization,
    iterations: usize,
}

fn run_gemm_benchmark_case(foundry: &mut Foundry, ctx: &mut Context<LegacyF16>, cfg: BenchmarkConfig) {
    let mode_str = match cfg.quant_b {
        Quantization::F16 => "F16",
        Quantization::Q8 => "Q8",
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
    let _b_weights: Option<FoundryTensor<metallic_foundry::tensor::dtypes::U8, Pooled>>;
    let _b_scales: Option<FoundryTensor<F16, Pooled>>;

    let (b_ref, b_scales_ref) = if cfg.quant_b == Quantization::Q8 {
        let blocks_per_k = cfg.k.div_ceil(32);
        let bw = FoundryTensor::<metallic_foundry::tensor::dtypes::U8, Pooled>::new(
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
        output: Ref("output".to_string()),
        b_scales: b_scales_ref,
        m_dim: DynamicValue::Literal(cfg.m as u32),
        n_dim: DynamicValue::Literal(cfg.n as u32),
        k_dim: DynamicValue::Literal(cfg.k as u32),
        b_quant: cfg.quant_b,
        transpose_a: cfg.transpose_a,
        transpose_b: cfg.transpose_b,
        alpha: 1.0,
        beta: 0.0,
        bias: None,
        c: None,
        weights_per_block: 32,
        tile_config: None,
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
            c_step.execute(f, &fast_bindings, &bindings, &symbols).unwrap();
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
    buf.waitUntilCompleted();
    let total_time = start_v2.elapsed();

    let v2_micros = total_time.as_micros() as f64 / cfg.iterations as f64;
    let cpu_micros = cpu_time.as_micros() as f64 / cfg.iterations as f64;
    let v2_tflops = (2.0 * cfg.m as f64 * cfg.n as f64 * cfg.k as f64) / (v2_micros * 1e6);

    println!(
        "  -> V2:  {:>8.2} us (Overhead: {:>8.2} us) | {:>6.2} TFLOPS",
        v2_micros, cpu_micros, v2_tflops
    );

    // 4. MLX Comparison
    let mut mlx_micros = 0.0;

    // LegacyTensor uses dedicated storage on F16 ctx for F16 tensors
    let a_leg =
        LegacyTensor::<LegacyF16>::new(vec![a_rows, a_cols], LegacyStorage::Dedicated(ctx), LegacyInit::CopyFrom(&a_data)).unwrap();

    let run_mlx_f16 = |c: &mut Context<LegacyF16>, b_leg: &LegacyTensor<LegacyF16>| -> Result<(), metallic_foundry::MetalError> {
        c.call::<MatMulMlxOp>(
            (
                &a_leg,
                TensorType::Dense(b_leg),
                None,
                None,
                cfg.transpose_a,
                cfg.transpose_b,
                1.0,
                0.0,
            ),
            None,
        )
        .map_err(|e| metallic_foundry::MetalError::OperationFailed(format!("Context Error: {:?}", e)))?;
        Ok(())
    };

    let run_mlx_q8 = |c: &mut Context<LegacyF16>, b_quant: &QuantizedQ8_0Tensor| -> Result<(), metallic_foundry::MetalError> {
        c.call::<MatMulMlxOp>(
            (
                &a_leg,
                TensorType::Quant(LegacyQuantizedTensor::Q8_0(b_quant)),
                None,
                None,
                cfg.transpose_a,
                cfg.transpose_b,
                1.0,
                0.0,
            ),
            None,
        )
        .map_err(|e| metallic_foundry::MetalError::OperationFailed(format!("Context Error: {:?}", e)))?;
        Ok(())
    };

    if cfg.quant_b == Quantization::F16 {
        let b_leg =
            LegacyTensor::<LegacyF16>::new(vec![b_rows, b_cols], LegacyStorage::Dedicated(ctx), LegacyInit::CopyFrom(&b_data)).unwrap();

        let mut skipped = false;
        for _ in 0..50 {
            if run_mlx_f16(ctx, &b_leg).is_err() {
                skipped = true;
                break;
            }
        }

        if skipped {
            println!("  -> MLX: Skipped (Not Supported)");
        } else {
            ctx.synchronize();
            let start_mlx = Instant::now();
            for _ in 0..cfg.iterations {
                let _ = run_mlx_f16(ctx, &b_leg);
            }
            ctx.synchronize();
            mlx_micros = start_mlx.elapsed().as_micros() as f64 / cfg.iterations as f64;
            let mlx_tflops = (2.0 * cfg.m as f64 * cfg.n as f64 * cfg.k as f64) / (mlx_micros * 1e6);
            println!("  -> MLX: {:>8.2} us | {:>6.2} TFLOPS (Legacy)", mlx_micros, mlx_tflops);
        }
    } else {
        // Q8 Setup for MLX
        let ctx_u8 = metallic_context::Context::<metallic_context::tensor::U8>::new().unwrap();

        let blocks_per_k = cfg.k.div_ceil(32);
        let bw = LegacyTensor::<metallic_context::tensor::U8>::new(
            vec![cfg.n, blocks_per_k * 32],
            LegacyStorage::Dedicated(&ctx_u8),
            LegacyInit::Uninitialized,
        )
        .unwrap();
        let bs = LegacyTensor::<metallic_context::tensor::U8>::new(
            vec![cfg.n, blocks_per_k * 2],
            LegacyStorage::Dedicated(&ctx_u8),
            LegacyInit::Uninitialized,
        )
        .unwrap();

        // MLX usually needs logical dims to match the semantic shape of B (N, K)
        let logical_dims = if cfg.transpose_b { vec![cfg.n, cfg.k] } else { vec![cfg.k, cfg.n] };

        let q8_tensor = QuantizedQ8_0Tensor {
            data: bw,
            scales: bs,
            logical_dims,
            blocks_per_k,
        };

        let mut skipped = false;
        // Try once to see if supported
        if run_mlx_q8(ctx, &q8_tensor).is_err() {
            skipped = true;
        } else {
            // Warmup
            for _ in 0..50 {
                let _ = run_mlx_q8(ctx, &q8_tensor);
            }
            ctx.synchronize();

            let start_mlx = Instant::now();
            for _ in 0..cfg.iterations {
                let _ = run_mlx_q8(ctx, &q8_tensor);
            }
            ctx.synchronize();
            mlx_micros = start_mlx.elapsed().as_micros() as f64 / cfg.iterations as f64;
        }

        if skipped {
            println!("  -> MLX: Skipped (Not Supported)");
        } else {
            let mlx_tflops = (2.0 * cfg.m as f64 * cfg.n as f64 * cfg.k as f64) / (mlx_micros * 1e6);
            println!("  -> MLX: {:>8.2} us | {:>6.2} TFLOPS (Legacy)", mlx_micros, mlx_tflops);
        }
    }
}

#[test]
fn benchmark_qwen25_shapes() {
    let mut foundry = Foundry::new().unwrap();
    let mut ctx = Context::<LegacyF16>::new().unwrap();
    let iterations = 1000;

    let hidden = 896;
    let intermediate = 4864;
    let vocab_subset = 16384;

    let batch_sizes = vec![1, 32, 512];

    for m in batch_sizes {
        println!("\n--- Batch Size M={} ---", m);

        // MLP Up
        run_gemm_benchmark_case(
            &mut foundry,
            &mut ctx,
            BenchmarkConfig {
                m,
                n: intermediate,
                k: hidden,
                transpose_a: false,
                transpose_b: true,
                quant_b: Quantization::F16,
                iterations,
            },
        );
        run_gemm_benchmark_case(
            &mut foundry,
            &mut ctx,
            BenchmarkConfig {
                m,
                n: intermediate,
                k: hidden,
                transpose_a: false,
                transpose_b: true,
                quant_b: Quantization::Q8,
                iterations,
            },
        );

        // MLP Down
        run_gemm_benchmark_case(
            &mut foundry,
            &mut ctx,
            BenchmarkConfig {
                m,
                n: hidden,
                k: intermediate,
                transpose_a: false,
                transpose_b: true,
                quant_b: Quantization::F16,
                iterations,
            },
        );
        run_gemm_benchmark_case(
            &mut foundry,
            &mut ctx,
            BenchmarkConfig {
                m,
                n: hidden,
                k: intermediate,
                transpose_a: false,
                transpose_b: true,
                quant_b: Quantization::Q8,
                iterations,
            },
        );

        // LM Head
        run_gemm_benchmark_case(
            &mut foundry,
            &mut ctx,
            BenchmarkConfig {
                m,
                n: vocab_subset,
                k: hidden,
                transpose_a: false,
                transpose_b: true,
                quant_b: Quantization::F16,
                iterations,
            },
        );
        run_gemm_benchmark_case(
            &mut foundry,
            &mut ctx,
            BenchmarkConfig {
                m,
                n: vocab_subset,
                k: hidden,
                transpose_a: false,
                transpose_b: true,
                quant_b: Quantization::Q8,
                iterations,
            },
        );

        // Extra Q8 benchmarks with transpose_b = false for MLX comparison
        // (Pre-transposed weights scenario)
        run_gemm_benchmark_case(
            &mut foundry,
            &mut ctx,
            BenchmarkConfig {
                m,
                n: vocab_subset,
                k: hidden,
                transpose_a: false,
                transpose_b: false, // Explicitly false for comparison
                quant_b: Quantization::Q8,
                iterations,
            },
        );
    }
}
