use std::time::Instant;

use metallic_foundry::{
    Foundry, compound::stages::Layout, metals::{gemm::step::GemmV2Step, gemv::step::GemvV2Step, matmul::MatMulStep}, policy::f16::PolicyF16, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::Pooled, tensor::{Tensor, TensorInit, dtypes::F16}, types::TensorArg
};
use objc2_metal::MTLCommandBuffer as _;

struct Shape {
    m: usize,
    n: usize,
    k: usize,
}

fn measure_step(
    foundry: &mut Foundry,
    compiled: &[Box<dyn CompiledStep>],
    fast_bindings: &FastBindings,
    bindings: &TensorBindings,
    symbols: &SymbolTable,
    iterations: usize,
) -> f64 {
    // Warmup
    for _ in 0..10 {
        for s in compiled {
            s.execute(foundry, fast_bindings, bindings, symbols).unwrap();
        }
    }

    // Explicit sync to clear queue
    if let Ok(buf) = foundry.end_capture() {
        buf.waitUntilCompleted();
    } else {
        foundry.synchronize().unwrap();
    }

    let start = Instant::now();
    foundry.start_capture().unwrap();
    for _ in 0..iterations {
        for s in compiled {
            s.execute(foundry, fast_bindings, bindings, symbols).unwrap();
        }
    }
    let buf = foundry.end_capture().unwrap();
    buf.waitUntilCompleted();
    let duration = start.elapsed();

    duration.as_micros() as f64 / iterations as f64
}

fn run_comparison(foundry: &mut Foundry, shape: Shape, iterations: usize) {
    println!("\n--- Shape: M={}, N={}, K={} ---", shape.m, shape.n, shape.k);

    // Setup Tensors
    let dims_a = vec![shape.m, shape.k];
    let dims_b = vec![shape.k, shape.n];
    let dims_out = vec![shape.m, shape.n];

    let a = Tensor::<F16, Pooled>::new(foundry, dims_a.clone(), TensorInit::Uninitialized).unwrap();
    let b = Tensor::<F16, Pooled>::new(foundry, dims_b.clone(), TensorInit::Uninitialized).unwrap();
    let out = Tensor::<F16, Pooled>::new(foundry, dims_out.clone(), TensorInit::Uninitialized).unwrap();

    let mut bindings = TensorBindings::new();
    bindings.insert("a".to_string(), TensorArg::from_tensor(&a));
    bindings.insert("b".to_string(), TensorArg::from_tensor(&b));
    bindings.insert("output".to_string(), TensorArg::from_tensor(&out));

    // 1. GEMM V2
    let gemm = GemmV2Step {
        a: Ref("a".into()),
        b: Ref("b".into()),
        output: Ref("output".into()),
        bias: None,
        b_scales: None,
        c: None,
        m_dim: DynamicValue::Literal(shape.m as u32),
        n_dim: DynamicValue::Literal(shape.n as u32),
        k_dim: DynamicValue::Literal(shape.k as u32),
        b_quant: std::sync::Arc::new(PolicyF16),
        transpose_a: false,
        transpose_b: false,
        alpha: 1.0,
        beta: 0.0,
        weights_per_block: 32,
        tile_config: None, // Auto
    };
    let mut sym_gemm = SymbolTable::new();
    let compiled_gemm = gemm.compile(&mut bindings, &mut sym_gemm);
    let mut fb_gemm = FastBindings::new(sym_gemm.len());
    for (name, id) in sym_gemm.iter() {
        if let Ok(arg) = bindings.get(name) {
            fb_gemm.set(*id, arg);
        }
    }

    let lat_gemm = measure_step(foundry, &compiled_gemm, &fb_gemm, &bindings, &sym_gemm, iterations);
    println!("GEMM V2:    {:>8.2} us", lat_gemm);

    // 2. GEMV V2 (Only valid if M=1)
    let lat_gemv = if shape.m == 1 {
        let gemv = GemvV2Step {
            input: Ref("a".into()),
            weights: Ref("b".into()), // Transpose B=false -> ColMajor KxN
            output: Ref("output".into()),
            bias: None,
            residual: None,
            scale_bytes: None,
            k_dim: DynamicValue::Literal(shape.k as u32),
            n_dim: DynamicValue::Literal(shape.n as u32),
            weights_per_block: 32,
            layout: Layout::ColMajor, // Matches GEMM TransposeB=false
            strategy: None,
            alpha: 1.0,
            beta: 0.0,
        };
        let mut sym_gemv = SymbolTable::new();
        let compiled_gemv = gemv.compile(&mut bindings, &mut sym_gemv);
        let mut fb_gemv = FastBindings::new(sym_gemv.len());
        for (name, id) in sym_gemv.iter() {
            if let Ok(arg) = bindings.get(name) {
                fb_gemv.set(*id, arg);
            }
        }

        let val = measure_step(foundry, &compiled_gemv, &fb_gemv, &bindings, &sym_gemv, iterations);
        println!("GEMV V2:    {:>8.2} us", val);
        Some(val)
    } else {
        println!("GEMV V2:    {:>8}    (Skipped: M != 1)", "N/A");
        None
    };

    // 3. Unified MatMul
    let unified = MatMulStep {
        a: Ref("a".into()),
        b: Ref("b".into()),
        output: Ref("output".into()),
        bias: None,
        b_scales: None,
        c: None,
        m: DynamicValue::Literal(shape.m as u32),
        n: DynamicValue::Literal(shape.n as u32),
        k: DynamicValue::Literal(shape.k as u32),
        transpose_a: false,
        transpose_b: false,
        alpha: 1.0,
        beta: 0.0,
        weights_per_block: 32,
    };
    let mut sym_uni = SymbolTable::new();
    let compiled_uni = unified.compile(&mut bindings, &mut sym_uni);
    let mut fb_uni = FastBindings::new(sym_uni.len());
    for (name, id) in sym_uni.iter() {
        if let Ok(arg) = bindings.get(name) {
            fb_uni.set(*id, arg);
        }
    }

    let lat_uni = measure_step(foundry, &compiled_uni, &fb_uni, &bindings, &sym_uni, iterations);
    println!("Unified:    {:>8.2} us", lat_uni);

    // Verification
    if let Some(v_lat) = lat_gemv {
        // M=1 case: Unified should be close to GEMV
        let diff = (lat_uni - v_lat).abs();
        if diff < 10.0 {
            // Tolerance 10us
            println!("-> VERDICT: MATCHES GEMV (Optimal)");
        } else if lat_uni < lat_gemm {
            println!("-> VERDICT: FASTER THAN GEMM but not quite GEMV?");
        } else {
            println!("-> VERDICT: SLOWER (Matches GEMM?)");
        }
    } else {
        // M>1 case: Unified should be close to GEMM
        let diff = (lat_uni - lat_gemm).abs();
        if diff < 10.0 {
            println!("-> VERDICT: MATCHES GEMM (Optimal)");
        } else {
            println!("-> VERDICT: MISMATCH ({:.2} diff)", diff);
        }
    }
}

#[test]
fn benchmark_compare_approaches() {
    let mut foundry = Foundry::new().unwrap();

    // 1. Single Token Decoding (M=1)
    run_comparison(&mut foundry, Shape { m: 1, n: 4864, k: 896 }, 1000);
    run_comparison(&mut foundry, Shape { m: 1, n: 16384, k: 896 }, 1000); // Head projection (large N)

    // 2. Small Batch (M=32)
    run_comparison(&mut foundry, Shape { m: 32, n: 4864, k: 896 }, 500);

    // 3. Prompt Processing (M=512)
    run_comparison(&mut foundry, Shape { m: 512, n: 4864, k: 896 }, 100);
}
