use std::time::Instant;

use metallic_foundry::{
    Foundry, MetalError, metals::{flashattention::step::FlashDecodeKernel, sdpa::step::SdpaMaterializedStep}, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::Pooled, tensor::{Tensor, TensorInit, dtypes::F16}, types::TensorArg
};
use objc2_metal::MTLCommandBuffer as _;

fn bind_all_symbols(symbols: &SymbolTable, bindings: &TensorBindings) -> FastBindings {
    let mut fast = FastBindings::new(symbols.len());
    for (name, id) in symbols.iter() {
        if let Ok(arg) = bindings.get(name) {
            fast.set(*id, arg);
        }
    }
    fast
}

fn execute_steps(
    foundry: &mut Foundry,
    steps: &[Box<dyn CompiledStep>],
    fast: &FastBindings,
    bindings: &TensorBindings,
    symbols: &SymbolTable,
) {
    for step in steps {
        step.execute(foundry, fast, bindings, symbols).unwrap();
    }
}

fn bench(foundry: &mut Foundry, label: &str, iterations: usize, f: impl Fn(&mut Foundry)) -> f64 {
    // Warmup
    for _ in 0..10 {
        f(foundry);
    }
    foundry.synchronize().unwrap();

    let start = Instant::now();
    foundry.start_capture().unwrap();
    for _ in 0..iterations {
        f(foundry);
    }
    let buf = foundry.end_capture().unwrap();
    buf.waitUntilCompleted();
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    println!("  -> {label}: {avg_us:.2} us");
    avg_us
}

#[test]
#[ignore]
fn benchmark_sdpa_m1_fused_vs_materialized() {
    let mut foundry = match Foundry::new() {
        Ok(foundry) => foundry,
        Err(MetalError::DeviceNotFound) => {
            eprintln!("Skipping: Metal device not available (DeviceNotFound)");
            return;
        }
        Err(e) => panic!("Failed to create Foundry: {e:?}"),
    };

    // Decode-like case: M=1, batch=1, heads>1.
    let heads = 14usize;
    let head_dim = 64usize;
    let kv_len = 1024usize;
    let m = 1usize;

    assert_eq!(head_dim % 4, 0, "sdpa decode kernel expects head_dim % 4 == 0");

    let d_model = heads * head_dim;
    let iterations = 1000usize;

    println!("Benchmarking SDPA M=1: heads={heads}, head_dim={head_dim}, kv_len={kv_len}, iters={iterations}");

    // Allocate head-major Q/K/V: [H, M, D] and [H, S, D].
    // Output is token-major [M, H*D] for both variants so we can compare apples-to-apples.
    let q = Tensor::<F16, Pooled>::new(&mut foundry, vec![heads, m, head_dim], TensorInit::Uninitialized).unwrap();
    let k = Tensor::<F16, Pooled>::new(&mut foundry, vec![heads, kv_len, head_dim], TensorInit::Uninitialized).unwrap();
    let v = Tensor::<F16, Pooled>::new(&mut foundry, vec![heads, kv_len, head_dim], TensorInit::Uninitialized).unwrap();

    let out_fused = Tensor::<F16, Pooled>::new(&mut foundry, vec![m, d_model], TensorInit::Uninitialized).unwrap();
    let out_mat = Tensor::<F16, Pooled>::new(&mut foundry, vec![m, d_model], TensorInit::Uninitialized).unwrap();

    let mut bindings = TensorBindings::new();
    bindings.insert("q".to_string(), TensorArg::from_tensor(&q));
    bindings.insert("k".to_string(), TensorArg::from_tensor(&k));
    bindings.insert("v".to_string(), TensorArg::from_tensor(&v));
    bindings.insert("out_fused".to_string(), TensorArg::from_tensor(&out_fused));
    bindings.insert("out_mat".to_string(), TensorArg::from_tensor(&out_mat));

    // Fused decode-style SDPA (online softmax + PV accumulation).
    // NOTE: This is the standalone SDPA op (no RoPE), intended to be comparable to SdpaMaterialized
    // when Q is already rotated (e.g. q_rot in the DSL).
    let fused_step = FlashDecodeKernel {
        q: Ref("q".into()),
        k: Ref("k".into()),
        v: Ref("v".into()),
        output: Ref("out_fused".into()),
        causal: true,
        n_heads: DynamicValue::Literal(heads as u32),
        head_dim: DynamicValue::Literal(head_dim as u32),
        kv_seq_len: DynamicValue::Literal(kv_len as u32),
        query_offset: DynamicValue::Literal(0),
        kv_head_major: true,
    };

    // Materialized SDPA (GEMM QK^T -> softmax -> GEMM PV) forced to M=1.
    let mat_step = SdpaMaterializedStep {
        q: Ref("q".into()),
        k: Ref("k".into()),
        v: Ref("v".into()),
        output: Ref("out_mat".into()),
        causal: true,
        query_offset: DynamicValue::Literal(0),
        n_heads: DynamicValue::Literal(heads as u32),
        head_dim: DynamicValue::Literal(head_dim as u32),
        kv_seq_len: DynamicValue::Literal(kv_len as u32),
        m: DynamicValue::Literal(1),
        kv_head_major: true,
    };

    let mut symbols = SymbolTable::new();
    let fused_compiled = fused_step.compile(&mut bindings, &mut symbols);
    let mat_compiled = mat_step.compile(&mut bindings, &mut symbols);
    let fast = bind_all_symbols(&symbols, &bindings);

    let run_fused = |f: &mut Foundry| execute_steps(f, &fused_compiled, &fast, &bindings, &symbols);
    let run_mat = |f: &mut Foundry| execute_steps(f, &mat_compiled, &fast, &bindings, &symbols);

    let fused_us = bench(&mut foundry, "Fused (online)", iterations, run_fused);
    let mat_us = bench(&mut foundry, "Materialized (QK/softmax/PV)", iterations, run_mat);

    println!("Speedup: {:.2}x (materialized / fused)", mat_us / fused_us);
}
