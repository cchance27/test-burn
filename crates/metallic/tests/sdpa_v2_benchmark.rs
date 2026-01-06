use std::time::Instant;

use metallic::{
    foundry::{Foundry, storage::Pooled, tensor::Tensor}, metals::{
        rope::{Rope, RopeParamsResolved}, sdpa::sdpa::scaled_dot_product_attention, v2::attention::{stages::SdpaParamsResolved, step::FusedMhaStep}
    }, tensor::{TensorInit, dtypes::F16}, types::TensorArg
};

fn run_benchmark_case(foundry: &mut Foundry, batch: usize, heads: usize, kv_len: usize, head_dim: usize, iterations: usize) {
    println!(
        "Benchmarking: Batch={}, Heads={}, KV_Len={}, HeadDim={}",
        batch, heads, kv_len, head_dim
    );

    let q_len = 1;
    let total_batch = batch * heads;
    let q_dims_3d = vec![total_batch, q_len, head_dim];
    let k_dims_3d = vec![total_batch, kv_len, head_dim];
    let v_dims_3d = vec![total_batch, kv_len, head_dim];
    let rope_cache_dims = vec![kv_len, head_dim / 2];

    // Dummy Allocation (Content irrelevant for perf)
    let q = Tensor::<F16, Pooled>::new(foundry, q_dims_3d.clone(), TensorInit::Uninitialized).unwrap();
    let k = Tensor::<F16, Pooled>::new(foundry, k_dims_3d.clone(), TensorInit::Uninitialized).unwrap();
    let v = Tensor::<F16, Pooled>::new(foundry, v_dims_3d.clone(), TensorInit::Uninitialized).unwrap();
    let cos = Tensor::<F16, Pooled>::new(foundry, rope_cache_dims.clone(), TensorInit::Uninitialized).unwrap();
    let sin = Tensor::<F16, Pooled>::new(foundry, rope_cache_dims.clone(), TensorInit::Uninitialized).unwrap();
    let output_v2 = Tensor::<F16, Pooled>::new(foundry, q_dims_3d.clone(), TensorInit::Uninitialized).unwrap();

    // Aux tensors
    let q_roped = Tensor::<F16, Pooled>::new(foundry, q_dims_3d.clone(), TensorInit::Uninitialized).unwrap();
    let k_roped = Tensor::<F16, Pooled>::new(foundry, k_dims_3d.clone(), TensorInit::Uninitialized).unwrap();

    // ----------------------------------------------------------------
    // Warmup & Setup V2
    // ----------------------------------------------------------------
    let rope_params = RopeParamsResolved {
        dim: head_dim as u32,
        seq_len: kv_len as u32,
        position_offset: (kv_len - 1) as u32,
        total_elements: (total_batch * head_dim) as u32,
    };
    let sdpa_params = SdpaParamsResolved {
        kv_len: kv_len as u32,
        head_dim: head_dim as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
        stride_k_s: k.strides()[1] as u32,
        stride_v_s: v.strides()[1] as u32,
    };
    let q_strides = (q.strides()[0] as u32, q.strides()[1] as u32);
    let k_strides = (k.strides()[0] as u32, k.strides()[1] as u32);
    let v_strides = (v.strides()[0] as u32, v.strides()[1] as u32);
    let out_strides = (output_v2.strides()[0] as u32, output_v2.strides()[1] as u32);

    // Pre-Rope K for V2 (simulating cache state)
    let rope_k_params = RopeParamsResolved {
        dim: head_dim as u32,
        seq_len: kv_len as u32,
        position_offset: 0,
        total_elements: 0,
    };
    let rope_k_kernel = Rope::new(
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&k_roped),
        &TensorArg::from_tensor(&cos),
        &TensorArg::from_tensor(&sin),
        rope_k_params,
    );
    // Run once setup
    foundry.run(&rope_k_kernel).unwrap();

    let v2_step = FusedMhaStep::compile(
        foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k_roped),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&cos),
        &TensorArg::from_tensor(&sin),
        &TensorArg::from_tensor(&output_v2),
        rope_params,
        sdpa_params,
        total_batch as u32,
        1,
        head_dim as u32,
        q_strides,
        k_strides,
        v_strides,
        out_strides,
    )
    .unwrap();

    // ----------------------------------------------------------------
    // Setup Legacy (Dispatch 3 kernels)
    // ----------------------------------------------------------------
    let rope_q_params = RopeParamsResolved {
        dim: head_dim as u32,
        seq_len: 1,
        position_offset: (kv_len - 1) as u32,
        total_elements: 0,
    };
    let rope_q_kernel = Rope::new(
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&q_roped),
        &TensorArg::from_tensor(&cos),
        &TensorArg::from_tensor(&sin),
        rope_q_params,
    );

    // Legacy Loop Closure
    let run_legacy = |f: &mut Foundry| {
        // 1. Rope Q
        f.run(&rope_q_kernel).unwrap();
        // 2. Rope K (Assume K is dynamic in legacy? Usually cached.
        // Logic: Legacy usually Ropes Q, reads Cache K (roped?).
        // Parity test ran Rope K every time. Let's maximize fairness.
        // If Legacy assumes K is cached, we skip Rope K here.
        // But SDPA Legacy takes `k_roped`. So we assume k_roped is ready.

        // 3. SDPA
        scaled_dot_product_attention(
            f,
            &q_roped,
            &k_roped,
            &v,
            true,                // causal
            (kv_len - 1) as u32, // query_offset
        )
        .unwrap();
    };

    // V2 Loop Closure
    use metallic::foundry::spec::{CompiledStep, FastBindings, TensorBindings};
    let run_v2 = |f: &mut Foundry| {
        v2_step.execute(f, &FastBindings::default(), &TensorBindings::default()).unwrap();
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
fn benchmark_sdpa_v2_perf() {
    let mut foundry = Foundry::new().unwrap();
    let iterations = 100;

    // Cases: (Batch, Heads, KV_Len, HeadDim)
    let cases = vec![
        (1, 1, 128, 64),
        (8, 8, 128, 64),    // Standard Decoding
        (1, 32, 1024, 64),  // Long Context (Llama 7B style)
        (1, 32, 1024, 128), // Larger Head Dim
        (32, 1, 128, 64),   // Large Batch
    ];

    for (b, h, kv, d) in cases {
        run_benchmark_case(&mut foundry, b, h, kv, d, iterations);
    }
}
