use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use metallic::{
    Context, F16Element, Tensor, TensorElement, TensorInit, TensorStorage, kernels::{elemwise_add::BroadcastElemwiseAddInplaceOp, matmul_dispatcher::MatmulDispatchOp}
};
use metallic_env::{FORCE_MATMUL_BACKEND_VAR, MATMUL_SMALLN_MAX_N_VAR};

const DISPATCH_BACKENDS: &[&str] = &["mlx", "mps", "gemv", "gemm_tiled", "auto", "noop"];

fn bench_matmul_dispatcher_smalln<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("matmul_dispatcher_smalln_{dtype_name}"));

    // Small-N shapes that should trigger the SmallN path
    let smalln_shapes = [
        (128, 1024, 1),  // M=128, K=1024, N=1
        (512, 2048, 2),  // M=512, K=2048, N=2
        (2048, 4096, 4), // M=2048, K=4096, N=4
        (1024, 2048, 8), // M=1024, K=2048, N=8
        (512, 1024, 16), // M=512, K=1024, N=16
    ];

    let alpha_beta_cases = [
        (1.0f32, 0.0f32, "a1_b0"), // Standard matmul
        (2.5f32, 0.0f32, "aN_b0"), // Alpha scaling
        (1.0f32, 1.0f32, "a1_b1"), // Add to result
        (2.5f32, 1.5f32, "aN_bN"), // Both alpha and beta
    ];

    for &(m, k, n) in &smalln_shapes {
        let flops = (m * n * k) as f64 * 2.0;
        group.throughput(Throughput::Elements(flops as u64));
        let shape_label = format!("{m}x{k}x{n}");

        for &(alpha, beta, case_name) in &alpha_beta_cases {
            let label = format!("{shape_label}_{case_name}");

            // Benchmark each backend for comparison (include custom GEMV and 'noop')
            for &backend in DISPATCH_BACKENDS {
                group.bench_with_input(BenchmarkId::new(backend, &label), &label, |bi, _| {
                    let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard(backend.to_string()).unwrap();

                    let mut ctx = Context::<T>::new().expect("ctx setup");
                    // Reset pool before creating tensors to ensure clean state
                    ctx.reset_pool();

                    let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("A");
                    let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("B");
                    let out: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("C");
                    let bias: Tensor<T> = Tensor::new(vec![n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("bias");

                    // Warmup
                    let mut _warmup_out = ctx
                        .call::<MatmulDispatchOp>((&a, &b, Some(&bias), Some(&out), false, false, alpha, beta), None)
                        .expect("warmup");
                    _warmup_out = ctx
                        .call::<BroadcastElemwiseAddInplaceOp>((_warmup_out, bias.clone()), None)
                        .unwrap();
                    ctx.synchronize();

                    bi.iter(|| {
                        let mut _iter_out = ctx
                            .call::<MatmulDispatchOp>((&a, &b, Some(&bias), Some(&out), false, false, alpha, beta), None)
                            .unwrap();
                        _iter_out = ctx.call::<BroadcastElemwiseAddInplaceOp>((_iter_out, bias.clone()), None).unwrap();
                        ctx.synchronize();
                    });

                    // Reset pool to free memory after each iteration
                    ctx.reset_pool();
                });
            }
        }
    }

    group.finish();
}

fn bench_matmul_dispatcher_dispatcher_tuning<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("matmul_dispatcher_tuning_{dtype_name}"));

    // Benchmark threshold tuning - shapes that should cross the small-N and SIMD boundaries
    let tuning_shapes = [
        // Small-N boundary (N <= 8 vs > 8)
        (512, 2048, 8),  // Should use SmallN
        (512, 2048, 16), // Should use GEMM (not SmallN)
        (512, 2048, 32), // Should use GEMM
        // SIMD boundary (M >= 64, N >= 16)
        (32, 2048, 16),  // M too small for SIMD
        (64, 2048, 8),   // N too small for SIMD
        (64, 2048, 16),  // Should use SIMD (at boundary)
        (128, 2048, 32), // Should use SIMD (well within boundary)
    ];

    for &(m, k, n) in &tuning_shapes {
        let flops = (m * n * k) as f64 * 2.0;
        group.throughput(Throughput::Elements(flops as u64));
        let shape_label = format!("{m}x{k}x{n}");

        // Test different threshold values to tune dispatcher
        let thresholds = [1, 2, 4, 8, 16]; // Different values for METALLIC_MATMUL_SMALLN_MAX_N

        for &thresh in &thresholds {
            let label = format!("{}thresh{}", shape_label, thresh);

            group.bench_with_input(BenchmarkId::new(format!("smalln_max_{}", thresh), &label), &label, |bi, _| {
                let _thresh_guard = MATMUL_SMALLN_MAX_N_VAR.set_guard(thresh.to_string()).unwrap();
                let _backend_guard = FORCE_MATMUL_BACKEND_VAR.set_guard("auto".to_string()).unwrap();

                let mut ctx = Context::<T>::new().expect("ctx setup");
                // Reset pool before creating tensors to ensure clean state
                ctx.reset_pool();

                let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("A");
                let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("B");
                let out: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("C");

                // Warmup
                let _warmup_out = ctx
                    .call::<MatmulDispatchOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0), None)
                    .expect("warmup");
                ctx.synchronize();

                bi.iter(|| {
                    let _iter_out = ctx
                        .call::<MatmulDispatchOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0), None)
                        .unwrap();
                    ctx.synchronize();
                });

                // Reset pool to free memory after each iteration
                ctx.reset_pool();
            });
        }
    }

    group.finish();
}

fn bench_matmul_dispatcher_qwen_shapes<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("matmul_dispatcher_qwen_{dtype_name}"));

    // Qwen2.5 inference shapes that may trigger different paths
    let qwen_shapes = [
        (1, 896, 1152), // decode_bias_add
        (1, 896, 896),  // kv_proj_self
        (1, 896, 9728), // kv_cache_expand - Note: This is quite large
        (1, 4864, 896), // kv_cache_reduce
        (1, 64, 1),     // attn_qk_skinny (batch=14, m=1, k=64, n=1)
        (1, 1, 64),     // attn_kv_skinny (batch=14, m=1, k=1, n=64)
    ];

    // These would be for batch=14 but simplified to batch=1 for benchmarking
    let batch_qwen_shapes = [
        (14, 1, 64, 1), // attn_qk_skinny
        (14, 1, 1, 64), // attn_kv_skinny
    ];

    for &(m, k, n) in &qwen_shapes {
        let flops = (m * n * k) as f64 * 2.0;
        group.throughput(Throughput::Elements(flops as u64));
        let label = format!("{m}x{k}x{n}");

        group.bench_with_input(BenchmarkId::new("MatmulDispatch", &label), &label, |bi, _| {
            let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("auto".to_string()).unwrap();

            let mut ctx = Context::<T>::new().expect("ctx setup");
            // Reset pool before creating tensors to ensure clean state
            ctx.reset_pool();

            let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("A");
            let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("B");
            let out: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("C");

            // Warmup
            let _warmup_out = ctx
                .call::<MatmulDispatchOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0), None)
                .expect("warmup");
            ctx.synchronize();

            bi.iter(|| {
                let _iter_out = ctx
                    .call::<MatmulDispatchOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0), None)
                    .unwrap();
                ctx.synchronize();
            });

            // Reset pool to free memory after each iteration
            ctx.reset_pool();
        });
    }

    // Test batched shapes
    for &(batch, m, k, n) in &batch_qwen_shapes {
        let flops = (batch * m * n * k) as f64 * 2.0;
        group.throughput(Throughput::Elements(flops as u64));
        let label = format!("batch{}_{}x{}x{}", batch, m, k, n);

        group.bench_with_input(BenchmarkId::new("MatmulDispatch", &label), &label, |bi, _| {
            let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("auto".to_string()).unwrap();

            let mut ctx = Context::<T>::new().expect("ctx setup");
            // Reset pool before creating tensors to ensure clean state
            ctx.reset_pool();

            let a: Tensor<T> = Tensor::new(vec![batch, m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("A");
            let b: Tensor<T> = Tensor::new(vec![batch, k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("B");
            let out: Tensor<T> = Tensor::new(vec![batch, m, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("C");

            // Warmup
            let _warmup_out = ctx
                .call::<MatmulDispatchOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0), None)
                .expect("warmup");
            ctx.synchronize();

            bi.iter(|| {
                let _iter_out = ctx
                    .call::<MatmulDispatchOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0), None)
                    .unwrap();
                ctx.synchronize();
            });

            // Reset pool to free memory after each iteration
            ctx.reset_pool();
        });
    }

    group.finish();
}

fn bench_matmul_dispatcher_gemm_shapes<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("matmul_dispatcher_gemm_{dtype_name}"));

    // Match M,K pairs used in direct GEMM benches; sweep non-small-N values
    let mk_pairs = [(128, 1024), (512, 2048), (2048, 4096), (1024, 2048), (512, 1024)];
    let n_values = [32usize, 64usize];

    for &(m, k) in &mk_pairs {
        for &n in &n_values {
            let flops = (m * n * k) as f64 * 2.0;
            group.throughput(Throughput::Elements(flops as u64));
            let shape_label = format!("{}x{}x{}_a1_b0", m, k, n);

            // Benchmark each backend for comparison (dispatcher path, include 'noop')
            for &backend in DISPATCH_BACKENDS {
                group.bench_with_input(BenchmarkId::new(backend, &shape_label), &shape_label, |bi, _| {
                    let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard(backend.to_string()).unwrap();

                    let mut ctx = Context::<T>::new().expect("ctx setup");
                    // Reset pool before creating tensors to ensure clean state
                    ctx.reset_pool();

                    let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("A");
                    let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("B");
                    let out: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("C");

                    // Warmup
                    let _warmup_out = ctx
                        .call::<MatmulDispatchOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0), None)
                        .expect("warmup");
                    ctx.synchronize();

                    bi.iter(|| {
                        let _iter_out = ctx
                            .call::<MatmulDispatchOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0), None)
                            .unwrap();
                        ctx.synchronize();
                    });

                    // Reset pool to free memory after each iteration
                    ctx.reset_pool();
                });
            }
        }
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    // Run dispatcher benchmarks
    bench_matmul_dispatcher_smalln::<F16Element>(c, "f16");

    bench_matmul_dispatcher_dispatcher_tuning::<F16Element>(c, "f16");

    bench_matmul_dispatcher_qwen_shapes::<F16Element>(c, "f16");
    bench_matmul_dispatcher_gemm_shapes::<F16Element>(c, "f16");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
