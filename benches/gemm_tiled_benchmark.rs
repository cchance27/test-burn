use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use metallic::{
    Context, F16Element, Tensor, TensorElement, TensorInit, TensorStorage, kernels::{matmul_dispatcher::MatmulDispatchOp, matmul_gemm_tiled::MatmulGemmTiledOp}
};
use metallic_env::FORCE_MATMUL_BACKEND_VAR;

fn bench_gemm_tiled_configurations<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("matmul_gemm_tiled_{}", dtype_name));

    // Test different matrix sizes to evaluate the effectiveness of the tiled approach
    let shapes = [
        // Medium sizes where tiling should start to show benefits
        (256, 256, 256),
        (512, 512, 512),
        // Large sizes where tiling should be clearly beneficial
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        // Rectangular matrices that are common in transformer models
        (1, 1024, 4096),    // skinny M
        (4096, 1024, 1),    // skinny N
        (512, 4096, 11008), // transformer-like
        (11008, 4096, 512), // transformer-like transpose
    ];

    for &(m, k, n) in &shapes {
        let flops = (m * n * k) as f64 * 2.0;
        group.throughput(Throughput::Elements(flops as u64));
        let label = format!("{}_{}x{}x{}", dtype_name, m, k, n);

        // Benchmark the GEMM tiled kernel directly to avoid dispatcher side-effects
        group.bench_with_input(BenchmarkId::new("GemmTiledKernel", &label), &label, |bi, _| {
            let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("gemm_tiled".to_string()).unwrap();

            let mut ctx = Context::<T>::new().expect("context creation");
            ctx.reset_pool();

            let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("A");
            let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("B");
            let out: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("C");

            ctx.call::<MatmulGemmTiledOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0))
                .expect("warmup");
            ctx.synchronize();

            bi.iter(|| {
                ctx.call::<MatmulGemmTiledOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0))
                    .unwrap();
                ctx.synchronize();
            });

            ctx.reset_pool();
        });

        // Compare with MLX baseline to make sure we don't regress performance
        group.bench_with_input(BenchmarkId::new("MLX", &label), &label, |bi, _| {
            let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mlx".to_string()).unwrap();

            let mut ctx = Context::<T>::new().expect("context creation");
            ctx.reset_pool();

            let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("A");
            let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("B");
            let out: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("C");

            // Warmup
            let _warmup_out = ctx
                .call::<MatmulDispatchOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0))
                .expect("warmup");
            ctx.synchronize();

            bi.iter(|| {
                let _result = ctx
                    .call::<MatmulDispatchOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0))
                    .unwrap();
                ctx.synchronize();
            });

            ctx.reset_pool();
        });
    }

    group.finish();
}

fn bench_gemm_tiled_tuning<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("matmul_gemm_tiled_tuning_{}", dtype_name));

    // Specific shapes to test the threshold selection logic
    let threshold_test_shapes = [
        // Just below the threshold (should not use GemmTiled)
        (256, 256, 256),
        (511, 511, 511),
        // Just above the threshold (should use GemmTiled)
        (512, 512, 512),
        (1024, 1024, 1024),
    ];

    for &(m, k, n) in &threshold_test_shapes {
        let flops = (m * n * k) as f64 * 2.0;
        group.throughput(Throughput::Elements(flops as u64));
        let label = format!("{}_{}x{}x{}", dtype_name, m, k, n);

        group.bench_with_input(BenchmarkId::new("GemmTiledKernel", &label), &label, |bi, _| {
            let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("gemm_tiled".to_string()).unwrap();

            let mut ctx = Context::<T>::new().expect("context creation");
            ctx.reset_pool();

            let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("A");
            let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("B");
            let out: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("C");

            ctx.call::<MatmulGemmTiledOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0))
                .expect("warmup");
            ctx.synchronize();

            bi.iter(|| {
                ctx.call::<MatmulGemmTiledOp>((&a, &b, None, Some(&out), false, false, 1.0, 0.0))
                    .unwrap();
                ctx.synchronize();
            });

            ctx.reset_pool();
        });
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_gemm_tiled_configurations::<F16Element>(c, "f16");
    bench_gemm_tiled_tuning::<F16Element>(c, "f16");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
