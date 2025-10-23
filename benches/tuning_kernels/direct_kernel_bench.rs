use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use metallic::{
    Context, F16Element, Tensor, TensorElement, TensorInit, TensorStorage, kernels::{
        matmul_dispatcher::MatmulDispatchOp, matmul_gemm_tiled::MatmulGemmTiledOp, matmul_gemv_smalln::{MatmulGemvSmallN1Op, MatmulGemvSmallN2Op, MatmulGemvSmallN4Op, MatmulGemvSmallN8Op, MatmulGemvSmallN16Op}, matmul_mlx::MatMulMlxOp, matmul_mps::MatMulMpsOp, softmax_block::SoftmaxBlockOp, softmax_vec::SoftmaxVecOp
    }
};

/// Direct kernel benchmarks that test raw kernel performance without dispatcher overhead.
/// This helps identify optimal crossover points between different kernel variants.
fn bench_smalln_gemv_kernels_directly<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("smalln_gemv_direct_{dtype_name}"));

    // Match dispatcher small-N shapes for consistent sweeps across charts
    let smalln_shapes = [
        (128, 1024, 1),  // M=128, K=1024, N=1
        (512, 2048, 2),  // M=512, K=2048, N=2
        (2048, 4096, 4), // M=2048, K=4096, N=4  - Reduced to avoid memory overload
        (1024, 2048, 8), // M=1024, K=2048, N=8
        (512, 1024, 16), // M=512, K=1024, N=16
    ];

    for &(m, k, n) in &smalln_shapes {
        let flops = (m * k * n) as f64 * 2.0; // Estimate of multiply-add operations
        group.throughput(Throughput::Elements(flops as u64));
        // Align label format with analysis parser and dispatcher labels
        let label = format!("{}x{}x{}_a1_b0", m, k, n);

        let bench_name = match n {
            1 => "SmallN_Direct_N1",
            2 => "SmallN_Direct_N2",
            4 => "SmallN_Direct_N4",
            8 => "SmallN_Direct_N8",
            16 => "SmallN_Direct_N16",
            _ => "SmallN_Direct",
        };
        group.bench_with_input(BenchmarkId::new(bench_name, &label), &label, |bi, _| {
            let mut ctx = Context::<T>::new().expect("ctx setup");
            // Reset pool before creating tensors to ensure clean state
            ctx.reset_pool();

            // Create input tensors with pool allocation to limit memory usage
            let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("tensor A");
            let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("tensor B");

            // Warmup - use correct interface for Small-N GEMV operations
            let _warmup_out = match n {
                1 => ctx.call::<MatmulGemvSmallN1Op>((&a, &b)).expect("warmup"),
                2 => ctx.call::<MatmulGemvSmallN2Op>((&a, &b)).expect("warmup"),
                4 => ctx.call::<MatmulGemvSmallN4Op>((&a, &b)).expect("warmup"),
                8 => ctx.call::<MatmulGemvSmallN8Op>((&a, &b)).expect("warmup"),
                16 => ctx.call::<MatmulGemvSmallN16Op>((&a, &b)).expect("warmup"),
                _ => unreachable!(),
            };
            ctx.synchronize();

            bi.iter(|| {
                let _iter_out = match n {
                    1 => ctx.call::<MatmulGemvSmallN1Op>((&a, &b)).unwrap(),
                    2 => ctx.call::<MatmulGemvSmallN2Op>((&a, &b)).unwrap(),
                    4 => ctx.call::<MatmulGemvSmallN4Op>((&a, &b)).unwrap(),
                    8 => ctx.call::<MatmulGemvSmallN8Op>((&a, &b)).unwrap(),
                    16 => ctx.call::<MatmulGemvSmallN16Op>((&a, &b)).unwrap(),
                    _ => unreachable!(),
                };
                ctx.synchronize();
            });

            // Reset pool to free memory after each iteration
            ctx.reset_pool();
        });
    }

    group.finish();
}

/// Direct GEMM kernels (non-small-N) swept at matching M,K shapes and N > 16
fn bench_gemm_kernels_directly<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("gemm_kernels_direct_{dtype_name}"));

    // Reuse M,K from small-N shapes, use N values that trigger GEMM paths
    let mk_pairs = [(128, 1024), (512, 2048), (2048, 4096), (1024, 2048), (512, 1024)];
    let n_values = [32usize, 64usize];

    for &(m, k) in &mk_pairs {
        for &n in &n_values {
            let flops = (m * k * n) as f64 * 2.0;
            group.throughput(Throughput::Elements(flops as u64));
            let label = format!("{}x{}x{}_a1_b0", m, k, n);

            // MLX GEMM direct
            group.bench_with_input(BenchmarkId::new("Gemm_Direct_MLX", &label), &label, |bi, _| {
                let mut ctx = Context::<T>::new().expect("ctx setup");
                // Reset pool before creating tensors to ensure clean state
                ctx.reset_pool();

                let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("A");
                let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("B");

                // Warmup
                let _warmup_out = ctx
                    .call::<MatMulMlxOp>((&a, &b, None, None, false, false, 1.0f32, 0.0f32))
                    .expect("warmup");
                ctx.synchronize();

                bi.iter(|| {
                    let _iter_out = ctx.call::<MatMulMlxOp>((&a, &b, None, None, false, false, 1.0f32, 0.0f32)).unwrap();
                    ctx.synchronize();
                });

                // Reset pool to free memory after each iteration
                ctx.reset_pool();
            });

            // MPS GEMM direct
            group.bench_with_input(BenchmarkId::new("Gemm_Direct_MPS", &label), &label, |bi, _| {
                let mut ctx = Context::<T>::new().expect("ctx setup");
                // Reset pool before creating tensors to ensure clean state
                ctx.reset_pool();

                let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("A");
                let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("B");

                // Warmup
                let _warmup_out = ctx.call::<MatMulMpsOp>((&a, &b, false, false)).expect("warmup");
                ctx.synchronize();

                bi.iter(|| {
                    let _iter_out = ctx.call::<MatMulMpsOp>((&a, &b, false, false)).unwrap();
                    ctx.synchronize();
                });

                // Reset pool to free memory after each iteration
                ctx.reset_pool();
            });

            // Gemm Tiled direct
            group.bench_with_input(BenchmarkId::new("Gemm_Direct_Tiled", &label), &label, |bi, _| {
                let mut ctx = Context::<T>::new().expect("ctx setup");
                ctx.reset_pool();

                let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("A");
                let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("B");
                let out: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("C");

                let _warmup_out = ctx
                    .call::<MatmulGemmTiledOp>((&a, &b, None, Some(&out), false, false, 1.0f32, 0.0f32))
                    .expect("warmup");
                ctx.synchronize();

                bi.iter(|| {
                    let _iter_out = ctx
                        .call::<MatmulGemmTiledOp>((&a, &b, None, Some(&out), false, false, 1.0f32, 0.0f32))
                        .unwrap();
                    ctx.synchronize();
                });

                ctx.reset_pool();
            });
        }
    }

    group.finish();
}

fn bench_softmax_kernels_directly<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("softmax_kernels_direct_{dtype_name}"));

    // Different sequence lengths to find optimal crossover points
    let seq_lengths = [
        64,   // Should favor vec-softmax
        128,  // Should favor vec-softmax
        256,  // Should favor vec-softmax
        512,  // Should favor vec-softmax
        1024, // Edge case between vec and block
        2048, // Should favor block-softmax
        4096, // Should favor block-softmax
        8192, // Should favor block-softmax
    ];

    let seq_q = 128; // Fixed number of rows (queries)

    for &seq_k in &seq_lengths {
        let flops = (seq_q * seq_k) as f64 * 3.0; // Estimate of ops (read+max+exp+sum+div)
        group.throughput(Throughput::Elements(flops as u64));
        let label = format!("seqq{}_seqk{}", seq_q, seq_k);

        // Test vec-softmax
        group.bench_with_input(BenchmarkId::new("VecSoftmax_Direct", &label), &label, |bi, _| {
            let mut ctx = Context::<T>::new().expect("ctx setup");
            // Reset pool before creating tensors to ensure clean state
            ctx.reset_pool();

            let input: Tensor<T> =
                Tensor::new(vec![seq_q, seq_k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("input");

            let rows_total = seq_q as u32;
            let seq_q = seq_q as u32;
            let seq_k = seq_k as u32;

            // Warmup
            let _warmup_out = ctx.call::<SoftmaxVecOp>((&input, rows_total, seq_q, seq_k, 0, 0)).expect("warmup");
            ctx.synchronize();

            bi.iter(|| {
                let _iter_out = ctx.call::<SoftmaxVecOp>((&input, rows_total, seq_q, seq_k, 0, 0)).unwrap();
                ctx.synchronize();
            });

            // Reset pool to free memory after each iteration
            ctx.reset_pool();
        });

        // Test block-softmax (only for longer sequences where it makes sense)
        if seq_k >= 1024 {
            group.bench_with_input(BenchmarkId::new("BlockSoftmax_Direct", &label), &label, |bi, _| {
                let mut ctx = Context::<T>::new().expect("ctx setup");
                // Reset pool before creating tensors to ensure clean state
                ctx.reset_pool();

                let input: Tensor<T> =
                    Tensor::new(vec![seq_q, seq_k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("input");

                let rows_total = seq_q as u32;
                let seq_q = seq_q as u32;
                let seq_k = seq_k as u32;

                // Warmup
                let _warmup_out = ctx
                    .call::<SoftmaxBlockOp>((&input, rows_total, seq_q, seq_k, 0, 0, 0))
                    .expect("warmup");
                ctx.synchronize();

                bi.iter(|| {
                    let _iter_out = ctx.call::<SoftmaxBlockOp>((&input, rows_total, seq_q, seq_k, 0, 0, 0)).unwrap();
                    ctx.synchronize();
                });

                // Reset pool to free memory after each iteration
                ctx.reset_pool();
            });
        }
    }

    group.finish();
}

fn bench_smalln_vs_dispatcher_comparison<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("smalln_vs_dispatcher_{dtype_name}"));

    // Compare direct kernel calls vs dispatcher overhead for key shapes
    let test_cases = [
        (128, 1024, 8),  // Shape where N=8 should be selected
        (128, 1024, 16), // Shape where N=16 should be selected
    ];

    for &(m, k, n) in &test_cases {
        let flops = (m * k * n) as f64 * 2.0;
        group.throughput(Throughput::Elements(flops as u64));
        // Include case suffix to match analysis parsing and charts
        let label = format!("{}x{}x{}_a1_b0", m, k, n);

        // Direct kernel call
        group.bench_with_input(BenchmarkId::new("Direct_Kernel", &label), &label, |bi, _| {
            let mut ctx = Context::<T>::new().expect("ctx setup");
            // Reset pool before creating tensors to ensure clean state
            ctx.reset_pool();

            let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("tensor A");
            let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("tensor B");

            // Warmup - use correct interface for Small-N GEMV operations
            let _warmup_out = match n {
                8 => ctx.call::<MatmulGemvSmallN8Op>((&a, &b)).expect("warmup"),
                16 => ctx.call::<MatmulGemvSmallN16Op>((&a, &b)).expect("warmup"),
                _ => unreachable!(),
            };
            ctx.synchronize();

            bi.iter(|| {
                let _iter_out = match n {
                    8 => ctx.call::<MatmulGemvSmallN8Op>((&a, &b)).unwrap(),
                    16 => ctx.call::<MatmulGemvSmallN16Op>((&a, &b)).unwrap(),
                    _ => unreachable!(),
                };
                ctx.synchronize();
            });

            // Reset pool to free memory after each iteration
            ctx.reset_pool();
        });

        // Dispatcher call (for comparison) - use MatmulDispatchOp
        group.bench_with_input(BenchmarkId::new("Via_Dispatcher", &label), &label, |bi, _| {
            let mut ctx = Context::<T>::new().expect("ctx setup");
            // Reset pool before creating tensors to ensure clean state
            ctx.reset_pool();

            let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("tensor A");
            let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("tensor B");
            let c: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("tensor C");

            // Warmup - use dispatcher with correct signature
            let _warmup_out = ctx
                .call::<MatmulDispatchOp>((&a, &b, None, Some(&c), false, false, 1.0f32, 0.0f32))
                .expect("warmup");
            ctx.synchronize();

            bi.iter(|| {
                let _iter_out = ctx
                    .call::<MatmulDispatchOp>((&a, &b, None, Some(&c), false, false, 1.0f32, 0.0f32))
                    .unwrap();
                ctx.synchronize();
            });

            // Reset pool to free memory after each iteration
            ctx.reset_pool();
        });
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    // Direct kernel performance benchmarks
    bench_smalln_gemv_kernels_directly::<F16Element>(c, "f16");
    bench_softmax_kernels_directly::<F16Element>(c, "f16");
    bench_gemm_kernels_directly::<F16Element>(c, "f16");

    // Comparison benchmarks
    bench_smalln_vs_dispatcher_comparison::<F16Element>(c, "f16");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
