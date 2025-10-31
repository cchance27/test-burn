use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use metallic::{Context, F16Element, Tensor, TensorElement, TensorInit, TensorStorage, kernels::softmax_dispatcher::SoftmaxDispatchOp};
use metallic_env::SOFTMAX_BACKEND_VAR;

fn bench_softmax_dispatcher_seq_lengths<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("softmax_dispatcher_seq_{dtype_name}"));

    // Different key sequence lengths (seq_k) to test vec vs block
    let seq_k_lengths = [
        64,   // Should use vectorized approach
        128,  // Should use vectorized approach
        256,  // Should use vectorized approach
        512,  // Should use vectorized approach
        1024, // Edge case between vec and block
        2048, // Should use block approach
        4096, // Should use block approach
        8192, // Should use block approach
    ];

    // Rows_total (e.g., seq_q) to test
    let rows_totals = [
        64,  // number of rows / queries
        128, // number of rows / queries
    ];

    for &seq_k in &seq_k_lengths {
        for &rows_total in &rows_totals {
            let flops = (rows_total * seq_k) as f64 * 3.0; // softmax ops per element
            group.throughput(Throughput::Elements(flops as u64));
            let label = format!("rows{}_seqk{}", rows_total, seq_k);

            // Benchmark softmax with different backends (include 'noop' for dispatcher overhead)
            for backend in ["auto", "kernel", "mps", "noop"] {
                group.bench_with_input(BenchmarkId::new(format!("Softmax_{}", backend), &label), &label, |bi, _| {
                    let _guard = SOFTMAX_BACKEND_VAR.set_guard(backend.to_string()).unwrap();

                    let mut ctx = Context::<T>::new().expect("ctx setup");
                    // Reset pool before creating tensors to ensure clean state
                    ctx.reset_pool();

                    // Shape: [rows_total, seq_k] (vary seq_k on last axis)
                    let input: Tensor<T> =
                        Tensor::new(vec![rows_total, seq_k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("input");

                    // Warmup
                    let _warmup_out = ctx.call::<SoftmaxDispatchOp>((&input, false, 0)).expect("warmup");
                    ctx.synchronize();

                    bi.iter(|| {
                        let _iter_out = ctx.call::<SoftmaxDispatchOp>((&input, false, 0)).unwrap();
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

fn bench_softmax_dispatcher_batch_variants<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("softmax_dispatcher_batch_{dtype_name}"));

    // Batch sizes with different seq_k to test threading behavior
    let test_cases = [
        (1, 1024, 64),  // Single batch, seq_q=64, seq_k=1024
        (4, 1024, 64),  // Small batch, seq_q=64, seq_k=1024
        (1, 2048, 128), // Single batch, seq_q=128, seq_k=2048
        (2, 2048, 128), // Small batch, seq_q=128, seq_k=2048
    ];

    for &(batch_size, seq_k, seq_q) in &test_cases {
        let flops = (batch_size * seq_q * seq_k) as f64 * 3.0;
        group.throughput(Throughput::Elements(flops as u64));
        let label = format!("batch{}_seqq{}_seqk{}", batch_size, seq_q, seq_k);

        // Test with auto backend selection
        group.bench_with_input(BenchmarkId::new("Softmax_Auto", &label), &label, |bi, _| {
            let _guard = SOFTMAX_BACKEND_VAR.set_guard("auto".to_string()).unwrap();

            let mut ctx = Context::<T>::new().expect("ctx setup");
            // Reset pool before creating tensors to ensure clean state
            ctx.reset_pool();

            let input: Tensor<T> = Tensor::new(
                vec![batch_size, seq_q, seq_k],
                TensorStorage::Pooled(&mut ctx),
                TensorInit::Uninitialized,
            )
            .expect("input");

            // Warmup
            let _warmup_out = ctx.call::<SoftmaxDispatchOp>((&input, false, 0)).expect("warmup");
            ctx.synchronize();

            bi.iter(|| {
                let _iter_out = ctx.call::<SoftmaxDispatchOp>((&input, false, 0)).unwrap();
                ctx.synchronize();
            });

            // Reset pool to free memory after each iteration
            ctx.reset_pool();
        });
    }

    group.finish();
}

fn bench_softmax_dispatcher_causal_vs_normal<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("softmax_dispatcher_causal_{dtype_name}"));

    let test_cases = [
        (512, 64),   // seq_k, rows_total
        (1024, 128), // seq_k, rows_total
        (2048, 64),  // seq_k, rows_total
    ];

    for &(seq_k, rows_total) in &test_cases {
        let flops = (rows_total * seq_k) as f64 * 3.0;
        group.throughput(Throughput::Elements(flops as u64));
        let label = format!("rows{}_seqk{}", rows_total, seq_k);

        // Test normal softmax
        group.bench_with_input(BenchmarkId::new("Softmax_Normal", &label), &label, |bi, _| {
            let mut ctx = Context::<T>::new().expect("ctx setup");
            // Reset pool before creating tensors to ensure clean state
            ctx.reset_pool();

            let input: Tensor<T> =
                Tensor::new(vec![rows_total, seq_k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("input");

            // Warmup
            let _warmup_out = ctx.call::<SoftmaxDispatchOp>((&input, false, 0)).expect("warmup");
            ctx.synchronize();

            bi.iter(|| {
                let _iter_out = ctx.call::<SoftmaxDispatchOp>((&input, false, 0)).unwrap();
                ctx.synchronize();
            });

            // Reset pool to free memory after each iteration
            ctx.reset_pool();
        });

        // Test causal softmax
        group.bench_with_input(BenchmarkId::new("Softmax_Causal", &label), &label, |bi, _| {
            let mut ctx = Context::<T>::new().expect("ctx setup");
            // Reset pool before creating tensors to ensure clean state
            ctx.reset_pool();

            let input: Tensor<T> =
                Tensor::new(vec![rows_total, seq_k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("input");

            // Warmup
            let _warmup_out = ctx.call::<SoftmaxDispatchOp>((&input, true, 0)).expect("warmup");
            ctx.synchronize();

            bi.iter(|| {
                let _iter_out = ctx.call::<SoftmaxDispatchOp>((&input, true, 0)).unwrap();
                ctx.synchronize();
            });

            // Reset pool to free memory after each iteration
            ctx.reset_pool();
        });
    }

    group.finish();
}

fn bench_softmax_dispatcher_variant_selection<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("softmax_dispatcher_variant_selection_{dtype_name}"));

    // Test dispatcher selection at different seq_k lengths
    let seq_k_lengths = [
        64,   // Should use vec-softmax
        128,  // Should use vec-softmax
        256,  // Should use vec-softmax
        512,  // Should use vec-softmax
        1024, // Edge case - seq_k <= 1024
        2048, // Should use block-softmax
        4096, // Should use block-softmax
        8192, // Should use block-softmax - Note: This is quite large for memory-constrained systems
    ];

    let rows_total = 128;

    for &seq_k in &seq_k_lengths {
        let flops = (rows_total * seq_k) as f64 * 3.0;
        group.throughput(Throughput::Elements(flops as u64));
        let label = format!("rows{}_seqk{}", rows_total, seq_k);

        // Test with forced variants to verify dispatcher selection
        for variant in ["auto", "vec", "block"] {
            group.bench_with_input(BenchmarkId::new(format!("SoftmaxDispatch_{}", variant), &label), &label, |bi, _| {
                let _guard = SOFTMAX_BACKEND_VAR.set_guard(variant.to_string()).unwrap();

                let mut ctx = Context::<T>::new().expect("ctx setup");
                // Reset pool before creating tensors to ensure clean state
                ctx.reset_pool();

                let input: Tensor<T> =
                    Tensor::new(vec![rows_total, seq_k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("input");

                // Warmup
                let _warmup_out = ctx.call::<SoftmaxDispatchOp>((&input, false, 0)).expect("warmup");
                ctx.synchronize();

                bi.iter(|| {
                    let _iter_out = ctx.call::<SoftmaxDispatchOp>((&input, false, 0)).unwrap();
                    ctx.synchronize();
                });

                // Reset pool to free memory after each iteration
                ctx.reset_pool();
            });
        }
    }

    group.finish();
}

fn bench_softmax_dispatcher_crossover_analysis<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("softmax_dispatcher_crossover_{dtype_name}"));

    // Focus on the crossover region around 1024 to find optimal seq_k threshold
    let crossover_seqk = [
        512,  // Well within vec territory
        768,  // Approaching crossover
        896,  // Near crossover
        1024, // At current threshold
        1152, // Just past crossover
        1280, // Well within block territory
        1536, // Further into block territory
    ];

    let rows_total = 128;

    for &seq_k in &crossover_seqk {
        let flops = (rows_total * seq_k) as f64 * 3.0;
        group.throughput(Throughput::Elements(flops as u64));
        let label = format!("rows{}_seqk{}", rows_total, seq_k);

        // Compare vec vs block performance in crossover region
        for variant in ["vec", "block"] {
            group.bench_with_input(BenchmarkId::new(format!("SoftmaxDispatch_{}", variant), &label), &label, |bi, _| {
                let _guard = SOFTMAX_BACKEND_VAR.set_guard(variant.to_string()).unwrap();

                let mut ctx = Context::<T>::new().expect("ctx setup");
                // Reset pool before creating tensors to ensure clean state
                ctx.reset_pool();

                let input: Tensor<T> =
                    Tensor::new(vec![rows_total, seq_k], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("input");

                // Warmup
                let _warmup_out = ctx.call::<SoftmaxDispatchOp>((&input, false, 0)).expect("warmup");
                ctx.synchronize();

                bi.iter(|| {
                    let _iter_out = ctx.call::<SoftmaxDispatchOp>((&input, false, 0)).unwrap();
                    ctx.synchronize();
                });

                // Reset pool to free memory after each iteration
                ctx.reset_pool();
            });
        }
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    // Run enhanced softmax dispatcher benchmarks
    bench_softmax_dispatcher_seq_lengths::<F16Element>(c, "f16");

    bench_softmax_dispatcher_batch_variants::<F16Element>(c, "f16");

    bench_softmax_dispatcher_causal_vs_normal::<F16Element>(c, "f16");

    // New dispatcher-focused benchmarks
    bench_softmax_dispatcher_variant_selection::<F16Element>(c, "f16");
    bench_softmax_dispatcher_crossover_analysis::<F16Element>(c, "f16");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
