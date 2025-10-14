use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use metallic::kernels::softmax::SoftmaxOp;
use metallic::{Context, F16Element, F32Element, Tensor, TensorElement, TensorInit, TensorStorage};
use metallic_env::SOFTMAX_BACKEND_VAR;

fn bench_softmax_dispatcher_seq_lengths<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("softmax_dispatcher_seq_{dtype_name}"));

    // Different sequence lengths to test vector vs block softmax variants
    let seq_lengths = [
        64,    // Should use vectorized approach
        128,   // Should use vectorized approach
        256,   // Should use vectorized approach
        512,   // Should use vectorized approach
        1024,  // Edge case between vec and block
        2048,  // Should use block approach
        4096,  // Should use block approach
        8192,  // Should use block approach
    ];

    // Head sizes to test
    let head_sizes = [
        64,  // Common in attention mechanisms
        128, // Common in attention mechanisms
    ];

    for &seq_len in &seq_lengths {
        for &head_size in &head_sizes {
            let flops = (seq_len * head_size) as f64 * 3.0; // Estimate of ops (read+max+exp+sum+div)
            group.throughput(Throughput::Elements(flops as u64));
            let label = format!("seq{}_head{}", seq_len, head_size);

            // Benchmark softmax with different backends
            for backend in ["auto", "kernel", "mps"] {
                group.bench_with_input(
                    BenchmarkId::new(format!("Softmax_{}", backend), &label), 
                    &label, 
                    |bi, _| {
                        let _guard = SOFTMAX_BACKEND_VAR.set_guard(backend.to_string()).unwrap();

                        let mut ctx = Context::<T>::new().expect("ctx setup");
                        let input: Tensor<T> = Tensor::new(
                            vec![seq_len, head_size], 
                            TensorStorage::Dedicated(&ctx), 
                            TensorInit::Uninitialized
                        ).expect("input");

                        // Warmup
                        let _warmup_out = ctx.call::<SoftmaxOp>((&input, (seq_len * head_size) as u32, seq_len as u32, head_size as u32, 0, 0)).expect("warmup");
                        ctx.synchronize();

                        bi.iter(|| {
                            let _iter_out = ctx.call::<SoftmaxOp>((&input, (seq_len * head_size) as u32, seq_len as u32, head_size as u32, 0, 0)).unwrap();
                            ctx.synchronize();
                        });
                    }
                );
            }
        }
    }

    group.finish();
}

fn bench_softmax_dispatcher_batch_variants<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("softmax_dispatcher_batch_{dtype_name}"));

    // Batch sizes with different sequence lengths to test threading behavior
    let test_cases = [
        (1, 1024, 64),   // Single batch, medium sequence, small head
        (4, 1024, 64),   // Small batch, medium sequence, small head
        (1, 2048, 128),  // Single batch, long sequence, medium head
        (2, 2048, 128),  // Small batch, long sequence, medium head
    ];

    for &(batch_size, seq_len, head_size) in &test_cases {
        let flops = (batch_size * seq_len * head_size) as f64 * 3.0;
        group.throughput(Throughput::Elements(flops as u64));
        let label = format!("batch{}_seq{}_head{}", batch_size, seq_len, head_size);

        // Test with auto backend selection
        group.bench_with_input(
            BenchmarkId::new("Softmax_Auto", &label), 
            &label, 
            |bi, _| {
                let _guard = SOFTMAX_BACKEND_VAR.set_guard("auto".to_string()).unwrap();

                let mut ctx = Context::<T>::new().expect("ctx setup");
                let input: Tensor<T> = Tensor::new(
                    vec![batch_size, seq_len, head_size], 
                    TensorStorage::Dedicated(&ctx), 
                    TensorInit::Uninitialized
                ).expect("input");

                let rows_total = (batch_size * seq_len) as u32;
                let seq_q = seq_len as u32;
                let seq_k = head_size as u32;

                // Warmup
                let _warmup_out = ctx.call::<SoftmaxOp>((&input, rows_total, seq_q, seq_k, 0, 0)).expect("warmup");
                ctx.synchronize();

                bi.iter(|| {
                    let _iter_out = ctx.call::<SoftmaxOp>((&input, rows_total, seq_q, seq_k, 0, 0)).unwrap();
                    ctx.synchronize();
                });
            }
        );
    }

    group.finish();
}

fn bench_softmax_dispatcher_causal_vs_normal<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("softmax_dispatcher_causal_{dtype_name}"));

    let test_cases = [
        (512, 64),   // Medium sequence, small head
        (1024, 128), // Long sequence, medium head
        (2048, 64),  // Very long sequence, small head
    ];

    for &(seq_len, head_size) in &test_cases {
        let flops = (seq_len * head_size) as f64 * 3.0;
        group.throughput(Throughput::Elements(flops as u64));
        let label = format!("seq{}_head{}", seq_len, head_size);

        // Test normal softmax
        group.bench_with_input(
            BenchmarkId::new("Softmax_Normal", &label), 
            &label, 
            |bi, _| {
                let mut ctx = Context::<T>::new().expect("ctx setup");
                let input: Tensor<T> = Tensor::new(
                    vec![seq_len, head_size], 
                    TensorStorage::Dedicated(&ctx), 
                    TensorInit::Uninitialized
                ).expect("input");

                let rows_total = seq_len as u32;
                let seq_q = seq_len as u32;
                let seq_k = head_size as u32;

                // Warmup
                let _warmup_out = ctx.call::<SoftmaxOp>((&input, rows_total, seq_q, seq_k, 0, 0)).expect("warmup");
                ctx.synchronize();

                bi.iter(|| {
                    let _iter_out = ctx.call::<SoftmaxOp>((&input, rows_total, seq_q, seq_k, 0, 0)).unwrap();
                    ctx.synchronize();
                });
            }
        );

        // Test causal softmax 
        group.bench_with_input(
            BenchmarkId::new("Softmax_Causal", &label), 
            &label, 
            |bi, _| {
                let mut ctx = Context::<T>::new().expect("ctx setup");
                let input: Tensor<T> = Tensor::new(
                    vec![seq_len, head_size], 
                    TensorStorage::Dedicated(&ctx), 
                    TensorInit::Uninitialized
                ).expect("input");

                let rows_total = seq_len as u32;
                let seq_q = seq_len as u32;
                let seq_k = head_size as u32;

                // Warmup
                let _warmup_out = ctx.call::<SoftmaxOp>((&input, rows_total, seq_q, seq_k, 1, 0)).expect("warmup");
                ctx.synchronize();

                bi.iter(|| {
                    let _iter_out = ctx.call::<SoftmaxOp>((&input, rows_total, seq_q, seq_k, 1, 0)).unwrap();
                    ctx.synchronize();
                });
            }
        );
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    // Run softmax dispatcher benchmarks
    bench_softmax_dispatcher_seq_lengths::<F16Element>(c, "f16");
    bench_softmax_dispatcher_seq_lengths::<F32Element>(c, "f32");

    bench_softmax_dispatcher_batch_variants::<F16Element>(c, "f16");
    bench_softmax_dispatcher_batch_variants::<F32Element>(c, "f32");

    bench_softmax_dispatcher_causal_vs_normal::<F16Element>(c, "f16");
    bench_softmax_dispatcher_causal_vs_normal::<F32Element>(c, "f32");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);