use criterion::{Criterion, criterion_group, criterion_main};
use metallic::kernels::tensors::NoopOp;
use metallic::{Context, F16Element, Tensor, TensorElement, TensorInit, TensorStorage};

fn benchmark_noop<T: TensorElement>(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_testing");

    group.bench_function("100_noop_1_sync", |b| {
        let mut ctx = Context::<T>::new().unwrap();
        let tensor = Tensor::new(vec![1, 1], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).unwrap();
        b.iter(|| {
            for _i in 0..100 {
                ctx.call::<NoopOp>(tensor.clone()).unwrap();
            }
            ctx.synchronize();
        })
    });

    group.bench_function("1_noop_1_sync", |b| {
        let mut ctx = Context::<T>::new().unwrap();
        let tensor = Tensor::new(vec![1, 1], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).unwrap();
        b.iter(|| {
            ctx.call::<NoopOp>(tensor.clone()).unwrap();
            ctx.synchronize();
        })
    });

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    // Run dispatcher benchmarks
    benchmark_noop::<F16Element>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
