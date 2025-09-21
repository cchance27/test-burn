//! Benchmark for the SDPA implementations
//!
//! We run multiple iterations of each operation within each benchmark measurement
//! to reduce the impact of benchmark framework overhead and get more stable measurements.
//! This is especially important for faster operations like SDPA on smaller tensors.
use burn::prelude::Backend;
use burn::tensor::{Distribution, Float, Int, Tensor as BurnTensor};
use criterion::{Criterion, criterion_group, criterion_main};
use test_burn::metallic::{Context, Tensor};
type MyBackend = burn::backend::Metal;

fn benchmark_random_uniform_creation(c: &mut Criterion) {
    let batch = 8;
    let seq = 512;
    let dim = 64;
    let device = <MyBackend as burn::prelude::Backend>::Device::default();
    let mut context = Context::new().unwrap();
    let mut group = c.benchmark_group("random_uniform_creation");

    group.bench_function("burn", |b| {
        b.iter(|| {
            let _query = BurnTensor::<MyBackend, 3, Float>::random(
                [batch, seq, dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            MyBackend::sync(&device);
        })
    });

    group.bench_function("metallic", |b| {
        b.iter(|| {
            context.pool.reset();
            let _v_tensor = Tensor::random_uniform(vec![batch, seq, dim], &mut context).unwrap();
            context.synchronize();
        })
    });
}

fn benchmark_arange_creation(c: &mut Criterion) {
    let num_elements: usize = 8 * 512 * 64;
    let device = <MyBackend as burn::prelude::Backend>::Device::default();
    let mut context = Context::new().unwrap();
    let mut group = c.benchmark_group("arange_creation");

    group.bench_function("burn", |b| {
        b.iter(|| {
            let _t = BurnTensor::<MyBackend, 1, Int>::arange(0..num_elements as i64, &device);
            MyBackend::sync(&device);
        })
    });

    group.bench_function("metallic", |b| {
        b.iter(|| {
            context.pool.reset();
            let _t = Tensor::arange(num_elements, vec![num_elements], &mut context).unwrap();
            context.synchronize();
        })
    });

    group.bench_function("metallic_legacy", |b| {
        b.iter(|| {
            #[allow(deprecated)]
            let _t = Tensor::arange_cpu(num_elements, vec![num_elements], &context).unwrap();
            context.synchronize();
        })
    });
}

fn benchmark_zeros_creation(c: &mut Criterion) {
    let shape = [8, 512, 64];
    let device = <MyBackend as burn::prelude::Backend>::Device::default();
    let mut context = Context::new().unwrap();
    let mut group = c.benchmark_group("zeros_creation");

    group.bench_function("burn", |b| {
        b.iter(|| {
            let t = BurnTensor::<MyBackend, 3>::zeros(shape, &device);
            let _data = t.to_data();
        });
    });

    group.bench_function("metallic", |b| {
        b.iter(|| {
            context.pool.reset();
            let t = Tensor::zeros(shape.to_vec(), &mut context).unwrap();
            context.synchronize();
            let _vec = t.to_vec();
        });
    });
}

fn benchmark_ones_creation(c: &mut Criterion) {
    let shape = [8, 512, 64];
    let device = <MyBackend as burn::prelude::Backend>::Device::default();
    let mut context = Context::new().unwrap();
    let mut group = c.benchmark_group("ones_creation");

    group.bench_function("burn", |b| {
        b.iter(|| {
            let t = BurnTensor::<MyBackend, 3>::ones(shape, &device);
            let _data = t.to_data();
        });
    });

    group.bench_function("metallic", |b| {
        b.iter(|| {
            context.pool.reset();
            let t = Tensor::ones(shape.to_vec(), &mut context).unwrap();
            context.synchronize();
            let _vec = t.to_vec();
        });
    });
}

fn benchmark_batched_operations(c: &mut Criterion) {
    let shape = [8, 512, 64];
    let mut context = Context::new().unwrap();
    let mut group = c.benchmark_group("batched_operations");

    // Benchmark individual operations
    group.bench_function("individual", |b| {
        b.iter(|| {
            context.pool.reset();
            let _t1 = Tensor::zeros(shape.to_vec(), &mut context).unwrap();
            let _t2 = Tensor::ones(shape.to_vec(), &mut context).unwrap();
            let _t3 = Tensor::random_uniform(shape.to_vec(), &mut context).unwrap();
            let _t4 = Tensor::arange(shape.iter().product(), shape.to_vec(), &mut context).unwrap();
        });
        context.synchronize();
    });

    // Benchmark batched operations
    group.bench_function("batched", |b| {
        b.iter(|| {
            context.pool.reset();
            let cb =
                test_burn::metallic::operation::CommandBuffer::new(&context.command_queue).unwrap();
            let _t1 = Tensor::zeros_batched(shape.to_vec(), &cb, &mut context).unwrap();
            let _t2 = Tensor::ones_batched(shape.to_vec(), &cb, &mut context).unwrap();
            let _t3 = Tensor::random_uniform_batched(shape.to_vec(), &cb, &mut context).unwrap();
            let _t4 =
                Tensor::arange_batched(shape.iter().product(), shape.to_vec(), &cb, &mut context)
                    .unwrap();
            cb.commit();
            cb.wait();
        })
    });
}

fn benchmark_large_tensor_gpu_fallback(c: &mut Criterion) {
    let large_shape = [64, 512, 512]; // ~67MB tensor - should trigger GPU path
    let mut context = Context::new().unwrap();
    let mut group = c.benchmark_group("large_tensor_gpu_fallback");

    group.bench_function("zeros_large", |b| {
        b.iter(|| {
            context.pool.reset();
            let _t = Tensor::zeros(large_shape.to_vec(), &mut context).unwrap();
        });
        context.synchronize();
    });

    group.bench_function("ones_large", |b| {
        b.iter(|| {
            context.pool.reset();
            let _t = Tensor::ones(large_shape.to_vec(), &mut context).unwrap();
        });
        context.synchronize();
    });
}

fn benchmark_ones_scaling(c: &mut Criterion) {
    let sizes = [
        ("128kb", 128 * 1024 / 4),
        ("512kb", 512 * 1024 / 4),
        ("768kb", 768 * 1024 / 4),
        ("1mb", 1024 * 1024 / 4),
        ("4mb", 4 * 1024 * 1024 / 4),
        ("8mb", 8 * 1024 * 1024 / 4),
        ("16mb", 16 * 1024 * 1024 / 4),
    ];
    let device = <MyBackend as burn::prelude::Backend>::Device::default();
    let mut context = Context::new().unwrap();
    let mut group = c.benchmark_group("ones_scaling");

    for (name, num_elements) in sizes {
        group.bench_function(format!("{}_burn_ones", name), |b| {
            b.iter(|| {
                let t = BurnTensor::<MyBackend, 1>::ones([num_elements], &device);
                let _data = t.to_data().to_vec::<f32>();
            });
        });

        group.bench_function(format!("{}_metallic_ones", name), |b| {
            b.iter(|| {
                context.pool.reset();
                let t = Tensor::ones(vec![num_elements], &mut context).unwrap();
                context.synchronize();
                let _vec = t.to_vec();
            });
        });
    }
}

fn benchmark_zeros_scaling(c: &mut Criterion) {
    let sizes = [
        ("128kb", 128 * 1024 / 4),
        ("512kb", 512 * 1024 / 4),
        ("768kb", 768 * 1024 / 4),
        ("1mb", 1024 * 1024 / 4),
        ("4mb", 4 * 1024 * 1024 / 4),
        ("8mb", 8 * 1024 * 1024 / 4),
        ("16mb", 16 * 1024 * 1024 / 4),
    ];
    let device = <MyBackend as burn::prelude::Backend>::Device::default();
    let mut context = Context::new().unwrap();

    let mut group = c.benchmark_group("zero_scaling");
    for (name, num_elements) in sizes {
        group.bench_function(format!("{}_burn_zeros", name), |b| {
            b.iter(|| {
                let t = BurnTensor::<MyBackend, 1>::zeros([num_elements], &device);
                let _data = t.to_data();
            });
        });

        group.bench_function(format!("{}_metallic_zeros", name), |b| {
            b.iter(|| {
                context.pool.reset();
                let t = Tensor::zeros(vec![num_elements], &mut context).unwrap();
                context.synchronize();
                let _vec = t.to_vec();
            });
        });
    }
}

criterion_group!(
    benches,
    benchmark_random_uniform_creation,
    benchmark_arange_creation,
    benchmark_zeros_creation,
    benchmark_ones_creation,
    benchmark_batched_operations,
    benchmark_large_tensor_gpu_fallback,
    benchmark_ones_scaling,
    benchmark_zeros_scaling
);
criterion_main!(benches);
