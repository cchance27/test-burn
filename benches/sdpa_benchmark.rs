//! Benchmark for the SDPA implementations
//!
//! We run multiple iterations of each operation within each benchmark measurement
//! to reduce the impact of benchmark framework overhead and get more stable measurements.
//! This is especially important for faster operations like SDPA on smaller tensors.

use std::ffi::c_void;

use burn::{
    backend::wgpu::WgpuDevice, prelude::*, tensor::{Distribution, Float, Tensor as BurnTensor}
};
use criterion::{Criterion, criterion_group, criterion_main};
use metallic::{
    Context, F32Element, Tensor, alternatives::{sdpa_burn::scaled_dot_product_attention_burn, sdpa_metal::scaled_dot_product_attention_metal}
};

/// Number of iterations to run within each benchmark measurement
/// This helps reduce measurement noise for faster operations
const ITERATIONS: usize = 1;

fn sdpa_metallic(batch: usize, seq: usize, dim: usize, causal: bool, context: &mut Context<F32Element>) {
    let q_tensor = Tensor::random_uniform(vec![batch, seq, dim], context).unwrap();
    let k_tensor = Tensor::random_uniform(vec![batch, seq, dim], context).unwrap();
    let v_tensor = Tensor::random_uniform(vec![batch, seq, dim], context).unwrap();

    for _ in 0..ITERATIONS {
        context
            .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, causal)
            .unwrap();
    }
}

type MyBackend = burn::backend::Metal;
fn sdpa_burn(batch: usize, seq: usize, dim: usize, device: &WgpuDevice, causal: bool) {
    let query = BurnTensor::<MyBackend, 3, Float>::random([batch, seq, dim], Distribution::Normal(0.0, 1.0), device);
    let key = BurnTensor::<MyBackend, 3, Float>::random([batch, seq, dim], Distribution::Normal(0.0, 1.0), device);
    let value = BurnTensor::<MyBackend, 3, Float>::random([batch, seq, dim], Distribution::Normal(0.0, 1.0), device);

    for _ in 0..ITERATIONS {
        scaled_dot_product_attention_burn(query.clone(), key.clone(), value.clone(), None, causal);
        MyBackend::sync(device);
    }
}

fn sdpa_metal(batch: usize, seq: usize, dim: usize, _causal: bool) {
    //sdpa_metal we didn't bother adding causal to since we were just testing it.
    let query: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let key: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let value: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let q_ptr = std::ptr::NonNull::new(query.as_ptr() as *mut c_void).expect("query pointer is null");
    let k_ptr = std::ptr::NonNull::new(key.as_ptr() as *mut c_void).expect("key pointer is null");
    let v_ptr = std::ptr::NonNull::new(value.as_ptr() as *mut c_void).expect("value pointer is null");

    for _ in 0..ITERATIONS {
        scaled_dot_product_attention_metal(q_ptr, k_ptr, v_ptr, batch, seq, seq, dim);
    }
}

fn benchmark_sdpa_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdpa_showdown_small");
    let device = <MyBackend as burn::prelude::Backend>::Device::default();

    let batch = 4;
    let seq = 128;
    let dim = 64;

    let mut context = Context::<F32Element>::new().unwrap();

    let causal = false;
    group.bench_function("metallic", |b| {
        b.iter(|| {
            context.pool.reset();
            sdpa_metallic(batch, seq, dim, causal, &mut context)
        })
    });
    group.bench_function("burn", |b| b.iter(|| sdpa_burn(batch, seq, dim, &device, causal)));
    group.bench_function("metal", |b| b.iter(|| sdpa_metal(batch, seq, dim, causal)));
}

fn benchmark_sdpa_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdpa_showdown_medium");
    let device = <MyBackend as burn::prelude::Backend>::Device::default();

    let batch = 8;
    let seq = 512;
    let dim = 64;

    let mut context = Context::<F32Element>::new().unwrap();

    let causal = false;
    group.bench_function("metallic", |b| {
        b.iter(|| {
            context.pool.reset();
            sdpa_metallic(batch, seq, dim, causal, &mut context)
        })
    });
    group.bench_function("burn", |b| b.iter(|| sdpa_burn(batch, seq, dim, &device, causal)));
    group.bench_function("metal", |b| b.iter(|| sdpa_metal(batch, seq, dim, causal)));
}

fn benchmark_sdpa_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdpa_showdown_large");
    let device = <MyBackend as burn::prelude::Backend>::Device::default();

    let batch = 16;
    let seq = 1024;
    let dim = 96;

    let mut context = Context::<F32Element>::new().unwrap();

    let causal = false;
    group.bench_function("metallic", |b| {
        b.iter(|| {
            context.pool.reset();
            sdpa_metallic(batch, seq, dim, causal, &mut context)
        })
    });
    group.bench_function("burn", |b| b.iter(|| sdpa_burn(batch, seq, dim, &device, causal)));
    group.bench_function("metal", |b| b.iter(|| sdpa_metal(batch, seq, dim, causal)));
}

fn benchmark_sdpa_metallic(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdpa_metallic");

    let mut context = Context::<F32Element>::new().unwrap();
    let causal = true;
    group.bench_function("small_non_causal", |b| {
        b.iter(|| {
            context.pool.reset();
            sdpa_metallic(4, 128, 64, causal, &mut context)
        })
    });
    group.bench_function("medium_non_causal", |b| {
        b.iter(|| {
            context.pool.reset();
            sdpa_metallic(8, 512, 64, causal, &mut context)
        })
    });
    group.bench_function("large_non_causal", |b| {
        b.iter(|| {
            context.pool.reset();
            sdpa_metallic(16, 1024, 96, causal, &mut context)
        })
    });

    let causal = false;
    group.bench_function("small_causal", |b| {
        b.iter(|| {
            context.pool.reset();
            sdpa_metallic(4, 128, 64, causal, &mut context)
        })
    });
    group.bench_function("medium_causal", |b| {
        b.iter(|| {
            context.pool.reset();
            sdpa_metallic(8, 512, 64, causal, &mut context)
        })
    });
    group.bench_function("large_causal", |b| {
        b.iter(|| {
            context.pool.reset();
            sdpa_metallic(16, 1024, 96, causal, &mut context)
        })
    });
    group.finish();
}

fn benchmark_sdpa_burn(c: &mut Criterion) {
    let device = <MyBackend as burn::prelude::Backend>::Device::default();

    let mut group = c.benchmark_group("sdpa_burn");
    let causal = false;
    group.bench_function("small_non_causal", |b| b.iter(|| sdpa_burn(4, 128, 64, &device, causal)));
    group.bench_function("medium_non_causal", |b| b.iter(|| sdpa_burn(8, 512, 64, &device, causal)));
    group.bench_function("large_non_causal", |b| b.iter(|| sdpa_burn(16, 1024, 96, &device, causal)));

    let causal = false;
    group.bench_function("small_causal", |b| b.iter(|| sdpa_burn(4, 128, 64, &device, causal)));
    group.bench_function("medium_causal", |b| b.iter(|| sdpa_burn(8, 512, 64, &device, causal)));
    group.bench_function("large_causal", |b| b.iter(|| sdpa_burn(16, 1024, 96, &device, causal)));
    group.finish();
}

fn benchmark_sdpa_metal(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdpa_metal");
    let causal = false;
    group.bench_function("small_non_causal", |b| b.iter(|| sdpa_metal(4, 128, 64, causal)));
    group.bench_function("medium_non_causal", |b| b.iter(|| sdpa_metal(8, 512, 64, causal)));
    group.bench_function("large_non_causal", |b| b.iter(|| sdpa_metal(16, 1024, 96, causal)));

    group.finish();
}

criterion_group!(
    benches,
    benchmark_sdpa_metallic,
    benchmark_sdpa_burn,
    benchmark_sdpa_metal,
    benchmark_sdpa_small,
    benchmark_sdpa_medium,
    benchmark_sdpa_large
);
criterion_main!(benches);
