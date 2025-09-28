//! Benchmarks the SwiGLU composite with and without an explicitly threaded cache.
//!
//! This helps quantify the benefit of sharing a [`ResourceCache`] across the individual
//! matmul and elementwise kernels that make up the composite implementation.

use criterion::{criterion_group, criterion_main, Criterion};
use test_burn::metallic::kernels::swiglu::swiglu_with_optional_cache;
use test_burn::metallic::resource_cache::ResourceCache;
use test_burn::metallic::{Context, Tensor};

const BATCH: usize = 4;
const SEQ: usize = 128;
const D_MODEL: usize = 1024;
const FF_DIM: usize = 4096;

fn prepare_inputs(ctx: &mut Context) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    let m = BATCH * SEQ;
    let x_normed_flat = Tensor::random_uniform(vec![m, D_MODEL], ctx).unwrap();
    let ffn_gate = Tensor::random_uniform(vec![FF_DIM, D_MODEL], ctx).unwrap();
    let ffn_gate_bias = Tensor::random_uniform(vec![FF_DIM], ctx).unwrap();
    let ffn_up = Tensor::random_uniform(vec![FF_DIM, D_MODEL], ctx).unwrap();
    let ffn_up_bias = Tensor::random_uniform(vec![FF_DIM], ctx).unwrap();
    let ffn_down = Tensor::random_uniform(vec![D_MODEL, FF_DIM], ctx).unwrap();
    let ffn_down_bias = Tensor::random_uniform(vec![D_MODEL], ctx).unwrap();

    (x_normed_flat, ffn_gate, ffn_gate_bias, ffn_up, ffn_up_bias, ffn_down, ffn_down_bias)
}

fn benchmark_swiglu_cache(c: &mut Criterion) {
    let mut ctx = Context::new().unwrap();
    let (x_normed_flat, ffn_gate, ffn_gate_bias, ffn_up, ffn_up_bias, ffn_down, ffn_down_bias) = prepare_inputs(&mut ctx);
    ctx.synchronize();

    let mut group = c.benchmark_group("swiglu_cache");

    group.bench_function("with_cache", |b| {
        b.iter(|| {
            ctx.reset_pool();
            ctx.clear_cache();
            let mut cache = ResourceCache::with_device(ctx.device.clone());
            swiglu_with_optional_cache(
                &mut ctx,
                &x_normed_flat,
                &ffn_gate,
                &ffn_gate_bias,
                &ffn_up,
                &ffn_up_bias,
                &ffn_down,
                &ffn_down_bias,
                Some(&mut cache),
            )
            .unwrap();
            ctx.synchronize();
        })
    });

    group.bench_function("without_cache", |b| {
        b.iter(|| {
            ctx.reset_pool();
            ctx.clear_cache();
            swiglu_with_optional_cache(
                &mut ctx,
                &x_normed_flat,
                &ffn_gate,
                &ffn_gate_bias,
                &ffn_up,
                &ffn_up_bias,
                &ffn_down,
                &ffn_down_bias,
                None,
            )
            .unwrap();
            ctx.synchronize();
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_swiglu_cache);
criterion_main!(benches);
