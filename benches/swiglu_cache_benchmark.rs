//! Benchmarks the SwiGLU composite with and without an explicitly threaded cache.
//!
//! This helps quantify the benefit of sharing a [`ResourceCache`] across the individual
//! matmul and elementwise kernels that make up the composite implementation.

use criterion::{Criterion, criterion_group, criterion_main};
use test_burn::metallic::kernels::swiglu::swiglu_with_optional_cache;
use test_burn::metallic::resource_cache::ResourceCache;
use test_burn::metallic::{Context, F32Element, Tensor};

const BATCH: usize = 4;
const SEQ: usize = 128;
const D_MODEL: usize = 1024;
const FF_DIM_VECTOR: usize = 4096;
const FF_DIM_SCALAR: usize = 4098;

#[allow(clippy::type_complexity)]
fn prepare_inputs(
    ctx: &mut Context<F32Element>,
    ff_dim: usize,
) -> (
    Tensor<F32Element>,
    Tensor<F32Element>,
    Tensor<F32Element>,
    Tensor<F32Element>,
    Tensor<F32Element>,
    Tensor<F32Element>,
    Tensor<F32Element>,
) {
    let m = BATCH * SEQ;
    let x_normed_flat = Tensor::<F32Element>::random_uniform(vec![m, D_MODEL], ctx).unwrap();
    let ffn_gate = Tensor::<F32Element>::random_uniform(vec![ff_dim, D_MODEL], ctx).unwrap();
    let ffn_gate_bias = Tensor::<F32Element>::random_uniform(vec![ff_dim], ctx).unwrap();
    let ffn_up = Tensor::<F32Element>::random_uniform(vec![ff_dim, D_MODEL], ctx).unwrap();
    let ffn_up_bias = Tensor::<F32Element>::random_uniform(vec![ff_dim], ctx).unwrap();
    let ffn_down = Tensor::<F32Element>::random_uniform(vec![D_MODEL, ff_dim], ctx).unwrap();
    let ffn_down_bias = Tensor::<F32Element>::random_uniform(vec![D_MODEL], ctx).unwrap();

    (x_normed_flat, ffn_gate, ffn_gate_bias, ffn_up, ffn_up_bias, ffn_down, ffn_down_bias)
}

fn benchmark_variant(c: &mut Criterion, label: &str, ff_dim: usize) {
    let mut ctx = Context::new().unwrap();
    let (x_normed_flat, ffn_gate, ffn_gate_bias, ffn_up, ffn_up_bias, ffn_down, ffn_down_bias) = prepare_inputs(&mut ctx, ff_dim);
    ctx.synchronize();

    let mut group = c.benchmark_group(format!("swiglu_cache_{label}"));

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
                None,
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
                None,
            )
            .unwrap();
            ctx.synchronize();
        })
    });

    group.finish();
}

fn benchmark_swiglu_cache(c: &mut Criterion) {
    benchmark_variant(c, "vectorized", FF_DIM_VECTOR);
    benchmark_variant(c, "scalar_fallback", FF_DIM_SCALAR);
}

criterion_group!(benches, benchmark_swiglu_cache);
criterion_main!(benches);
