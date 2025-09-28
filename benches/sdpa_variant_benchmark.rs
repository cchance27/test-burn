use criterion::{Criterion, criterion_group, criterion_main};
use test_burn::metallic::kernels::KernelInvocable;
use test_burn::metallic::kernels::scaled_dot_product_attention::{
    ScaledDotProductAttentionMpsSoftmaxOp, ScaledDotProductAttentionNoPermuteOp, ScaledDotProductAttentionOp,
    ScaledDotProductAttentionOptimizedOp, ScaledDotProductAttentionWorkspaceOp,
};
use test_burn::metallic::{Context, Tensor};

const ITERATIONS: usize = 1;

fn run_variant<O: KernelInvocable<Args = (Tensor, Tensor, Tensor, bool, u32)>>(
    ctx: &mut Context,
    batch: usize,
    seq_q: usize,
    seq_k: usize,
    dim: usize,
    causal: bool,
) {
    let q_tensor = Tensor::random_uniform(vec![batch, seq_q, dim], ctx).unwrap();
    let k_tensor = Tensor::random_uniform(vec![batch, seq_k, dim], ctx).unwrap();
    let v_tensor = Tensor::random_uniform(vec![batch, seq_k, dim], ctx).unwrap();

    let mut last_output = None;

    for _ in 0..ITERATIONS {
        let out = ctx
            .call::<O>((q_tensor.clone(), k_tensor.clone(), v_tensor.clone(), causal, 0))
            .unwrap();
        last_output = Some(out);
    }

    if let Some(tensor) = last_output {
        let _ = tensor.to_vec();
    }
}

fn benchmark_sdpa_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdpa_variant_comparison");
    let mut context = Context::new().unwrap();

    let batch = 8;
    let seq_q = 512;
    let seq_k = 512;
    let dim = 64;
    let causal = false;

    group.bench_function("baseline", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant::<ScaledDotProductAttentionOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("no_permute", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant::<ScaledDotProductAttentionNoPermuteOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("workspace_reuse", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant::<ScaledDotProductAttentionWorkspaceOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("mps_softmax", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant::<ScaledDotProductAttentionMpsSoftmaxOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("all_optimizations", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant::<ScaledDotProductAttentionOptimizedOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.finish();
}

criterion_group!(sdpa_variant_benches, benchmark_sdpa_variants);
criterion_main!(sdpa_variant_benches);
