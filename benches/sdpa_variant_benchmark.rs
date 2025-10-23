use criterion::{Criterion, criterion_group, criterion_main};
use metallic::{
    Context, F32Element, Tensor, kernels::{
        KernelInvocable, scaled_dot_product_attention::{
            ScaledDotProductAttentionMpsSoftmaxOp, ScaledDotProductAttentionNoPermuteOp, ScaledDotProductAttentionOp, ScaledDotProductAttentionOptimizedOp, ScaledDotProductAttentionWorkspaceOp
        }
    }
};

const ITERATIONS: usize = 1;

fn run_variant_batched<O>(ctx: &mut Context<F32Element>, batch: usize, seq_q: usize, seq_k: usize, dim: usize, causal: bool)
where
    O: for<'a> KernelInvocable<Args<'a, F32Element> = (&'a Tensor<F32Element>, &'a Tensor<F32Element>, &'a Tensor<F32Element>, bool, u32)>,
{
    let q_tensor = Tensor::<F32Element>::random_uniform(vec![batch, seq_q, dim], ctx).unwrap();
    let k_tensor = Tensor::<F32Element>::random_uniform(vec![batch, seq_k, dim], ctx).unwrap();
    let v_tensor = Tensor::<F32Element>::random_uniform(vec![batch, seq_k, dim], ctx).unwrap();

    let mut last_output = None;

    for _ in 0..ITERATIONS {
        let out = ctx.call::<O>((&q_tensor, &k_tensor, &v_tensor, causal, 0u32)).unwrap();
        last_output = Some(out);
    }

    if let Some(tensor) = last_output {
        let _ = tensor.to_vec();
    }
}

fn run_variant_per_batch<O>(ctx: &mut Context<F32Element>, batch: usize, seq_q: usize, seq_k: usize, dim: usize, causal: bool)
where
    O: for<'a> KernelInvocable<Args<'a, F32Element> = (&'a Tensor<F32Element>, &'a Tensor<F32Element>, &'a Tensor<F32Element>, bool, u32)>,
{
    let q_tensor = Tensor::<F32Element>::random_uniform(vec![batch, seq_q, dim], ctx).unwrap();
    let k_tensor = Tensor::<F32Element>::random_uniform(vec![batch, seq_k, dim], ctx).unwrap();
    let v_tensor = Tensor::<F32Element>::random_uniform(vec![batch, seq_k, dim], ctx).unwrap();

    let q_batches: Vec<Tensor<F32Element>> = (0..batch)
        .map(|i| q_tensor.get_batch(i).unwrap().reshape(vec![1, seq_q, dim]).unwrap())
        .collect();
    let k_batches: Vec<Tensor<F32Element>> = (0..batch)
        .map(|i| k_tensor.get_batch(i).unwrap().reshape(vec![1, seq_k, dim]).unwrap())
        .collect();
    let v_batches: Vec<Tensor<F32Element>> = (0..batch)
        .map(|i| v_tensor.get_batch(i).unwrap().reshape(vec![1, seq_k, dim]).unwrap())
        .collect();

    let mut last_output = None;

    for _ in 0..ITERATIONS {
        for i in 0..batch {
            let out = ctx.call::<O>((&q_batches[i], &k_batches[i], &v_batches[i], causal, 0u32)).unwrap();
            last_output = Some(out);
        }
    }

    if let Some(tensor) = last_output {
        let _ = tensor.to_vec();
    }
}

fn benchmark_sdpa_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdpa_variant_comparison");
    let mut context = Context::<F32Element>::new().unwrap();

    let batch = 8;
    let seq_q = 512;
    let seq_k = 512;
    let dim = 64;
    let causal = false;

    group.bench_function("baseline_batched", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant_batched::<ScaledDotProductAttentionOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("baseline_per_batch", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant_per_batch::<ScaledDotProductAttentionOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("no_permute_batched", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant_batched::<ScaledDotProductAttentionNoPermuteOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("no_permute_per_batch", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant_per_batch::<ScaledDotProductAttentionNoPermuteOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("workspace_reuse_batched", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant_batched::<ScaledDotProductAttentionWorkspaceOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("workspace_reuse_per_batch", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant_per_batch::<ScaledDotProductAttentionWorkspaceOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("mps_softmax_batched", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant_batched::<ScaledDotProductAttentionMpsSoftmaxOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("mps_softmax_per_batch", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant_per_batch::<ScaledDotProductAttentionMpsSoftmaxOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("all_optimizations_batched", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant_batched::<ScaledDotProductAttentionOptimizedOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.bench_function("all_optimizations_per_batch", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_variant_per_batch::<ScaledDotProductAttentionOptimizedOp>(&mut context, batch, seq_q, seq_k, dim, causal);
            context.synchronize();
        })
    });

    group.finish();
}

criterion_group!(sdpa_variant_benches, benchmark_sdpa_variants);
criterion_main!(sdpa_variant_benches);
