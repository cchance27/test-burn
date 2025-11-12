use criterion::{Criterion, criterion_group, criterion_main};
use metallic::{
    Context, F16Element, Tensor, kernels::{DefaultKernelInvocable, scaled_dot_product_attention::ScaledDotProductAttentionOptimizedOp, sdpa_mps_graph::SdpaMpsGraphOp}
};

const ITERATIONS: usize = 1;

fn run_variant<O>(
    ctx: &mut Context<F16Element>,
    q_tensor: &Tensor<F16Element>,
    k_tensor: &Tensor<F16Element>,
    v_tensor: &Tensor<F16Element>,
    causal: bool,
) -> Tensor<F16Element>
where
    O: for<'a> DefaultKernelInvocable<
        Args<'a, F16Element> = (&'a Tensor<F16Element>, &'a Tensor<F16Element>, &'a Tensor<F16Element>, bool, u32),
    >,
{
    let mut last_output = None;
    for _ in 0..ITERATIONS {
        let out = ctx.call::<O>((q_tensor, k_tensor, v_tensor, causal, 0u32), None).unwrap();
        last_output = Some(out);
    }
    last_output.unwrap()
}

fn benchmark_sdpa_mpsgraph_vs_current(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdpa_mpsgraph_vs_current_f16");
    let mut ctx = Context::<F16Element>::new().unwrap();

    // Use same shapes as comparison function to avoid cache issues
    let batch = 4;
    let seq_q = 256;
    let seq_k = 256;
    let dim = 16;
    let causal = true; // match current capability as requested

    // Pre-create tensors to reduce per-iteration initialization overhead
    let q_tensor = Tensor::<F16Element>::random_uniform(vec![batch, seq_q, dim], &mut ctx).unwrap();
    let k_tensor = Tensor::<F16Element>::random_uniform(vec![batch, seq_k, dim], &mut ctx).unwrap();
    let v_tensor = Tensor::<F16Element>::random_uniform(vec![batch, seq_k, dim], &mut ctx).unwrap();

    group.bench_function("current_optimized_f16", |b| {
        b.iter(|| {
            ctx.synchronize();
            ctx.pool.reset();
            run_variant::<ScaledDotProductAttentionOptimizedOp>(&mut ctx, &q_tensor, &k_tensor, &v_tensor, causal);
            ctx.synchronize();
        })
    });

    group.bench_function("mpsgraph_f16", |b| {
        b.iter(|| {
            ctx.synchronize();
            ctx.pool.reset();
            run_variant::<SdpaMpsGraphOp>(&mut ctx, &q_tensor, &k_tensor, &v_tensor, causal);
            ctx.synchronize();
        })
    });

    group.finish();
}

criterion_group!(sdpa_mpsgraph_benches, benchmark_sdpa_mpsgraph_vs_current);
criterion_main!(sdpa_mpsgraph_benches);
