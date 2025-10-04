use criterion::{Criterion, black_box, criterion_group, criterion_main};
use test_burn::metallic::kernels::sampling::MAX_TOP_K;
use test_burn::metallic::sampling::{SamplerBuffers, sample_top_k_top_p_with_random_value};
use test_burn::metallic::{Context, F32Element, Tensor, TensorInit, TensorStorage};

const ITERATIONS: usize = 32;
const VOCAB_SIZE: usize = MAX_TOP_K * 4;
const TOP_K: usize = 40;
const TOP_P: f32 = 0.9;
const TEMPERATURE: f32 = 0.85;
const RNG_SEED: u64 = 0x5EED_FACED;

fn prepare_logits_tensor(ctx: &Context<F32Element>) -> Tensor<F32Element> {
    let logits: Vec<f32> = (0..VOCAB_SIZE).map(|i| ((i % 97) as f32) * 0.021 - 1.3).collect();
    Tensor::new(vec![VOCAB_SIZE], TensorStorage::Dedicated(ctx), TensorInit::CopyFrom(&logits)).expect("failed to allocate logits tensor")
}

fn bench_cpu_sampling(ctx: &mut Context<F32Element>, logits_tensor: &Tensor<F32Element>) {
    let mut buffers = SamplerBuffers::default();
    ctx.reseed_sampler(RNG_SEED);

    for _ in 0..ITERATIONS {
        let random = ctx.next_sampler_random();
        let logits_host = logits_tensor.to_vec();
        let token =
            sample_top_k_top_p_with_random_value::<F32Element>(&logits_host[..VOCAB_SIZE], TOP_K, TOP_P, TEMPERATURE, random, &mut buffers);
        black_box(token);
    }
}

fn bench_gpu_sampling(ctx: &mut Context<F32Element>, logits_tensor: &Tensor<F32Element>) {
    ctx.reseed_sampler(RNG_SEED);

    for _ in 0..ITERATIONS {
        let random = ctx.next_sampler_random();
        let token = ctx
            .sample_top_k_top_p_device(logits_tensor, VOCAB_SIZE, TOP_K, TOP_P, TEMPERATURE, random)
            .expect("device sampling should not fail")
            .expect("kernel must return a token");
        black_box(token);
    }
}

fn sampling_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_device_vs_cpu");
    let mut context = Context::<F32Element>::new().expect("metal context");
    let logits_tensor = prepare_logits_tensor(&context);

    group.bench_function("cpu_host_sampling", |b| {
        b.iter(|| {
            context.pool.reset();
            bench_cpu_sampling(&mut context, &logits_tensor);
        })
    });

    group.bench_function("gpu_kernel_sampling", |b| {
        b.iter(|| {
            context.pool.reset();
            bench_gpu_sampling(&mut context, &logits_tensor);
        })
    });

    group.finish();
}

criterion_group!(sampling_benches, sampling_benchmarks);
criterion_main!(sampling_benches);
