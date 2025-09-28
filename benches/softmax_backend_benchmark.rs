use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use test_burn::metallic::kernels::softmax::{apply_softmax, METALLIC_SOFTMAX_BACKEND_ENV};
use test_burn::metallic::resource_cache::ResourceCache;
use test_burn::metallic::{Context, Tensor};

const ITERATIONS: usize = 5;
const ROWS: usize = 512;
const COLUMNS: usize = 512;

fn run_softmax(ctx: &mut Context, cache: Option<&mut ResourceCache>, allow_mps: bool) {
    let attn = Tensor::random_uniform(vec![ROWS, COLUMNS], ctx).unwrap();

    match cache {
        Some(cache_ref) => {
            for _ in 0..ITERATIONS {
                let _ = apply_softmax(ctx, Some(cache_ref), &attn, 1, ROWS, COLUMNS, false, 0, allow_mps).unwrap();
            }
        }
        None => {
            for _ in 0..ITERATIONS {
                let _ = apply_softmax(ctx, None, &attn, 1, ROWS, COLUMNS, false, 0, allow_mps).unwrap();
            }
        }
    }

    ctx.synchronize();
}

fn benchmark_softmax_backends(c: &mut Criterion) {
    // Ensure the backend selector defaults to automatic behavior for the benchmark run.
    unsafe {
        std::env::set_var(METALLIC_SOFTMAX_BACKEND_ENV, "auto");
    }

    let mut group = c.benchmark_group("softmax_backend_comparison");
    let mut context = Context::new().unwrap();
    let mut cache = ResourceCache::with_device(context.device.clone());

    group.bench_function("kernel", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_softmax(&mut context, None, false);
        })
    });

    group.bench_function("mps", |b| {
        b.iter(|| {
            context.synchronize();
            context.pool.reset();
            run_softmax(&mut context, Some(&mut cache), true);
        })
    });

    group.finish();
}

fn benchmark_fused_softmax_kernel(c: &mut Criterion) {
    unsafe {
        std::env::set_var(METALLIC_SOFTMAX_BACKEND_ENV, "kernel");
    }

    let mut group = c.benchmark_group("fused_softmax_kernel");
    let mut context = Context::new().unwrap();

    let configs: [(&str, usize, usize, usize); 3] = [
        ("rows256_cols256", 1, 256, 256),
        ("rows512_cols512", 1, 512, 512),
        ("rows1024_cols128", 1, 1024, 128),
    ];

    for &(label, batch, rows, columns) in &configs {
        group.bench_with_input(BenchmarkId::new("apply", label), &(), |b, _| {
            let attn = Tensor::random_uniform(vec![rows, columns], &mut context).unwrap();
            b.iter(|| {
                context.synchronize();
                context.pool.reset();
                apply_softmax(&mut context, None, &attn, batch, rows, columns, false, 0, false).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(softmax_backend_benches, benchmark_softmax_backends, benchmark_fused_softmax_kernel);
criterion_main!(softmax_backend_benches);
