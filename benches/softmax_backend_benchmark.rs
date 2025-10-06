use criterion::{Criterion, criterion_group, criterion_main};
use metallic::kernels::softmax::{METALLIC_SOFTMAX_BACKEND_ENV, apply_softmax};
use metallic::resource_cache::ResourceCache;
use metallic::{Context, F32Element, Tensor};

const ITERATIONS: usize = 5;
const ROWS: usize = 512;
const COLUMNS: usize = 512;

fn run_softmax(ctx: &mut Context<F32Element>, cache: Option<&mut ResourceCache>, allow_mps: bool) {
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

criterion_group!(softmax_backend_benches, benchmark_softmax_backends);
criterion_main!(softmax_backend_benches);
