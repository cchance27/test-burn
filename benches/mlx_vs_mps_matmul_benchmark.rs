use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use test_burn::metallic::{Context, F16Element, F32Element, Tensor, TensorElement, TensorInit, TensorStorage};

use std::sync::{Mutex, OnceLock};

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn bytes_for_shape<T: TensorElement>(m: usize, k: usize, n: usize) -> usize {
    let es = T::DTYPE.size_bytes();
    // A: m x k, B: k x n, C: m x n
    m * k * es + k * n * es + m * n * es
}

fn bench_shapes<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("mlx_vs_mps_matmul_{dtype_name}"));

    // Keep memory budget conservative for 16GB VRAM; stay under ~2GB per shape.
    const MAX_BYTES_PER_SHAPE: usize = 2 * 1024 * 1024 * 1024; // 2GB

    // Shape candidates; we will filter by memory budget per dtype below.
    let candidates: &[(usize, usize, usize)] = &[
        // square baselines
        (256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
        // skinny M
        (1, 4096, 4096), (2, 4096, 4096), (4, 4096, 4096), (8, 4096, 4096),
        // skinny N
        (4096, 4096, 1), (4096, 4096, 2), (4096, 4096, 4), (4096, 4096, 8),
        // transformer-ish
        (1, 4096, 11008), (1, 11008, 4096),
    ];

    let shapes: Vec<(usize, usize, usize)> = candidates
        .iter()
        .copied()
        .filter(|&(m, k, n)| bytes_for_shape::<T>(m, k, n) <= MAX_BYTES_PER_SHAPE)
        .collect();

    let alpha_beta_cases = [
        (1.0f32, 0.0f32, "a1_b0"),
        (2.5f32, 0.0f32, "aN_b0"),
        (1.0f32, 1.0f32, "a1_b1"),
        (2.5f32, 1.5f32, "aN_bN"),
    ];

    for &(m, k, n) in &shapes {
        let flops = (m * n * k) as f64 * 2.0;
        group.throughput(Throughput::Elements(flops as u64));
        let shape_label = format!("{m}x{k}x{n}");

        for &(alpha, beta, case_name) in &alpha_beta_cases {
            let label = format!("{shape_label}_{case_name}");

            // Benchmark MPS
            group.bench_with_input(BenchmarkId::new("MPS", &label), &label, |bi, _| {
                // Create context and tensors once per backend/shape (reuse in b.iter)
                let _guard = env_lock().lock().unwrap();
                let prev = std::env::var("FORCE_MATMUL_BACKEND").ok();
                unsafe { std::env::set_var("FORCE_MATMUL_BACKEND", "mps"); }
                drop(_guard);

                let mut ctx = Context::<T>::new().expect("ctx mps");
                let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("A");
                let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("B");
                let out: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("C");
                // Warmup
                ctx.matmul_alpha_beta(&a, &b, &out, false, false, alpha, beta).expect("warmup");
                ctx.synchronize();

                bi.iter(|| {
                    ctx.matmul_alpha_beta(&a, &b, &out, false, false, alpha, beta).unwrap();
                    ctx.synchronize();
                });

                // Restore env
                let _guard = env_lock().lock().unwrap();
                unsafe {if let Some(prev) = prev { std::env::set_var("FORCE_MATMUL_BACKEND", prev); } else { std::env::remove_var("FORCE_MATMUL_BACKEND"); }}
            });

            // Benchmark MLX
            group.bench_with_input(BenchmarkId::new("MLX", &label), &label, |bi, _| {
                let _guard = env_lock().lock().unwrap();
                let prev = std::env::var("FORCE_MATMUL_BACKEND").ok();
                unsafe { std::env::set_var("FORCE_MATMUL_BACKEND", "mlx"); }
                drop(_guard);

                let mut ctx = Context::<T>::new().expect("ctx mlx");
                let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("A");
                let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("B");
                let out: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("C");
                ctx.matmul_alpha_beta(&a, &b, &out, false, false, alpha, beta).expect("warmup");
                ctx.synchronize();

                bi.iter(|| {
                    ctx.matmul_alpha_beta(&a, &b, &out, false, false, alpha, beta).unwrap();
                    ctx.synchronize();
                });

                let _guard = env_lock().lock().unwrap();
                unsafe {if let Some(prev) = prev { std::env::set_var("FORCE_MATMUL_BACKEND", prev); } else { std::env::remove_var("FORCE_MATMUL_BACKEND"); }}
            });
        }
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_shapes::<F16Element>(c, "f16");
    bench_shapes::<F32Element>(c, "f32");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
