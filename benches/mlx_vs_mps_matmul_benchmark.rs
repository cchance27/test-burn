use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use metallic::{
    Context, F16Element, F32Element, Tensor, TensorElement, TensorInit, TensorStorage, context::MatmulAlphaBeta, kernels::elemwise_add::BroadcastElemwiseAddInplaceOp, tensor::TensorType
};
use metallic_env::FORCE_MATMUL_BACKEND_VAR;

fn bytes_for_shape<T: TensorElement>(m: usize, k: usize, n: usize) -> usize {
    let es = T::DTYPE.size_bytes();
    // A: m x k, B: k x n, C: m x n, bias: n - accounting for all tensors created in benchmarks
    m * k * es + k * n * es + m * n * es + n * es
}

fn bench_generic_shapes<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("mlx_vs_mps_matmul_generic_{dtype_name}"));

    // Keep memory budget conservative for 16GB VRAM; stay under ~2GB per shape.
    const MAX_BYTES_PER_SHAPE: usize = 2 * 1024 * 1024 * 1024; // 2GB

    // Shape candidates; we will filter by memory budget per dtype below.
    let candidates: &[(usize, usize, usize)] = &[
        // square baselines
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        // skinny M
        (1, 4096, 4096),
        (2, 4096, 4096),
        (4, 4096, 4096),
        (8, 4096, 4096),
        // skinny N
        (4096, 4096, 1),
        (4096, 4096, 2),
        (4096, 4096, 4),
        (4096, 4096, 8),
        // transformer-ish
        (1, 4096, 11008),
        (1, 11008, 4096),
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

            group.bench_with_input(BenchmarkId::new("MPS", &label), &label, |bi, _| {
                let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mps".to_string()).unwrap();

                let mut ctx = Context::<T>::new().expect("ctx mps");
                let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("A");
                let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("B");
                let out: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("C");
                let bias: Tensor<T> = Tensor::new(vec![n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("bias");

                // Warmup
                let mut _warmup_out = ctx
                    .matmul(
                        &a,
                        &TensorType::Dense(&b),
                        false,
                        false,
                        None,
                        Some(MatmulAlphaBeta { output: &out, alpha, beta }),
                        None,
                    )
                    .expect("warmup");
                _warmup_out = ctx
                    .call::<BroadcastElemwiseAddInplaceOp>((_warmup_out, bias.clone()), None)
                    .unwrap();
                ctx.synchronize();

                bi.iter(|| {
                    let mut _iter_out = ctx
                        .matmul(
                            &a,
                            &TensorType::Dense(&b),
                            false,
                            false,
                            None,
                            Some(MatmulAlphaBeta { output: &out, alpha, beta }),
                            None,
                        )
                        .unwrap();
                    _iter_out = ctx.call::<BroadcastElemwiseAddInplaceOp>((_iter_out, bias.clone()), None).unwrap();
                    ctx.synchronize();
                });
            });

            // Benchmark MLX
            group.bench_with_input(BenchmarkId::new("MLX", &label), &label, |bi, _| {
                let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mlx".to_string()).unwrap();

                let mut ctx = Context::<T>::new().expect("ctx mlx");
                let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("A");
                let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("B");
                let out: Tensor<T> = Tensor::new(vec![m, n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("C");
                let bias: Tensor<T> = Tensor::new(vec![n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("bias");

                // Warmup
                let mut _warmup_out = ctx
                    .matmul(
                        &a,
                        &TensorType::Dense(&b),
                        false,
                        false,
                        None,
                        Some(MatmulAlphaBeta { output: &out, alpha, beta }),
                        None,
                    )
                    .expect("warmup");
                _warmup_out = ctx
                    .call::<BroadcastElemwiseAddInplaceOp>((_warmup_out, bias.clone()), None)
                    .unwrap();
                ctx.synchronize();

                bi.iter(|| {
                    let mut _iter_out = ctx
                        .matmul(
                            &a,
                            &TensorType::Dense(&b),
                            false,
                            false,
                            None,
                            Some(MatmulAlphaBeta { output: &out, alpha, beta }),
                            None,
                        )
                        .unwrap();
                    _iter_out = ctx.call::<BroadcastElemwiseAddInplaceOp>((_iter_out, bias.clone()), None).unwrap();
                    ctx.synchronize();
                });
            });
        }
    }

    // Measure fused matmul + bias path explicitly to compare MLX and MPS end-to-end epilogues.
    for &(m, k, n) in &shapes {
        let label = format!("{m}x{k}x{n}_bias");

        group.bench_with_input(BenchmarkId::new("MPS-bias", &label), &label, |bi, _| {
            let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mps".to_string()).unwrap();

            let mut ctx = Context::<T>::new().expect("ctx mps bias");
            let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("A");
            let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("B");
            let bias: Tensor<T> = Tensor::new(vec![n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("bias");

            let _warmup = ctx
                .matmul(&a, &TensorType::Dense(&b), false, false, Some(&bias), None, None)
                .expect("warmup");
            ctx.synchronize();

            bi.iter(|| {
                let _ = ctx
                    .matmul(&a, &TensorType::Dense(&b), false, false, Some(&bias), None, None)
                    .unwrap();
                ctx.synchronize();
            });
        });

        group.bench_with_input(BenchmarkId::new("MLX-bias", &label), &label, |bi, _| {
            let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mlx".to_string()).unwrap();

            let mut ctx = Context::<T>::new().expect("ctx mlx bias");
            let a: Tensor<T> = Tensor::new(vec![m, k], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("A");
            let b: Tensor<T> = Tensor::new(vec![k, n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("B");
            let bias: Tensor<T> = Tensor::new(vec![n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("bias");

            let _warmup = ctx
                .matmul(&a, &TensorType::Dense(&b), false, false, Some(&bias), None, None)
                .expect("warmup");
            ctx.synchronize();

            bi.iter(|| {
                let _ = ctx
                    .matmul(&a, &TensorType::Dense(&b), false, false, Some(&bias), None, None)
                    .unwrap();
                ctx.synchronize();
            });
        });
    }

    group.finish();
}

fn bench_qwen_shapes<T: TensorElement>(c: &mut Criterion, dtype_name: &str) {
    let mut group = c.benchmark_group(format!("mlx_vs_mps_matmul_qwen_{dtype_name}"));

    #[derive(Clone, Copy)]
    enum CaseKind {
        Matmul,
        MatmulBias,
    }

    #[derive(Clone, Copy)]
    struct TroubleCase {
        name: &'static str,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        kind: CaseKind,
    }

    const CASES: &[TroubleCase] = &[
        TroubleCase {
            name: "decode_bias_add",
            batch: 1,
            m: 1,
            k: 896,
            n: 1152,
            kind: CaseKind::MatmulBias,
        },
        TroubleCase {
            name: "kv_proj_self",
            batch: 1,
            m: 1,
            k: 896,
            n: 896,
            kind: CaseKind::Matmul,
        },
        TroubleCase {
            name: "kv_cache_expand",
            batch: 1,
            m: 1,
            k: 896,
            n: 9728,
            kind: CaseKind::Matmul,
        },
        TroubleCase {
            name: "kv_cache_reduce",
            batch: 1,
            m: 1,
            k: 4864,
            n: 896,
            kind: CaseKind::Matmul,
        },
        TroubleCase {
            name: "attn_qk_skinny",
            batch: 14,
            m: 1,
            k: 64,
            n: 1,
            kind: CaseKind::Matmul,
        },
        TroubleCase {
            name: "attn_kv_skinny",
            batch: 14,
            m: 1,
            k: 1,
            n: 64,
            kind: CaseKind::Matmul,
        },
    ];

    fn dims(batch: usize, rows: usize, cols: usize) -> Vec<usize> {
        if batch > 1 { vec![batch, rows, cols] } else { vec![rows, cols] }
    }

    const MAX_BYTES_PER_CASE: usize = 2 * 1024 * 1024 * 1024; // 2GB cap per configuration

    let bytes_for_case = |case: &TroubleCase| -> usize {
        let elem_size = T::DTYPE.size_bytes();
        let batch = case.batch;
        // A: batch x m x k, B: batch x k x n, C: batch x m x n, bias: n (only if MatmulBias)
        let a_elems = batch * case.m * case.k;
        let b_elems = batch * case.k * case.n;
        let c_elems = batch * case.m * case.n;
        let bias_elems = match case.kind {
            CaseKind::MatmulBias => case.n,
            CaseKind::Matmul => 0,
        };

        let total_elems = a_elems.saturating_add(b_elems).saturating_add(c_elems).saturating_add(bias_elems);

        total_elems.saturating_mul(elem_size)
    };

    for case in CASES {
        if bytes_for_case(case) > MAX_BYTES_PER_CASE {
            continue;
        }

        let flops = (case.batch as u128)
            .saturating_mul(case.m as u128)
            .saturating_mul(case.n as u128)
            .saturating_mul(case.k as u128)
            .saturating_mul(2);
        let throughput = flops.min(u128::from(u64::MAX)) as u64;
        group.throughput(Throughput::Elements(throughput));

        let bench_id = |backend: &str| BenchmarkId::new(backend, case.name);

        let run_case = |backend: &str, bencher: &mut criterion::Bencher<'_>| {
            let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard(backend.to_lowercase()).unwrap();

            let mut ctx = Context::<T>::new().expect("ctx setup");
            let a: Tensor<T> = Tensor::new(
                dims(case.batch, case.m, case.k),
                TensorStorage::Dedicated(&ctx),
                TensorInit::Uninitialized,
            )
            .expect("A");
            let b: Tensor<T> = Tensor::new(
                dims(case.batch, case.k, case.n),
                TensorStorage::Dedicated(&ctx),
                TensorInit::Uninitialized,
            )
            .expect("B");

            match case.kind {
                CaseKind::Matmul => {
                    let _warmup = ctx
                        .matmul(&a, &TensorType::Dense(&b), false, false, None, None, None)
                        .expect("warmup matmul");
                    ctx.synchronize();

                    bencher.iter(|| {
                        let _ = ctx.matmul(&a, &TensorType::Dense(&b), false, false, None, None, None).unwrap();
                        ctx.synchronize();
                    });
                }
                CaseKind::MatmulBias => {
                    let bias: Tensor<T> =
                        Tensor::new(vec![case.n], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).expect("bias");
                    let _warmup = ctx
                        .matmul(&a, &TensorType::Dense(&b), false, false, Some(&bias), None, None)
                        .expect("warmup bias");
                    ctx.synchronize();

                    bencher.iter(|| {
                        let _ = ctx
                            .matmul(&a, &TensorType::Dense(&b), false, false, Some(&bias), None, None)
                            .unwrap();
                        ctx.synchronize();
                    });
                }
            }
        };

        group.bench_function(bench_id("MPS"), |bencher| run_case("mps", bencher));
        group.bench_function(bench_id("MLX"), |bencher| run_case("mlx", bencher));
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    // Run generic shape benchmarks
    bench_generic_shapes::<F16Element>(c, "f16");
    bench_generic_shapes::<F32Element>(c, "f32");

    // Run Qwen-specific shape benchmarks
    bench_qwen_shapes::<F16Element>(c, "f16");
    bench_qwen_shapes::<F32Element>(c, "f32");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
