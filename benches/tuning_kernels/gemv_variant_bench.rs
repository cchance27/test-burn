use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use half::f16;
use metallic::{
    Context, F16Element, Tensor, TensorStorage, kernels::matmul_gemv::MatmulGemvOp, tensor::{
        QuantizedTensor, TensorType, quantized::{Q8_0_SCALE_BYTES_PER_BLOCK, QuantizedQ8_0Tensor}
    }
};

fn make_f16(ctx: &mut Context<F16Element>, dims: Vec<usize>, seed: u64) -> Tensor<F16Element> {
    let len = dims.iter().product();
    let mut data = Vec::with_capacity(len);
    let mut x = seed;
    for _ in 0..len {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = ((x >> 32) as u32) as f32 / (u32::MAX as f32) - 0.5;
        data.push(v);
    }
    Tensor::<F16Element>::from_f32_slice(dims, TensorStorage::Pooled(ctx), &data).expect("tensor")
}

fn quantize_q8_0_canonical_from_f32(ctx: &Context<F16Element>, k: usize, n: usize, weights_kn: &[f32]) -> QuantizedQ8_0Tensor {
    use metallic::tensor::Q8_0_WEIGHTS_PER_BLOCK;
    let blocks_per_k = (k + Q8_0_WEIGHTS_PER_BLOCK - 1) / Q8_0_WEIGHTS_PER_BLOCK;
    let total_blocks = blocks_per_k * n;
    let mut data = vec![0u8; total_blocks * Q8_0_WEIGHTS_PER_BLOCK];
    let mut scales = vec![0u8; total_blocks * Q8_0_SCALE_BYTES_PER_BLOCK];

    for bk in 0..blocks_per_k {
        let base_k = bk * Q8_0_WEIGHTS_PER_BLOCK;
        let blk_len = Q8_0_WEIGHTS_PER_BLOCK.min(k - base_k);
        for col in 0..n {
            let mut max_abs = 0f32;
            for i in 0..blk_len {
                let v = weights_kn[(base_k + i) * n + col];
                max_abs = max_abs.max(v.abs());
            }
            let d = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            let d_bits = f16::from_f32(d).to_bits().to_le_bytes();
            let sb = (bk * n + col) * Q8_0_SCALE_BYTES_PER_BLOCK;
            scales[sb..sb + 2].copy_from_slice(&d_bits);
            let base = (bk * n + col) * Q8_0_WEIGHTS_PER_BLOCK;
            for i in 0..Q8_0_WEIGHTS_PER_BLOCK {
                let val = if i < blk_len { weights_kn[(base_k + i) * n + col] } else { 0.0 };
                let q = if d > 0.0 { (val / d).round().clamp(-127.0, 127.0) as i8 } else { 0 };
                data[base + i] = q as u8;
            }
        }
    }

    QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![k, n], &data, &scales, ctx).expect("q8")
}

fn bench_gemv_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_variant");
    let shapes = [(896usize, 4864usize), (896usize, 896usize), (896usize, 1152usize)];

    for &(k, n) in &shapes {
        let flops = (k * n) as f64 * 2.0;
        group.throughput(Throughput::Elements(flops as u64));
        let label = format!("k{}_n{}", k, n);

        // Dense GEMV
        group.bench_with_input(BenchmarkId::new("dense", &label), &label, |b, _| {
            let mut ctx = Context::<F16Element>::new().expect("ctx");
            ctx.reset_pool();
            let x = make_f16(&mut ctx, vec![1, k], 0x1111_2222);
            let w = make_f16(&mut ctx, vec![k, n], 0x3333_4444);

            let _warm = ctx
                .call::<MatmulGemvOp>((&x, TensorType::Dense(&w), false, None), None)
                .expect("warm");
            ctx.synchronize();

            b.iter(|| {
                let _out = ctx.call::<MatmulGemvOp>((&x, TensorType::Dense(&w), false, None), None).unwrap();
                ctx.synchronize();
            });
        });

        // Q8 GEMV
        group.bench_with_input(BenchmarkId::new("q8", &label), &label, |b, _| {
            let mut ctx = Context::<F16Element>::new().expect("ctx");
            ctx.reset_pool();
            let x = make_f16(&mut ctx, vec![1, k], 0x5555_6666);
            let w = make_f16(&mut ctx, vec![k, n], 0x7777_8888);
            let w_q8 = quantize_q8_0_canonical_from_f32(&ctx, k, n, &w.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>());

            let _warm = ctx
                .call::<MatmulGemvOp>((&x, TensorType::Quant(QuantizedTensor::Q8_0(&w_q8)), false, None), None)
                .expect("warm");
            ctx.synchronize();

            b.iter(|| {
                let _out = ctx
                    .call::<MatmulGemvOp>((&x, TensorType::Quant(QuantizedTensor::Q8_0(&w_q8)), false, None), None)
                    .unwrap();
                ctx.synchronize();
            });
        });
    }

    group.finish();
}

criterion_group!(gemv_variant_benches, bench_gemv_variants);
criterion_main!(gemv_variant_benches);
