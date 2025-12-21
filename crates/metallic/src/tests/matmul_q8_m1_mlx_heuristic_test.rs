#![cfg(test)]

use half::f16;

use crate::{
    Context, MetalError, Tensor, TensorStorage, tensor::{Q8_0_WEIGHTS_PER_BLOCK, TensorType, quantized::QuantizedQ8_0Tensor}
};

fn make_fp16_tensor(ctx: &mut Context<crate::tensor::F16>, dims: Vec<usize>, seed: u64) -> Result<Tensor<crate::tensor::F16>, MetalError> {
    let len = dims.iter().product();
    let mut data = Vec::with_capacity(len);
    let mut x = seed;
    for _ in 0..len {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = ((x >> 32) as u32) as f32 / (u32::MAX as f32) - 0.5;
        data.push(v);
    }
    Tensor::<crate::tensor::F16>::from_f32_slice(dims, TensorStorage::Pooled(ctx), &data)
}

fn quantize_q8_0_canonical_from_f32(
    ctx: &Context<crate::tensor::F16>,
    k: usize,
    n: usize,
    weights_kn: &[f32],
) -> Result<QuantizedQ8_0Tensor, MetalError> {
    assert_eq!(weights_kn.len(), k * n);
    let blocks_per_k = (k + Q8_0_WEIGHTS_PER_BLOCK - 1) / Q8_0_WEIGHTS_PER_BLOCK;
    let total_blocks = blocks_per_k * n;
    let mut data = Vec::with_capacity(total_blocks * Q8_0_WEIGHTS_PER_BLOCK);
    let mut scales = Vec::with_capacity(total_blocks * 2);

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
            scales.extend_from_slice(&d_bits);

            for i in 0..Q8_0_WEIGHTS_PER_BLOCK {
                let val = if i < blk_len { weights_kn[(base_k + i) * n + col] } else { 0.0 };
                let q = if d > 0.0 { (val / d).round().clamp(-127.0, 127.0) as i8 } else { 0 };
                data.push(q as u8);
            }
        }
    }

    QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![k, n], &data, &scales, ctx)
}

#[test]
fn q8_m1_large_n_dispatches_without_mlx_heuristic() -> Result<(), MetalError> {
    let mut ctx = Context::<crate::tensor::F16>::new()?;

    let k = 32usize;
    let n = 4096usize;
    let x = make_fp16_tensor(&mut ctx, vec![1, k], 0xBEEF_1234)?;
    let w = make_fp16_tensor(&mut ctx, vec![k, n], 0xACDC_F00D)?;
    let w_f: Vec<f32> = w.as_slice().iter().map(|v| v.to_f32()).collect();
    let q8 = quantize_q8_0_canonical_from_f32(&ctx, k, n, &w_f)?;

    let y = ctx.matmul(
        &x,
        &TensorType::Quant(crate::tensor::QuantizedTensor::Q8_0(&q8)),
        false,
        false,
        None,
        None,
        None,
    )?;
    ctx.synchronize();

    assert_eq!(y.dims(), &[1, n]);
    Ok(())
}
