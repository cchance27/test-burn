#![cfg(test)]

use half::f16;
use serial_test::serial;

use crate::{
    Context, MetalError, Tensor, TensorStorage, kernels::{
        matmul_gemv::{MatmulGemvOp, MatmulGemvQ8SwiGluOp, MatmulGemvQ8SwiGluRmsnormOp, MatmulGemvRmsnormOp}, matmul_gemv_qkv_fused::{MatmulGemvQkvFusedOp, MatmulGemvQkvFusedRmsnormOp}, rmsnorm::RMSNormOp
    }, tensor::{
        Dtype, QuantizedTensor, TensorType, quantized::{Q8_0_SCALE_BYTES_PER_BLOCK, QuantizedQ8_0Tensor}
    }
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
    use crate::tensor::Q8_0_WEIGHTS_PER_BLOCK;
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

    QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![k, n], &data, &scales, ctx)
}

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff <= tol, "idx={} diff={} x={} y={}", i, diff, x, y);
    }
}

#[test]
#[serial]
fn qkv_fused_rmsnorm_parity() -> Result<(), MetalError> {
    let mut ctx = Context::<crate::tensor::F16>::new()?;
    let k = 64usize;
    let n = 64usize;

    let x = make_fp16_tensor(&mut ctx, vec![1, k], 0x1234_5678)?;
    let gamma = make_fp16_tensor(&mut ctx, vec![k], 0xA0A0_A0A0)?;

    let wq = make_fp16_tensor(&mut ctx, vec![k, n], 0x1111_2222)?;
    let wk = make_fp16_tensor(&mut ctx, vec![k, n], 0x3333_4444)?;
    let wv = make_fp16_tensor(&mut ctx, vec![k, n], 0x5555_6666)?;
    let q_bias = make_fp16_tensor(&mut ctx, vec![n], 0x7777_8888)?;
    let k_bias = make_fp16_tensor(&mut ctx, vec![n], 0x9999_AAAA)?;
    let v_bias = make_fp16_tensor(&mut ctx, vec![n], 0xBBBB_CCCC)?;

    let wq_q8 = quantize_q8_0_canonical_from_f32(&ctx, k, n, &wq.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;
    let wk_q8 = quantize_q8_0_canonical_from_f32(&ctx, k, n, &wk.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;
    let wv_q8 = quantize_q8_0_canonical_from_f32(&ctx, k, n, &wv.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;

    let x_normed = ctx.call::<RMSNormOp>((x.clone(), gamma.clone(), k as u32), None)?;
    let y_ref = ctx.call::<MatmulGemvQkvFusedOp>(
        (
            &x_normed,
            (
                &QuantizedTensor::Q8_0(&wq_q8),
                &QuantizedTensor::Q8_0(&wk_q8),
                &QuantizedTensor::Q8_0(&wv_q8),
            ),
            (Some(&q_bias), Some(&k_bias), Some(&v_bias)),
        ),
        None,
    )?;
    let y_fused = ctx.call::<MatmulGemvQkvFusedRmsnormOp>(
        (
            &x,
            &gamma,
            (
                &QuantizedTensor::Q8_0(&wq_q8),
                &QuantizedTensor::Q8_0(&wk_q8),
                &QuantizedTensor::Q8_0(&wv_q8),
            ),
            (Some(&q_bias), Some(&k_bias), Some(&v_bias)),
        ),
        None,
    )?;
    ctx.synchronize();

    let elem = Dtype::F16.size_bytes();
    let yq_ref = y_ref.build_view(vec![1, n], vec![n, 1], y_ref.offset);
    let yk_ref = y_ref.build_view(vec![1, n], vec![n, 1], y_ref.offset + n * elem);
    let yv_ref = y_ref.build_view(vec![1, n], vec![n, 1], y_ref.offset + 2 * n * elem);

    let yq_fused = y_fused.build_view(vec![1, n], vec![n, 1], y_fused.offset);
    let yk_fused = y_fused.build_view(vec![1, n], vec![n, 1], y_fused.offset + n * elem);
    let yv_fused = y_fused.build_view(vec![1, n], vec![n, 1], y_fused.offset + 2 * n * elem);

    let fq = yq_ref.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let fk = yk_ref.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let fv = yv_ref.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();

    let fqr = yq_fused.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let fkr = yk_fused.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let fvr = yv_fused.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();

    assert_close(&fq, &fqr, 0.15);
    assert_close(&fk, &fkr, 0.15);
    assert_close(&fv, &fvr, 0.15);
    Ok(())
}

#[test]
#[serial]
fn swiglu_rmsnorm_parity() -> Result<(), MetalError> {
    let mut ctx = Context::<crate::tensor::F16>::new()?;
    let k = 64usize;
    let n = 64usize;

    let x = make_fp16_tensor(&mut ctx, vec![1, k], 0xFACE_CAFE)?;
    let gamma = make_fp16_tensor(&mut ctx, vec![k], 0xBEEF_BEEF)?;
    let w_gate = make_fp16_tensor(&mut ctx, vec![k, n], 0xAAAA_0001)?;
    let w_up = make_fp16_tensor(&mut ctx, vec![k, n], 0xAAAA_0002)?;
    let b_gate = make_fp16_tensor(&mut ctx, vec![n], 0xAAAA_0003)?;
    let b_up = make_fp16_tensor(&mut ctx, vec![n], 0xAAAA_0004)?;

    let w_gate_q8 = quantize_q8_0_canonical_from_f32(&ctx, k, n, &w_gate.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;
    let w_up_q8 = quantize_q8_0_canonical_from_f32(&ctx, k, n, &w_up.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;

    let x_normed = ctx.call::<RMSNormOp>((x.clone(), gamma.clone(), k as u32), None)?;
    let y_ref = ctx.call::<MatmulGemvQ8SwiGluOp>(
        (
            &x_normed,
            (&QuantizedTensor::Q8_0(&w_gate_q8), &QuantizedTensor::Q8_0(&w_up_q8)),
            (Some(&b_gate), Some(&b_up)),
        ),
        None,
    )?;
    let y_fused = ctx.call::<MatmulGemvQ8SwiGluRmsnormOp>(
        (
            &x,
            &gamma,
            (&QuantizedTensor::Q8_0(&w_gate_q8), &QuantizedTensor::Q8_0(&w_up_q8)),
            (Some(&b_gate), Some(&b_up)),
        ),
        None,
    )?;
    ctx.synchronize();

    let ref_vals = y_ref.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let fused_vals = y_fused.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    assert_close(&ref_vals, &fused_vals, 0.15);
    Ok(())
}

#[test]
#[serial]
fn dense_gemv_rmsnorm_parity() -> Result<(), MetalError> {
    let mut ctx = Context::<crate::tensor::F16>::new()?;
    let k = 64usize;
    let n = 64usize;

    let x = make_fp16_tensor(&mut ctx, vec![1, k], 0xDEAD_BEEF)?;
    let gamma = make_fp16_tensor(&mut ctx, vec![k], 0xCAF0_BABE)?;
    let w = make_fp16_tensor(&mut ctx, vec![k, n], 0x1111_1111)?;
    let bias = make_fp16_tensor(&mut ctx, vec![n], 0x2222_2222)?;

    let x_normed = ctx.call::<RMSNormOp>((x.clone(), gamma.clone(), k as u32), None)?;
    let y_ref = ctx.call::<MatmulGemvOp>((&x_normed, TensorType::Dense(&w), true, Some(&bias)), None)?;
    let y_fused = ctx.call::<MatmulGemvRmsnormOp>((&x, &gamma, TensorType::Dense(&w), true, Some(&bias)), None)?;
    ctx.synchronize();

    let ref_vals = y_ref.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let fused_vals = y_fused.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    assert_close(&ref_vals, &fused_vals, 0.01);
    Ok(())
}
