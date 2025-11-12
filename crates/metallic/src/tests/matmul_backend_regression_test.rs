use half::f16;

use crate::{
    Context, F16Element, MetalError, Tensor, TensorElement, TensorInit, TensorStorage, kernels::{
        elemwise_add::BroadcastElemwiseAddOp, matmul_gemv::{MatmulGemvOp, MatmulGemvSmallN8Op}, matmul_mlx::MatMulMlxOp, matmul_mps::MatMulMpsOp
    }, tensor::TensorType
};

fn make_tensor<T: TensorElement>(ctx: &mut Context<T>, dims: Vec<usize>, data: Vec<T::Scalar>) -> Result<Tensor<T>, MetalError> {
    Tensor::new(dims, TensorStorage::Dedicated(&*ctx), TensorInit::CopyFrom(&data))
}

fn deterministic_f16(len: usize, seed: u32) -> Vec<f16> {
    (0..len)
        .map(|i| {
            let raw = ((i as u64 * 1_664_525_u64) + (seed as u64 * 1_013_904_223_u64)) & 0xFFFF_FFFF;
            let v = (raw as f32 / 4_294_967_295.0) * 2.0 - 1.0;
            f16::from_f32(v)
        })
        .collect()
}

fn max_abs_diff(a: &Tensor<F16Element>, b: &Tensor<F16Element>) -> f32 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(lhs, rhs)| (f16::to_f32(*lhs) - f16::to_f32(*rhs)).abs())
        .fold(0.0_f32, f32::max)
}

/// Compares MatMulMlxOp (modern path) against MatMulMpsOp (legacy baseline) for a representative
/// FFN gate projection shape. Any divergence here indicates a regression in the MLX kernel or
/// dispatcher routing.
#[test]
fn mlx_gate_projection_matches_mps_baseline() -> Result<(), MetalError> {
    const M: usize = 37;
    const K: usize = 896;
    const N: usize = 4864;

    let mut ctx = Context::<F16Element>::new()?;

    let x = make_tensor(&mut ctx, vec![M, K], deterministic_f16(M * K, 1))?;
    let w = make_tensor(&mut ctx, vec![N, K], deterministic_f16(N * K, 7))?;
    let bias = make_tensor(&mut ctx, vec![N], deterministic_f16(N, 13))?;

    // MLX path (transpose_right=true so we treat W as (K, N))
    let mlx_out = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&w), None, None, false, true, 1.0, 0.0), None)?;
    let mlx_out = ctx.call::<BroadcastElemwiseAddOp>((mlx_out, bias.clone()), None)?;

    // Legacy baseline: MPS matmul + bias add
    let mps_out = ctx.call::<MatMulMpsOp>((&x, &w, false, true), None)?;
    let mps_out = ctx.call::<BroadcastElemwiseAddOp>((mps_out, bias), None)?;

    ctx.synchronize();

    let diff = max_abs_diff(&mlx_out, &mps_out);
    assert!(diff <= 5e-3, "MLX vs MPS mismatch for gate projection: max abs diff {:.6}", diff);

    Ok(())
}

/// Validates that the small-N GEMV kernel produces identical results to the generic GEMV baseline.
/// This protects the new dispatcher route that prefers the specialized kernel.
#[test]
fn small_n8_gemv_matches_generic_gemv() -> Result<(), MetalError> {
    const M: usize = 1;
    const K: usize = 32;
    const N: usize = 8;

    let mut ctx = Context::<F16Element>::new()?;

    let a = make_tensor(&mut ctx, vec![M, K], deterministic_f16(M * K, 23))?;
    let b = make_tensor(&mut ctx, vec![K, N], deterministic_f16(K * N, 31))?;

    let small_n = ctx.call::<MatmulGemvSmallN8Op>((&a, &b), None)?;
    let generic = ctx.call::<MatmulGemvOp>((&a, TensorType::Dense(&b), None), None)?;
    ctx.synchronize();

    let diff = max_abs_diff(&small_n, &generic);
    assert!(diff <= 2e-3, "Small-N GEMV diverged from generic GEMV: max abs diff {:.6}", diff);

    Ok(())
}

/// Regression guard for attention o_proj: verifies MatMulMlxOp matches MatMulMpsOp for the
/// head-projected attention output matrix multiply used in the attention block.
#[test]
fn mlx_attention_oproj_matches_mps_baseline() -> Result<(), MetalError> {
    const BATCH: usize = 1;
    const SEQ: usize = 37;
    const D_MODEL: usize = 896;

    let mut ctx = Context::<F16Element>::new()?;

    // Simulated attention heads output after SDPA, before projection: [batch, seq, d_model]
    let attn_heads = make_tensor(&mut ctx, vec![BATCH, SEQ, D_MODEL], deterministic_f16(BATCH * SEQ * D_MODEL, 97))?;

    // Attention output weight: [d_model, d_model]
    let attn_weight = make_tensor(&mut ctx, vec![D_MODEL, D_MODEL], deterministic_f16(D_MODEL * D_MODEL, 103))?;

    // Flatten input to [seq, d_model] to match how the model uses it.
    let attn_flat = attn_heads.reshape(vec![SEQ, D_MODEL])?;

    let mlx_proj = ctx.call::<MatMulMlxOp>(
        (&attn_flat, TensorType::Dense(&attn_weight), None, None, false, true, 1.0, 0.0),
        None,
    )?;
    let mps_proj = ctx.call::<MatMulMpsOp>((&attn_flat, &attn_weight, false, true), None)?;

    ctx.synchronize();

    let diff = max_abs_diff(&mlx_proj, &mps_proj);
    assert!(diff <= 5e-3, "MLX vs MPS mismatch for attention o_proj: max abs diff {:.6}", diff);

    Ok(())
}
