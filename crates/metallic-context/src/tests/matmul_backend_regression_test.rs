use half::f16;

use crate::{
    Context, F16Element, MetalError, Tensor, TensorElement, TensorInit, TensorStorage, kernels::matmul_gemv::{MatmulGemvOp, MatmulGemvSmallN8Op}, tensor::TensorType
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
    let generic = ctx.call::<MatmulGemvOp>((&a, TensorType::Dense(&b), false, None), None)?;
    ctx.synchronize();

    let diff = max_abs_diff(&small_n, &generic);
    assert!(diff <= 2e-3, "Small-N GEMV diverged from generic GEMV: max abs diff {:.6}", diff);

    Ok(())
}
