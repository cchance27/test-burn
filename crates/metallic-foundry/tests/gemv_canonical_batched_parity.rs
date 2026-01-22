use std::sync::Arc;

use half::f16;
use metallic_foundry::{
    Foundry, MetalError, compound::stages::Layout, metals::gemv::{GemvStrategy, GemvV2Args, get_gemv_v2_kernel, warp_dispatch_config_2d}, policy::{activation::Activation, f16::PolicyF16}, tensor::{Tensor as FoundryTensor, TensorInit}
};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[test]
fn test_gemv_canonical_batched_parity_m4() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    let m: usize = 4;
    let n: usize = 256;
    let k: usize = 384;
    let weights_per_block: usize = 32;
    let blocks_per_k = k.div_ceil(weights_per_block);

    // Input X: [m, k]
    let x_f16: Vec<f16> = (0..m * k).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();

    // Dense weights W: [n, k] (row-major in CPU space)
    let w_dense_f32: Vec<f32> = (0..n * k).map(|_| rng.random_range(-0.25f32..0.25f32)).collect();

    // Canonical packed weights buffer: blocks ordered by block_k then row (N), each block is 32 elements.
    let mut w_canon_f16 = vec![f16::from_f32(0.0); blocks_per_k * n * weights_per_block];
    for row in 0..n {
        for kk in 0..k {
            let blk = kk / weights_per_block;
            let in_blk = kk % weights_per_block;
            let idx = in_blk + weights_per_block * (row + blk * n);
            w_canon_f16[idx] = f16::from_f32(w_dense_f32[row * k + kk]);
        }
    }

    // CPU reference: Y = X @ W^T => [m, n] where y[t, row] = sum_k x[t,k] * w[row,k]
    let mut y_cpu = vec![0.0f32; m * n];
    for t in 0..m {
        for row in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x_f16[t * k + kk].to_f32() * w_dense_f32[row * k + kk];
            }
            y_cpu[t * n + row] = acc;
        }
    }

    // Upload tensors
    let x = FoundryTensor::<metallic_foundry::F16, metallic_foundry::storage::Pooled>::new(
        &mut foundry,
        vec![m, k],
        TensorInit::CopyFrom(&x_f16),
    )?;
    let w = FoundryTensor::<metallic_foundry::F16, metallic_foundry::storage::Pooled>::new(
        &mut foundry,
        vec![blocks_per_k * n * weights_per_block],
        TensorInit::CopyFrom(&w_canon_f16),
    )?;
    let y = FoundryTensor::<metallic_foundry::F16, metallic_foundry::storage::Pooled>::new(
        &mut foundry,
        vec![m, n],
        TensorInit::Uninitialized,
    )?;

    let kernel = get_gemv_v2_kernel(Arc::new(PolicyF16), Layout::Canonical, GemvStrategy::Canonical, Activation::None);
    let dispatch = warp_dispatch_config_2d(n as u32, m as u32);

    let y_arg = metallic_foundry::TensorArg::from_tensor(&y);
    let args = GemvV2Args {
        weights: metallic_foundry::TensorArg::from_tensor(&w),
        scale_bytes: metallic_foundry::TensorArg::from_tensor(&w), // Dummy for F16
        input: metallic_foundry::TensorArg::from_tensor(&x),
        output: y_arg.clone(),
        k_dim: k as u32,
        n_dim: n as u32,
        weights_per_block: weights_per_block as u32,
        bias: y_arg, // Dummy
        has_bias: 0,
        alpha: 1.0,
        residual: metallic_foundry::TensorArg::from_tensor(&y),
        has_residual: 0,
        beta: 0.0,
    };

    foundry.run(&kernel.bind_arc(args, dispatch))?;
    foundry.synchronize()?;

    let y_gpu: Vec<f16> = y.to_vec(&foundry);

    let mut max_diff = 0.0f32;
    for (g, c) in y_gpu.iter().zip(y_cpu.iter()) {
        max_diff = max_diff.max((g.to_f32() - c).abs());
    }
    assert!(max_diff < 0.25, "max diff too high: {}", max_diff);

    Ok(())
}
