// Test GEMV V2 with SDPA-specific shapes using Foundry tensors (like gemv_v2_context_parity)
// This should pass if the kernel itself is correct.

use std::sync::Arc;

use half::f16;
use metallic_foundry::{
    Foundry, MetalError, compound::stages::Layout, metals::gemv::{GemvStrategy, GemvV2Args, get_gemv_v2_kernel, warp_dispatch_config}, policy::{activation::Activation, f16::PolicyF16}, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[test]
fn test_gemv_v2_sdpa_foundry_qk() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;
    let mut rng = StdRng::seed_from_u64(42);

    let head_dim = 64;
    let seq_len = 8;

    // RowMajor GEMV: weights [N, K] = [seq_len, head_dim], input [K], output [N]
    let n = seq_len;
    let k = head_dim;

    let q_data: Vec<f16> = (0..k).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();
    let k_data: Vec<f16> = (0..n * k).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();

    // Flat 1D for weights like gemv_v2_context_parity
    let q_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k], TensorInit::CopyFrom(&q_data))?;
    let k_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n * k], TensorInit::CopyFrom(&k_data))?;
    let output_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n], TensorInit::Uninitialized)?;

    let kernel = get_gemv_v2_kernel(Arc::new(PolicyF16), Layout::RowMajor, GemvStrategy::Vectorized, Activation::None);
    let dispatch = warp_dispatch_config(n as u32);

    let args = GemvV2Args {
        weights: TensorArg::from_tensor(&k_tensor),
        scale_bytes: TensorArg::from_tensor(&k_tensor), // dummy
        input: TensorArg::from_tensor(&q_tensor),
        output: TensorArg::from_tensor(&output_tensor),
        bias: TensorArg::from_tensor(&output_tensor), // dummy
        has_bias: 0,
        k_dim: k as u32,
        n_dim: n as u32,
        weights_per_block: 32,
        alpha: 1.0,
        residual: TensorArg::from_tensor(&output_tensor),
        has_residual: 0,
        beta: 0.0,
    };

    foundry.run(&kernel.bind(args, dispatch))?;

    // Verify
    let gpu_out: Vec<f16> = output_tensor.to_vec(&foundry);
    let gpu_out_f32: Vec<f32> = gpu_out.iter().map(|x| x.to_f32()).collect();

    let mut cpu_out = vec![0.0f32; n];
    // RowMajor: weights[row * K + k]
    for row in 0..n {
        let mut sum = 0.0;
        for kk in 0..k {
            sum += k_data[row * k + kk].to_f32() * q_data[kk].to_f32();
        }
        cpu_out[row] = sum;
    }

    let max_diff = max_diff(&gpu_out_f32, &cpu_out);
    println!("Foundry QK Max Diff: {}", max_diff);
    assert!(max_diff < 0.05, "Max diff {} exceeds tolerance", max_diff);

    Ok(())
}

#[test]
fn test_gemv_v2_sdpa_foundry_av() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;
    let mut rng = StdRng::seed_from_u64(42);

    let head_dim = 64;
    let seq_len = 8;

    // ColMajor GEMV: weights [K, N] = [seq_len, head_dim], input [K], output [N]
    // Probs @ V: Probs [seq] x V [seq, head_dim] -> Output [head_dim]
    // This is row-vector @ matrix = [1, K] @ [K, N] = [1, N]
    // ColMajor means weights[k * N + n]
    let k = seq_len;
    let n = head_dim;

    let probs_data: Vec<f16> = (0..k).map(|_| f16::from_f32(rng.random_range(0.0..1.0))).collect();
    let v_data: Vec<f16> = (0..k * n).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();

    let probs_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k], TensorInit::CopyFrom(&probs_data))?;
    let v_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k * n], TensorInit::CopyFrom(&v_data))?;
    let output_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n], TensorInit::Uninitialized)?;

    let kernel = get_gemv_v2_kernel(Arc::new(PolicyF16), Layout::ColMajor, GemvStrategy::Vectorized, Activation::None);
    let dispatch = warp_dispatch_config(n as u32);

    let args = GemvV2Args {
        weights: TensorArg::from_tensor(&v_tensor),
        scale_bytes: TensorArg::from_tensor(&v_tensor),
        input: TensorArg::from_tensor(&probs_tensor),
        output: TensorArg::from_tensor(&output_tensor),
        bias: TensorArg::from_tensor(&output_tensor),
        has_bias: 0,
        k_dim: k as u32,
        n_dim: n as u32,
        weights_per_block: 32,
        alpha: 1.0,
        residual: TensorArg::from_tensor(&output_tensor),
        has_residual: 0,
        beta: 0.0,
    };

    foundry.run(&kernel.bind(args, dispatch))?;

    // Verify
    let gpu_out: Vec<f16> = output_tensor.to_vec(&foundry);
    let gpu_out_f32: Vec<f32> = gpu_out.iter().map(|x| x.to_f32()).collect();

    let mut cpu_out = vec![0.0f32; n];
    // ColMajor: weights[k * N + n]
    for nn in 0..n {
        let mut sum = 0.0;
        for kk in 0..k {
            sum += v_data[kk * n + nn].to_f32() * probs_data[kk].to_f32();
        }
        cpu_out[nn] = sum;
    }

    let max_diff = max_diff(&gpu_out_f32, &cpu_out);
    println!("Foundry AV Max Diff: {}", max_diff);
    assert!(max_diff < 0.05, "Max diff {} exceeds tolerance", max_diff);

    Ok(())
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
