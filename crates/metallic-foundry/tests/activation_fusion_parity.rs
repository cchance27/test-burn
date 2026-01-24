use std::sync::Arc;

use half::f16;
use metallic_foundry::{
    Foundry, MetalError, compound::Layout, metals::gemv::{GemvStrategy, GemvV2Args, get_gemv_v2_kernel, warp_dispatch_config}, policy::{activation::Activation, f16::PolicyF16}, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn run_activation_parity_test(
    n: usize,
    k: usize,
    layout: Layout,
    activation: Activation,
    ref_fn: fn(f32) -> f32,
) -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;
    let mut rng = StdRng::seed_from_u64(42);

    let x_data: Vec<f16> = (0..k).map(|_| f16::from_f32(rng.random_range(-2.0..2.0))).collect();
    let w_data: Vec<f16> = (0..n * k).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();

    let x_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k], TensorInit::CopyFrom(&x_data))?;
    let w_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n * k], TensorInit::CopyFrom(&w_data))?;
    let output_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n], TensorInit::Uninitialized)?;

    let kernel = get_gemv_v2_kernel(Arc::new(PolicyF16), layout, GemvStrategy::Vectorized, activation);
    let dispatch = warp_dispatch_config(n as u32);

    let args = GemvV2Args {
        weights: TensorArg::from_tensor(&w_tensor),
        scale_bytes: TensorArg::from_tensor(&w_tensor),
        input: TensorArg::from_tensor(&x_tensor),
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

    foundry.run(&kernel.bind_arc(args, dispatch))?;

    let gpu_out: Vec<f16> = output_tensor.to_vec(&foundry);
    let gpu_out_f32: Vec<f32> = gpu_out.iter().map(|x| x.to_f32()).collect();

    let mut cpu_out = vec![0.0f32; n];
    for row in 0..n {
        let mut sum = 0.0;
        for kk in 0..k {
            let w_val = match layout {
                Layout::RowMajor => w_data[row * k + kk].to_f32(),
                Layout::ColMajor => w_data[kk * n + row].to_f32(),
                _ => panic!("Unsupported layout"),
            };
            sum += w_val * x_data[kk].to_f32();
        }
        cpu_out[row] = ref_fn(sum);
    }

    for (g, c) in gpu_out_f32.iter().zip(cpu_out.iter()) {
        let diff = (g - c).abs();
        assert!(diff < 0.05, "Mismatch: GPU={} CPU={} diff={} at layout={:?}", g, c, diff, layout);
    }

    Ok(())
}

#[test]
fn test_gemv_fused_silu() -> Result<(), MetalError> {
    let silu = |x: f32| x / (1.0 + (-x).exp());
    run_activation_parity_test(128, 256, Layout::RowMajor, Activation::SiLU, silu)?;
    run_activation_parity_test(128, 256, Layout::ColMajor, Activation::SiLU, silu)
}

#[test]
fn test_gemv_fused_relu() -> Result<(), MetalError> {
    let relu = |x: f32| x.max(0.0);
    run_activation_parity_test(128, 256, Layout::RowMajor, Activation::ReLU, relu)?;
    run_activation_parity_test(128, 256, Layout::ColMajor, Activation::ReLU, relu)
}

#[test]
fn test_gemv_none() -> Result<(), MetalError> {
    let none = |x: f32| x;
    run_activation_parity_test(128, 256, Layout::RowMajor, Activation::None, none)?;
    run_activation_parity_test(128, 256, Layout::ColMajor, Activation::None, none)
}
