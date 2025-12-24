//! Comprehensive SwiGLU Fused Activation Test Suite for Foundry.
//!
//! Tests the SwiGLU fused activation kernel against legacy and CPU reference.

use half::f16;
use metallic::{
    Context, F16Element, foundry::{Foundry, storage::Pooled, tensor::Tensor as FoundryTensor}, kernels::swiglu::SwiGLUFusedActivationOp, metals::swiglu::{SwigluFusedActivation, SwigluParams}, tensor::{F16, Tensor, TensorInit, TensorStorage as LegacyStorage}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

const TOLERANCE: f32 = 5e-3; // SwiGLU has activation, small tolerance for f16

// ============================================================================
// Test Configuration
// ============================================================================

struct TestConfig {
    batch: usize,
    hidden_dim: usize,
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================

fn cpu_swiglu_fused(gate: &[f16], up: &[f16], gate_bias: &[f16], up_bias: &[f16], batch: usize, hidden_dim: usize) -> Vec<f16> {
    let total = batch * hidden_dim;
    let mut output = vec![f16::from_f32(0.0); total];

    for gid in 0..total {
        let row = gid / hidden_dim;
        let col = gid % hidden_dim;

        let gate_val = gate[row * hidden_dim + col].to_f32() + gate_bias[col].to_f32();
        let up_val = up[row * hidden_dim + col].to_f32() + up_bias[col].to_f32();

        // SiLU: x * sigmoid(x)
        let sigmoid = 1.0 / (1.0 + (-gate_val).exp());
        let activated = gate_val * sigmoid;
        let result = activated * up_val;

        output[gid] = f16::from_f32(result);
    }

    output
}

// ============================================================================
// Parity Test Helper
// ============================================================================

fn run_parity_test(cfg: TestConfig) {
    let mut ctx = Context::<F16Element>::new().unwrap();
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rng();

    let total_elements = cfg.batch * cfg.hidden_dim;

    // Generate random data
    let gate_data: Vec<f16> = (0..total_elements).map(|_| f16::from_f32(rng.random_range(-2.0..2.0))).collect();
    let up_data: Vec<f16> = (0..total_elements).map(|_| f16::from_f32(rng.random_range(-2.0..2.0))).collect();
    let gate_bias_data: Vec<f16> = (0..cfg.hidden_dim).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();
    let up_bias_data: Vec<f16> = (0..cfg.hidden_dim).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();

    // =========================================================================
    // Legacy Kernel
    // =========================================================================
    let gate_legacy = Tensor::<F16>::new(
        vec![cfg.batch, cfg.hidden_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&gate_data),
    )
    .unwrap();
    let up_legacy = Tensor::<F16>::new(
        vec![cfg.batch, cfg.hidden_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&up_data),
    )
    .unwrap();
    let gate_bias_legacy = Tensor::<F16>::new(
        vec![cfg.hidden_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&gate_bias_data),
    )
    .unwrap();
    let up_bias_legacy = Tensor::<F16>::new(
        vec![cfg.hidden_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&up_bias_data),
    )
    .unwrap();

    let out_legacy = ctx
        .call::<SwiGLUFusedActivationOp>(
            (
                gate_legacy,
                gate_bias_legacy,
                up_legacy,
                up_bias_legacy,
                cfg.hidden_dim as u32, // gate_leading_stride
                cfg.hidden_dim as u32, // up_leading_stride
            ),
            None,
        )
        .unwrap();
    let legacy_result = out_legacy.to_vec();

    // =========================================================================
    // Foundry Kernel
    // =========================================================================
    let gate_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![total_elements], TensorInit::CopyFrom(&gate_data)).unwrap();
    let up_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![total_elements], TensorInit::CopyFrom(&up_data)).unwrap();
    let gate_bias_foundry =
        FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.hidden_dim], TensorInit::CopyFrom(&gate_bias_data)).unwrap();
    let up_bias_foundry =
        FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.hidden_dim], TensorInit::CopyFrom(&up_bias_data)).unwrap();

    let vector_width = if cfg.hidden_dim % 4 == 0 { 4 } else { 1 };
    let params = SwigluParams {
        total_elements: total_elements as u32,
        bias_len: cfg.hidden_dim as u32,
        vector_width,
        gate_leading_stride: cfg.hidden_dim as u32,
        up_leading_stride: cfg.hidden_dim as u32,
    };

    let gate_arg = TensorArg::from_tensor(&gate_foundry);
    let up_arg = TensorArg::from_tensor(&up_foundry);
    let gate_bias_arg = TensorArg::from_tensor(&gate_bias_foundry);
    let up_bias_arg = TensorArg::from_tensor(&up_bias_foundry);

    let kernel = SwigluFusedActivation::new(&gate_arg, &up_arg, &gate_bias_arg, &up_bias_arg, params);
    foundry.run(&kernel).unwrap();

    let foundry_result = FoundryTensor::to_vec(&up_foundry, &foundry);

    // =========================================================================
    // CPU Reference
    // =========================================================================
    let cpu_result = cpu_swiglu_fused(&gate_data, &up_data, &gate_bias_data, &up_bias_data, cfg.batch, cfg.hidden_dim);

    // =========================================================================
    // Comparison
    // =========================================================================
    let mut legacy_vs_foundry: f32 = 0.0;
    let mut cpu_vs_foundry: f32 = 0.0;

    for i in 0..total_elements {
        legacy_vs_foundry = legacy_vs_foundry.max((legacy_result[i].to_f32() - foundry_result[i].to_f32()).abs());
        cpu_vs_foundry = cpu_vs_foundry.max((cpu_result[i].to_f32() - foundry_result[i].to_f32()).abs());
    }

    println!("\n[SwiGLU batch={} hidden={}]", cfg.batch, cfg.hidden_dim);
    println!("  Legacy vs Foundry max diff: {:.6}", legacy_vs_foundry);
    println!("  CPU vs Foundry max diff:    {:.6}", cpu_vs_foundry);

    assert!(legacy_vs_foundry <= TOLERANCE, "Legacy vs Foundry mismatch: {}", legacy_vs_foundry);
    assert!(cpu_vs_foundry <= TOLERANCE, "CPU vs Foundry mismatch: {}", cpu_vs_foundry);
}

// ============================================================================
// F16 Parity Tests
// ============================================================================

#[test]
#[serial]
fn test_swiglu_qwen05b() {
    // Qwen2.5-0.5B: hidden_dim = 4864
    run_parity_test(TestConfig {
        batch: 1,
        hidden_dim: 4864,
    });
}

#[test]
#[serial]
fn test_swiglu_small() {
    run_parity_test(TestConfig { batch: 1, hidden_dim: 256 });
}

#[test]
#[serial]
fn test_swiglu_medium() {
    run_parity_test(TestConfig {
        batch: 1,
        hidden_dim: 2048,
    });
}

#[test]
#[serial]
fn test_swiglu_batched() {
    run_parity_test(TestConfig {
        batch: 4,
        hidden_dim: 1024,
    });
}

#[test]
#[serial]
fn test_swiglu_non_aligned() {
    // Non-vectorizable (not divisible by 4)
    run_parity_test(TestConfig { batch: 1, hidden_dim: 257 });
}

#[test]
#[serial]
fn test_swiglu_large() {
    run_parity_test(TestConfig {
        batch: 8,
        hidden_dim: 4096,
    });
}

// ============================================================================
// Sanity Edge Cases (Varied Shapes & Batches)
// ============================================================================

#[test]
#[serial]
fn test_swiglu_1x257() {
    // Unaligned case
    run_parity_test(TestConfig { batch: 1, hidden_dim: 257 });
}

#[test]
#[serial]
fn test_swiglu_8x1024() {
    run_parity_test(TestConfig {
        batch: 8,
        hidden_dim: 1024,
    });
}
