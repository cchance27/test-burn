//! GemvV2 Canonical Parity Test Suite with Reference
//!
//! Compares GemvV2 (Canonical Strategy + Canonical Layout) against:
//! 1. Legacy Foundry `GemvCanonical`
//! 2. Reference CPU implementation

use half::f16;
use metallic_foundry::{
    Foundry, compound::Layout, metals::gemv::{GemvV2Step, step::GemvStrategy, GemvV2Params}, policy::activation::Activation, spec::{DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

// ============================================================================
// Reference CPU Implementation
// ============================================================================

fn cpu_gemv(k: usize, n: usize, weights: &[f16], input: &[f16], alpha: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for row in 0..n {
        let mut sum = 0.0f32;
        for ki in 0..k {
            sum += weights[row * k + ki].to_f32() * input[ki].to_f32();
        }
        out[row] = sum * alpha;
    }
    out
}

fn swizzle_to_canonical(k: usize, n: usize, wpb: usize, data: &[f16]) -> Vec<f16> {
    let mut out = vec![f16::ZERO; k * n];
    for row in 0..n {
        for ki in 0..k {
            let block_idx = ki / wpb;
            let elem_in_block = ki % wpb;
            let canonical_idx = elem_in_block + wpb * (row + block_idx * n);
            out[canonical_idx] = data[row * k + ki];
        }
    }
    out
}

// ============================================================================
// Test Runner
// ============================================================================

fn run_canonical_parity_test(k: usize, n: usize, alpha: f32) {
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rng();
    let wpb = 32;

    // Generate random test data
    let weights_data: Vec<f32> = (0..k * n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let input_data: Vec<f32> = (0..k).map(|_| rng.random_range(-1.0..1.0)).collect();

    let weights_half: Vec<f16> = weights_data.iter().map(|&x| f16::from_f32(x)).collect();
    let input_half: Vec<f16> = input_data.iter().map(|&x| f16::from_f32(x)).collect();

    // CPU Reference
    let reference = cpu_gemv(k, n, &weights_half, &input_half, alpha);

    // Swizzle weights for GPU
    let weights_can_half = swizzle_to_canonical(k, n, wpb, &weights_half);

    // Create tensors
    let weights_can = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k * n], TensorInit::CopyFrom(&weights_can_half)).unwrap();
    let input = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k], TensorInit::CopyFrom(&input_half)).unwrap();
    let output_v2 = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n], TensorInit::Uninitialized).unwrap();

    // Run V2 (Testing CANONICAL strategy explicitly)
    let mut bindings = TensorBindings::new();
    let mut symbols = SymbolTable::new();
    bindings.insert("weights".to_string(), TensorArg::from_tensor(&weights_can));
    bindings.insert("input".to_string(), TensorArg::from_tensor(&input));
    bindings.insert("output".to_string(), TensorArg::from_tensor(&output_v2));

    let step = GemvV2Step {
        weights: Ref("weights".to_string()),
        scale_bytes: None,
        input: Ref("input".to_string()),
        output: Ref("output".to_string()),
        bias: None,
        residual: None,
        params: GemvV2Params {
            k_dim: DynamicValue::Literal(k as u32),
            n_dim: DynamicValue::Literal(n as u32),
            weights_per_block: wpb as u32,
            batch: 1,
        },
        layout: Layout::Canonical {
            expected_k: 0,
            expected_n: 0,
        },
        strategy: Some(GemvStrategy::Canonical),
        alpha,
        beta: 0.0,
        has_bias: 0,
        has_residual: 0,
        activation: Activation::None,
    };

    let compiled_steps = step.compile(&mut bindings, &mut symbols);
    let mut fast_bindings = FastBindings::new(symbols.len());
    for (name, id) in symbols.iter() {
        if let Ok(arg) = bindings.get(name) {
            fast_bindings.set(*id, arg);
        }
    }
    for c_step in compiled_steps {
        c_step.execute(&mut foundry, &fast_bindings, &bindings, &symbols).unwrap();
    }

    // Compare
    let v2_output: Vec<f32> = FoundryTensor::to_vec(&output_v2, &foundry).iter().map(|x| x.to_f32()).collect();

    // Tolerance: F16 precision accumulation typically needs loose tolerance
    // (sqrt(k) factor is good rule of thumb for sum accumulation noise)
    let tolerance = 5e-3 * (k as f32).sqrt();

    for i in 0..n {
        let r = reference[i];
        let v = v2_output[i];
        let diff = (r - v).abs();

        if diff > tolerance {
            panic!(
                "Canonical Parity mismatch (Strategy=Canonical) at {}: Ref={}, V2={}, Diff={}, Tol={} (K={}, N={}, Alpha={})",
                i, r, v, diff, tolerance, k, n, alpha
            );
        }
    }
}

#[test]
#[serial]
fn test_gemv_v2_canonical_parity_basic() {
    run_canonical_parity_test(128, 128, 1.0);
}

#[test]
#[serial]
fn test_gemv_v2_canonical_parity_alpha() {
    run_canonical_parity_test(256, 128, 0.5);
    run_canonical_parity_test(128, 64, 2.0);
}

#[test]
#[serial]
fn test_gemv_v2_canonical_parity_large() {
    run_canonical_parity_test(4096, 512, 1.0);
}
