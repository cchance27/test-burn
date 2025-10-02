use crate::metallic::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage};
use std::fs;

use serde::{Deserialize, Serialize};

#[test]
fn test_swiglu_small_uniform() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    // Small dimensions matching PyTorch verification example
    let d_model: usize = 4;
    let ff_dim: usize = 8;
    let m: usize = 1;

    // Create uniform weights matching PyTorch small example
    // gate_weight all 0.1, shape [ff_dim, d_model]
    let gate_data: Vec<f32> = vec![0.1; ff_dim * d_model];
    let ffn_gate = Tensor::new(
        vec![ff_dim, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&gate_data),
    )?;

    // up_weight all 0.2
    let up_data: Vec<f32> = vec![0.2; ff_dim * d_model];
    let ffn_up = Tensor::new(
        vec![ff_dim, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&up_data),
    )?;

    // down_weight all 0.3, shape [d_model, ff_dim]
    let down_data: Vec<f32> = vec![0.3; d_model * ff_dim];
    let ffn_down = Tensor::new(
        vec![d_model, ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&down_data),
    )?;

    // Input all 2.0, [m, d_model]
    let input_data: Vec<f32> = vec![2.0; m * d_model];
    let x_normed_flat = Tensor::new(vec![m, d_model], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&input_data))?;

    // Bias tensors (zeros) for this small test
    let ffn_gate_bias = Tensor::new(
        vec![ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.0f32; ff_dim]),
    )?;
    let ffn_up_bias = Tensor::new(
        vec![ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.0f32; ff_dim]),
    )?;
    let ffn_down_bias = Tensor::new(
        vec![d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.0f32; d_model]),
    )?;
    let output = ctx.SwiGLU(
        &x_normed_flat,
        &ffn_gate,
        &ffn_gate_bias,
        &ffn_up,
        &ffn_up_bias,
        &ffn_down,
        &ffn_down_bias,
        None,
    )?;
    ctx.synchronize();

    // Expected approx 2.1196 (silu(0.8) ≈0.55198 * 1.6 * 0.3 * 8 ≈2.1196; slight FP diff OK)
    let expected = 2.1196_f32;
    let tol = 1e-3_f32; // Tolerant for Metal FP precision
    let output_slice = output.as_slice();
    assert_eq!(output_slice.len(), m * d_model);
    for &val in output_slice {
        assert!(
            (val - expected).abs() < tol,
            "Expected ≈{:.4}, got {:.4} (diff {:.6})",
            expected,
            val,
            (val - expected).abs()
        );
    }

    Ok(())
}

#[test]
fn test_swiglu_zero_input() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    let d_model: usize = 4;
    let ff_dim: usize = 8;
    let m: usize = 1;

    // Dummy zero weights
    let gate_data: Vec<f32> = vec![0.0; ff_dim * d_model];
    let ffn_gate = Tensor::new(
        vec![ff_dim, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&gate_data),
    )?;

    let up_data: Vec<f32> = vec![0.0; ff_dim * d_model];
    let ffn_up = Tensor::new(
        vec![ff_dim, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&up_data),
    )?;

    let down_data: Vec<f32> = vec![0.0; d_model * ff_dim];
    let ffn_down = Tensor::new(
        vec![d_model, ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&down_data),
    )?;

    // Biases (zeros)
    let ffn_gate_bias = Tensor::new(
        vec![ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.0f32; ff_dim]),
    )?;
    let ffn_up_bias = Tensor::new(
        vec![ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.0f32; ff_dim]),
    )?;
    let ffn_down_bias = Tensor::new(
        vec![d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.0f32; d_model]),
    )?;

    // Zero input
    let input_data: Vec<f32> = vec![0.0; m * d_model];
    let x_normed_flat = Tensor::new(vec![m, d_model], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&input_data))?;

    let output = ctx.SwiGLU(
        &x_normed_flat,
        &ffn_gate,
        &ffn_gate_bias,
        &ffn_up,
        &ffn_up_bias,
        &ffn_down,
        &ffn_down_bias,
        None,
    )?;

    let output_slice = output.as_slice();
    for &val in output_slice {
        assert!((val - 0.0).abs() < 1e-6);
    }

    Ok(())
}

#[test]
fn test_swiglu_scalar_fallback_path() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    let d_model: usize = 4;
    let ff_dim: usize = 6; // Not divisible by the vector width
    let m: usize = 1;

    let gate_data: Vec<f32> = vec![0.1; ff_dim * d_model];
    let ffn_gate = Tensor::new(
        vec![ff_dim, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&gate_data),
    )?;

    let up_data: Vec<f32> = vec![0.2; ff_dim * d_model];
    let ffn_up = Tensor::new(
        vec![ff_dim, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&up_data),
    )?;

    let down_data: Vec<f32> = vec![0.3; d_model * ff_dim];
    let ffn_down = Tensor::new(
        vec![d_model, ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&down_data),
    )?;

    let input_data: Vec<f32> = vec![2.0; m * d_model];
    let x_normed_flat = Tensor::new(vec![m, d_model], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&input_data))?;

    let ffn_gate_bias = Tensor::new(
        vec![ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.0f32; ff_dim]),
    )?;
    let ffn_up_bias = Tensor::new(
        vec![ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.0f32; ff_dim]),
    )?;
    let ffn_down_bias = Tensor::new(
        vec![d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.0f32; d_model]),
    )?;

    let output = ctx.SwiGLU(
        &x_normed_flat,
        &ffn_gate,
        &ffn_gate_bias,
        &ffn_up,
        &ffn_up_bias,
        &ffn_down,
        &ffn_down_bias,
        None,
    )?;
    ctx.synchronize();

    let expected = 1.5897012_f32;
    let tol = 1e-3_f32;
    let output_slice = output.as_slice();
    assert_eq!(output_slice.len(), m * d_model);
    for &val in output_slice {
        assert!(
            (val - expected).abs() < tol,
            "Expected ≈{:.4}, got {:.4} (diff {:.6})",
            expected,
            val,
            (val - expected).abs()
        );
    }

    Ok(())
}

#[test]
fn test_swiglu_fused_matches_unfused() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    let d_model: usize = 4;
    let ff_dim: usize = 8;
    let m: usize = 3;

    let gate_data: Vec<f32> = (0..ff_dim * d_model).map(|i| 0.05 + 0.01 * (i as f32)).collect();
    let up_data: Vec<f32> = (0..ff_dim * d_model).map(|i| -0.02 * (i as f32) + 0.15).collect();
    let down_data: Vec<f32> = (0..d_model * ff_dim).map(|i| 0.03 * (i as f32) - 0.4).collect();

    let ffn_gate = Tensor::new(
        vec![ff_dim, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&gate_data),
    )?;
    let ffn_up = Tensor::new(
        vec![ff_dim, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&up_data),
    )?;
    let ffn_down = Tensor::new(
        vec![d_model, ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&down_data),
    )?;

    let fused_data: Vec<f32> = gate_data.iter().copied().chain(up_data.iter().copied()).collect();
    let fused_gate_up = Tensor::new(
        vec![ff_dim * 2, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&fused_data),
    )?;

    let x_data: Vec<f32> = (0..m * d_model).map(|i| ((i % d_model) as f32) * 0.25 - 0.5).collect();
    let x_normed_flat = Tensor::new(vec![m, d_model], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&x_data))?;

    let ffn_gate_bias = Tensor::new(
        vec![ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.01f32; ff_dim]),
    )?;
    let ffn_up_bias = Tensor::new(
        vec![ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.02f32; ff_dim]),
    )?;
    let ffn_down_bias = Tensor::new(
        vec![d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![-0.03f32; d_model]),
    )?;

    let baseline = ctx.SwiGLU(
        &x_normed_flat,
        &ffn_gate,
        &ffn_gate_bias,
        &ffn_up,
        &ffn_up_bias,
        &ffn_down,
        &ffn_down_bias,
        None,
    )?;
    ctx.synchronize();
    let baseline_vals = baseline.to_vec();

    let fused = ctx.SwiGLU(
        &x_normed_flat,
        &ffn_gate,
        &ffn_gate_bias,
        &ffn_up,
        &ffn_up_bias,
        &ffn_down,
        &ffn_down_bias,
        Some(&fused_gate_up),
    )?;
    ctx.synchronize();
    let fused_vals = fused.as_slice();

    assert_eq!(baseline_vals.len(), fused_vals.len());
    for (expected, actual) in baseline_vals.iter().zip(fused_vals.iter()) {
        let diff = (expected - actual).abs();
        assert!(diff < 1e-4, "Mismatch between fused and unfused outputs: diff={diff}");
    }

    Ok(())
}

#[test]
fn test_swiglu_fused_scalar_path_matches_unfused() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    let d_model: usize = 4;
    let ff_dim: usize = 6; // Not divisible by vector width
    let m: usize = 2;

    let gate_data: Vec<f32> = (0..ff_dim * d_model).map(|i| 0.07 * (i as f32) - 0.25).collect();
    let up_data: Vec<f32> = (0..ff_dim * d_model).map(|i| 0.09 - 0.03 * (i as f32)).collect();
    let down_data: Vec<f32> = (0..d_model * ff_dim).map(|i| 0.02 * (i as f32) + 0.11).collect();

    let ffn_gate = Tensor::new(
        vec![ff_dim, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&gate_data),
    )?;
    let ffn_up = Tensor::new(
        vec![ff_dim, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&up_data),
    )?;
    let ffn_down = Tensor::new(
        vec![d_model, ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&down_data),
    )?;

    let fused_data: Vec<f32> = gate_data.iter().copied().chain(up_data.iter().copied()).collect();
    let fused_gate_up = Tensor::new(
        vec![ff_dim * 2, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&fused_data),
    )?;

    let x_data: Vec<f32> = (0..m * d_model).map(|i| (i as f32) * 0.11 - 0.6).collect();
    let x_normed_flat = Tensor::new(vec![m, d_model], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&x_data))?;

    let ffn_gate_bias = Tensor::new(
        vec![ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.005f32; ff_dim]),
    )?;
    let ffn_up_bias = Tensor::new(
        vec![ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![-0.01f32; ff_dim]),
    )?;
    let ffn_down_bias = Tensor::new(
        vec![d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.02f32; d_model]),
    )?;

    let baseline = ctx.SwiGLU(
        &x_normed_flat,
        &ffn_gate,
        &ffn_gate_bias,
        &ffn_up,
        &ffn_up_bias,
        &ffn_down,
        &ffn_down_bias,
        None,
    )?;
    ctx.synchronize();
    let baseline_vals = baseline.to_vec();

    let fused = ctx.SwiGLU(
        &x_normed_flat,
        &ffn_gate,
        &ffn_gate_bias,
        &ffn_up,
        &ffn_up_bias,
        &ffn_down,
        &ffn_down_bias,
        Some(&fused_gate_up),
    )?;
    ctx.synchronize();
    let fused_vals = fused.as_slice();

    assert_eq!(baseline_vals.len(), fused_vals.len());
    for (expected, actual) in baseline_vals.iter().zip(fused_vals.iter()) {
        let diff = (expected - actual).abs();
        assert!(diff < 1e-4, "Scalar fused mismatch: diff={diff}");
    }

    Ok(())
}

#[cfg(test)]
#[derive(Debug, Serialize, Deserialize)]
struct TestCase {
    test_id: usize,
    input_shape: Vec<usize>,
    input: Vec<f32>,
    output_shape: Vec<usize>,
    output: Vec<f32>,
    name: String,
    parameter_shapes: ParameterShapes,
}

#[cfg(test)]
#[derive(Debug, Serialize, Deserialize)]
struct ParameterShapes {
    gate_weight: Vec<usize>,
    up_weight: Vec<usize>,
    down_weight: Vec<usize>,
}

#[cfg(test)]
#[derive(Debug, Serialize, Deserialize)]
struct SwiGluWeights {
    gate_weight: Vec<f32>,
    up_weight: Vec<f32>,
    down_weight: Vec<f32>,
    shapes: ParameterShapes,
}

#[test]
fn test_swiglu_pytorch_data() -> Result<(), MetalError> {
    // Load weights from JSON
    let weights_json = fs::read_to_string("pytorch/swiglu_qwen25_weights_full.json").expect("Failed to read weights JSON");
    let weights: SwiGluWeights = serde_json::from_str(&weights_json).expect("Failed to parse weights JSON");

    let d_model = weights.shapes.gate_weight[1]; // 896
    let ff_dim = weights.shapes.gate_weight[0]; // 4864

    // Transpose PyTorch weights to Rust format [d_model, ff_dim] for gate/up, [ff_dim, d_model] for down
    let gate_weight_py = weights.gate_weight;
    let up_weight_py = weights.up_weight;
    let down_weight_py = weights.down_weight;

    let gate_weight_rust = gate_weight_py; // already [ff_dim, d_model]
    let up_weight_rust = up_weight_py; // already [ff_dim, d_model]
    let down_weight_rust = down_weight_py; // already [d_model, ff_dim]

    // Create weight Tensors
    let mut ctx = Context::<F32Element>::new()?;
    let ffn_gate = Tensor::new(
        vec![ff_dim, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&gate_weight_rust),
    )?;
    let ffn_up = Tensor::new(
        vec![ff_dim, d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&up_weight_rust),
    )?;
    let ffn_down = Tensor::new(
        vec![d_model, ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&down_weight_rust),
    )?;

    // Bias tensors (assume zero since JSON doesn't include biases)
    let ffn_gate_bias = Tensor::new(
        vec![ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.0f32; ff_dim]),
    )?;
    let ffn_up_bias = Tensor::new(
        vec![ff_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.0f32; ff_dim]),
    )?;
    let ffn_down_bias = Tensor::new(
        vec![d_model],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&vec![0.0f32; d_model]),
    )?;

    // Load test cases
    let cases_json = fs::read_to_string("pytorch/swiglu_full_qwen25_comparison_data.json").expect("Failed to read cases JSON");
    let cases: Vec<TestCase> = serde_json::from_str(&cases_json).expect("Failed to parse cases JSON");

    for case in cases {
        let input_len = case.input.len();
        let m = input_len / d_model;
        assert_eq!(input_len, m * d_model, "Input length mismatch for {}", case.name);

        // Create input Tensor [m, d_model]
        let x_normed_flat = Tensor::new(vec![m, d_model], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&case.input))?;

        // Run swiglu
        let rust_output = ctx.SwiGLU(
            &x_normed_flat,
            &ffn_gate,
            &ffn_gate_bias,
            &ffn_up,
            &ffn_up_bias,
            &ffn_down,
            &ffn_down_bias,
            None,
        )?;
        ctx.synchronize(); // Sync to ensure GPU ops complete before CPU read
        let rust_output_flat = rust_output.to_vec();

        // Compare to PyTorch expected
        assert_eq!(
            rust_output_flat.len(),
            case.output.len(),
            "Output length mismatch for {}",
            case.name
        );
        for (i, (&rust_val, &py_val)) in rust_output_flat.iter().zip(case.output.iter()).enumerate() {
            let diff = (rust_val - py_val).abs();
            if diff > 1e-5 {
                panic!(
                    "Mismatch at index {} for {}: Rust {:.7} vs PyTorch {:.7} (diff {:.7})",
                    i, case.name, rust_val, py_val, diff
                );
            }
        }
    }

    Ok(())
}
