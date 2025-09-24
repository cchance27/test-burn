use super::elemwise_mul::ElemwiseMul;
use super::silu::Silu;
use super::{Context, MetalError, Tensor};
use serde::{Deserialize, Serialize};
use std::fs;

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
struct Weights {
    gate_weight: Vec<f32>,
    up_weight: Vec<f32>,
    down_weight: Vec<f32>,
    shapes: ParameterShapes,
}

#[cfg(test)]
fn transpose_2d(data: &[f32], orig_rows: usize, orig_cols: usize) -> Vec<f32> {
    let new_rows = orig_cols;
    let new_cols = orig_rows;
    let mut transposed = vec![0.0; data.len()];
    for i in 0..new_rows {
        for j in 0..new_cols {
            transposed[i * new_cols + j] = data[j * orig_cols + i];
        }
    }
    transposed
}

/// SwiGLU implementation extracted from Qwen25 FFN block.
/// Computes: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
///
/// # Arguments
/// * `x_normed_flat` - Flattened input [m, d_model] where m = batch * seq
/// * `ffn_gate` - Gate projection weight [d_model, ff_dim] (row-major; transpose if from PyTorch [ff_dim, d_model])
/// * `ffn_up` - Up projection weight [d_model, ff_dim] (row-major; transpose if from PyTorch [ff_dim, d_model])
/// * `ffn_down` - Down projection weight [ff_dim, d_model] (row-major; transpose if from PyTorch [d_model, ff_dim])
/// * `ctx` - Metal context for operations
///
/// # Returns
/// Flat output [m, d_model] (reshape externally to [batch, seq, d_model])
pub fn swiglu(
    x_normed_flat: &Tensor,
    ffn_gate: &Tensor,
    ffn_up: &Tensor,
    ffn_down: &Tensor,
    ctx: &mut Context,
) -> Result<Tensor, MetalError> {
    // Ensure required pipelines
    super::silu::ensure_silu_pipeline(ctx)?;
    super::elemwise_mul::ensure_mul_pipeline(ctx)?;

    // gate_proj: [m, d_model] @ [d_model, ff_dim] -> [m, ff_dim]
    let gate_proj = ctx.matmul(x_normed_flat, ffn_gate, false, false)?;

    // up_proj: [m, d_model] @ [d_model, ff_dim] -> [m, ff_dim]
    let up_proj = ctx.matmul(x_normed_flat, ffn_up, false, false)?;

    // SiLU activation on gate_proj
    let gate_act = {
        let out = Tensor::create_tensor_pooled(gate_proj.dims().to_vec(), ctx)?;
        let op = Silu::new(
            gate_proj,
            out.clone(),
            ctx.silu_pipeline.as_ref().unwrap().clone(),
        )?;
        ctx.with_command_buffer(|cb, cache| cb.record(&op, cache))?;
        out
    };

    // Element-wise multiplication: SiLU(gate_proj) * up_proj -> [m, ff_dim]
    let hidden = {
        let out = Tensor::create_tensor_pooled(gate_act.dims().to_vec(), ctx)?;
        let op = ElemwiseMul::new(
            gate_act,
            up_proj,
            out.clone(),
            ctx.mul_pipeline.as_ref().unwrap().clone(),
        )?;
        ctx.with_command_buffer(|cb, cache| cb.record(&op, cache))?;
        out
    };

    // down_proj: [m, ff_dim] @ [ff_dim, d_model] -> [m, d_model]
    let ffn_output = ctx.matmul(&hidden, ffn_down, false, false)?;

    Ok(ffn_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metallic::Context;

    #[test]
    fn test_swiglu_small_uniform() -> Result<(), MetalError> {
        let mut ctx = Context::new()?;

        // Small dimensions matching PyTorch verification example
        let d_model: usize = 4;
        let ff_dim: usize = 8;
        let m: usize = 1;

        // Create uniform weights matching PyTorch small example
        // gate_weight all 0.1, shape [d_model, ff_dim]
        let gate_data: Vec<f32> = vec![0.1; d_model * ff_dim];
        let ffn_gate = Tensor::create_tensor_from_slice(&gate_data, vec![d_model, ff_dim], &ctx)?;

        // up_weight all 0.2
        let up_data: Vec<f32> = vec![0.2; d_model * ff_dim];
        let ffn_up = Tensor::create_tensor_from_slice(&up_data, vec![d_model, ff_dim], &ctx)?;

        // down_weight all 0.3, shape [ff_dim, d_model]
        let down_data: Vec<f32> = vec![0.3; ff_dim * d_model];
        let ffn_down = Tensor::create_tensor_from_slice(&down_data, vec![ff_dim, d_model], &ctx)?;

        // Input all 2.0, [m, d_model]
        let input_data: Vec<f32> = vec![2.0; m * d_model];
        let x_normed_flat = Tensor::create_tensor_from_slice(&input_data, vec![m, d_model], &ctx)?;

        let output = swiglu(&x_normed_flat, &ffn_gate, &ffn_up, &ffn_down, &mut ctx)?;

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
        let mut ctx = Context::new()?;

        let d_model: usize = 4;
        let ff_dim: usize = 8;
        let m: usize = 1;

        // Dummy zero weights
        let gate_data: Vec<f32> = vec![0.0; d_model * ff_dim];
        let ffn_gate = Tensor::create_tensor_from_slice(&gate_data, vec![d_model, ff_dim], &ctx)?;

        let up_data: Vec<f32> = vec![0.0; d_model * ff_dim];
        let ffn_up = Tensor::create_tensor_from_slice(&up_data, vec![d_model, ff_dim], &ctx)?;

        let down_data: Vec<f32> = vec![0.0; ff_dim * d_model];
        let ffn_down = Tensor::create_tensor_from_slice(&down_data, vec![ff_dim, d_model], &ctx)?;

        // Zero input
        let input_data: Vec<f32> = vec![0.0; m * d_model];
        let x_normed_flat = Tensor::create_tensor_from_slice(&input_data, vec![m, d_model], &ctx)?;

        let output = swiglu(&x_normed_flat, &ffn_gate, &ffn_up, &ffn_down, &mut ctx)?;

        let output_slice = output.as_slice();
        for &val in output_slice {
            assert!((val - 0.0).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_swiglu_pytorch_data() -> Result<(), MetalError> {
        // Load weights from JSON
        let weights_json = fs::read_to_string("pytorch/swiglu_qwen25_weights_full.json")
            .expect("Failed to read weights JSON");
        let weights: Weights =
            serde_json::from_str(&weights_json).expect("Failed to parse weights JSON");

        let d_model = weights.shapes.gate_weight[1]; // 896
        let ff_dim = weights.shapes.gate_weight[0]; // 4864

        // Transpose PyTorch weights to Rust format [d_model, ff_dim] for gate/up, [ff_dim, d_model] for down
        let gate_weight_py = weights.gate_weight;
        let up_weight_py = weights.up_weight;
        let down_weight_py = weights.down_weight;

        let gate_weight_rust = transpose_2d(&gate_weight_py, ff_dim, d_model);
        let up_weight_rust = transpose_2d(&up_weight_py, ff_dim, d_model);
        let down_weight_rust = transpose_2d(&down_weight_py, d_model, ff_dim);

        // Create weight Tensors
        let mut ctx = Context::new()?;
        let ffn_gate =
            Tensor::create_tensor_from_slice(&gate_weight_rust, vec![d_model, ff_dim], &ctx)?;
        let ffn_up =
            Tensor::create_tensor_from_slice(&up_weight_rust, vec![d_model, ff_dim], &ctx)?;
        let ffn_down =
            Tensor::create_tensor_from_slice(&down_weight_rust, vec![ff_dim, d_model], &ctx)?;

        // Load test cases
        let cases_json = fs::read_to_string("pytorch/swiglu_full_qwen25_comparison_data.json")
            .expect("Failed to read cases JSON");
        let cases: Vec<TestCase> =
            serde_json::from_str(&cases_json).expect("Failed to parse cases JSON");

        for case in cases {
            let input_len = case.input.len();
            let m = input_len / d_model;
            assert_eq!(
                input_len,
                m * d_model,
                "Input length mismatch for {}",
                case.name
            );

            // Create input Tensor [m, d_model]
            let x_normed_flat =
                Tensor::create_tensor_from_slice(&case.input, vec![m, d_model], &ctx)?;

            // Run swiglu
            let rust_output = swiglu(&x_normed_flat, &ffn_gate, &ffn_up, &ffn_down, &mut ctx)?;
            ctx.synchronize(); // Sync to ensure GPU ops complete before CPU read
            let rust_output_flat = rust_output.as_slice().to_vec();

            // Compare to PyTorch expected
            assert_eq!(
                rust_output_flat.len(),
                case.output.len(),
                "Output length mismatch for {}",
                case.name
            );
            for (i, (&rust_val, &py_val)) in
                rust_output_flat.iter().zip(case.output.iter()).enumerate()
            {
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
}
