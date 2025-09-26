use super::{Context, MetalError, Tensor};
use crate::metallic::kernels::elemwise_add::BroadcastElemwiseAddOp;
use crate::metallic::kernels::silu::SiluOp;
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

/// SwiGLU implementation extracted from Qwen25 FFN block.
/// Computes: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
///
/// # Arguments
/// * `x_normed_flat` - Flattened input [m, d_model] where m = batch * seq
/// * `ffn_gate` - Gate projection weight [ff_dim, d_model] (row-major; transpose if source stored as [d_model, ff_dim])
/// * `ffn_up` - Up projection weight [ff_dim, d_model] (row-major; transpose if source stored as [d_model, ff_dim])
/// * `ffn_down` - Down projection weight [d_model, ff_dim] (row-major; transpose if source stored as [ff_dim, d_model])
/// * `ctx` - Metal context for operations
///
/// # Returns
/// Flat output [m, d_model] (reshape externally to [batch, seq, d_model])
#[allow(clippy::too_many_arguments)]
pub fn swiglu(
    x_normed_flat: &Tensor,
    ffn_gate: &Tensor,
    ffn_gate_bias: &Tensor,
    ffn_up: &Tensor,
    ffn_up_bias: &Tensor,
    ffn_down: &Tensor,
    ffn_down_bias: &Tensor,
    ctx: &mut Context,
) -> Result<Tensor, MetalError> {
    // gate_proj: [m, d_model] @ weight -> [m, ff_dim]
    // Choose transpose_b based on weight layout to ensure b_rows == d_model
    let d_model = x_normed_flat.dims()[1];
    let gate_dims = ffn_gate.dims();
    let gate_transpose_b = if gate_dims[0] == d_model {
        false
    } else if gate_dims[1] == d_model {
        true
    } else {
        return Err(MetalError::DimensionMismatch {
            expected: d_model,
            actual: gate_dims[0],
        });
    };
    let gate_temp = ctx.matmul(x_normed_flat, ffn_gate, false, gate_transpose_b)?;
    ctx.synchronize();
    // Add gate bias (broadcast over last dim)
    let gate_out = ctx.call::<BroadcastElemwiseAddOp>((gate_temp, ffn_gate_bias.clone()))?;
    ctx.synchronize();
    //println!("SWIGLU DEBUG: gate_proj dims={:?}", gate_out.dims());
    //{
    //    let s = gate_out.as_slice();
    //    //let n = std::cmp::min(10, s.len());
    //    //println!("SWIGLU DEBUG: gate_proj first {}: {:?}", n, &s[0..n]);
    //    // Dump full gate_proj for offline analysis
    //    let _ = std::fs::create_dir_all("debug_dumps");
    //    let mut content = format!("{:?}\n", gate_out.dims());
    //    for &v in s {
    //        content.push_str(&format!("{:.8},", v));
    //    }
    //    //let _ = std::fs::write("debug_dumps/gate_proj.txt", content);
    //}

    // up_proj: [m, d_model] @ weight -> [m, ff_dim]
    // Choose transpose_b based on weight layout to ensure b_rows == d_model
    let up_dims = ffn_up.dims();
    let up_transpose_b = if up_dims[0] == d_model {
        false
    } else if up_dims[1] == d_model {
        true
    } else {
        return Err(MetalError::DimensionMismatch {
            expected: d_model,
            actual: up_dims[0],
        });
    };
    let up_temp = ctx.matmul(x_normed_flat, ffn_up, false, up_transpose_b)?;
    ctx.synchronize();
    // Add up bias
    let up_out = ctx.call::<BroadcastElemwiseAddOp>((up_temp, ffn_up_bias.clone()))?;
    ctx.synchronize();
    //println!("SWIGLU DEBUG: up_proj dims={:?}", up_out.dims());
    //{
    //    let s = up_out.as_slice();
    //    //let n = std::cmp::min(10, s.len());
    //    //println!("SWIGLU DEBUG: up_proj first {}: {:?}", n, &s[0..n]);
    //    // Dump full up_proj for offline analysis
    //    let _ = std::fs::create_dir_all("debug_dumps");
    //    let mut content = format!("{:?}\n", up_out.dims());
    //    for &v in s {
    //        content.push_str(&format!("{:.8},", v));
    //    }
    //    let _ = std::fs::write("debug_dumps/up_proj.txt", content);
    //}

    // SiLU activation on gate_proj
    let gate_act = ctx.call::<SiluOp>(gate_out)?;
    ctx.synchronize();
    //println!("SWIGLU DEBUG: gate_act dims={:?}", gate_act.dims());
    {
        //let s = gate_act.as_slice();
        //let n = std::cmp::min(10, s.len());
        //println!("SWIGLU DEBUG: gate_act first {}: {:?}", n, &s[0..n]);
    }

    // Element-wise multiplication: SiLU(gate_proj) * up_proj -> [m, ff_dim]
    let hidden = ctx.call::<crate::metallic::kernels::elemwise_mul::ElemwiseMulOp>((gate_act, up_out))?;
    ctx.synchronize();
    //println!("SWIGLU DEBUG: hidden dims={:?}", hidden.dims());
    //{
    //    let s = hidden.as_slice();
    //    //let n = std::cmp::min(10, s.len());
    //    //println!("SWIGLU DEBUG: hidden first {}: {:?}", n, &s[0..n]);
    //    // Dump full hidden for offline analysis
    //    let _ = std::fs::create_dir_all("debug_dumps");
    //    let mut content = format!("{:?}\n", hidden.dims());
    //    for &v in s {
    //        content.push_str(&format!("{:.8},", v));
    //    }
    //    let _ = std::fs::write("debug_dumps/hidden.txt", content);
    //}

    // down_proj: [m, ff_dim] @ [ff_dim, d_model] -> [m, d_model]
    // Choose transpose_b based on weight layout so that b_rows == hidden_cols (ff_dim)
    let hidden_cols = hidden.dims()[1];
    let ffn_down_rows = ffn_down.dims()[0];
    let ffn_down_cols = ffn_down.dims()[1];
    //println!(
    //    "SWIGLU DEBUG: down_proj matmul shapes: hidden_cols={}, ffn_down_rows={}, ffn_down_cols={}",
    //    hidden_cols, ffn_down_rows, ffn_down_cols
    //);
    let ffn_temp = if hidden_cols == ffn_down_rows {
        // Hidden [m, ff_dim] @ ffn_down [ff_dim, d_model] -> [m, d_model]
        ctx.matmul(&hidden, ffn_down, false, false)?
    } else if hidden_cols == ffn_down_cols {
        // Hidden [m, ff_dim] @ ffn_down^T [ff_dim, d_model] where stored as [d_model, ff_dim]
        ctx.matmul(&hidden, ffn_down, false, true)?
    } else {
        return Err(MetalError::DimensionMismatch {
            expected: hidden_cols,
            actual: ffn_down_rows,
        });
    };
    ctx.synchronize();
    // Add down bias to final projection output
    let ffn_out = ctx.call::<BroadcastElemwiseAddOp>((ffn_temp, ffn_down_bias.clone()))?;
    ctx.synchronize();
    //println!("SWIGLU DEBUG: ffn_output dims={:?}", ffn_out.dims());
    //{
    //    let s = ffn_out.as_slice();
    //    //let n = std::cmp::min(10, s.len());
    //    //println!("SWIGLU DEBUG: ffn_output first {}: {:?}", n, &s[0..n]);
    //    // Dump full ffn_output for offline analysis
    //    let _ = std::fs::create_dir_all("debug_dumps");
    //    let mut content = format!("{:?}\n", ffn_out.dims());
    //    for &v in s {
    //        content.push_str(&format!("{:.8},", v));
    //    }
    //    let _ = std::fs::write("debug_dumps/ffn_output.txt", content);
    //}

    Ok(ffn_out)
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
        // gate_weight all 0.1, shape [ff_dim, d_model]
        let gate_data: Vec<f32> = vec![0.1; ff_dim * d_model];
        let ffn_gate = Tensor::create_tensor_from_slice(&gate_data, vec![ff_dim, d_model], &ctx)?;

        // up_weight all 0.2
        let up_data: Vec<f32> = vec![0.2; ff_dim * d_model];
        let ffn_up = Tensor::create_tensor_from_slice(&up_data, vec![ff_dim, d_model], &ctx)?;

        // down_weight all 0.3, shape [d_model, ff_dim]
        let down_data: Vec<f32> = vec![0.3; d_model * ff_dim];
        let ffn_down = Tensor::create_tensor_from_slice(&down_data, vec![d_model, ff_dim], &ctx)?;

        // Input all 2.0, [m, d_model]
        let input_data: Vec<f32> = vec![2.0; m * d_model];
        let x_normed_flat = Tensor::create_tensor_from_slice(&input_data, vec![m, d_model], &ctx)?;

        // Bias tensors (zeros) for this small test
        let ffn_gate_bias = Tensor::create_tensor_from_slice(&vec![0.0f32; ff_dim], vec![ff_dim], &ctx)?;
        let ffn_up_bias = Tensor::create_tensor_from_slice(&vec![0.0f32; ff_dim], vec![ff_dim], &ctx)?;
        let ffn_down_bias = Tensor::create_tensor_from_slice(&vec![0.0f32; d_model], vec![d_model], &ctx)?;
        let output = swiglu(
            &x_normed_flat,
            &ffn_gate,
            &ffn_gate_bias,
            &ffn_up,
            &ffn_up_bias,
            &ffn_down,
            &ffn_down_bias,
            &mut ctx,
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
        let mut ctx = Context::new()?;

        let d_model: usize = 4;
        let ff_dim: usize = 8;
        let m: usize = 1;

        // Dummy zero weights
        let gate_data: Vec<f32> = vec![0.0; ff_dim * d_model];
        let ffn_gate = Tensor::create_tensor_from_slice(&gate_data, vec![ff_dim, d_model], &ctx)?;

        let up_data: Vec<f32> = vec![0.0; ff_dim * d_model];
        let ffn_up = Tensor::create_tensor_from_slice(&up_data, vec![ff_dim, d_model], &ctx)?;

        let down_data: Vec<f32> = vec![0.0; d_model * ff_dim];
        let ffn_down = Tensor::create_tensor_from_slice(&down_data, vec![d_model, ff_dim], &ctx)?;

        // Biases (zeros)
        let ffn_gate_bias = Tensor::create_tensor_from_slice(&vec![0.0f32; ff_dim], vec![ff_dim], &ctx)?;
        let ffn_up_bias = Tensor::create_tensor_from_slice(&vec![0.0f32; ff_dim], vec![ff_dim], &ctx)?;
        let ffn_down_bias = Tensor::create_tensor_from_slice(&vec![0.0f32; d_model], vec![d_model], &ctx)?;

        // Zero input
        let input_data: Vec<f32> = vec![0.0; m * d_model];
        let x_normed_flat = Tensor::create_tensor_from_slice(&input_data, vec![m, d_model], &ctx)?;

        let output = swiglu(
            &x_normed_flat,
            &ffn_gate,
            &ffn_gate_bias,
            &ffn_up,
            &ffn_up_bias,
            &ffn_down,
            &ffn_down_bias,
            &mut ctx,
        )?;

        let output_slice = output.as_slice();
        for &val in output_slice {
            assert!((val - 0.0).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_swiglu_pytorch_data() -> Result<(), MetalError> {
        // Load weights from JSON
        let weights_json = fs::read_to_string("pytorch/swiglu_qwen25_weights_full.json").expect("Failed to read weights JSON");
        let weights: Weights = serde_json::from_str(&weights_json).expect("Failed to parse weights JSON");

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
        let mut ctx = Context::new()?;
        let ffn_gate = Tensor::create_tensor_from_slice(&gate_weight_rust, vec![ff_dim, d_model], &ctx)?;
        let ffn_up = Tensor::create_tensor_from_slice(&up_weight_rust, vec![ff_dim, d_model], &ctx)?;
        let ffn_down = Tensor::create_tensor_from_slice(&down_weight_rust, vec![d_model, ff_dim], &ctx)?;

        // Bias tensors (assume zero since JSON doesn't include biases)
        let ffn_gate_bias = Tensor::create_tensor_from_slice(&vec![0.0f32; ff_dim], vec![ff_dim], &ctx)?;
        let ffn_up_bias = Tensor::create_tensor_from_slice(&vec![0.0f32; ff_dim], vec![ff_dim], &ctx)?;
        let ffn_down_bias = Tensor::create_tensor_from_slice(&vec![0.0f32; d_model], vec![d_model], &ctx)?;

        // Load test cases
        let cases_json = fs::read_to_string("pytorch/swiglu_full_qwen25_comparison_data.json").expect("Failed to read cases JSON");
        let cases: Vec<TestCase> = serde_json::from_str(&cases_json).expect("Failed to parse cases JSON");

        for case in cases {
            let input_len = case.input.len();
            let m = input_len / d_model;
            assert_eq!(input_len, m * d_model, "Input length mismatch for {}", case.name);

            // Create input Tensor [m, d_model]
            let x_normed_flat = Tensor::create_tensor_from_slice(&case.input, vec![m, d_model], &ctx)?;

            // Run swiglu
            let rust_output = swiglu(
                &x_normed_flat,
                &ffn_gate,
                &ffn_gate_bias,
                &ffn_up,
                &ffn_up_bias,
                &ffn_down,
                &ffn_down_bias,
                &mut ctx,
            )?;
            ctx.synchronize(); // Sync to ensure GPU ops complete before CPU read
            let rust_output_flat = rust_output.as_slice().to_vec();

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
}
