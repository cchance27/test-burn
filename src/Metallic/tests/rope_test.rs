use super::*;
use crate::metallic::rope::{RoPE, ensure_rope_pipeline};

// CPU RoPE reference implementation for testing
fn cpu_rope(
    input: &[f32],
    batch: usize,
    seq_len: usize,
    dim: usize,
    cos: &[f32],
    sin: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    let rows = batch * seq_len;
    for row in 0..rows {
        let pos = row % seq_len;
        for p in 0..(dim / 2) {
            let i = row * dim + 2 * p;
            let j = i + 1;
            let cosv = cos[pos * (dim / 2) + p];
            let sinv = sin[pos * (dim / 2) + p];
            let xi = input[i];
            let xj = input[j];
            out[i] = xi * cosv - xj * sinv;
            out[j] = xj * cosv + xi * sinv;
        }
    }
    out
}

#[test]
fn test_rope_basic() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_rope_pipeline(&mut context)?;

    let batch = 2usize;
    let seq_len = 3usize;
    let dim = 4usize; // must be even

    // Construct simple input: sequential values per row to make results predictable
    let mut input_data: Vec<f32> = Vec::with_capacity(batch * seq_len * dim);
    for r in 0..(batch * seq_len) {
        for d in 0..dim {
            input_data.push((r * dim + d) as f32);
        }
    }

    // Build simple cos/sin tables: cos=cos(theta), sin=sin(theta) for theta = p*0.1 + pos*0.01
    let mut cos_data = vec![0.0f32; seq_len * (dim / 2)];
    let mut sin_data = vec![0.0f32; seq_len * (dim / 2)];
    for pos in 0..seq_len {
        for p in 0..(dim / 2) {
            let theta = (p as f32) * 0.1 + (pos as f32) * 0.01;
            cos_data[pos * (dim / 2) + p] = theta.cos();
            sin_data[pos * (dim / 2) + p] = theta.sin();
        }
    }

    let dims = vec![batch, seq_len, dim];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;
    let output_tensor = Tensor::create_tensor(dims.clone(), &context)?;
    let cos_tensor = Tensor::create_tensor_from_slice(&cos_data, vec![seq_len, dim / 2], &context)?;
    let sin_tensor = Tensor::create_tensor_from_slice(&sin_data, vec![seq_len, dim / 2], &context)?;

    let rope_op = RoPE::new(
        input_tensor,
        output_tensor.clone(),
        cos_tensor,
        sin_tensor,
        dim as u32,
        seq_len as u32,
        context.rope_pipeline.as_ref().unwrap().clone(),
    )?;

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    rope_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = output_tensor.as_slice();
    let cpu_output = cpu_rope(&input_data, batch, seq_len, dim, &cos_data, &sin_data);

    let rtol = 1e-4f64;
    let atol = 1e-6f64;
    for i in 0..metal_output.len() {
        let m = metal_output[i] as f64;
        let c = cpu_output[i] as f64;
        let diff = (m - c).abs();
        let rel = if c.abs() > 1e-8 { diff / c.abs() } else { diff };
        assert!(
            diff <= atol || rel <= rtol,
            "RoPE mismatch idx {}: metal={}, cpu={}, diff={}",
            i,
            m,
            c,
            diff
        );
    }

    Ok(())
}
