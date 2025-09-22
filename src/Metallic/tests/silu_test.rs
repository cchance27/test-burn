use super::*;
use crate::metallic::silu::{Silu, ensure_silu_pipeline};

// CPU SiLU
fn cpu_silu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            let x64 = x as f64;
            let sig = 1.0 / (1.0 + (-x64).exp());
            (x64 * sig) as f32
        })
        .collect()
}

#[test]
fn test_silu_basic() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_silu_pipeline(&mut context)?;

    let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -50.0, 50.0];
    let dims = vec![input_data.len()];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;
    let output_tensor = Tensor::create_tensor(dims.clone(), &context)?;

    let silu_op = Silu::new(
        input_tensor,
        output_tensor.clone(),
        context.silu_pipeline.as_ref().unwrap().clone(),
    )?;

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    silu_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = output_tensor.as_slice();
    let cpu_output = cpu_silu(&input_data);

    let rtol = 1e-4f64;
    let atol = 1e-6f64;
    for i in 0..input_data.len() {
        let m = metal_output[i] as f64;
        let c = cpu_output[i] as f64;
        let diff = (m - c).abs();
        let rel = if c.abs() > 1e-8 { diff / c.abs() } else { diff };
        assert!(
            diff <= atol || rel <= rtol,
            "Mismatch at {}: metal={}, cpu={}, diff={}",
            i,
            m,
            c,
            diff
        );
    }

    Ok(())
}
