use crate::metallic::kernels::rope::RoPEOp;
use crate::metallic::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage};

// CPU RoPE reference implementation for testing (pairs elements dim/2 apart)
fn cpu_rope(input: &[f32], batch: usize, seq_len: usize, dim: usize, cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    let rows = batch * seq_len;
    let half_dim = dim / 2;
    for row in 0..rows {
        let pos = row % seq_len;
        for p in 0..half_dim {
            let i = row * dim + p; // index in first half
            let j = row * dim + p + half_dim; // index in second half (i + half_dim)
            let cosv = cos[pos * half_dim + p];
            let sinv = sin[pos * half_dim + p];
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
    let mut context = Context::<F32Element>::new()?;

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
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;
    let cos_tensor = Tensor::new(
        vec![seq_len, dim / 2],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&cos_data),
    )?;
    let sin_tensor = Tensor::new(
        vec![seq_len, dim / 2],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&sin_data),
    )?;

    let output_tensor = context.call::<RoPEOp>((input_tensor, cos_tensor, sin_tensor, dim as u32, seq_len as u32, 0))?;
    context.synchronize();

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

#[test]
fn test_rope_extreme_large_position_values() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;

    let batch = 1usize;
    let seq_len = 10000usize; // Very large sequence length
    let dim = 128usize; // Even dimension

    // Construct simple input
    let mut input_data: Vec<f32> = Vec::with_capacity(batch * seq_len * dim);
    for r in 0..(batch * seq_len) {
        for _d in 0..dim {
            input_data.push((r % 10) as f32 * 0.1); // Small values to avoid overflow
        }
    }

    // Build cos/sin tables with extremely large position values
    let mut cos_data = vec![0.0f32; seq_len * (dim / 2)];
    let mut sin_data = vec![0.0f32; seq_len * (dim / 2)];
    for pos in 0..seq_len {
        for p in 0..(dim / 2) {
            // Using large position values with high frequency
            let theta = (p as f32) * 0.01 + (pos as f32) * 1000.0;
            cos_data[pos * (dim / 2) + p] = theta.cos();
            sin_data[pos * (dim / 2) + p] = theta.sin();
        }
    }

    let dims = vec![batch, seq_len, dim];
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;
    let cos_tensor = Tensor::new(
        vec![seq_len, dim / 2],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&cos_data),
    )?;
    let sin_tensor = Tensor::new(
        vec![seq_len, dim / 2],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&sin_data),
    )?;

    let output_tensor = context.call::<RoPEOp>((input_tensor, cos_tensor, sin_tensor, dim as u32, seq_len as u32, 0))?;
    context.synchronize();

    let metal_output = output_tensor.as_slice();
    let cpu_output = cpu_rope(&input_data, batch, seq_len, dim, &cos_data, &sin_data);

    let rtol = 1e-4f64;
    let atol = 1e-6f64;
    for i in 0..metal_output.len() {
        let m = metal_output[i] as f64;
        let c = cpu_output[i] as f64;
        let diff = (m - c).abs();
        let rel = if c.abs() > 1e-8 { diff / c.abs() } else { diff };

        // Verify finite values
        assert!(m.is_finite(), "Metal output contains non-finite value at index {}: {}", i, m);
        assert!(c.is_finite(), "CPU output contains non-finite value at index {}: {}", i, c);

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

#[test]
fn test_rope_extreme_angle_values() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;

    let batch = 1usize;
    let seq_len = 8usize;
    let dim = 64usize; // Even dimension

    // Construct simple input
    let mut input_data: Vec<f32> = Vec::with_capacity(batch * seq_len * dim);
    for r in 0..(batch * seq_len) {
        for d in 0..dim {
            input_data.push((r * dim + d) as f32);
        }
    }

    // Build cos/sin tables with extreme angle values (near pi/2 where cos/sin can be problematic)
    let mut cos_data = vec![0.0f32; seq_len * (dim / 2)];
    let mut sin_data = vec![0.0f32; seq_len * (dim / 2)];
    for pos in 0..seq_len {
        for p in 0..(dim / 2) {
            // Use angles that are very large (multiple of 2*pi) to test periodicity
            let theta = (p as f32) * 1000.0 + (pos as f32) * 100.0;
            cos_data[pos * (dim / 2) + p] = theta.cos();
            sin_data[pos * (dim / 2) + p] = theta.sin();
        }
    }

    let dims = vec![batch, seq_len, dim];
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;
    let cos_tensor = Tensor::new(
        vec![seq_len, dim / 2],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&cos_data),
    )?;
    let sin_tensor = Tensor::new(
        vec![seq_len, dim / 2],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&sin_data),
    )?;

    let output_tensor = context.call::<RoPEOp>((input_tensor, cos_tensor, sin_tensor, dim as u32, seq_len as u32, 0))?;
    context.synchronize();

    let metal_output = output_tensor.as_slice();
    let cpu_output = cpu_rope(&input_data, batch, seq_len, dim, &cos_data, &sin_data);

    let rtol = 1e-4f64;
    let atol = 1e-6f64;
    for i in 0..metal_output.len() {
        let m = metal_output[i] as f64;
        let c = cpu_output[i] as f64;
        let diff = (m - c).abs();
        let rel = if c.abs() > 1e-8 { diff / c.abs() } else { diff };

        // Verify finite values
        assert!(m.is_finite(), "Metal output contains non-finite value at index {}: {}", i, m);
        assert!(c.is_finite(), "CPU output contains non-finite value at index {}: {}", i, c);

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

#[test]
fn test_rope_extreme_input_values() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;

    let batch = 1usize;
    let seq_len = 4usize;
    let dim = 16usize; // Even dimension

    // Construct input with extreme values (very large)
    let extreme_value = 1e10f32;
    let mut input_data: Vec<f32> = Vec::with_capacity(batch * seq_len * dim);
    for _r in 0..(batch * seq_len) {
        for _d in 0..dim {
            input_data.push(if _d % 2 == 0 { extreme_value } else { -extreme_value });
        }
    }

    // Small cos/sin values
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
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;
    let cos_tensor = Tensor::new(
        vec![seq_len, dim / 2],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&cos_data),
    )?;
    let sin_tensor = Tensor::new(
        vec![seq_len, dim / 2],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&sin_data),
    )?;

    let output_tensor = context.call::<RoPEOp>((input_tensor, cos_tensor, sin_tensor, dim as u32, seq_len as u32, 0))?;
    context.synchronize();

    let metal_output = output_tensor.as_slice();

    // Verify output does not contain infinities or NaNs
    for (i, &val) in metal_output.iter().enumerate() {
        assert!(val.is_finite(), "Metal output contains non-finite value at index {}: {}", i, val);
    }

    Ok(())
}

#[test]
fn test_rope_extreme_cos_sin_values() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;

    let batch = 1usize;
    let seq_len = 2usize;
    let dim = 8usize; // Even dimension

    // Construct simple input
    let mut input_data: Vec<f32> = Vec::with_capacity(batch * seq_len * dim);
    for r in 0..(batch * seq_len) {
        for d in 0..dim {
            input_data.push((r * dim + d) as f32);
        }
    }

    // Build cos/sin tables with extreme values (near the boundaries of -1.0 to 1.0)
    let mut cos_data = vec![0.0f32; seq_len * (dim / 2)];
    let mut sin_data = vec![0.0f32; seq_len * (dim / 2)];
    for pos in 0..seq_len {
        for p in 0..(dim / 2) {
            // Use angles that yield near edge values (like 0, pi/2, pi, 3pi/2)
            let theta = (p as f32) * std::f32::consts::PI / 2.0;
            cos_data[pos * (dim / 2) + p] = theta.cos();
            sin_data[pos * (dim / 2) + p] = theta.sin();
        }
    }

    let dims = vec![batch, seq_len, dim];
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;
    let cos_tensor = Tensor::new(
        vec![seq_len, dim / 2],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&cos_data),
    )?;
    let sin_tensor = Tensor::new(
        vec![seq_len, dim / 2],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&sin_data),
    )?;

    let output_tensor = context.call::<RoPEOp>((input_tensor, cos_tensor, sin_tensor, dim as u32, seq_len as u32, 0))?;
    context.synchronize();

    let metal_output = output_tensor.as_slice();
    let cpu_output = cpu_rope(&input_data, batch, seq_len, dim, &cos_data, &sin_data);

    let rtol = 1e-4f64;
    let atol = 1e-6f64;
    for i in 0..metal_output.len() {
        let m = metal_output[i] as f64;
        let c = cpu_output[i] as f64;
        let diff = (m - c).abs();
        let rel = if c.abs() > 1e-8 { diff / c.abs() } else { diff };

        // Verify finite values
        assert!(m.is_finite(), "Metal output contains non-finite value at index {}: {}", i, m);
        assert!(c.is_finite(), "CPU output contains non-finite value at index {}: {}", i, c);

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
