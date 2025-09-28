#![cfg(test)]
use super::elemwise_div::ElemwiseDivOp;
use crate::metallic::{Context, MetalError, Tensor, TensorInit, TensorStorage};

fn cpu_elemwise_div(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
}

#[test]
fn test_elemwise_div_basic() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![0.5, 1.0, 1.5, 2.0];

    let a_tensor = Tensor::new(vec![2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_elemwise_div(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseDivOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}

#[test]
fn test_elemwise_div_1d() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

    let a_tensor = Tensor::new(vec![6], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![6], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_elemwise_div(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseDivOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}

#[test]
fn test_elemwise_div_3d() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    let a_tensor = Tensor::new(vec![2, 2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![2, 2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_elemwise_div(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseDivOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}

#[test]
fn test_elemwise_div_by_one() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![1.0, 1.0, 1.0, 1.0]; // Dividing by 1 should leave values unchanged

    let a_tensor = Tensor::new(vec![2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_elemwise_div(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseDivOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}

#[test]
fn test_elemwise_div_with_fractions() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![2.0, 4.0, 6.0, 8.0];

    let a_tensor = Tensor::new(vec![2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_elemwise_div(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseDivOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    // Using a tolerance for floating point comparison
    for i in 0..metal_output.len() {
        let diff = (metal_output[i] - cpu_result[i]).abs();
        assert!(
            diff < 1e-6,
            "Mismatch at index {}: metal={}, cpu={}, diff={}",
            i,
            metal_output[i],
            cpu_result[i],
            diff
        );
    }

    Ok(())
}

#[test]
fn test_elemwise_div_large_tensor() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let size = 1024;
    let a_data: Vec<f32> = (1..=size).map(|i| i as f32 * 0.5).collect();
    let b_data: Vec<f32> = (1..=size).map(|i| i as f32 * 0.3).collect();

    let a_tensor = Tensor::new(vec![32, 32], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![32, 32], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_elemwise_div(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseDivOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    // Using a tolerance for floating point comparison due to potential precision issues
    assert_eq!(metal_output.len(), cpu_result.len());
    for i in 0..metal_output.len() {
        let diff = (metal_output[i] - cpu_result[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: metal={}, cpu={}, diff={}",
            i,
            metal_output[i],
            cpu_result[i],
            diff
        );
    }

    Ok(())
}

#[test]
fn test_elemwise_div_floating_precision() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    // Using floating point numbers that may introduce precision errors
    let a_data = vec![1.1, 2.2, 3.3, 4.4];
    let b_data = vec![0.1, 0.2, 0.3, 0.4];

    let a_tensor = Tensor::new(vec![2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_elemwise_div(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseDivOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    // Using a tolerance for floating point comparison
    for i in 0..metal_output.len() {
        let diff = (metal_output[i] - cpu_result[i]).abs();
        assert!(
            diff < 1e-6,
            "Mismatch at index {}: metal={}, cpu={}, diff={}",
            i,
            metal_output[i],
            cpu_result[i],
            diff
        );
    }

    Ok(())
}