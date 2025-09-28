#![cfg(test)]
use crate::metallic::kernels::elemwise_add::elemwise_broadcast_add::BroadcastElemwiseAddOp;
use crate::metallic::{Context, MetalError, Tensor, TensorInit, TensorStorage};

fn cpu_broadcast_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().enumerate().map(|(i, x)| x + b[i % b.len()]).collect()
}

#[test]
fn test_broadcast_add_1d_bias() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Shape [2, 3]
    let b_data = vec![0.5, 1.0, 1.5]; // Shape [3] - broadcast along last dimension

    let a_tensor = Tensor::new(vec![2, 3], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![3], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_broadcast_add(&a_data, &b_data);

    let result_tensor = context.call::<BroadcastElemwiseAddOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}

#[test]
fn test_broadcast_add_2d_bias() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // Shape [2, 2, 3]
    let b_data = vec![0.5, 1.0, 1.5]; // Shape [3] - broadcast along last dimension

    let a_tensor = Tensor::new(vec![2, 2, 3], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![3], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_broadcast_add(&a_data, &b_data);

    let result_tensor = context.call::<BroadcastElemwiseAddOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}

#[test]
fn test_broadcast_add_singleton_broadcast() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0]; // Shape [4]
    let b_data = vec![2.5]; // Shape [1] - broadcast to all elements

    let a_tensor = Tensor::new(vec![4], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![1], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_broadcast_add(&a_data, &b_data);

    let result_tensor = context.call::<BroadcastElemwiseAddOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}