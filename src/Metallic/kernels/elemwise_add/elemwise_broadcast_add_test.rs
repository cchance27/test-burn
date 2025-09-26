#![cfg(test)]
use crate::metallic::kernels::elemwise_add::elemwise_broadcast_add::BroadcastElemwiseAddOp;
use crate::metallic::{Context, MetalError, Tensor};

fn cpu_broadcast_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().enumerate().map(|(i, x)| x + b[i % b.len()]).collect()
}

#[test]
fn test_broadcast_add_1d_bias() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Shape [2, 3]
    let b_data = vec![0.5, 1.0, 1.5]; // Shape [3] - broadcast along last dimension

    let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 3], &context)?;
    let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![3], &context)?;

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

    let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2, 3], &context)?;
    let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![3], &context)?;

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

    let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![4], &context)?;
    let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![1], &context)?;

    let cpu_result = cpu_broadcast_add(&a_data, &b_data);

    let result_tensor = context.call::<BroadcastElemwiseAddOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}
