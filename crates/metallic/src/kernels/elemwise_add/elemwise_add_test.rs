use crate::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage, kernels::elemwise_add::elemwise_add::ElemwiseAddOp};

fn cpu_elemwise_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[test]
fn test_elemwise_add_basic() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![0.5, 1.5, 2.5, 3.5];

    let a_tensor = Tensor::new(vec![2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_elemwise_add(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseAddOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}

#[test]
fn test_elemwise_add_1d() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

    let a_tensor = Tensor::new(vec![6], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![6], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_elemwise_add(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseAddOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}

#[test]
fn test_elemwise_add_3d() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    let a_tensor = Tensor::new(vec![2, 2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![2, 2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_elemwise_add(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseAddOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}

#[test]
fn test_elemwise_add_large_tensor() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;

    let size = 1024;
    let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.5).collect();
    let b_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.3).collect();

    let a_tensor = Tensor::new(vec![32, 32], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![32, 32], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_elemwise_add(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseAddOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}
