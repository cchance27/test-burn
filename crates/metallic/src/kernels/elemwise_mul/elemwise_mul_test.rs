use crate::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage, kernels::elemwise_mul::ElemwiseMulOp};

fn cpu_elemwise_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

#[test]
fn test_elemwise_mul_basic() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![0.5, 1.5, 2.5, 3.5];

    let a_tensor = Tensor::new(vec![2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![2, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_result = cpu_elemwise_mul(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseMulOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}
