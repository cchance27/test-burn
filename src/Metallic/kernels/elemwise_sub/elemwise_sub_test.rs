#![cfg(test)]
use crate::metallic::kernels::elemwise_sub::ElemwiseSubOp;
use crate::metallic::{Context, MetalError, Tensor};

fn cpu_elemwise_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

#[test]
fn test_elemwise_sub_basic() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![0.5, 1.5, 2.5, 3.5];

    let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2], &context)?;
    let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2], &context)?;

    let cpu_result = cpu_elemwise_sub(&a_data, &b_data);

    let result_tensor = context.call::<ElemwiseSubOp>((a_tensor, b_tensor))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, &cpu_result[..]);

    Ok(())
}
