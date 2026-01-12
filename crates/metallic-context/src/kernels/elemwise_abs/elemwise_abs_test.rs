use crate::{Context, F32Element, Tensor, TensorInit, TensorStorage, kernels::elemwise_abs::ElemwiseAbsOp};

#[test]
fn test_elemwise_abs() {
    let mut ctx = Context::<F32Element>::new().unwrap();
    let input = Tensor::new(vec![3], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&[-2.0, -5.0, 3.0])).unwrap();

    let output = ctx.call::<ElemwiseAbsOp>((input,), None).unwrap();
    ctx.synchronize();

    let result = output.to_f32_vec();
    assert_eq!(result, vec![2.0, 5.0, 3.0]);
}
