#![cfg(test)]
use metallic_env::FORCE_MATMUL_BACKEND_VAR;

use crate::{Context, F32Element, Tensor, TensorInit, TensorStorage, kernels::matmul_dispatcher::dispatch_op::MatmulDispatchOp};

fn make_tensor_f32(ctx: &Context<F32Element>, data: &[f32], dims: &[usize]) -> Tensor<F32Element> {
    Tensor::new(dims.to_vec(), TensorStorage::Dedicated(ctx), TensorInit::CopyFrom(data)).unwrap()
}

#[test]
fn alpha_beta_parity_mlx() {
    let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mlx").unwrap();

    let mut ctx = Context::<F32Element>::new().unwrap();
    let a = make_tensor_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let fused = ctx.call::<MatmulDispatchOp>((&a, &b, None, None, false, false, 0.5, 0.0)).unwrap();
    let unfused = {
        let tmp = ctx.call::<MatmulDispatchOp>((&a, &b, None, None, false, false, 1.0, 0.0)).unwrap();
        tmp.mul_scalar(0.5, &mut ctx).unwrap()
    };

    let diff = fused.sub_elem(&unfused, &mut ctx).unwrap();
    let abs_diff = diff.abs(&mut ctx).unwrap();
    let max_abs = abs_diff.max_scalar();
    assert!(max_abs < 1e-5, "max_abs {} too large", max_abs);
}

#[test]
fn alpha_beta_parity_mps() {
    let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mps").unwrap();

    let mut ctx = Context::<F32Element>::new().unwrap();
    let a = make_tensor_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let fused = ctx.call::<MatmulDispatchOp>((&a, &b, None, None, false, false, 0.5, 0.0)).unwrap();
    let unfused = {
        let tmp = ctx.call::<MatmulDispatchOp>((&a, &b, None, None, false, false, 1.0, 0.0)).unwrap();
        tmp.mul_scalar(0.5, &mut ctx).unwrap()
    };

    let diff = fused.sub_elem(&unfused, &mut ctx).unwrap();
    let abs_diff = diff.abs(&mut ctx).unwrap();
    let max_abs = abs_diff.max_scalar();
    assert!(max_abs < 1e-5, "max_abs {} too large", max_abs);
}

#[test]
fn alpha_beta_parity_with_beta_mlx() {
    let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mlx").unwrap();

    let mut ctx = Context::<F32Element>::new().unwrap();
    let a_data = [1.0, 2.0, 3.0, 4.0];
    let b_data = [5.0, 6.0, 7.0, 8.0];
    let c_data = [0.1, 0.2, 0.3, 0.4];
    let a = make_tensor_f32(&ctx, &a_data, &[2, 2]);
    let b = make_tensor_f32(&ctx, &b_data, &[2, 2]);
    let c = make_tensor_f32(&ctx, &c_data, &[2, 2]);

    let fused = ctx
        .call::<MatmulDispatchOp>((&a, &b, None, Some(&c), false, false, 1.0, 1.0))
        .unwrap();

    let mut unfused_data = vec![0.0; 4];
    unfused_data[0] = a_data[0] * b_data[0] + a_data[1] * b_data[2] + c_data[0];
    unfused_data[1] = a_data[0] * b_data[1] + a_data[1] * b_data[3] + c_data[1];
    unfused_data[2] = a_data[2] * b_data[0] + a_data[3] * b_data[2] + c_data[2];
    unfused_data[3] = a_data[2] * b_data[1] + a_data[3] * b_data[3] + c_data[3];
    let unfused = make_tensor_f32(&ctx, &unfused_data, &[2, 2]);

    let diff = fused.sub_elem(&unfused, &mut ctx).unwrap();
    let abs_diff = diff.abs(&mut ctx).unwrap();
    let max_abs = abs_diff.max_scalar();
    assert!(max_abs < 1e-5, "max_abs {} too large", max_abs);
}

#[test]
fn alpha_beta_parity_with_beta_mps() {
    let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mps").unwrap();

    let mut ctx = Context::<F32Element>::new().unwrap();
    let a_data = [1.0, 2.0, 3.0, 4.0];
    let b_data = [5.0, 6.0, 7.0, 8.0];
    let c_data = [0.1, 0.2, 0.3, 0.4];
    let a = make_tensor_f32(&ctx, &a_data, &[2, 2]);
    let b = make_tensor_f32(&ctx, &b_data, &[2, 2]);
    let c = make_tensor_f32(&ctx, &c_data, &[2, 2]);

    let fused = ctx
        .call::<MatmulDispatchOp>((&a, &b, None, Some(&c), false, false, 1.0, 1.0))
        .unwrap();

    let mut unfused_data = vec![0.0; 4];
    unfused_data[0] = a_data[0] * b_data[0] + a_data[1] * b_data[2] + c_data[0];
    unfused_data[1] = a_data[0] * b_data[1] + a_data[1] * b_data[3] + c_data[1];
    unfused_data[2] = a_data[2] * b_data[0] + a_data[3] * b_data[2] + c_data[2];
    unfused_data[3] = a_data[2] * b_data[1] + a_data[3] * b_data[3] + c_data[3];
    let unfused = make_tensor_f32(&ctx, &unfused_data, &[2, 2]);

    let diff = fused.sub_elem(&unfused, &mut ctx).unwrap();
    let abs_diff = diff.abs(&mut ctx).unwrap();
    let max_abs = abs_diff.max_scalar();
    assert!(max_abs < 1e-5, "max_abs {} too large", max_abs);
}

#[test]
fn alpha_beta_parity_with_beta_auto() {
    let mut ctx = Context::<F32Element>::new().unwrap();
    let a_data = [1.0, 2.0, 3.0, 4.0];
    let b_data = [5.0, 6.0, 7.0, 8.0];
    let c_data = [0.1, 0.2, 0.3, 0.4];
    let a = make_tensor_f32(&ctx, &a_data, &[2, 2]);
    let b = make_tensor_f32(&ctx, &b_data, &[2, 2]);
    let c = make_tensor_f32(&ctx, &c_data, &[2, 2]);

    let fused = ctx
        .call::<MatmulDispatchOp>((&a, &b, None, Some(&c), false, false, 1.0, 1.0))
        .unwrap();

    let mut unfused_data = vec![0.0; 4];
    unfused_data[0] = a_data[0] * b_data[0] + a_data[1] * b_data[2] + c_data[0];
    unfused_data[1] = a_data[0] * b_data[1] + a_data[1] * b_data[3] + c_data[1];
    unfused_data[2] = a_data[2] * b_data[0] + a_data[3] * b_data[2] + c_data[2];
    unfused_data[3] = a_data[2] * b_data[1] + a_data[3] * b_data[3] + c_data[3];
    let unfused = make_tensor_f32(&ctx, &unfused_data, &[2, 2]);

    let diff = fused.sub_elem(&unfused, &mut ctx).unwrap();
    let abs_diff = diff.abs(&mut ctx).unwrap();
    let max_abs = abs_diff.max_scalar();
    assert!(max_abs < 1e-5, "max_abs {} too large", max_abs);
}

#[test]
fn alpha_beta_gpu_parity_with_beta_mlx() {
    let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mlx").unwrap();

    let mut ctx = Context::<F32Element>::new().unwrap();
    let a_data = [1.0, 2.0, 3.0, 4.0];
    let b_data = [5.0, 6.0, 7.0, 8.0];
    let c_data = [0.1, 0.2, 0.3, 0.4];
    let a = make_tensor_f32(&ctx, &a_data, &[2, 2]);
    let b = make_tensor_f32(&ctx, &b_data, &[2, 2]);
    let c = make_tensor_f32(&ctx, &c_data, &[2, 2]);

    let fused = ctx
        .call::<MatmulDispatchOp>((&a, &b, None, Some(&c), false, false, 1.0, 1.0))
        .unwrap();

    let mut unfused_data = vec![0.0; 4];
    unfused_data[0] = a_data[0] * b_data[0] + a_data[1] * b_data[2] + c_data[0];
    unfused_data[1] = a_data[0] * b_data[1] + a_data[1] * b_data[3] + c_data[1];
    unfused_data[2] = a_data[2] * b_data[0] + a_data[3] * b_data[2] + c_data[2];
    unfused_data[3] = a_data[2] * b_data[1] + a_data[3] * b_data[3] + c_data[3];
    let unfused = make_tensor_f32(&ctx, &unfused_data, &[2, 2]);

    let diff = fused.sub_elem(&unfused, &mut ctx).unwrap();
    let abs_diff = diff.abs(&mut ctx).unwrap();
    let max_abs = abs_diff.max_scalar();
    assert!(max_abs < 1e-5, "max_abs {} too large", max_abs);
}

#[test]
fn alpha_beta_gpu_parity_with_beta_mps() {
    let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mps").unwrap();

    let mut ctx = Context::<F32Element>::new().unwrap();
    let a_data = [1.0, 2.0, 3.0, 4.0];
    let b_data = [5.0, 6.0, 7.0, 8.0];
    let c_data = [0.1, 0.2, 0.3, 0.4];
    let a = make_tensor_f32(&ctx, &a_data, &[2, 2]);
    let b = make_tensor_f32(&ctx, &b_data, &[2, 2]);
    let c = make_tensor_f32(&ctx, &c_data, &[2, 2]);

    let fused = ctx
        .call::<MatmulDispatchOp>((&a, &b, None, Some(&c), false, false, 1.0, 1.0))
        .unwrap();

    let mut unfused_data = vec![0.0; 4];
    unfused_data[0] = a_data[0] * b_data[0] + a_data[1] * b_data[2] + c_data[0];
    unfused_data[1] = a_data[0] * b_data[1] + a_data[1] * b_data[3] + c_data[1];
    unfused_data[2] = a_data[2] * b_data[0] + a_data[3] * b_data[2] + c_data[2];
    unfused_data[3] = a_data[2] * b_data[1] + a_data[3] * b_data[3] + c_data[3];
    let unfused = make_tensor_f32(&ctx, &unfused_data, &[2, 2]);

    let diff = fused.sub_elem(&unfused, &mut ctx).unwrap();
    let abs_diff = diff.abs(&mut ctx).unwrap();
    let max_abs = abs_diff.max_scalar();
    assert!(max_abs < 1e-5, "max_abs {} too large", max_abs);
}
