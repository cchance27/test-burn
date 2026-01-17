#![cfg(test)]
use metallic_context::{Context, F32Element, Tensor, TensorInit, TensorStorage, context::MatmulAlphaBeta, tensor::TensorType};
use metallic_env::FORCE_MATMUL_BACKEND_VAR;

fn make_tensor_f32(ctx: &Context<F32Element>, data: &[f32], dims: &[usize]) -> Tensor<F32Element> {
    Tensor::new(dims.to_vec(), TensorStorage::Dedicated(ctx), TensorInit::CopyFrom(data)).unwrap()
}

fn make_zero_tensor_f32(ctx: &Context<F32Element>, dims: &[usize]) -> Tensor<F32Element> {
    let len = dims.iter().product::<usize>();
    let zeros = vec![0.0f32; len];
    make_tensor_f32(ctx, &zeros, dims)
}

#[test]
fn alpha_beta_parity_mlx_forced() {
    let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mlx").unwrap();

    let mut ctx = Context::<F32Element>::new().unwrap();
    let a = make_tensor_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor_f32(&ctx, &[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let out = make_zero_tensor_f32(&ctx, &[2, 2]);

    let fused = ctx
        .matmul(
            &a,
            &TensorType::Dense(&b),
            false,
            false,
            None,
            Some(MatmulAlphaBeta {
                output: &out,
                alpha: 0.5,
                beta: 0.0,
            }),
            None,
        )
        .unwrap();

    let unfused = {
        let tmp = ctx.matmul(&a, &TensorType::Dense(&b), false, false, None, None, None).unwrap();
        tmp.mul_scalar(0.5, &mut ctx).unwrap()
    };

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
        .matmul(
            &a,
            &TensorType::Dense(&b),
            false,
            false,
            None,
            Some(MatmulAlphaBeta {
                output: &c,
                alpha: 1.0,
                beta: 1.0,
            }),
            None,
        )
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
