#![cfg(test)]
use std::env;

use half::f16;
use metallic_env::FORCE_MATMUL_BACKEND_VAR;

use crate::{Context, F16Element, Tensor, TensorInit, TensorStorage, kernels::matmul_dispatcher::dispatch_op::MatmulDispatchOp};

fn make_tensor_f16(ctx: &Context<F16Element>, data: &[f32], dims: &[usize]) -> Tensor<F16Element> {
    let data = data.iter().map(|&d| f16::from_f32(d)).collect::<Vec<_>>();
    Tensor::new(dims.to_vec(), TensorStorage::Dedicated(ctx), TensorInit::CopyFrom(&data)).unwrap()
}

#[test]
fn dispatch_routes_to_mlx_when_forced() {
    let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mlx").unwrap();

    let mut ctx = Context::<F16Element>::new().unwrap();
    let a = make_tensor_f16(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor_f16(&ctx, &[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let out = ctx
        .call::<MatmulDispatchOp>((&a, &b, None, None, false, false, 1.0, 0.0), None)
        .unwrap();
    assert_eq!(out.dims(), &[2, 2]);
}

#[test]
fn dispatch_routes_to_mps_when_forced() {
    let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("mps").unwrap();

    let mut ctx = Context::<F16Element>::new().unwrap();
    let a = make_tensor_f16(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor_f16(&ctx, &[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let out = ctx
        .call::<MatmulDispatchOp>((&a, &b, None, None, false, false, 1.0, 0.0), None)
        .unwrap();
    assert_eq!(out.dims(), &[2, 2]);
}

#[test]
fn dispatch_routes_to_gemv_when_forced() {
    let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard("gemv").unwrap();

    let mut ctx = Context::<F16Element>::new().unwrap();
    // Use a degenerate shape compatible with GEMV fallback
    let a = make_tensor_f16(&ctx, &[1.0, 2.0], &[1, 2]);
    let b = make_tensor_f16(&ctx, &[5.0, 6.0], &[2, 1]);

    let out = ctx
        .call::<MatmulDispatchOp>((&a, &b, None, None, false, false, 1.0, 0.0), None)
        .unwrap();
    assert_eq!(out.dims(), &[1, 1]);
}

#[test]
fn dispatch_routes_to_smalln_when_forced() {
    let _backend_guard: metallic_env::TypedEnvVarGuard<'_, String> = FORCE_MATMUL_BACKEND_VAR.set_guard("gemv").unwrap();
    unsafe {
        env::set_var("METALLIC_MATMUL_FORCE_SMALLN", "true");
    }

    let mut ctx = Context::<F16Element>::new().unwrap();
    // Use a shape where N=8
    let a_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let a = make_tensor_f16(&ctx, &a_data, &[1, 16]);
    let b = make_tensor_f16(&ctx, &b_data, &[16, 8]);

    let out = ctx
        .call::<MatmulDispatchOp>((&a, &b, None, None, false, false, 1.0, 0.0), None)
        .unwrap();
    assert_eq!(out.dims(), &[1, 8]);

    // Clean up env var
    unsafe {
        env::remove_var("METALLIC_MATMUL_FORCE_SMALLN");
    }
}
