use crate::metallic::kernels::rmsnorm::RMSNormOp;
use crate::metallic::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage};

#[test]
fn fused_rmsnorm_matches_separate_ops() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    let rows = 2usize;
    let feature_dim = 4usize;
    let kv_dim = 3usize;
    let total_out_dim = feature_dim + 2 * kv_dim;

    let input_data = vec![
        0.5, -1.0, 2.0, -0.5, // row 0
        1.5, 0.0, -0.75, 0.25, // row 1
    ];
    let gamma_data = vec![1.0, 1.1, 0.9, 1.2];

    let weight_data: Vec<f32> = (0..feature_dim * total_out_dim).map(|idx| (idx as f32 * 0.013) - 0.5).collect();
    let bias_data: Vec<f32> = (0..total_out_dim).map(|idx| (idx as f32 * 0.01) - 0.1).collect();

    let input = Tensor::new(
        vec![rows, feature_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )?;
    let gamma = Tensor::new(vec![feature_dim], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&gamma_data))?;
    let weight = Tensor::new(
        vec![feature_dim, total_out_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&weight_data),
    )?;
    let bias = Tensor::new(
        vec![total_out_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&bias_data),
    )?;

    let input_clone = input.clone();
    let gamma_clone = gamma.clone();
    let weight_clone = weight.clone();
    let bias_clone = bias.clone();

    let (q_fused, k_fused, v_fused) =
        ctx.fused_rmsnorm_qkv_projection(&input_clone, &gamma_clone, &weight_clone, &bias_clone, feature_dim, kv_dim)?;
    ctx.synchronize();

    let fused_slice = {
        let mut combined = Vec::with_capacity(rows * total_out_dim);
        combined.extend_from_slice(q_fused.as_slice());
        combined.extend_from_slice(k_fused.as_slice());
        combined.extend_from_slice(v_fused.as_slice());
        combined
    };

    let rms_input = ctx.call::<RMSNormOp>((input, gamma, feature_dim as u32))?;
    let rms_flat = rms_input.reshape(vec![rows, feature_dim])?;
    let (q_expected, k_expected, v_expected) = ctx.fused_qkv_projection(&rms_flat, &weight, &bias, feature_dim, kv_dim)?;
    ctx.synchronize();

    let total_flat_expected = {
        let mut combined = Vec::with_capacity(rows * total_out_dim);
        combined.extend_from_slice(q_expected.as_slice());
        combined.extend_from_slice(k_expected.as_slice());
        combined.extend_from_slice(v_expected.as_slice());
        combined
    };

    for (idx, (fused, expected)) in fused_slice.iter().zip(total_flat_expected.iter()).enumerate() {
        let diff = (fused - expected).abs();
        assert!(
            diff <= 1e-4,
            "Mismatch at index {}: fused={} expected={} diff={}",
            idx,
            fused,
            expected,
            diff
        );
    }

    Ok(())
}
