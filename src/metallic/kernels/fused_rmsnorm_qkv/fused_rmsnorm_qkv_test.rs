use crate::metallic::kernels::fused_rmsnorm_qkv::FusedRmsNormQkvProjectionOp;
use crate::metallic::kernels::kv_rearrange::KvRearrangeOp;
use crate::metallic::kernels::rmsnorm::RMSNormOp;
use crate::metallic::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage};

#[test]
fn fused_rmsnorm_matches_separate_ops() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    let batch = 2usize;
    let seq = 2usize;
    let rows = batch * seq;
    let feature_dim = 4usize;
    let kv_dim = 2usize;
    let total_out_dim = feature_dim + 2 * kv_dim;

    let head_dim = 2usize;
    let n_heads = feature_dim / head_dim;
    let n_kv_heads = 1usize;
    let kv_head_dim = kv_dim / n_kv_heads;

    let input_data: Vec<f32> = (0..rows * feature_dim).map(|idx| idx as f32 * 0.25 - 1.0).collect();
    let gamma_data: Vec<f32> = (0..feature_dim).map(|i| 0.8 + i as f32 * 0.1).collect();

    let weight_data: Vec<f32> = (0..feature_dim * total_out_dim).map(|idx| (idx as f32 * 0.017) - 0.3).collect();
    let bias_data: Vec<f32> = (0..total_out_dim).map(|idx| (idx as f32 * 0.02) - 0.05).collect();

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

    let (q_fused, k_fused, v_fused) = ctx.fused_rmsnorm_qkv_projection(&input, &gamma, &weight, &bias, feature_dim, kv_dim)?;

    assert_eq!(q_fused.dims(), &[rows, feature_dim]);
    assert_eq!(k_fused.dims(), &[rows, kv_dim]);
    assert_eq!(v_fused.dims(), &[rows, kv_dim]);

    assert_eq!(q_fused.strides[0], total_out_dim);
    assert_eq!(k_fused.strides[0], total_out_dim);
    assert_eq!(v_fused.strides[0], total_out_dim);

    assert_eq!(q_fused.strides[1], 1);
    assert_eq!(k_fused.strides[1], 1);
    assert_eq!(v_fused.strides[1], 1);

    let rms_input = ctx.call::<RMSNormOp>((input.clone(), gamma.clone(), feature_dim as u32))?;
    let rms_flat = rms_input.reshape(vec![rows, feature_dim])?;
    let (q_expected, k_expected, v_expected) = ctx.fused_qkv_projection(&rms_flat, &weight, &bias, feature_dim, kv_dim)?;

    let combined_direct = ctx.call::<FusedRmsNormQkvProjectionOp>((
        input.clone(),
        gamma.clone(),
        weight.clone(),
        bias.clone(),
        feature_dim as u32,
        total_out_dim as u32,
    ))?;

    let q_heads_fused = ctx.call::<KvRearrangeOp>((
        q_fused.clone(),
        feature_dim as u32,
        head_dim as u32,
        n_heads as u32,
        n_heads as u32,
        head_dim as u32,
        seq as u32,
    ))?;
    let k_heads_fused = ctx.call::<KvRearrangeOp>((
        k_fused.clone(),
        kv_dim as u32,
        kv_head_dim as u32,
        n_kv_heads as u32,
        n_kv_heads as u32,
        kv_head_dim as u32,
        seq as u32,
    ))?;
    let v_heads_fused = ctx.call::<KvRearrangeOp>((
        v_fused.clone(),
        kv_dim as u32,
        kv_head_dim as u32,
        n_kv_heads as u32,
        n_kv_heads as u32,
        kv_head_dim as u32,
        seq as u32,
    ))?;

    let q_heads_expected = ctx.call::<KvRearrangeOp>((
        q_expected.clone(),
        feature_dim as u32,
        head_dim as u32,
        n_heads as u32,
        n_heads as u32,
        head_dim as u32,
        seq as u32,
    ))?;
    let k_heads_expected = ctx.call::<KvRearrangeOp>((
        k_expected.clone(),
        kv_dim as u32,
        kv_head_dim as u32,
        n_kv_heads as u32,
        n_kv_heads as u32,
        kv_head_dim as u32,
        seq as u32,
    ))?;
    let v_heads_expected = ctx.call::<KvRearrangeOp>((
        v_expected.clone(),
        kv_dim as u32,
        kv_head_dim as u32,
        n_kv_heads as u32,
        n_kv_heads as u32,
        kv_head_dim as u32,
        seq as u32,
    ))?;

    ctx.synchronize();

    // Materialize contiguous buffers for the unfused tensors before comparing. The views
    // returned by `fused_qkv_projection` retain the combined QKV stride (total_out_dim) so
    // reading them directly as slices would interleave rows with K/V segments. By
    // re-materializing we get row-major [rows, dim] buffers that match the logical layout
    // of the fused kernel's output.
    let q_expected_host = ctx.materialize_contiguous_view(q_expected.clone())?;
    let k_expected_host = ctx.materialize_contiguous_view(k_expected.clone())?;
    let v_expected_host = ctx.materialize_contiguous_view(v_expected.clone())?;

    println!(
        "expected strides (Q/K/V): {:?} / {:?} / {:?}",
        q_fused.strides, k_fused.strides, v_fused.strides
    );

    let q_expected_slice = q_expected_host.as_slice();
    let k_expected_slice = k_expected_host.as_slice();
    let v_expected_slice = v_expected_host.as_slice();

    let mut total_flat_expected = Vec::with_capacity(rows * total_out_dim);

    for row in 0..rows {
        let q_base = row * feature_dim;
        total_flat_expected.extend_from_slice(&q_expected_slice[q_base..q_base + feature_dim]);

        let k_base = row * kv_dim;
        total_flat_expected.extend_from_slice(&k_expected_slice[k_base..k_base + kv_dim]);

        let v_base = row * kv_dim;
        total_flat_expected.extend_from_slice(&v_expected_slice[v_base..v_base + kv_dim]);
    }

    for (idx, (fused, expected)) in combined_direct.as_slice().iter().zip(total_flat_expected.iter()).enumerate() {
        let diff = (fused - expected).abs();
        assert!(
            diff <= 1e-4,
            "Combined mismatch at index {}: fused={} expected={} diff={}",
            idx,
            fused,
            expected,
            diff
        );
    }

    let compare_rearranged = |label: &str, fused: &[f32], expected: &[f32]| {
        for (idx, (f, e)) in fused.iter().zip(expected.iter()).enumerate() {
            let diff = (f - e).abs();
            assert!(
                diff <= 1e-4,
                "{} mismatch at index {}: fused={} expected={} diff={}",
                label,
                idx,
                f,
                e,
                diff
            );
        }
    };

    compare_rearranged("Q heads", q_heads_fused.as_slice(), q_heads_expected.as_slice());
    compare_rearranged("K heads", k_heads_fused.as_slice(), k_heads_expected.as_slice());
    compare_rearranged("V heads", v_heads_fused.as_slice(), v_heads_expected.as_slice());

    Ok(())
}
