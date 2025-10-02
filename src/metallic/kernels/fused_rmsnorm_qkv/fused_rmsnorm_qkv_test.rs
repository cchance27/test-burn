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

    let q_expected_slice = q_expected.as_slice();
    let k_expected_slice = k_expected.as_slice();
    let v_expected_slice = v_expected.as_slice();

    let mut total_flat_expected = Vec::with_capacity(rows * total_out_dim);
    // The `as_slice` view on a strided tensor exposes the underlying buffer
    // starting at the slice offset, so subsequent rows remain `stride[0]`
    // elements apart. Use those strides instead of assuming the data is tightly
    // packed so we read the correct row segments when rebuilding the
    // interleaved layout.
    let q_row_stride = q_expected.strides[0];
    let q_col_stride = *q_expected.strides.get(1).unwrap_or(&1);
    let k_row_stride = k_expected.strides[0];
    let k_col_stride = *k_expected.strides.get(1).unwrap_or(&1);
    let v_row_stride = v_expected.strides[0];
    let v_col_stride = *v_expected.strides.get(1).unwrap_or(&1);

    for row in 0..rows {
        let q_base = row * q_row_stride;
        for col in 0..feature_dim {
            total_flat_expected.push(q_expected_slice[q_base + col * q_col_stride]);
        }

        let k_base = row * k_row_stride;
        for col in 0..kv_dim {
            total_flat_expected.push(k_expected_slice[k_base + col * k_col_stride]);
        }

        let v_base = row * v_row_stride;
        for col in 0..kv_dim {
            total_flat_expected.push(v_expected_slice[v_base + col * v_col_stride]);
        }
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
