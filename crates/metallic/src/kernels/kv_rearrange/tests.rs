use super::*;
use crate::F32Element;

#[test]
fn test_kv_rearrange_logic() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;
    let batch = 2usize;
    let seq = 3usize;
    let n_heads = 4usize;
    let n_kv_heads = 2usize;
    let head_dim = 2usize;
    let kv_head_dim = 3usize;
    let d_model = n_heads * head_dim;
    let kv_dim = n_kv_heads * kv_head_dim;
    let fused_dim = d_model + 2 * kv_dim;
    let rows = batch * seq;

    // Create a fused QKV tensor so slicing produces strided views.
    let fused_data: Vec<f32> = (0..rows * fused_dim).map(|i| i as f32).collect();
    let fused = Tensor::new(
        vec![rows, fused_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&fused_data),
    )?;

    let q_mat = fused.slice_last_dim(0..d_model)?;
    let k_mat = fused.slice_last_dim(d_model..d_model + kv_dim)?;
    let v_mat = fused.slice_last_dim(d_model + kv_dim..fused_dim)?;

    let q_result = ctx.call::<KvRearrangeOp>((
        q_mat.clone(),
        d_model as u32,
        head_dim as u32,
        n_heads as u32,
        n_heads as u32,
        head_dim as u32,
        seq as u32,
    ))?;
    let k_result = ctx.call::<KvRearrangeOp>((
        k_mat.clone(),
        kv_dim as u32,
        kv_head_dim as u32,
        n_kv_heads as u32,
        n_kv_heads as u32,
        kv_head_dim as u32,
        seq as u32,
    ))?;
    let v_result = ctx.call::<KvRearrangeOp>((
        v_mat.clone(),
        kv_dim as u32,
        kv_head_dim as u32,
        n_kv_heads as u32,
        n_kv_heads as u32,
        kv_head_dim as u32,
        seq as u32,
    ))?;

    ctx.synchronize();

    // Verify output dimensions for each projection.
    assert_eq!(q_result.dims(), &[batch * n_heads, seq, head_dim]);
    assert_eq!(k_result.dims(), &[batch * n_kv_heads, seq, kv_head_dim]);
    assert_eq!(v_result.dims(), &[batch * n_kv_heads, seq, kv_head_dim]);

    // CPU reference for verification using the original fused buffer.
    let row_stride = fused_dim;
    let q_expected = cpu_reference_rearrange(&fused_data, row_stride, batch, seq, n_heads, n_heads, head_dim, head_dim, 0);
    let k_expected = cpu_reference_rearrange(
        &fused_data,
        row_stride,
        batch,
        seq,
        n_kv_heads,
        n_kv_heads,
        kv_head_dim,
        kv_head_dim,
        d_model,
    );
    let v_expected = cpu_reference_rearrange(
        &fused_data,
        row_stride,
        batch,
        seq,
        n_kv_heads,
        n_kv_heads,
        kv_head_dim,
        kv_head_dim,
        d_model + kv_dim,
    );

    assert_eq!(q_result.as_slice(), q_expected.as_slice());
    assert_eq!(k_result.as_slice(), k_expected.as_slice());
    assert_eq!(v_result.as_slice(), v_expected.as_slice());

    Ok(())
}
