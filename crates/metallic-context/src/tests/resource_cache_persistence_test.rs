use crate::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage};

fn build_tensor(ctx: &Context<F32Element>, dims: &[usize], data: &[f32]) -> Result<Tensor<F32Element>, MetalError> {
    Tensor::new(dims.to_vec(), TensorStorage::Dedicated(ctx), TensorInit::CopyFrom(data))
}

#[test]
fn resource_cache_survives_synchronize() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    let q_data: Vec<f32> = (0..24).map(|idx| idx as f32 * 0.01).collect();
    let k_data: Vec<f32> = (0..24).map(|idx| idx as f32 * -0.02).collect();
    let v_data: Vec<f32> = (0..24).map(|idx| idx as f32 * 0.03).collect();

    let q = build_tensor(&ctx, &[1, 3, 8], &q_data)?;
    let k = build_tensor(&ctx, &[1, 3, 8], &k_data)?;
    let v = build_tensor(&ctx, &[1, 3, 8], &v_data)?;

    // Populate the cache by issuing SDPA, which records a cache entry for the scale helper.
    let _ = ctx.scaled_dot_product_attention(&q, &k, &v, false)?;

    let stats_before = ctx.get_cache_stats().expect("resource cache should be initialized after dispatch");
    assert!(stats_before.sdpa.size > 0, "SDPA dispatch should populate the resource cache");

    ctx.synchronize();

    let stats_after = ctx.get_cache_stats().expect("resource cache should persist after synchronization");

    assert_eq!(
        stats_before.sdpa.size, stats_after.sdpa.size,
        "SDPA cache entries should survive a command buffer flush",
    );

    Ok(())
}
