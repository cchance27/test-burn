use crate::metallic::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage};

fn build_tensor(ctx: &Context<F32Element>, dims: &[usize], data: &[f32]) -> Result<Tensor<F32Element>, MetalError> {
    Tensor::new(dims.to_vec(), TensorStorage::Dedicated(ctx), TensorInit::CopyFrom(data))
}

#[test]
fn resource_cache_survives_synchronize() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    let a_data: Vec<f32> = (0..6).map(|idx| idx as f32 + 1.0).collect();
    let b_data: Vec<f32> = (0..6).map(|idx| (idx as f32 + 1.0) * 0.5).collect();

    let a = build_tensor(&ctx, &[2, 3], &a_data)?;
    let b = build_tensor(&ctx, &[3, 2], &b_data)?;

    // Populate the cache by issuing a matmul that exercises the MPS backend.
    let _ = ctx.matmul(&a, &b, false, false)?;

    let stats_before = ctx.get_cache_stats().expect("resource cache should be initialized after dispatch");
    assert!(
        stats_before.gemm_cache_size > 0 || stats_before.descriptor_cache_size > 0,
        "matmul dispatch should populate the resource cache"
    );

    ctx.synchronize();

    let stats_after = ctx.get_cache_stats().expect("resource cache should persist after synchronization");

    assert_eq!(
        stats_before.gemm_cache_size, stats_after.gemm_cache_size,
        "GEMM cache entries should survive a command buffer flush",
    );
    assert_eq!(
        stats_before.descriptor_cache_size, stats_after.descriptor_cache_size,
        "Descriptor cache entries should survive a command buffer flush",
    );
    assert_eq!(
        stats_before.softmax_cache_size, stats_after.softmax_cache_size,
        "Softmax cache entries should survive a command buffer flush",
    );
    assert_eq!(
        stats_before.sdpa_cache_size, stats_after.sdpa_cache_size,
        "SDPA cache entries should survive a command buffer flush",
    );

    Ok(())
}
