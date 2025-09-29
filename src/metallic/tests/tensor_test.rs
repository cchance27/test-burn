use super::*;
use crate::metallic::tensor::Dtype;
use crate::metallic::{Context, F32Element, Tensor, TensorInit, TensorStorage};

#[test]
fn zeros_and_ones() {
    let mut ctx = Context::<F32Element>::new().unwrap();
    let t0 = Tensor::zeros(vec![2, 3, 4], &mut ctx, true).unwrap();
    assert!(t0.as_slice().iter().all(|&x| x == 0.0));
    let t1 = Tensor::ones(vec![2, 3, 4], &mut ctx).unwrap();
    assert!(t1.as_slice().iter().all(|&x| x == 1.0));
}

#[test]
fn zeros_like_and_ones_like() {
    let mut ctx = Context::<F32Element>::new().unwrap();
    let base = Tensor::new(
        vec![2, 2],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&[1.0, 2.0, 3.0, 4.0]),
    )
    .unwrap();
    let z = base.zeros_like(&mut ctx).unwrap();
    assert_eq!(z.dims(), base.dims());
    assert!(z.as_slice().iter().all(|&x| x == 0.0));
    let o = base.ones_like(&mut ctx).unwrap();
    assert_eq!(o.dims(), base.dims());
    assert!(o.as_slice().iter().all(|&x| x == 1.0));
}

#[test]
fn elementwise_ops_and_fill() {
    let mut ctx = Context::<F32Element>::new().unwrap();
    let a = Tensor::new(
        vec![2, 2],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&[1.0, 2.0, 3.0, 4.0]),
    )
    .unwrap();
    let b = Tensor::new(
        vec![2, 2],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&[5.0, 6.0, 7.0, 8.0]),
    )
    .unwrap();
    let c = a.add_elem(&b, &mut ctx).unwrap();
    assert_eq!(c.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
    let d = a.mul_elem(&b, &mut ctx).unwrap();
    assert_eq!(d.as_slice(), &[5.0, 12.0, 21.0, 32.0]);
    let mut e = a.add_scalar(10.0, &mut ctx).unwrap();
    assert_eq!(e.as_slice(), &[11.0, 12.0, 13.0, 14.0]);
    e.fill(2.5);
    assert!(e.as_slice().iter().all(|&x| (x - 2.5).abs() < 1e-12));
}

#[test]
fn get_batch_and_from_existing_buffer() {
    let ctx = Context::<F32Element>::new().unwrap();
    // base tensor: shape [2,3,4], values 0..24
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let mut base = Tensor::new(vec![2, 3, 4], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized).unwrap();
    base.as_mut_slice().copy_from_slice(&data);

    // get second batch
    let b1 = base.get_batch(1).unwrap();
    assert_eq!(b1.dims(), &[3, 4]);
    // expected slice
    let expected: Vec<f32> = data[12..].to_vec();
    assert_eq!(b1.as_slice(), expected.as_slice());

    // wrap the second batch region using from_existing_buffer
    let offset_bytes = 12 * std::mem::size_of::<f32>();
    let view = Tensor::<F32Element>::from_existing_buffer(
        base.buf.clone(),
        vec![3, 4],
        Dtype::F32,
        &base.device,
        &ctx.command_queue,
        offset_bytes,
        false,
    )
    .unwrap();
    assert_eq!(view.as_slice(), expected.as_slice());
}

#[test]
fn arange_helper() {
    let mut ctx = Context::<F32Element>::new().unwrap();
    let t = Tensor::arange(6, vec![2, 3], &mut ctx).unwrap();
    ctx.synchronize();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn from_slice_helper() {
    let ctx = Context::<F32Element>::new().unwrap();
    let v = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let t = Tensor::new(vec![2, 3], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&v)).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.to_vec(), v);
}

#[test]
fn pool_growth_behavior() {
    let mut ctx = Context::<F32Element>::new().unwrap();

    // Initial chunk is 256MB, allocate tensors that would exceed this
    // Each f32 is 4 bytes, so 256MB = 67,108,864 elements
    // Let's allocate several large tensors to force chunk growth

    // First, allocate a tensor that takes most of the first chunk
    let large_dims = vec![8, 512, 512]; // 8*512*512*4 = 8MB
    let t1 = Tensor::zeros(large_dims.clone(), &mut ctx, true).unwrap();
    assert_eq!(t1.dims(), &large_dims);

    // Allocate more to fill first chunk and trigger growth
    let mut tensors = vec![t1];
    for _i in 0..35 {
        // 35 * 8MB = 280MB > 256MB initial chunk
        let t = Tensor::ones(large_dims.clone(), &mut ctx).unwrap();
        assert_eq!(t.dims(), &large_dims);
        tensors.push(t);
    }

    ctx.synchronize();

    // Verify all tensors are valid and have correct values
    assert!(tensors[0].as_slice().iter().all(|&x| x == 0.0));
    for t in &tensors[1..] {
        assert!(t.as_slice().iter().all(|&x| x == 1.0));
    }

    // Check that pool has grown
    assert!(ctx.pool.num_chunks() > 1, "Pool should have grown to multiple chunks");
}

#[test]
fn pool_reset_invalidates_tensors() {
    let mut ctx = Context::<F32Element>::new().unwrap();

    // Allocate some tensors
    let t1 = Tensor::zeros(vec![100, 100], &mut ctx, true).unwrap();
    let t2 = Tensor::ones(vec![100, 100], &mut ctx).unwrap();

    ctx.synchronize();

    // Verify they work
    assert!(t1.as_slice().iter().all(|&x| x == 0.0));
    assert!(t2.as_slice().iter().all(|&x| x == 1.0));

    // Reset pool
    ctx.pool.reset();

    // After reset, old tensors are invalidated - they may contain garbage data
    // We can't reliably test the contents, but we can verify the pool state
    assert_eq!(ctx.pool.current_chunk_index(), 0);
    assert_eq!(ctx.pool.chunk_cursors(), vec![0; ctx.pool.num_chunks()]);

    // Allocate new tensors after reset - should work fine
    let t3 = Tensor::zeros(vec![50, 50], &mut ctx, true).unwrap();
    let t4 = Tensor::ones(vec![50, 50], &mut ctx).unwrap();

    ctx.synchronize();
    
    assert!(t3.as_slice().iter().all(|&x| x == 0.0));
    assert!(t4.as_slice().iter().all(|&x| x == 1.0));
}

#[test]
fn offset_correctness_across_chunks() {
    let mut ctx = Context::<F32Element>::new().unwrap();

    // Allocate first tensor to take some space
    let t1 = Tensor::zeros(vec![1000, 1000], &mut ctx, true).unwrap(); // ~4MB
    // Allocate second tensor - should be at non-zero offset
    let t2 = Tensor::ones(vec![100, 100], &mut ctx).unwrap(); // ~40KB

    ctx.synchronize();

    // Verify t2 has non-zero offset (proves it's not at buffer start)
    assert!(t2.offset > 0, "Second tensor should have non-zero offset");

    // Perform elementwise operation on t2 (uses GPU kernel with offset)
    let t3 = t2.add_elem(&t2, &mut ctx).unwrap(); // Should result in tensor of 2.0s

    ctx.synchronize();

    // Verify the operation worked correctly despite offset
    assert!(t3.as_slice().iter().all(|&x| (x - 2.0).abs() < 1e-6));

    // Also verify t1 is still correct (zeros)
    assert!(t1.as_slice().iter().all(|&x| x == 0.0));
}

#[test]
fn cpu_fill_with_offsets() {
    let mut ctx = Context::<F32Element>::new().unwrap();

    // Allocate first tensor to create offset
    let _t1 = Tensor::zeros(vec![500, 500], &mut ctx, true).unwrap();
    // Allocate second tensor - should use CPU fill due to small size
    let t2 = Tensor::ones(vec![10, 10], &mut ctx).unwrap();

    ctx.synchronize();

    // Verify offset is non-zero
    assert!(t2.offset > 0);

    // Verify CPU fill worked correctly despite offset
    assert!(t2.as_slice().iter().all(|&x| x == 1.0));

    // Test zeros as well
    let t3 = Tensor::zeros(vec![5, 5], &mut ctx, true).unwrap();
    ctx.synchronize();

    assert!(t3.offset > 0);
    assert!(t3.as_slice().iter().all(|&x| x == 0.0));
}

#[test]
fn random_determinism_with_seed_counter() {
    let mut ctx1 = Context::<F32Element>::new().unwrap();
    let mut ctx2 = Context::<F32Element>::new().unwrap();

    // Create tensors with same seed counter state
    let t1 = Tensor::random_uniform(vec![10, 10], &mut ctx1).unwrap();
    let t2 = Tensor::random_uniform(vec![10, 10], &mut ctx2).unwrap();

    // Ensure we synchronize our backend
    ctx1.synchronize();
    ctx2.synchronize();

    // They should be identical since both start with seed_counter = 0
    assert_eq!(t1.as_slice(), t2.as_slice());

    // Create another tensor in ctx1 - should be different due to incremented counter
    let t3 = Tensor::random_uniform(vec![10, 10], &mut ctx1).unwrap();
    assert_ne!(t1.as_slice(), t3.as_slice());

    // Create tensor in ctx2 with same counter state as t3 - should match
    let t4 = Tensor::random_uniform(vec![10, 10], &mut ctx2).unwrap();
    assert_eq!(t3.as_slice(), t4.as_slice());
}

#[test]
fn batched_operations_correctness() {
    let mut ctx = Context::<F32Element>::new().unwrap();

    // Test batched operations
    let cb = crate::metallic::operation::CommandBuffer::new(&ctx.command_queue).unwrap();
    let t1 = Tensor::zeros_batched(vec![5, 5], &cb, &mut ctx).unwrap();
    let t2 = Tensor::ones_batched(vec![5, 5], &cb, &mut ctx).unwrap();
    let t3 = Tensor::arange_batched(10, vec![2, 5], &cb, &mut ctx).unwrap();

    // Commit and wait for completion
    cb.commit();
    cb.wait();

    ctx.synchronize();

    // Verify results
    assert!(t1.as_slice().iter().all(|&x| x == 0.0));
    assert!(t2.as_slice().iter().all(|&x| x == 1.0));
    assert_eq!(t3.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn synchronization_contract() {
    let mut ctx = Context::<F32Element>::new().unwrap();

    // Immediate helpers should not wait - create tensor but don't read immediately
    let t1 = Tensor::random_uniform(vec![100, 100], &mut ctx).unwrap();

    // Do some other work to ensure GPU operations complete
    let _t2 = Tensor::zeros(vec![10, 10], &mut ctx, true).unwrap();
    let _t3 = Tensor::ones(vec![10, 10], &mut ctx).unwrap();

    // Now read t1 - should be fine since other operations forced synchronization
    let _val = t1.as_slice()[0]; // Just access to ensure it's readable

    // Test explicit synchronization with batched operations
    let cb = crate::metallic::operation::CommandBuffer::new(&ctx.command_queue).unwrap();
    let t4 = Tensor::random_uniform_batched(vec![50, 50], &cb, &mut ctx).unwrap();
    cb.commit();
    cb.wait();

    // Now safe to read t4
    let _val = t4.as_slice()[0];
}

#[test]
fn strides_and_dtype() {
    let mut ctx = Context::<F32Element>::new().unwrap();

    // Test contiguous strides computation
    let t = Tensor::zeros(vec![2, 3, 4], &mut ctx, true).unwrap();
    ctx.synchronize();

    assert_eq!(t.dims, &[2, 3, 4]);
    assert_eq!(t.dtype, Dtype::F32);
    // For contiguous [2,3,4], strides should be [12, 4, 1]
    assert_eq!(t.strides, &[12, 4, 1]);

    // Test size_bytes
    assert_eq!(t.size_bytes(), 2 * 3 * 4 * 4); // 96 bytes for f32

    // Test metal format
    assert_eq!(t.dtype.metal_format(), "float");
}

#[test]
fn borrow_host_buffer() {
    let mut ctx = Context::<F32Element>::new().unwrap();

    // Create data that will be managed by the caller
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dims = vec![2, 3];

    // Create tensor without copying - caller must ensure data lives long enough
    let t = Tensor::new(dims, TensorStorage::Dedicated(&ctx), TensorInit::BorrowHost(&data)).unwrap();

    ctx.synchronize();

    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Note: in real usage, the caller must ensure `data` outlives the tensor
    // This is just a basic functionality test
}
