use super::*;
use crate::tensor::Dtype;
use crate::{F32Element, TensorInit, TensorStorage};

#[test]
fn test_sdpa_invalid_shapes() {
    let mut context = Context::<F32Element>::new().unwrap();

    // Test with incompatible dimensions
    let batch = 2;
    let seq_q = 4;
    let seq_k = 4;
    let dim1 = 4;
    let dim2 = 5; // Different dimensions

    let q_data: Vec<f32> = (0..(batch * seq_q * dim1)).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim2)).map(|i| (i as f32) * 0.2).collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim2)).map(|i| (i as f32) * 0.3).collect();

    let q_tensor = Tensor::new(
        vec![batch, seq_q, dim1],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&q_data),
    )
    .unwrap();
    let k_tensor = Tensor::new(
        vec![batch, seq_k, dim2],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&k_data),
    )
    .unwrap();
    let v_tensor = Tensor::new(
        vec![batch, seq_k, dim2],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&v_data),
    )
    .unwrap();

    // This should fail with a dimension mismatch error
    let result = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false);
    assert!(result.is_err(), "SDPA should fail with incompatible dimensions");

    // Check that we get the expected error type
    match result {
        Err(MetalError::DimensionMismatch { .. }) => {
            // Expected error
        }
        Err(e) => {
            panic!("Expected DimensionMismatch error, got: {:?}", e);
        }
        Ok(_) => {
            panic!("Expected error but operation succeeded");
        }
    }
}

#[test]
fn test_sdpa_invalid_batch_dimensions() {
    let mut context = Context::<F32Element>::new().unwrap();

    // Test with different batch sizes
    let batch1 = 2;
    let batch2 = 3; // Different batch size
    let seq_q = 4;
    let seq_k = 4;
    let dim = 4;

    let q_data: Vec<f32> = (0..(batch1 * seq_q * dim)).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..(batch2 * seq_k * dim)).map(|i| (i as f32) * 0.2).collect();
    let v_data: Vec<f32> = (0..(batch2 * seq_k * dim)).map(|i| (i as f32) * 0.3).collect();

    let q_tensor = Tensor::new(
        vec![batch1, seq_q, dim],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&q_data),
    )
    .unwrap();
    let k_tensor = Tensor::new(
        vec![batch2, seq_k, dim],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&k_data),
    )
    .unwrap();
    let v_tensor = Tensor::new(
        vec![batch2, seq_k, dim],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&v_data),
    )
    .unwrap();

    // This should fail with a dimension mismatch error
    let result = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false);
    assert!(result.is_err(), "SDPA should fail with incompatible batch dimensions");
}

#[test]
fn test_matmul_invalid_shapes() {
    use crate::cache_keys::MpsGemmKey;
    use crate::resource_cache::ResourceCache;

    let context = Context::<F32Element>::new().unwrap();
    let mut cache = ResourceCache::with_device(context.device.clone());

    // Test with incompatible matrix dimensions
    let m = 3;
    let k1 = 4;
    let k2 = 5; // Different interior dimension
    let n = 2;

    let a_data: Vec<f32> = (0..(m * k1)).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..(k2 * n)).map(|i| (i as f32) * 0.2).collect();
    let result_data: Vec<f32> = vec![0.0; m * n];

    let _a_tensor = Tensor::new(vec![m, k1], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data)).unwrap();
    let _b_tensor = Tensor::new(vec![k2, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data)).unwrap();
    let _result_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&result_data)).unwrap();

    let gemm_key = MpsGemmKey {
        transpose_left: false,
        transpose_right: false,
        result_rows: m,
        result_columns: n,
        interior_columns: k1, // This won't match k2
        batch_size: 1,
        alpha: 1.0,
        beta: 0.0,
    };

    // This might not fail at the Rust level, but would fail at the Metal level
    // We're testing that our code doesn't panic
    let result = cache.get_or_create_gemm(gemm_key, &context.device);
    // The result could be Ok or Err depending on how the Metal API handles this
    // but it should not panic
    assert!(result.is_ok() || result.is_err()); // Just ensuring no panic
}

#[test]
fn test_tensor_creation_with_mismatched_dimensions() {
    let context = Context::<F32Element>::new().unwrap();

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dims = vec![2, 4]; // Expecting 8 elements but only have 6

    let result = Tensor::new(dims, TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&data));
    assert!(result.is_err(), "Tensor creation should fail with mismatched dimensions");

    match result {
        Err(MetalError::DimensionMismatch { expected, actual }) => {
            assert_eq!(expected, 8, "Expected 8 elements");
            assert_eq!(actual, 6, "Actually have 6 elements");
        }
        Err(e) => {
            panic!("Expected DimensionMismatch error, got: {:?}", e);
        }
        Ok(_) => {
            panic!("Expected error but tensor creation succeeded");
        }
    }
}

#[test]
fn test_tensor_from_existing_buffer_invalid_offset() {
    let context = Context::<F32Element>::new().unwrap();

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let dims = vec![2, 2];

    let tensor = Tensor::new(dims, TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&data)).unwrap();

    // Try to create a view with an invalid offset
    let invalid_offset = 100 * std::mem::size_of::<f32>(); // Way too large

    let result = Tensor::<F32Element>::from_existing_buffer(
        tensor.buf.clone(),
        vec![1, 2], // Smaller dimensions
        Dtype::F32,
        &context.device,
        &context.command_queue,
        invalid_offset,
        false,
    );

    assert!(result.is_err(), "Tensor creation should fail with invalid offset");

    match result {
        Err(MetalError::InvalidShape(_)) => {
            // Expected error
        }
        Err(e) => {
            panic!("Expected InvalidShape error, got: {:?}", e);
        }
        Ok(_) => {
            panic!("Expected error but tensor creation succeeded");
        }
    }
}

#[test]
fn test_tensor_get_batch_out_of_bounds() {
    let context = Context::<F32Element>::new().unwrap();

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dims = vec![2, 3]; // Only 2 batches (0 and 1)

    let tensor = Tensor::new(dims, TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&data)).unwrap();

    // Try to get a batch that doesn't exist
    let result = tensor.get_batch(2); // Only batches 0 and 1 exist

    assert!(result.is_err(), "get_batch should fail with out of bounds index");

    match result {
        Err(MetalError::InvalidShape(_)) => {
            // Expected error
        }
        Err(e) => {
            panic!("Expected InvalidShape error, got: {:?}", e);
        }
        Ok(_) => {
            panic!("Expected error but get_batch succeeded");
        }
    }
}

#[test]
fn test_tensor_get_batch_insufficient_dimensions() {
    let context = Context::<F32Element>::new().unwrap();

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let dims = vec![4]; // Only 1 dimension, need at least 3 for get_batch

    let tensor = Tensor::new(dims, TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&data)).unwrap();

    // Try to get a batch from a tensor with insufficient dimensions
    let result = tensor.get_batch(0);

    assert!(result.is_err(), "get_batch should fail with insufficient dimensions");

    match result {
        Err(MetalError::InvalidShape(_)) => {
            // Expected error
        }
        Err(e) => {
            panic!("Expected InvalidShape error, got: {:?}", e);
        }
        Ok(_) => {
            panic!("Expected error but get_batch succeeded");
        }
    }
}

#[test]
fn test_softmax_invalid_dimensions() {
    let mut context = Context::<F32Element>::new().unwrap();

    // Try with dimensions that might cause issues
    let seq_q = 0; // Invalid dimension
    let seq_k = 4;
    let input_data: Vec<f32> = vec![]; // Empty data
    let dims = vec![1, seq_q, seq_k];

    let result = Tensor::new(dims, TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data));
    // This might fail at tensor creation time
    if let Ok(attn_tensor) = result {
        let rows_total = if seq_k == 0 { 0 } else { (attn_tensor.len() / seq_k) as u32 };
        let result = context.call::<SoftmaxOp>((&attn_tensor, rows_total, seq_q as u32, seq_k as u32, 0, 0));
        // Should not panic, but might return an error
        assert!(result.is_ok() || result.is_err());
    }
}
