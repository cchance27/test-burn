use super::*;

#[test]
fn test_sdpa_determinism_non_causal() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let batch = 2;
    let seq_q = 4;
    let seq_k = 4;
    let dim = 4;

    // Fixed input data
    let q_data: Vec<f32> = (0..(batch * seq_q * dim))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.2)
        .collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.3)
        .collect();

    // Run SDPA multiple times
    let mut results: Vec<Vec<f32>> = Vec::new();

    for i in 0..5 {
        let q_tensor =
            Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &context)?;
        let k_tensor =
            Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &context)?;
        let v_tensor =
            Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &context)?;

        let result =
            context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)?;
        results.push(result.as_slice().to_vec());

        // Reinitialize context for each run to ensure clean state
        if i < 4 {
            context = Context::new()?;
        }
    }

    // All results should be identical
    let first_result = &results[0];
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(
            first_result.len(),
            result.len(),
            "Result {} has different length than first result",
            i
        );

        for (j, (&val1, &val2)) in first_result.iter().zip(result.iter()).enumerate() {
            let diff = (val1 - val2).abs();
            assert!(
                diff < 1e-6,
                "Non-deterministic result at index {} between run 0 and run {}: {} vs {}",
                j,
                i,
                val1,
                val2
            );
        }
    }

    Ok(())
}

#[test]
fn test_sdpa_determinism_causal() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let batch = 2;
    let seq_q = 4;
    let seq_k = 4;
    let dim = 4;

    // Fixed input data
    let q_data: Vec<f32> = (0..(batch * seq_q * dim))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.2)
        .collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.3)
        .collect();

    // Run SDPA multiple times
    let mut results: Vec<Vec<f32>> = Vec::new();

    for i in 0..5 {
        let q_tensor =
            Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &context)?;
        let k_tensor =
            Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &context)?;
        let v_tensor =
            Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &context)?;

        let result = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, true)?;
        results.push(result.as_slice().to_vec());

        // Reinitialize context for each run to ensure clean state
        if i < 4 {
            context = Context::new()?;
        }
    }

    // All results should be identical
    let first_result = &results[0];
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(
            first_result.len(),
            result.len(),
            "Result {} has different length than first result",
            i
        );

        for (j, (&val1, &val2)) in first_result.iter().zip(result.iter()).enumerate() {
            let diff = (val1 - val2).abs();
            assert!(
                diff < 1e-6,
                "Non-deterministic result at index {} between run 0 and run {}: {} vs {}",
                j,
                i,
                val1,
                val2
            );
        }
    }

    Ok(())
}

#[test]
fn test_matmul_determinism() -> Result<(), MetalError> {
    use crate::metallic::cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey};
    use crate::metallic::matmul::MatMulOperation;
    use crate::metallic::resource_cache::ResourceCache;

    let context = Context::new()?;

    let m = 4;
    let k = 4;
    let n = 4;

    // Fixed input data
    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.5).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.3).collect();
    let result_data: Vec<f32> = vec![0.0; m * n];

    // Run matmul multiple times
    let mut results: Vec<Vec<f32>> = Vec::new();

    for _ in 0..5 {
        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![m, k], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![k, n], &context)?;
        let result_tensor = Tensor::create_tensor_from_slice(&result_data, vec![m, n], &context)?;

        let mut cache = ResourceCache::new();

        let gemm_key = MpsGemmKey {
            transpose_left: false,
            transpose_right: false,
            result_rows: m,
            result_columns: n,
            interior_columns: k,
            alpha: 1.0,
            beta: 0.0,
        };
        let gemm_op = cache.get_or_create_gemm(gemm_key, &context.device)?;

        let bytes_per_elem: usize = core::mem::size_of::<f32>();

        let a_desc_key = MpsMatrixDescriptorKey {
            rows: m,
            columns: k,
            row_bytes: k * bytes_per_elem,
        };
        let b_desc_key = MpsMatrixDescriptorKey {
            rows: k,
            columns: n,
            row_bytes: n * bytes_per_elem,
        };
        let result_desc_key = MpsMatrixDescriptorKey {
            rows: m,
            columns: n,
            row_bytes: n * bytes_per_elem,
        };

        let matmul_op = MatMulOperation {
            left_buf: a_tensor.buf.clone(),
            left_offset: a_tensor.offset,
            right_buf: b_tensor.buf.clone(),
            right_offset: b_tensor.offset,
            result_buf: result_tensor.buf.clone(),
            result_offset: result_tensor.offset,
            left_desc: cache.get_or_create_descriptor(a_desc_key, &context.device)?,
            right_desc: cache.get_or_create_descriptor(b_desc_key, &context.device)?,
            result_desc: cache.get_or_create_descriptor(result_desc_key, &context.device)?,
            gemm: gemm_op,
        };

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        matmul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        results.push(result_tensor.as_slice().to_vec());
    }

    // All results should be identical
    let first_result = &results[0];
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(
            first_result.len(),
            result.len(),
            "Result {} has different length than first result",
            i
        );

        for (j, (&val1, &val2)) in first_result.iter().zip(result.iter()).enumerate() {
            let diff = (val1 - val2).abs();
            assert!(
                diff < 1e-6,
                "Non-deterministic result at index {} between run 0 and run {}: {} vs {}",
                j,
                i,
                val1,
                val2
            );
        }
    }

    Ok(())
}

#[test]
fn test_softmax_determinism() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    crate::metallic::ensure_fused_softmax_pipeline(&mut context)?;

    let seq_q = 4;
    let seq_k = 4;

    // Fixed input data
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.1).collect();
    let dims = vec![1, seq_q, seq_k];

    // Run softmax multiple times
    let mut results: Vec<Vec<f32>> = Vec::new();

    for _ in 0..5 {
        let attn_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;

        use crate::metallic::resource_cache::ResourceCache;
        use crate::metallic::softmax::SoftmaxOperation;

        let sm_op = SoftmaxOperation {
            attn_buf: attn_tensor.buf.clone(),
            attn_offset: attn_tensor.offset,
            seq_q: seq_q as u32,
            seq_k: seq_k as u32,
            causal: 0,
            pipeline: context.fused_softmax_pipeline.as_ref().unwrap().clone(),
        };

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        let mut cache = ResourceCache::new();
        sm_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        results.push(attn_tensor.as_slice().to_vec());
    }

    // All results should be identical
    let first_result = &results[0];
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(
            first_result.len(),
            result.len(),
            "Result {} has different length than first result",
            i
        );

        for (j, (&val1, &val2)) in first_result.iter().zip(result.iter()).enumerate() {
            let diff = (val1 - val2).abs();
            assert!(
                diff < 1e-6,
                "Non-deterministic result at index {} between run 0 and run {}: {} vs {}",
                j,
                i,
                val1,
                val2
            );
        }
    }

    Ok(())
}

#[test]
fn test_tensor_operations_determinism() -> Result<(), MetalError> {
    let context = Context::new()?;

    let dims = vec![2, 3];
    let data1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data2: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    // Run tensor operations multiple times
    let mut add_results: Vec<Vec<f32>> = Vec::new();
    let mut mul_results: Vec<Vec<f32>> = Vec::new();

    for _ in 0..5 {
        let tensor1 = Tensor::create_tensor_from_slice(&data1, dims.clone(), &context)?;
        let tensor2 = Tensor::create_tensor_from_slice(&data2, dims.clone(), &context)?;

        // Addition
        let add_result = (&tensor1 + &tensor2).to_vec();
        add_results.push(add_result);

        // Multiplication
        let mul_result = (&tensor1 * &tensor2).to_vec();
        mul_results.push(mul_result);
    }

    // All addition results should be identical
    let first_add_result = &add_results[0];
    for (i, result) in add_results.iter().enumerate().skip(1) {
        assert_eq!(
            first_add_result, result,
            "Non-deterministic addition result in run {}",
            i
        );
    }

    // All multiplication results should be identical
    let first_mul_result = &mul_results[0];
    for (i, result) in mul_results.iter().enumerate().skip(1) {
        assert_eq!(
            first_mul_result, result,
            "Non-deterministic multiplication result in run {}",
            i
        );
    }

    Ok(())
}
