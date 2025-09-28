use super::*;

#[test]
fn test_sdpa_determinism_non_causal() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let batch = 2;
    let seq_q = 4;
    let seq_k = 4;
    let dim = 4;

    // Fixed input data
    let q_data: Vec<f32> = (0..(batch * seq_q * dim)).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim)).map(|i| (i as f32) * 0.2).collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim)).map(|i| (i as f32) * 0.3).collect();

    // Run SDPA multiple times
    let mut results: Vec<Vec<f32>> = Vec::new();

    for i in 0..5 {
        let q_tensor = Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &context)?;
        let k_tensor = Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &context)?;
        let v_tensor = Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &context)?;

        let result = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)?;
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
    let q_data: Vec<f32> = (0..(batch * seq_q * dim)).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim)).map(|i| (i as f32) * 0.2).collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim)).map(|i| (i as f32) * 0.3).collect();

    // Run SDPA multiple times
    let mut results: Vec<Vec<f32>> = Vec::new();

    for i in 0..5 {
        let q_tensor = Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &context)?;
        let k_tensor = Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &context)?;
        let v_tensor = Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &context)?;

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
    use crate::metallic::kernels::matmul::MatMulOp;

    let mut context = Context::new()?;

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
        let _result_tensor = Tensor::create_tensor_from_slice(&result_data, vec![m, n], &context)?;

        // Use the new kernel system
        let result = context.matmul(&a_tensor, &b_tensor, false, false)?;
        context.synchronize();

        results.push(result.as_slice().to_vec());
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

    let seq_q = 4;
    let seq_k = 4;

    // Fixed input data
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.1).collect();

    // Run softmax multiple times
    let mut results: Vec<Vec<f32>> = Vec::new();

    for _ in 0..5 {
        let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
        let attn_tensor = input_tensor.clone();

        // Apply softmax using the new kernel system (in-place operation)
        let result = context.call::<SoftmaxOp>((&attn_tensor, seq_q as u32, seq_k as u32, 0, 0))?;
        context.synchronize();

        results.push(result.as_slice().to_vec());
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
    let mut context = Context::new()?;

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
        let add_result = tensor1.add_elem(&tensor2, &mut context)?.to_vec();
        add_results.push(add_result);

        // Multiplication
        let mul_result = tensor1.mul_elem(&tensor2, &mut context)?.to_vec();
        mul_results.push(mul_result);
    }

    // All addition results should be identical
    let first_add_result = &add_results[0];
    for (i, result) in add_results.iter().enumerate().skip(1) {
        assert_eq!(first_add_result, result, "Non-deterministic addition result in run {}", i);
    }

    // All multiplication results should be identical
    let first_mul_result = &mul_results[0];
    for (i, result) in mul_results.iter().enumerate().skip(1) {
        assert_eq!(first_mul_result, result, "Non-deterministic multiplication result in run {}", i);
    }

    Ok(())
}
