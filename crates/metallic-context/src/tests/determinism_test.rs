use super::*;
use crate::{F32Element, TensorElement, TensorInit, TensorStorage, tensor::TensorType};

fn softmax_rows_total<T: TensorElement>(attn_tensor: &Tensor<T>, seq_k: usize) -> u32 {
    if seq_k == 0 { 0 } else { (attn_tensor.len() / seq_k) as u32 }
}

#[test]
fn test_sdpa_determinism_non_causal() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;

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
        let q_tensor = Tensor::new(
            vec![batch, seq_q, dim],
            TensorStorage::Dedicated(&context),
            TensorInit::CopyFrom(&q_data),
        )?;
        let k_tensor = Tensor::new(
            vec![batch, seq_k, dim],
            TensorStorage::Dedicated(&context),
            TensorInit::CopyFrom(&k_data),
        )?;
        let v_tensor = Tensor::new(
            vec![batch, seq_k, dim],
            TensorStorage::Dedicated(&context),
            TensorInit::CopyFrom(&v_data),
        )?;

        let result = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)?;
        results.push(result.as_slice().to_vec());

        // Reinitialize context for each run to ensure clean state
        if i < 4 {
            context = Context::<F32Element>::new()?;
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
    let mut context = Context::<F32Element>::new()?;

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
        let q_tensor = Tensor::new(
            vec![batch, seq_q, dim],
            TensorStorage::Dedicated(&context),
            TensorInit::CopyFrom(&q_data),
        )?;
        let k_tensor = Tensor::new(
            vec![batch, seq_k, dim],
            TensorStorage::Dedicated(&context),
            TensorInit::CopyFrom(&k_data),
        )?;
        let v_tensor = Tensor::new(
            vec![batch, seq_k, dim],
            TensorStorage::Dedicated(&context),
            TensorInit::CopyFrom(&v_data),
        )?;

        let result = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, true)?;
        results.push(result.as_slice().to_vec());

        // Reinitialize context for each run to ensure clean state
        if i < 4 {
            context = Context::<F32Element>::new()?;
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
    let mut context = Context::<F32Element>::new()?;

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
        let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
        let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
        let _result_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&result_data))?;

        // Use the new kernel system
        let result = context.matmul(&a_tensor, &TensorType::Dense(&b_tensor), false, false, None, None, None)?;
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
    let mut context = Context::<F32Element>::new()?;

    let seq_q = 4;
    let seq_k = 4;

    // Fixed input data
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.1).collect();

    // Run softmax multiple times
    let mut results: Vec<Vec<f32>> = Vec::new();

    for _ in 0..5 {
        let input_tensor = Tensor::new(
            vec![seq_q, seq_k],
            TensorStorage::Dedicated(&context),
            TensorInit::CopyFrom(&input_data),
        )?; // Reshape to 2D as expected by kernel
        let attn_tensor = input_tensor.clone();

        // Apply softmax using the new kernel system (in-place operation)
        let rows_total = softmax_rows_total(&attn_tensor, seq_k);
        let result = context
            .call::<crate::kernels::softmax_kernel::SoftmaxKernelOp>((&attn_tensor, rows_total, seq_q as u32, seq_k as u32, 0, 0), None)?;
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
    let mut context = Context::<F32Element>::new()?;

    let dims = vec![2, 3];
    let data1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data2: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    // Run tensor operations multiple times
    let mut add_results: Vec<Vec<f32>> = Vec::new();
    let mut mul_results: Vec<Vec<f32>> = Vec::new();

    for _ in 0..5 {
        let tensor1 = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&data1))?;
        let tensor2 = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&data2))?;

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
