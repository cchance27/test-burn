//! Comprehensive unit tests to verify numerical consistency between softmax kernel variants
//!
//! This test suite ensures that SoftmaxVecOp and SoftmaxBlockOp produce numerically
//! identical results to the original SoftmaxKernelOp for the same inputs.

use crate::{
    F16Element, F32Element,
    context::Context,
    kernels::{softmax_block::SoftmaxBlockOp, softmax_kernel::SoftmaxKernelOp, softmax_vec::SoftmaxVecOp},
    tensor::{Tensor, TensorElement, TensorInit, TensorStorage},
};

/// Test element type for numerical consistency verification
type TestElement = F16Element;
// use rand::{Rng, SeedableRng};
// use rand_chacha::ChaCha8Rng;

/// Test numerical consistency between softmax kernel variants
#[test]
fn test_softmax_kernel_numerical_consistency() {
    let mut ctx = Context::<TestElement>::new().unwrap();

    // Test various configurations
    let test_configs = vec![
        // (batch_size, seq_len, head_dim, causal)
        (1, 64, 128, false),
        (1, 128, 64, false),
        (1, 256, 128, false),
        (1, 512, 64, false),
        (1, 1024, 128, false),
        (1, 2048, 64, false),
        (2, 128, 128, false),
        (4, 64, 64, false),
        // Causal masking tests
        (1, 64, 128, true),
        (1, 128, 64, true),
        (1, 256, 128, true),
        (2, 128, 128, true),
    ];

    for (batch_size, seq_len, head_dim, causal) in test_configs {
        println!(
            "Testing config: batch={}, seq_len={}, head_dim={}, causal={}",
            batch_size, seq_len, head_dim, causal
        );

        // Create test input with deterministic pattern
        let total_elements = batch_size * seq_len * head_dim;
        let input_data: Vec<half::f16> = (0..total_elements)
            .map(|i| TestElement::from_f32((i as f32 - total_elements as f32 / 2.0) / 100.0))
            .collect();

        let input_orig = Tensor::new(
            vec![batch_size, seq_len, head_dim],
            TensorStorage::Dedicated(&ctx),
            TensorInit::CopyFrom(&input_data),
        )
        .unwrap();
        let input_vec = Tensor::new(
            vec![batch_size, seq_len, head_dim],
            TensorStorage::Dedicated(&ctx),
            TensorInit::CopyFrom(&input_data),
        )
        .unwrap();

        // Test original softmax kernel
        let original_result = ctx
            .call::<SoftmaxKernelOp>((
                &input_orig,
                (batch_size * seq_len) as u32,
                seq_len as u32,
                head_dim as u32,
                causal as u32,
                0u32,
            ))
            .unwrap();

        // Test vec-softmax (for seq_len <= 1024)
        if seq_len <= 1024 {
            let vec_result = ctx
                .call::<SoftmaxVecOp>((
                    &input_vec,
                    (batch_size * seq_len) as u32,
                    seq_len as u32,
                    head_dim as u32,
                    causal as u32,
                    0u32,
                ))
                .unwrap();

            // Row width is the last dimension (seq_k/head_dim)
            compare_softmax_results(&original_result, &vec_result, "vec-softmax", head_dim);
        }

        // Test block-softmax (for seq_len > 1024)
        if seq_len > 1024 {
            let input_block = Tensor::new(
                vec![batch_size, seq_len, head_dim],
                TensorStorage::Dedicated(&ctx),
                TensorInit::CopyFrom(&input_data),
            )
            .unwrap();
            let block_result = ctx
                .call::<SoftmaxBlockOp>((
                    &input_block,
                    (batch_size * seq_len) as u32,
                    seq_len as u32,
                    head_dim as u32,
                    32u32,
                    causal as u32,
                    0u32,
                ))
                .unwrap();

            // Row width is the last dimension (seq_k/head_dim)
            compare_softmax_results(&original_result, &block_result, "block-softmax", head_dim);
        }

        ctx.synchronize();
    }
}

/// Test edge cases and extreme values
#[test]
fn test_softmax_kernel_edge_cases() {
    let mut ctx = Context::<TestElement>::new().unwrap();

    // Test extreme values that could cause numerical instability
    let edge_cases = vec![
        // Very small values (underflow)
        vec![TestElement::from_f32(-100.0); 128],
        // Very large values (overflow)
        vec![TestElement::from_f32(100.0); 128],
        // Mixed extreme values
        {
            let mut data = vec![TestElement::from_f32(0.0); 128];
            data[0] = TestElement::from_f32(-50.0);
            data[63] = TestElement::from_f32(50.0);
            data[127] = TestElement::from_f32(-25.0);
            data
        },
        // All zeros
        vec![TestElement::from_f32(0.0); 128],
        // Single non-zero value
        {
            let mut data = vec![TestElement::from_f32(0.0); 128];
            data[64] = TestElement::from_f32(1.0);
            data
        },
    ];

    for (i, input_data) in edge_cases.into_iter().enumerate() {
        println!("Testing edge case {}", i);

        let input_orig = Tensor::new(vec![1, 128, 1], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&input_data)).unwrap();
        let input_vec = Tensor::new(vec![1, 128, 1], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&input_data)).unwrap();

        // Test original vs vec-softmax
        let original_result = ctx.call::<SoftmaxKernelOp>((&input_orig, 128, 128, 1, false as u32, 0)).unwrap();

        let vec_result = ctx.call::<SoftmaxVecOp>((&input_vec, 128u32, 128u32, 1u32, 0u32, 0u32)).unwrap();

        // Row width is the last dimension (seq_k = 1)
        compare_softmax_results(&original_result, &vec_result, &format!("edge_case_{}", i), 1);

        ctx.synchronize();
    }
}

/// Test causal masking consistency
#[test]
fn test_softmax_causal_masking_consistency() {
    let mut ctx = Context::<TestElement>::new().unwrap();

    // Create input with known pattern for causal masking verification
    let seq_len = 128;
    let head_dim = 128; // ensure there are multiple columns to mask
    let mut input_data = vec![TestElement::from_f32(0.0); seq_len * head_dim];
    for r in 0..seq_len {
        for c in 0..head_dim {
            // Row-wise identical pattern across columns; mask will change distribution
            let val = (c as f32 - head_dim as f32 / 2.0) / 10.0;
            input_data[r * head_dim + c] = TestElement::from_f32(val);
        }
    }

    let input_nc_orig = Tensor::new(
        vec![1, seq_len, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )
    .unwrap();
    let input_nc_vec = Tensor::new(
        vec![1, seq_len, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )
    .unwrap();

    // Test non-causal case
    let non_causal_original = ctx
        .call::<SoftmaxKernelOp>((&input_nc_orig, seq_len as u32, seq_len as u32, head_dim as u32, false as u32, 0))
        .unwrap();

    let non_causal_vec = ctx
        .call::<SoftmaxVecOp>((&input_nc_vec, seq_len as u32, seq_len as u32, head_dim as u32, 0u32, 0u32))
        .unwrap();

    // Row width is the last dimension (seq_k = 1)
    compare_softmax_results(&non_causal_original, &non_causal_vec, "non_causal", head_dim);

    // Test causal case
    let input_c_orig = Tensor::new(
        vec![1, seq_len, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )
    .unwrap();
    let causal_original = ctx
        .call::<SoftmaxKernelOp>((&input_c_orig, seq_len as u32, seq_len as u32, head_dim as u32, true as u32, 0))
        .unwrap();

    let input_c_vec = Tensor::new(
        vec![1, seq_len, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )
    .unwrap();
    let causal_vec = ctx
        .call::<SoftmaxVecOp>((&input_c_vec, seq_len as u32, seq_len as u32, head_dim as u32, 1u32, 0u32))
        .unwrap();

    // Row width is the last dimension (seq_k = 1)
    compare_softmax_results(&causal_original, &causal_vec, "causal", head_dim);

    // Verify that causal and non-causal produce different results
    let original_data = non_causal_original.as_slice();
    let causal_data = causal_original.as_slice();
    let vec_data = non_causal_vec.as_slice();
    let causal_vec_data = causal_vec.as_slice();

    let mut diff_count = 0;
    for i in 0..original_data.len() {
        let orig_val = TestElement::to_f32(original_data[i]);
        let causal_val = TestElement::to_f32(causal_data[i]);
        let vec_val = TestElement::to_f32(vec_data[i]);
        let causal_vec_val = TestElement::to_f32(causal_vec_data[i]);

        if (orig_val - causal_val).abs() > 1e-6 {
            diff_count += 1;
        }
        if (vec_val - causal_vec_val).abs() > 1e-6 {
            diff_count += 1;
        }
    }

    assert!(diff_count > 0, "Causal and non-causal should produce different results");

    ctx.synchronize();
}

/// Helper function to compare two softmax results
fn compare_softmax_results(original: &Tensor<F16Element>, new_kernel: &Tensor<F16Element>, kernel_name: &str, row_width: usize) {
    let original_data = original.as_slice();
    let new_data = new_kernel.as_slice();

    assert_eq!(
        original_data.len(),
        new_data.len(),
        "Tensor sizes must match for {} comparison",
        kernel_name
    );

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut diff_count = 0;

    for (i, (orig, new)) in original_data.iter().zip(new_data.iter()).enumerate() {
        let orig_val = TestElement::to_f32(*orig);
        let new_val = TestElement::to_f32(*new);
        let diff = (orig_val - new_val).abs();

        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }

        if diff > 1e-5 {
            diff_count += 1;
        }
    }

    println!(
        "{} vs original: max_diff={:.8} at idx {}, significant_diffs={}/{}",
        kernel_name,
        max_diff,
        max_diff_idx,
        diff_count,
        original_data.len()
    );

    // Allow small f16 rounding differences; enforce tight but realistic threshold
    let epsilon = 2e-5;

    assert!(
        max_diff < epsilon,
        "{} kernel produces different results than original. Max diff: {} at index {}",
        kernel_name,
        max_diff,
        max_diff_idx
    );

    // Verify softmax properties: outputs should sum to 1 for each row
    let original_f32: Vec<f32> = original_data.iter().map(|&x| TestElement::to_f32(x)).collect();
    let new_f32: Vec<f32> = new_data.iter().map(|&x| TestElement::to_f32(x)).collect();
    verify_softmax_properties(&original_f32, "original", row_width);
    verify_softmax_properties(&new_f32, kernel_name, row_width);
}

/// Verify that softmax outputs have correct mathematical properties
fn verify_softmax_properties(data: &[f32], name: &str, row_width: usize) {
    let mut row_start = 0;
    while row_start < data.len() {
        let row_end = (row_start + row_width).min(data.len());
        let row = &data[row_start..row_end];

        let mut sum = 0.0f32;
        for &val in row {
            sum += val;
        }

        // Each row should sum to approximately 1.0
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "{} softmax row {} sum is {}, expected ~1.0",
            name,
            row_start / row_width,
            sum
        );

        row_start = row_end;
    }
}

/// Test with different data types
#[test]
fn test_softmax_kernel_dtype_consistency() {
    // This test would require creating contexts for different dtypes
    // For now, we'll test with the available TestElement type
    // In a full implementation, we'd test f16, f32, etc.

    let mut ctx = Context::<TestElement>::new().unwrap();

    let input_data: Vec<half::f16> = (0..256).map(|i| TestElement::from_f32((i as f32 - 128.0) / 10.0)).collect();

    let input_orig = Tensor::new(vec![1, 128, 2], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&input_data)).unwrap();
    let input_vec = Tensor::new(vec![1, 128, 2], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&input_data)).unwrap();

    // Test original kernel
    let original_result = ctx.call::<SoftmaxKernelOp>((&input_orig, 128, 128, 2, false as u32, 0)).unwrap();

    // Test vec kernel
    let vec_result = ctx.call::<SoftmaxVecOp>((&input_vec, 128u32, 128u32, 2u32, 0u32, 0u32)).unwrap();

    // Row width is the last dimension (seq_k = 2)
    compare_softmax_results(&original_result, &vec_result, "vec-softmax-dtype", 2);

    ctx.synchronize();
}

/// Test F32 dtype support for both vec and block softmax kernels
#[test]
fn test_softmax_f32_dtypes() {
    use crate::F32Element;

    let mut ctx = Context::<F32Element>::new().unwrap();

    // Test small sequences that use vec-softmax
    let seq_len = 128;
    let head_dim = 64;
    let input_data: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32 - (seq_len * head_dim) as f32 / 2.0) / 100.0)
        .collect();

    let input_orig = Tensor::new(
        vec![1, seq_len, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )
    .unwrap();
    let input_vec = Tensor::new(
        vec![1, seq_len, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )
    .unwrap();

    // Test original vs vec-softmax for F32
    let original_result = ctx
        .call::<SoftmaxKernelOp>((&input_orig, seq_len as u32, seq_len as u32, head_dim as u32, false as u32, 0))
        .unwrap();

    let vec_result = ctx
        .call::<SoftmaxVecOp>((&input_vec, seq_len as u32, seq_len as u32, head_dim as u32, 0u32, 0u32))
        .unwrap();

    compare_softmax_results_f32(&original_result, &vec_result, "vec-softmax-f32", head_dim);

    // Test longer sequences that use block-softmax
    let long_seq_len = 2048;
    let head_dim = 64;
    let long_input_data: Vec<f32> = (0..long_seq_len * head_dim)
        .map(|i| (i as f32 - (long_seq_len * head_dim) as f32 / 2.0) / 100.0)
        .collect();

    let input_orig_long = Tensor::new(
        vec![1, long_seq_len, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&long_input_data),
    )
    .unwrap();
    let input_block = Tensor::new(
        vec![1, long_seq_len, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&long_input_data),
    )
    .unwrap();

    // Test original vs block-softmax for F32
    let original_long_result = ctx
        .call::<SoftmaxKernelOp>((
            &input_orig_long,
            long_seq_len as u32,
            long_seq_len as u32,
            head_dim as u32,
            false as u32,
            0,
        ))
        .unwrap();

    let block_result = ctx
        .call::<SoftmaxBlockOp>((
            &input_block,
            long_seq_len as u32,
            long_seq_len as u32,
            head_dim as u32,
            1024u32,
            0u32,
            0u32,
        ))
        .unwrap();

    compare_softmax_results_f32(&original_long_result, &block_result, "block-softmax-f32", head_dim);

    ctx.synchronize();
}

/// Helper function to compare F32 softmax results
fn compare_softmax_results_f32(original: &Tensor<F32Element>, new_kernel: &Tensor<F32Element>, kernel_name: &str, row_width: usize) {
    let original_data = original.as_slice();
    let new_data = new_kernel.as_slice();

    assert_eq!(
        original_data.len(),
        new_data.len(),
        "Tensor sizes must match for {} comparison",
        kernel_name
    );

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut diff_count = 0;

    for (i, (orig, new)) in original_data.iter().zip(new_data.iter()).enumerate() {
        let orig_val = F32Element::to_f32(*orig);
        let new_val = F32Element::to_f32(*new);
        let diff = (orig_val - new_val).abs();

        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }

        if diff > 1e-5 {
            diff_count += 1;
        }
    }

    println!(
        "{} vs original: max_diff={:.8} at idx {}, significant_diffs={}/{}",
        kernel_name,
        max_diff,
        max_diff_idx,
        diff_count,
        original_data.len()
    );

    // Allow small rounding differences for F32
    let epsilon = 1e-5;

    assert!(
        max_diff < epsilon,
        "{} kernel produces different results than original. Max diff: {} at index {}",
        kernel_name,
        max_diff,
        max_diff_idx
    );

    // Verify softmax properties: outputs should sum to 1 for each row
    let original_f32: Vec<f32> = original_data.iter().map(|&x| F32Element::to_f32(x)).collect();
    let new_f32: Vec<f32> = new_data.iter().map(|&x| F32Element::to_f32(x)).collect();
    verify_softmax_properties(&original_f32, "original", row_width);
    verify_softmax_properties(&new_f32, kernel_name, row_width);
}

/// Test block-softmax causal masking with long sequences
#[test]
fn test_softmax_block_causal_masking_consistency() {
    let mut ctx = Context::<TestElement>::new().unwrap();

    // Use longer sequence to trigger block-softmax (seq_len > 1024)
    let batch = 1usize;
    let seq_q = 16usize; // Number of query rows
    let seq_k = 2048usize; // Number of key columns (this should trigger block-softmax)
    let causal = true;

    // Deterministic input pattern for causal testing
    let total = batch * seq_q * seq_k;
    let mut input_data = vec![TestElement::from_f32(0.0); total];
    (0..total).for_each(|i| {
        let r = (i / seq_k) % seq_q; // row index (query position)
        let c = i % seq_k; // column index (key position)

        // Create a pattern where causal masking should make a clear difference
        // Each query position should only see keys up to its own position
        let v = ((c as f32) - (r as f32)) / 10.0; // Create a gradient that varies by position
        input_data[i] = TestElement::from_f32(v);
    });

    // Test with original kernel
    let input_orig = Tensor::new(
        vec![batch, seq_q, seq_k],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )
    .unwrap();
    let original = ctx
        .call::<SoftmaxKernelOp>((&input_orig, (batch * seq_q) as u32, seq_q as u32, seq_k as u32, causal as u32, 0))
        .unwrap();

    // Test with block-softmax kernel
    let input_block = Tensor::new(
        vec![batch, seq_q, seq_k],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )
    .unwrap();
    let block = ctx
        .call::<SoftmaxBlockOp>((
            &input_block,
            (batch * seq_q) as u32,
            seq_q as u32,
            seq_k as u32,
            1024u32,
            causal as u32,
            0,
        ))
        .unwrap();

    // Compare results
    let original_data = original.as_slice();
    let block_data = block.as_slice();

    assert_eq!(
        original_data.len(),
        block_data.len(),
        "Tensor sizes must match for block-softmax comparison"
    );

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut diff_count = 0;

    for (i, (orig, block_val)) in original_data.iter().zip(block_data.iter()).enumerate() {
        let orig_val = TestElement::to_f32(*orig);
        let block_val = TestElement::to_f32(*block_val);
        let diff = (orig_val - block_val).abs();

        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }

        if diff > 1e-4 {
            // Slightly relaxed tolerance for block-softmax
            diff_count += 1;
        }
    }

    println!(
        "block-softmax vs original (causal): max_diff={:.8} at idx {}, significant_diffs={}/{}",
        max_diff,
        max_diff_idx,
        diff_count,
        original_data.len()
    );

    // Allow slightly more tolerance for block-softmax due to segmented reduction differences and f16 precision
    let epsilon = 1e-4; // Increased tolerance to account for segmented reduction precision differences

    assert!(
        max_diff < epsilon,
        "block-softmax kernel produces different results than original. Max diff: {} at index {}",
        max_diff,
        max_diff_idx
    );

    // Verify softmax properties: outputs should sum to 1 for each row
    let original_f32: Vec<f32> = original_data.iter().map(|&x| TestElement::to_f32(x)).collect();
    let block_f32: Vec<f32> = block_data.iter().map(|&x| TestElement::to_f32(x)).collect();
    verify_softmax_properties(&original_f32, "original", seq_k);
    verify_softmax_properties(&block_f32, "block-softmax", seq_k);

    ctx.synchronize();
}

/// Focused diagnostic to narrow down where vec-softmax diverges.
/// Runs a small configuration and prints per-row stats and top diffs.
#[test]
fn test_softmax_vec_debug_small() {
    let mut ctx = Context::<TestElement>::new().unwrap();

    let batch = 1usize;
    let seq_q = 8usize;
    let seq_k = 32usize;
    let causal = false;

    // Deterministic input: gentle gradient across columns
    let total = batch * seq_q * seq_k;
    let input_data: Vec<half::f16> = (0..total)
        .map(|i| {
            let r = (i / seq_k) % seq_q;
            let c = i % seq_k;
            let v = (c as f32 - seq_k as f32 / 2.0) / 20.0 + (r as f32) * 0.001; // slight row offset
            TestElement::from_f32(v)
        })
        .collect();

    let input_orig = Tensor::new(
        vec![batch, seq_q, seq_k],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )
    .unwrap();
    let input_vec = Tensor::new(
        vec![batch, seq_q, seq_k],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )
    .unwrap();

    // Run original and vec-softmax
    let original = ctx
        .call::<SoftmaxKernelOp>((&input_orig, (batch * seq_q) as u32, seq_q as u32, seq_k as u32, causal as u32, 0))
        .unwrap();
    let vec = ctx
        .call::<SoftmaxVecOp>((&input_vec, (batch * seq_q) as u32, seq_q as u32, seq_k as u32, causal as u32, 0))
        .unwrap();

    let o = original.as_slice();
    let v = vec.as_slice();

    // Analyze each row
    for r in 0..seq_q {
        let start = r * seq_k;
        let end = start + seq_k;
        let row_o: Vec<f32> = o[start..end].iter().map(|&x| TestElement::to_f32(x)).collect();
        let row_v: Vec<f32> = v[start..end].iter().map(|&x| TestElement::to_f32(x)).collect();
        let row_x: Vec<f32> = input_data[start..end].iter().map(|&x| TestElement::to_f32(x)).collect();

        let sum_o: f32 = row_o.iter().copied().sum();
        let sum_v: f32 = row_v.iter().copied().sum();

        // Estimate row_max via relation: log(o) = x - row_max - log(sum)
        // => row_max ~= x - log(o) - log(sum). Average over non-zero outputs.
        let log_sum_o = sum_o.ln();
        let log_sum_v = sum_v.ln();
        let mut est_max_o = 0.0f32;
        let mut est_max_v = 0.0f32;
        let mut count_o = 0usize;
        let mut count_v = 0usize;
        for c in 0..seq_k {
            if row_o[c] > 0.0 {
                est_max_o += row_x[c] - row_o[c].ln() - log_sum_o;
                count_o += 1;
            }
            if row_v[c] > 0.0 {
                est_max_v += row_x[c] - row_v[c].ln() - log_sum_v;
                count_v += 1;
            }
        }
        if count_o > 0 {
            est_max_o /= count_o as f32;
        }
        if count_v > 0 {
            est_max_v /= count_v as f32;
        }

        let mut max_diff = 0.0f32;
        let mut max_idx = 0usize;
        let mut diffs: Vec<(usize, f32, f32, f32)> = Vec::with_capacity(seq_k);
        for c in 0..seq_k {
            let d = (row_o[c] - row_v[c]).abs();
            if d > max_diff {
                max_diff = d;
                max_idx = c;
            }
            diffs.push((c, row_o[c], row_v[c], d));
        }
        diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        let top = diffs.iter().take(6).cloned().collect::<Vec<_>>();

        let actual_max = row_x.iter().copied().fold(f32::NEG_INFINITY, |a, b| a.max(b));
        println!(
            "row {}: sum_orig={:.6}, sum_vec={:.6}, est_max_orig={:.6}, est_max_vec={:.6}, actual_max={:.6}, max_diff={:.6} at col {}",
            r, sum_o, sum_v, est_max_o, est_max_v, actual_max, max_diff, max_idx
        );
        for (c, oval, vval, d) in top {
            println!("  col {:>2}: orig={:.6}, vec={:.6}, diff={:.6}", c, oval, vval, d);
        }
    }

    ctx.synchronize();
}
