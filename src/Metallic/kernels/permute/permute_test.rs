#![cfg(test)]
use crate::metallic::kernels::permute::PermuteOp;
use crate::metallic::{Context, Tensor};

#[test]
fn test_permute_2d_transpose() -> Result<(), crate::metallic::MetalError> {
    let mut ctx = Context::new()?;
    // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let src = Tensor::create_tensor_from_slice(&data, vec![2, 3], &ctx)?;

    // Permute dimensions [1, 0] to transpose: [[1, 4], [2, 5], [3, 6]]
    let permute_indices = vec![1, 0];
    let result_tensor = ctx.call::<PermuteOp>((src, permute_indices))?;

    // Expected result: [1, 4, 2, 5, 3, 6] (row-major order after transpose)
    let result = result_tensor.as_slice();
    assert_eq!(result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    Ok(())
}

#[test]
fn test_permute_3d() -> Result<(), crate::metallic::MetalError> {
    let mut ctx = Context::new()?;
    // Create a 2x2x2 tensor
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let src = Tensor::create_tensor_from_slice(&data, vec![2, 2, 2], &ctx)?;

    // Permute dimensions [2, 0, 1]
    let permute_indices = vec![2, 0, 1];
    let result_tensor = ctx.call::<PermuteOp>((src, permute_indices))?;

    // The result should have shape [2, 2, 2] but with dimensions permuted
    assert_eq!(result_tensor.dims(), &[2, 2, 2]);
    Ok(())
}

#[test]
fn test_permute_identity() -> Result<(), crate::metallic::MetalError> {
    let mut ctx = Context::new()?;
    // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let src = Tensor::create_tensor_from_slice(&data, vec![2, 3], &ctx)?;

    // Permute with identity: [0, 1] - should be unchanged
    let permute_indices = vec![0, 1];
    let result_tensor = ctx.call::<PermuteOp>((src, permute_indices))?;

    let result = result_tensor.as_slice();
    assert_eq!(result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

// Additional Permute Tests from src/metallic/tests/permute_reassembly_test.rs

fn manual_reassemble(out_heads: &[f32], batch: usize, n_heads: usize, seq: usize, head_dim: usize) -> Vec<f32> {
    let d_model = n_heads * head_dim;
    let mut manual = vec![0f32; batch * seq * d_model];
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..n_heads {
                let src_batch = b * n_heads + h; // [batch*n_heads, seq, head_dim]
                let src_base = (src_batch * seq + s) * head_dim;
                let dst_base = (b * seq + s) * d_model + h * head_dim;
                (0..head_dim).for_each(|d| {
                    manual[dst_base + d] = out_heads[src_base + d];
                });
            }
        }
    }
    manual
}

fn run_case(batch: usize, n_heads: usize, seq: usize, head_dim: usize) {
    let mut ctx = Context::new().unwrap();
    let bh = batch * n_heads;
    let d_model = n_heads * head_dim;
    let numel = bh * seq * head_dim;

    // Fill with arange pattern for deterministic mapping
    let t = Tensor::arange(numel, vec![bh, seq, head_dim], &mut ctx).unwrap();

    // Reshape -> Permute -> Reshape path (GPU)
    let reshaped = t.reshape(vec![batch, n_heads, seq, head_dim]).unwrap();
    let permuted = reshaped.permute(&[0, 2, 1, 3], &mut ctx).unwrap();
    let merged = permuted.reshape(vec![batch, seq, d_model]).unwrap();
    let gpu = merged.to_vec();

    // Manual reference on CPU
    let cpu = manual_reassemble(&t.to_vec(), batch, n_heads, seq, head_dim);

    assert_eq!(gpu.len(), cpu.len());
    let mut s = 0.0f32;
    for i in 0..gpu.len() {
        let d = gpu[i] - cpu[i];
        s += d * d;
    }
    let l2 = if s > 0.0 { (s / (gpu.len() as f32)).sqrt() } else { 0.0 };
    println!(
        "permute reassembly L2 (b={}, h={}, s={}, d={}): {}",
        batch, n_heads, seq, head_dim, l2
    );
    assert!(l2 < 1e-6, "permute-based reassembly mismatch: L2 = {}", l2);
}

#[test]
fn permute_reassembly_small() {
    run_case(1, 2, 3, 4);
}

#[test]
fn permute_reassembly_medium() {
    run_case(2, 3, 4, 5);
}
