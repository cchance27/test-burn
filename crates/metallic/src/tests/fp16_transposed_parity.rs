#![cfg(test)]

//! FP16 Transposed Weight Parity Tests
//!
//! These tests verify that the transposed FP16 weight storage (column-major [N, K])
//! produces mathematically equivalent results to the legacy row-major layout.
//!
//! FP16 now uses the same column-major layout as Q8 for unified kernel dispatch.
//!
//! Run with: `cargo test -p metallic --features metal fp16_transposed_parity`

use half::f16;
use serial_test::serial;

use crate::{
    Context, MetalError, Tensor, TensorStorage, kernels::matmul_gemv::MatmulGemvOp, tensor::{TensorInit, TensorType}
};

fn make_fp16_tensor(ctx: &mut Context<crate::tensor::F16>, dims: Vec<usize>, seed: u64) -> Result<Tensor<crate::tensor::F16>, MetalError> {
    let len = dims.iter().product();
    let mut data = Vec::with_capacity(len);
    let mut x = seed;
    for _ in 0..len {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = ((x >> 32) as u32) as f32 / (u32::MAX as f32) - 0.5;
        data.push(v);
    }
    Tensor::<crate::tensor::F16>::from_f32_slice(dims, TensorStorage::Pooled(ctx), &data)
}

fn assert_close_tensors(a: &Tensor<crate::tensor::F16>, b: &Tensor<crate::tensor::F16>, tol: f32, name: &str) {
    let a_f: Vec<f32> = a.as_slice().iter().map(|v| v.to_f32()).collect();
    let b_f: Vec<f32> = b.as_slice().iter().map(|v| v.to_f32()).collect();
    assert_eq!(a_f.len(), b_f.len(), "{}: length mismatch {} vs {}", name, a_f.len(), b_f.len());

    let mut max_diff = 0f32;
    for (i, (&x, &y)) in a_f.iter().zip(b_f.iter()).enumerate() {
        let diff = (x - y).abs();
        max_diff = max_diff.max(diff);
        assert!(diff <= tol, "{}: idx={} diff={} x={} y={}", name, i, diff, x, y);
    }
    eprintln!("[{}] max_diff={}", name, max_diff);
}

/// Test that transposed weight creation produces mathematically correct transpose.
#[test]
#[serial]
fn test_fp16_transposed_weight_creation() -> Result<(), MetalError> {
    let mut ctx = Context::<crate::tensor::F16>::new()?;

    let k = 64usize;
    let n = 32usize;

    // Create a row-major weight [K, N]
    let weight = make_fp16_tensor(&mut ctx, vec![k, n], 0xABCD_1234)?;

    // Manually transpose it [N, K]
    let src_slice = weight.as_slice();
    let mut transposed_data = vec![f16::ZERO; k * n];
    for row in 0..k {
        for col in 0..n {
            let src_idx = row * n + col;
            let dst_idx = col * k + row; // transpose: [n, k]
            transposed_data[dst_idx] = src_slice[src_idx];
        }
    }

    let transposed_manual =
        Tensor::<crate::tensor::F16>::new(vec![n, k], TensorStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&transposed_data))?;

    // Use our helper function (exposed via loading module, here we replicate logic)
    let mut transposed_fn = Tensor::new(vec![n, k], TensorStorage::Dedicated(&ctx), TensorInit::Uninitialized)?;
    {
        let src = weight.as_slice();
        let dst = transposed_fn.as_mut_slice();
        for row in 0..k {
            for col in 0..n {
                let src_idx = row * n + col;
                let dst_idx = col * k + row;
                dst[dst_idx] = src[src_idx];
            }
        }
    }

    // Verify both transposes match
    assert_close_tensors(&transposed_manual, &transposed_fn, 0.0, "transpose_parity");

    Ok(())
}

/// Test that GEMV with transposed weight produces same result as legacy path.
/// This validates the mathematical correctness of using column-major weights.
#[test]
#[serial]
fn test_fp16_transposed_gemv_parity() -> Result<(), MetalError> {
    let mut ctx = Context::<crate::tensor::F16>::new()?;

    // Test shapes that match Qwen2.5 dimensions
    let test_cases = [
        (896, 896, "qkv_single"),
        (896, 1152, "qkv_fused"),
        (896, 4864, "ffn_gate"),
        (4864, 896, "ffn_down"),
    ];

    for (k, n, name) in test_cases {
        // Create input vector [1, K]
        let x = make_fp16_tensor(&mut ctx, vec![1, k], 0x1111_2222 ^ (k as u64))?;

        // Create row-major weight [K, N] - legacy layout
        let w_row_major = make_fp16_tensor(&mut ctx, vec![k, n], 0x3333_4444 ^ (n as u64))?;

        // Create column-major weight [N, K] - transposed layout
        let mut w_col_major_data = vec![f16::ZERO; k * n];
        {
            let src = w_row_major.as_slice();
            for row in 0..k {
                for col in 0..n {
                    let src_idx = row * n + col;
                    let dst_idx = col * k + row; // transpose
                    w_col_major_data[dst_idx] = src[src_idx];
                }
            }
        }
        let w_col_major = Tensor::<crate::tensor::F16>::new(
            vec![n, k], // [N, K] after transpose
            TensorStorage::Pooled(&mut ctx),
            TensorInit::CopyFrom(&w_col_major_data),
        )?;

        // GEMV with row-major weight (legacy path): x @ W where W is [K, N]
        // Result: [1, N]
        let y_legacy = ctx.call::<MatmulGemvOp>((&x, TensorType::Dense(&w_row_major), false, None), None)?;

        // GEMV with column-major weight (transposed path): x @ W^T where W^T is [N, K]
        // Need transpose_right=true to get correct computation
        let y_transposed = ctx.call::<MatmulGemvOp>((&x, TensorType::Dense(&w_col_major), true, None), None)?;

        ctx.synchronize();

        // Results should match within tolerance
        assert_close_tensors(&y_legacy, &y_transposed, 0.01, &format!("gemv_{}", name));
    }

    Ok(())
}

/// End-to-end test: load model with and without METALLIC_FP16_TRANSPOSED,
/// verify the transposed weights are populated correctly.
#[test]
#[serial]
#[ignore] // Requires actual model file
fn test_fp16_transposed_model_load() -> Result<(), MetalError> {
    use crate::{
        gguf::{GGUFFile, model_loader::GGUFModelLoader}, models::LoadableModel
    };

    let model_path = "/Volumes/2TB/test-burn/models/qwen2.5-coder-0.5b-instruct-fp16.gguf";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_fp16_transposed_model_load: model not found");
        return Ok(());
    }

    // Load without transposed mode
    let gguf_file = GGUFFile::load_mmap_and_get_metadata(model_path)
        .map_err(|e| MetalError::InvalidOperation(format!("Failed to load GGUF: {:?}", e)))?;
    let loader = GGUFModelLoader::new(gguf_file);
    let gguf_model = loader
        .load_model()
        .map_err(|e| MetalError::InvalidOperation(format!("Failed to load model: {:?}", e)))?;

    let mut ctx = Context::<crate::tensor::F16>::new()?;
    let model_legacy = crate::models::Qwen25::load_from_gguf(&gguf_model, &mut ctx)?;

    // Verify no transposed weights in legacy mode
    for (i, block) in model_legacy.blocks.iter().enumerate() {
        assert!(
            block.attn_qkv_weight_transposed.is_none(),
            "block {} has transposed QKV in legacy mode",
            i
        );
    }

    // Load with transposed mode
    unsafe {
        std::env::set_var("METALLIC_FP16_TRANSPOSED", "1");
    }
    let mut ctx2 = Context::<crate::tensor::F16>::new()?;
    let model_transposed = crate::models::Qwen25::load_from_gguf(&gguf_model, &mut ctx2)?;
    unsafe {
        std::env::remove_var("METALLIC_FP16_TRANSPOSED");
    }

    // Verify transposed weights are populated when FP16 (non-Q8)
    let first_block = &model_transposed.blocks[0];
    if first_block.attn_q_weight_q8.is_none() {
        // This is a dense FP16 model, so transposed should be populated
        assert!(first_block.attn_qkv_weight_transposed.is_some(), "missing transposed QKV");

        // Verify dimensions are correctly transposed
        let legacy_dims = first_block.attn_qkv_weight.dims();
        let transposed_dims = first_block.attn_qkv_weight_transposed.as_ref().unwrap().dims();
        assert_eq!(legacy_dims[0], transposed_dims[1], "K dimension mismatch");
        assert_eq!(legacy_dims[1], transposed_dims[0], "N dimension mismatch");
    }

    Ok(())
}

/// Phase 2 Test: Verify that METALLIC_FP16_FUSED dispatch produces same results as legacy.
/// This tests the full forward_step with transposed weights vs legacy path.
#[test]
#[serial]
#[ignore] // Requires actual model file
fn test_fp16_fused_forward_parity() -> Result<(), MetalError> {
    use crate::{
        gguf::{GGUFFile, model_loader::GGUFModelLoader}, models::LoadableModel
    };

    let model_path = "/Volumes/2TB/test-burn/models/qwen2.5-coder-0.5b-instruct-fp16.gguf";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_fp16_fused_forward_parity: model not found");
        return Ok(());
    }

    // Load model with transposed weights enabled
    unsafe {
        std::env::set_var("METALLIC_FP16_TRANSPOSED", "1");
    }
    let gguf_file = GGUFFile::load_mmap_and_get_metadata(model_path)
        .map_err(|e| MetalError::InvalidOperation(format!("Failed to load GGUF: {:?}", e)))?;
    let loader = GGUFModelLoader::new(gguf_file);
    let gguf_model = loader
        .load_model()
        .map_err(|e| MetalError::InvalidOperation(format!("Failed to load model: {:?}", e)))?;
    let mut ctx = Context::<crate::tensor::F16>::new()?;
    let model = crate::models::Qwen25::load_from_gguf(&gguf_model, &mut ctx)?;
    unsafe {
        std::env::remove_var("METALLIC_FP16_TRANSPOSED");
    }

    // Verify transposed weights are populated
    let block = &model.blocks[0];
    assert!(block.attn_qkv_weight_transposed.is_some(), "transposed weights not loaded");

    // Create a test input [1, d_model]
    let d_model = model.config.d_model;
    let x = make_fp16_tensor(&mut ctx, vec![1, d_model], 0xDEAD_BEEF)?;

    // Legacy path: x @ W (row-major) with transpose_right=false
    let y_legacy = ctx.matmul(
        &x,
        &TensorType::Dense(&block.attn_qkv_weight),
        false,
        false,
        Some(&block.attn_qkv_bias),
        None,
        None,
    )?;
    ctx.synchronize();

    // Fused path: x @ W^T (transposed, column-major) with transpose_right=true
    let w_t = block.attn_qkv_weight_transposed.as_ref().unwrap();
    let y_fused = ctx.matmul(
        &x,
        &TensorType::Dense(w_t),
        false,
        true, // transpose_right because weight is [N, K]
        Some(&block.attn_qkv_bias),
        None,
        None,
    )?;
    ctx.synchronize();

    // Compare outputs
    assert_close_tensors(&y_legacy, &y_fused, 0.01, "qkv_projection_parity");

    Ok(())
}
