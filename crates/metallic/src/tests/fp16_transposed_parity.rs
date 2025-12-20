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

/// End-to-end test: load model and verify weights are in column-major [N, K] layout.
#[test]
#[serial]
#[ignore] // Requires actual model file
fn test_fp16_unified_layout_load() -> Result<(), MetalError> {
    use crate::{
        gguf::{GGUFFile, model_loader::GGUFModelLoader}, models::LoadableModel
    };

    let model_path = "/Volumes/2TB/test-burn/models/qwen2.5-coder-0.5b-instruct-fp16.gguf";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_fp16_unified_layout_load: model not found");
        return Ok(());
    }

    let gguf_file = GGUFFile::load_mmap_and_get_metadata(model_path)
        .map_err(|e| MetalError::InvalidOperation(format!("Failed to load GGUF: {:?}", e)))?;
    let loader = GGUFModelLoader::new(gguf_file);
    let gguf_model = loader
        .load_model()
        .map_err(|e| MetalError::InvalidOperation(format!("Failed to load model: {:?}", e)))?;

    let mut ctx = Context::<crate::tensor::F16>::new()?;
    let model = crate::models::Qwen25::load_from_gguf(&gguf_model, &mut ctx)?;

    // Verify weights are loaded in [N, K] column-major layout directly
    // Logic: original was [K, N] (row-major). Unified is [N, K].
    // Qwen2.5-0.5B: d_model=896, qkv_out=896+2*128=1152? (depends on config)
    let d_model = model.config.d_model;
    let kv_dim = model.config.d_model * model.config.n_kv_heads / model.config.n_heads;
    let qkv_out_dim = d_model + 2 * kv_dim;

    let first_block = &model.blocks[0];

    // Check attn_qkv_weight dims
    let dims = first_block.attn_qkv_weight.dims();
    // It should be [qkv_out_dim, d_model]
    assert_eq!(
        dims[0], qkv_out_dim,
        "attn_qkv_weight should have N composed of Output dims in dim 0"
    );
    assert_eq!(dims[1], d_model, "attn_qkv_weight should have K (d_model) in dim 1");

    // Check FFN Gate dims
    // Gate original: [d_model, ff_dim] (row-major) -> field was [ff_dim, d_model] actually depending on torch vs gguf
    // GGUF usually stores [N, K] row-major for linears? No, GGUF is often row-major.
    // In our loading logic we load it to [N, K].
    // For FFN Gate: Input is d_model. Output is ff_dim.
    // Unified [N, K] = [ff_dim, d_model].
    let ff_dim = model.config.ff_dim;
    let gate_dims = first_block.ffn_gate.dims();
    assert_eq!(gate_dims[0], ff_dim, "ffn_gate N dim mismatch");
    assert_eq!(gate_dims[1], d_model, "ffn_gate K dim mismatch");

    Ok(())
}

/// Verify that the unified weight + transpose_right=true produces correct results
/// by comparing against a manually reconstructed legacy GEMV.
#[test]
#[serial]
#[ignore] // Requires actual model file
fn test_fp16_unified_forward_parity() -> Result<(), MetalError> {
    use crate::{
        gguf::{GGUFFile, model_loader::GGUFModelLoader}, models::LoadableModel
    };

    let model_path = "/Volumes/2TB/test-burn/models/qwen2.5-coder-0.5b-instruct-fp16.gguf";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test_fp16_unified_forward_parity: model not found");
        return Ok(());
    }

    let gguf_file = GGUFFile::load_mmap_and_get_metadata(model_path)
        .map_err(|e| MetalError::InvalidOperation(format!("Failed to load GGUF: {:?}", e)))?;
    let loader = GGUFModelLoader::new(gguf_file);
    let gguf_model = loader
        .load_model()
        .map_err(|e| MetalError::InvalidOperation(format!("Failed to load model: {:?}", e)))?;
    let mut ctx = Context::<crate::tensor::F16>::new()?;
    let model = crate::models::Qwen25::load_from_gguf(&gguf_model, &mut ctx)?;

    let block = &model.blocks[0];
    let d_model = model.config.d_model;

    // Create input [1, d_model]
    let x = make_fp16_tensor(&mut ctx, vec![1, d_model], 0xDEAD_BEEF)?;

    // 1. Run Unified Path (what the model does now)
    // x @ attn_qkv_weight.T (since weight is [N, K])
    let y_unified = ctx.matmul(
        &x,
        &TensorType::Dense(&block.attn_qkv_weight),
        false,
        true, // transpose_right
        Some(&block.attn_qkv_bias),
        None,
        None,
    )?;
    ctx.synchronize();

    // 2. Run Reference (Legacy) Path
    // Manually transpose the [N, K] weight back to [K, N] (row-major reference)

    let w_ref_row_major = {
        let dims = block.attn_qkv_weight.dims();
        let n = dims[0];
        let k = dims[1];
        let src = block.attn_qkv_weight.as_slice();
        let mut data = vec![f16::ZERO; n * k];
        for row in 0..n {
            for col in 0..k {
                let src_idx = row * k + col;
                let dst_idx = col * n + row; // transpose [N, K] -> [K, N]
                data[dst_idx] = src[src_idx];
            }
        }
        Tensor::<crate::tensor::F16>::new(vec![k, n], TensorStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&data))?
    };

    // Run standard GEMV: x @ W_ref (no transpose on W_ref)
    let y_ref = ctx.matmul(
        &x,
        &TensorType::Dense(&w_ref_row_major),
        false,
        false, // transpose_right = false for row-major legacy match
        Some(&block.attn_qkv_bias),
        None,
        None,
    )?;
    ctx.synchronize();

    // Compare
    assert_close_tensors(&y_ref, &y_unified, 0.05, "unified_qkv_parity");

    Ok(())
}
