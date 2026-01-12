//! Comprehensive Embedding Lookup Test Suite for Foundry.
//!
//! Tests the embedding lookup kernel against legacy implementation and CPU reference.

use half::f16;
use metallic_context::{
    Context, F16Element, TensorElement as _, kernels::embedding_lookup::EmbeddingLookupOp, tensor::{Tensor as LegacyTensor, TensorInit as LegacyTensorInit, TensorStorage as LegacyStorage, U32 as LegacyU32}
};
use metallic_foundry::{
    Foundry, metals::embedding::{Embedding, EmbeddingParamsResolved}, storage::Pooled, tensor::{F16 as FoundryF16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use objc2_metal::MTLDevice as _;
use rand::{Rng, rng};
use serial_test::serial;

const TOLERANCE: f32 = 1e-6; // Embedding is exact copy, no precision loss expected

// ============================================================================
// Test Configuration
// ============================================================================

struct TestConfig {
    vocab_size: usize,
    d_model: usize,
    batch_size: usize,
    seq_len: usize,
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================

fn cpu_embedding_lookup(table: &[f16], indices: &[u32], vocab_size: usize, d_model: usize) -> Vec<f16> {
    let total_tokens = indices.len();
    let mut output = vec![f16::from_f32(0.0); total_tokens * d_model];

    for (pos, &tok) in indices.iter().enumerate() {
        let tok = tok as usize;
        if tok < vocab_size {
            for feat in 0..d_model {
                output[pos * d_model + feat] = table[tok * d_model + feat];
            }
        }
        // Out-of-vocab tokens get zeros (already initialized)
    }

    output
}

// ============================================================================
// Parity Test Helper
// ============================================================================

fn run_parity_test(cfg: TestConfig) {
    let mut ctx = Context::<F16Element>::new().unwrap();
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rng();

    let total_tokens = cfg.batch_size * cfg.seq_len;
    let total_elements = total_tokens * cfg.d_model;

    // Generate random embedding table
    let table_data: Vec<f16> = (0..cfg.vocab_size * cfg.d_model)
        .map(|_| f16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();

    // Generate random token indices (valid tokens only for parity)
    let indices_data: Vec<u32> = (0..total_tokens).map(|_| rng.random_range(0..cfg.vocab_size as u32)).collect();

    // =========================================================================
    // Legacy Kernel
    // =========================================================================
    let table_legacy = LegacyTensor::<F16Element>::new(
        vec![cfg.vocab_size, cfg.d_model],
        LegacyStorage::Pooled(&mut ctx),
        LegacyTensorInit::CopyFrom(&table_data),
    )
    .unwrap();

    // U32 indices: use from_existing_buffer with StorageModeShared (pattern from qwen25/mod.rs)
    let byte_len = std::mem::size_of::<u32>() * total_tokens;
    let buf = ctx
        .device
        .newBufferWithLength_options(byte_len, objc2_metal::MTLResourceOptions::StorageModeShared)
        .expect("Failed to create U32 indices buffer");
    let mut indices_legacy = LegacyTensor::<LegacyU32>::from_existing_buffer(
        buf,
        vec![total_tokens],
        LegacyU32::DTYPE,
        &ctx.device,
        &ctx.command_queue,
        0,
        true,
    )
    .unwrap();
    for (i, &tok) in indices_data.iter().enumerate() {
        indices_legacy.as_mut_slice()[i] = tok;
    }

    let out_legacy = ctx.call::<EmbeddingLookupOp>((&table_legacy, &indices_legacy), None).unwrap();
    let legacy_result = out_legacy.to_vec();

    // =========================================================================
    // Foundry Kernel
    // =========================================================================
    let table_foundry =
        FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![cfg.vocab_size * cfg.d_model], TensorInit::CopyFrom(&table_data))
            .unwrap();
    let indices_foundry =
        FoundryTensor::<metallic_foundry::tensor::U32, Pooled>::new(&mut foundry, vec![total_tokens], TensorInit::CopyFrom(&indices_data))
            .unwrap();
    let output_foundry = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![total_elements], TensorInit::Uninitialized).unwrap();

    let params = EmbeddingParamsResolved {
        d_model: cfg.d_model as u32,
        total_elements: total_elements as u32,
        vocab_size: cfg.vocab_size as u32,
    };

    let table_arg = TensorArg::from_tensor(&table_foundry);
    let indices_arg = TensorArg::from_tensor(&indices_foundry);
    let output_arg = TensorArg::from_tensor(&output_foundry);

    let kernel = Embedding::new(&table_arg, &indices_arg, &output_arg, params);
    foundry.run(&kernel).unwrap();

    let foundry_result: Vec<f16> = FoundryTensor::to_vec(&output_foundry, &foundry);

    // =========================================================================
    // CPU Reference
    // =========================================================================
    let cpu_result = cpu_embedding_lookup(&table_data, &indices_data, cfg.vocab_size, cfg.d_model);

    // =========================================================================
    // Comparison
    // =========================================================================
    let mut legacy_vs_foundry: f32 = 0.0;
    let mut cpu_vs_foundry: f32 = 0.0;

    for i in 0..total_elements {
        legacy_vs_foundry = legacy_vs_foundry.max((legacy_result[i].to_f32() - foundry_result[i].to_f32()).abs());
        cpu_vs_foundry = cpu_vs_foundry.max((cpu_result[i].to_f32() - foundry_result[i].to_f32()).abs());
    }

    println!(
        "\n[Embedding {}x{} batch={}x{}]",
        cfg.vocab_size, cfg.d_model, cfg.batch_size, cfg.seq_len
    );
    println!("  Legacy vs Foundry max diff: {:.6}", legacy_vs_foundry);
    println!("  CPU vs Foundry max diff:    {:.6}", cpu_vs_foundry);

    assert!(legacy_vs_foundry <= TOLERANCE, "Legacy vs Foundry mismatch: {}", legacy_vs_foundry);
    assert!(cpu_vs_foundry <= TOLERANCE, "CPU vs Foundry mismatch: {}", cpu_vs_foundry);
}

// ============================================================================
// F16 Parity Tests
// ============================================================================

#[test]
#[serial]
fn test_embedding_1000x128_1x4() {
    run_parity_test(TestConfig {
        vocab_size: 1000,
        d_model: 128,
        batch_size: 1,
        seq_len: 4,
    });
}

#[test]
#[serial]
fn test_embedding_5000x256_1x8() {
    run_parity_test(TestConfig {
        vocab_size: 5000,
        d_model: 256,
        batch_size: 1,
        seq_len: 8,
    });
}

#[test]
#[serial]
fn test_embedding_10000x512_2x16() {
    run_parity_test(TestConfig {
        vocab_size: 10000,
        d_model: 512,
        batch_size: 2,
        seq_len: 16,
    });
}

#[test]
#[serial]
fn test_embedding_50000x768_1x32() {
    run_parity_test(TestConfig {
        vocab_size: 50000,
        d_model: 768,
        batch_size: 1,
        seq_len: 32,
    });
}

// Edge cases
#[test]
#[serial]
fn test_embedding_small_100x64_1x1() {
    run_parity_test(TestConfig {
        vocab_size: 100,
        d_model: 64,
        batch_size: 1,
        seq_len: 1,
    });
}

#[test]
#[serial]
fn test_embedding_single_token() {
    run_parity_test(TestConfig {
        vocab_size: 1000,
        d_model: 128,
        batch_size: 1,
        seq_len: 1,
    });
}
