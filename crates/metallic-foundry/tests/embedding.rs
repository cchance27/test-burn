//! Comprehensive Embedding Lookup Test Suite for Foundry.
//!
//! Tests the embedding lookup kernel against CPU reference.

use std::sync::Arc;

use half::f16;
use metallic_foundry::{
    Foundry, metals::embedding::{
        EmbeddingParamsResolved, step::{EmbeddingGenericArgs, get_embedding_kernel}
    }, policy::f16::PolicyF16, storage::Pooled, tensor::{F16 as FoundryF16, Tensor as FoundryTensor, TensorInit}, types::{
        TensorArg, dispatch::{DispatchConfig, GridSize, ThreadgroupSize}
    }
};
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

    let args = EmbeddingGenericArgs {
        table: table_arg.clone(),
        scale_bytes: table_arg, // F16 uses same buffer for scales placeholder
        indices: indices_arg,
        output: output_arg,
        params,
    };

    let policy = Arc::new(PolicyF16);
    let kernel = get_embedding_kernel(policy);
    let dispatch = DispatchConfig {
        grid: GridSize::d1(total_elements),
        group: ThreadgroupSize::d1(256),
    };

    foundry.run(&kernel.bind_arc(args, dispatch)).unwrap();

    let foundry_result: Vec<f16> = FoundryTensor::to_vec(&output_foundry, &foundry);

    // =========================================================================
    // CPU Reference
    // =========================================================================
    let cpu_result = cpu_embedding_lookup(&table_data, &indices_data, cfg.vocab_size, cfg.d_model);

    // =========================================================================
    // Comparison
    // =========================================================================
    let mut cpu_vs_foundry: f32 = 0.0;

    for i in 0..total_elements {
        cpu_vs_foundry = cpu_vs_foundry.max((cpu_result[i].to_f32() - foundry_result[i].to_f32()).abs());
    }

    println!(
        "\n[Embedding {}x{} batch={}x{}]",
        cfg.vocab_size, cfg.d_model, cfg.batch_size, cfg.seq_len
    );
    println!("  CPU vs Foundry max diff:    {:.6}", cpu_vs_foundry);

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
