//! Comprehensive RepeatKvHeads Test Suite for Foundry.
//!
//! Tests the RepeatKvHeads kernel against legacy implementation and CPU reference.

use half::f16;
use metallic::{
    Context, F16Element, context::RepeatKvWorkspaceKind, foundry::{Foundry, storage::Pooled, tensor::Tensor as FoundryTensor}, kernels::repeat_kv_heads::RepeatKvHeadsOp, metals::repeat_kv_heads::{RepeatKvHeads, RepeatKvHeadsParams}, tensor::{F16, Tensor, TensorInit, TensorStorage as LegacyStorage}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

const TOLERANCE: f32 = 1e-6; // Exact copy, no precision loss

// ============================================================================
// Test Configuration
// ============================================================================

struct TestConfig {
    batch: usize,
    n_kv_heads: usize,
    n_heads: usize,
    seq: usize,
    head_dim: usize,
    cache_stride: usize, // max sequence capacity
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================

fn cpu_repeat_kv_heads(
    input: &[f16],
    group_size: usize,
    batch: usize,
    n_kv_heads: usize,
    n_heads: usize,
    seq: usize,
    head_dim: usize,
    cache_stride: usize,
) -> Vec<f16> {
    let total_output = batch * n_heads * seq * head_dim;
    let mut output = vec![f16::from_f32(0.0); total_output];

    for gid in 0..total_output {
        let dim_idx = gid % head_dim;
        let tmp = gid / head_dim;
        let seq_idx = tmp % seq;
        let batch_head_idx = tmp / seq;
        let b = batch_head_idx / n_heads;
        let h = batch_head_idx % n_heads;

        let kv_head = h / group_size;
        let input_batch_head = b * n_kv_heads + kv_head;
        let input_index = ((input_batch_head * cache_stride) + seq_idx) * head_dim + dim_idx;

        output[gid] = input[input_index];
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

    let group_size = cfg.n_heads / cfg.n_kv_heads;
    let input_elements = cfg.batch * cfg.n_kv_heads * cfg.cache_stride * cfg.head_dim;
    let output_elements = cfg.batch * cfg.n_heads * cfg.seq * cfg.head_dim;

    // Generate random input
    let input_data: Vec<f16> = (0..input_elements).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();

    // =========================================================================
    // Legacy Kernel
    // =========================================================================
    let mut input_legacy = Tensor::<F16>::new(
        vec![cfg.batch * cfg.n_kv_heads, cfg.seq, cfg.head_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::Uninitialized,
    )
    .unwrap();

    // Copy data with proper strides (cache_stride)
    {
        let slice = input_legacy.as_mut_slice();
        for b in 0..cfg.batch {
            for kv_h in 0..cfg.n_kv_heads {
                let batch_head = b * cfg.n_kv_heads + kv_h;
                for s in 0..cfg.seq {
                    for d in 0..cfg.head_dim {
                        let src_idx = ((batch_head * cfg.cache_stride) + s) * cfg.head_dim + d;
                        let dst_idx = ((batch_head * cfg.seq) + s) * cfg.head_dim + d;
                        if src_idx < input_data.len() && dst_idx < slice.len() {
                            slice[dst_idx] = input_data[src_idx];
                        }
                    }
                }
            }
        }
    }

    let out_legacy = ctx
        .call::<RepeatKvHeadsOp>(
            (
                input_legacy,
                group_size as u32,
                cfg.batch as u32,
                cfg.n_kv_heads as u32,
                cfg.n_heads as u32,
                cfg.seq as u32,
                cfg.head_dim as u32,
                cfg.seq as u32, // cache_stride = seq for simple test
                0,              // layer_idx
                RepeatKvWorkspaceKind::Key,
                false, // prefer_shared
            ),
            None,
        )
        .unwrap();
    let legacy_result = out_legacy.to_vec();

    // =========================================================================
    // Foundry Kernel
    // =========================================================================
    // For Foundry, use matching layout
    let input_foundry_data: Vec<f16> = (0..cfg.batch * cfg.n_kv_heads * cfg.seq * cfg.head_dim)
        .map(|i| {
            let batch_head = i / (cfg.seq * cfg.head_dim);
            let rem = i % (cfg.seq * cfg.head_dim);
            let s = rem / cfg.head_dim;
            let d = rem % cfg.head_dim;
            let src_idx = ((batch_head * cfg.cache_stride) + s) * cfg.head_dim + d;
            if src_idx < input_data.len() {
                input_data[src_idx]
            } else {
                f16::from_f32(0.0)
            }
        })
        .collect();

    let input_foundry = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![cfg.batch * cfg.n_kv_heads * cfg.seq * cfg.head_dim],
        TensorInit::CopyFrom(&input_foundry_data),
    )
    .unwrap();
    let output_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![output_elements], TensorInit::Uninitialized).unwrap();

    let params = RepeatKvHeadsParams {
        group_size: group_size as u32,
        batch: cfg.batch as u32,
        n_kv_heads: cfg.n_kv_heads as u32,
        n_heads: cfg.n_heads as u32,
        seq: cfg.seq as u32,
        head_dim: cfg.head_dim as u32,
        cache_stride: cfg.seq as u32, // Match legacy test
        total_elements: output_elements as u32,
    };

    let input_arg = TensorArg::from_tensor(&input_foundry);
    let output_arg = TensorArg::from_tensor(&output_foundry);

    let kernel = RepeatKvHeads::new(&input_arg, &output_arg, params);
    foundry.run(&kernel).unwrap();

    let foundry_result = FoundryTensor::to_vec(&output_foundry, &foundry);

    // =========================================================================
    // CPU Reference
    // =========================================================================
    let cpu_result = cpu_repeat_kv_heads(
        &input_foundry_data,
        group_size,
        cfg.batch,
        cfg.n_kv_heads,
        cfg.n_heads,
        cfg.seq,
        cfg.head_dim,
        cfg.seq, // cache_stride = seq for test
    );

    // =========================================================================
    // Comparison
    // =========================================================================
    let mut legacy_vs_foundry: f32 = 0.0;
    let mut cpu_vs_foundry: f32 = 0.0;

    for i in 0..output_elements {
        legacy_vs_foundry = legacy_vs_foundry.max((legacy_result[i].to_f32() - foundry_result[i].to_f32()).abs());
        cpu_vs_foundry = cpu_vs_foundry.max((cpu_result[i].to_f32() - foundry_result[i].to_f32()).abs());
    }

    println!(
        "\n[RepeatKvHeads batch={} heads={}/{} seq={} dim={}]",
        cfg.batch, cfg.n_heads, cfg.n_kv_heads, cfg.seq, cfg.head_dim
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
fn test_repeat_kv_heads_gqa_7x() {
    // 7x group (14 heads, 2 KV heads)
    run_parity_test(TestConfig {
        batch: 1,
        n_kv_heads: 2,
        n_heads: 14,
        seq: 8,
        head_dim: 64,
        cache_stride: 8,
    });
}

#[test]
#[serial]
fn test_repeat_kv_heads_gqa_8x() {
    // 8x group (32 heads, 4 KV heads)
    run_parity_test(TestConfig {
        batch: 1,
        n_kv_heads: 4,
        n_heads: 32,
        seq: 16,
        head_dim: 128,
        cache_stride: 16,
    });
}

#[test]
#[serial]
fn test_repeat_kv_heads_no_gqa() {
    // No GQA (n_heads == n_kv_heads, group_size = 1)
    run_parity_test(TestConfig {
        batch: 1,
        n_kv_heads: 8,
        n_heads: 8,
        seq: 8,
        head_dim: 64,
        cache_stride: 8,
    });
}

#[test]
#[serial]
fn test_repeat_kv_heads_single_token() {
    // Autoregressive: seq=1
    run_parity_test(TestConfig {
        batch: 1,
        n_kv_heads: 2,
        n_heads: 14,
        seq: 1,
        head_dim: 64,
        cache_stride: 1,
    });
}

#[test]
#[serial]
fn test_repeat_kv_heads_batched() {
    // Batched inference
    run_parity_test(TestConfig {
        batch: 4,
        n_kv_heads: 2,
        n_heads: 8,
        seq: 8,
        head_dim: 64,
        cache_stride: 8,
    });
}
