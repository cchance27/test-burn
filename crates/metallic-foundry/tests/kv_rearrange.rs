//! Comprehensive KV Rearrange Test Suite for Foundry.
//!
//! Tests the KV Rearrange kernel against legacy implementation and CPU reference.

use half::f16;
use metallic_context::{
    Context, F16Element as LegacyF16Element, kernels::kv_rearrange::KvRearrangeOp, tensor::{Tensor as LegacyTensor, TensorInit as LegacyTensorInit, TensorStorage as LegacyStorage}
};
use metallic_foundry::{
    Foundry, metals::kv_rearrange::{KvRearrange, KvRearrangeParamsResolved}, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

const TOLERANCE: f32 = 1e-6; // Exact rearrangement, no precision loss

// ============================================================================
// Test Configuration
// ============================================================================

struct TestConfig {
    batch: usize,
    seq: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,      // n_kv_heads * kv_head_dim
    kv_head_dim: usize, // usually == head_dim
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cpu_kv_rearrange(
    input: &[f16],
    row_stride: usize,
    batch: usize,
    seq: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_head_dim: usize,
    kv_dim: usize,
) -> Vec<f16> {
    let total_output = batch * n_heads * seq * head_dim;
    let mut output = vec![f16::from_f32(0.0); total_output];
    let group_size = n_heads / n_kv_heads;

    for (gid, out_val) in output.iter_mut().enumerate().take(total_output) {
        let hd = gid % head_dim;
        let tmp = gid / head_dim;
        let s = tmp % seq;
        let out_batch = tmp / seq;
        let b = out_batch / n_heads;
        let h = out_batch % n_heads;
        let kv_h = h / group_size;

        let base_offset = kv_h * kv_head_dim + hd;
        if base_offset >= kv_dim {
            continue;
        }

        let src_row = b * seq + s;
        let src_idx = src_row * row_stride + base_offset;
        *out_val = input[src_idx];
    }

    output
}

// ============================================================================
// Parity Test Helper
// ============================================================================

fn run_parity_test(cfg: TestConfig) {
    let mut ctx = Context::<LegacyF16Element>::new().unwrap();
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rng();

    let input_rows = cfg.batch * cfg.seq;
    let row_stride = cfg.kv_dim;
    let input_elements = input_rows * row_stride;
    let output_elements = cfg.batch * cfg.n_heads * cfg.seq * cfg.head_dim;

    // Generate random input
    let input_data: Vec<f16> = (0..input_elements).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();

    // =========================================================================
    // Legacy Kernel
    // =========================================================================
    let input_legacy = LegacyTensor::<LegacyF16Element>::new(
        vec![input_rows, row_stride],
        LegacyStorage::Pooled(&mut ctx),
        LegacyTensorInit::CopyFrom(&input_data),
    )
    .unwrap();

    let out_legacy = ctx
        .call::<KvRearrangeOp>(
            (
                input_legacy,
                cfg.kv_dim as u32,
                cfg.kv_head_dim as u32,
                cfg.n_heads as u32,
                cfg.n_kv_heads as u32,
                cfg.head_dim as u32,
                cfg.seq as u32,
            ),
            None,
        )
        .unwrap();
    let legacy_result = out_legacy.to_vec();

    // =========================================================================
    // Foundry Kernel
    // =========================================================================
    let input_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![input_elements], TensorInit::CopyFrom(&input_data)).unwrap();
    let output_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![output_elements], TensorInit::Uninitialized).unwrap();

    let params = KvRearrangeParamsResolved {
        kv_dim: cfg.kv_dim as u32,
        row_stride: row_stride as u32,
        kv_head_dim: cfg.kv_head_dim as u32,
        n_heads: cfg.n_heads as u32,
        n_kv_heads: cfg.n_kv_heads as u32,
        head_dim: cfg.head_dim as u32,
        seq: cfg.seq as u32,
        total_elements: output_elements as u32,
    };

    let input_arg = TensorArg::from_tensor(&input_foundry);
    let output_arg = TensorArg::from_tensor(&output_foundry);

    let kernel = KvRearrange {
        input: input_arg,
        output: output_arg,
        params,
    };
    foundry.run(&kernel).unwrap();

    let foundry_result = FoundryTensor::to_vec(&output_foundry, &foundry);

    // =========================================================================
    // CPU Reference
    // =========================================================================
    let cpu_result = cpu_kv_rearrange(
        &input_data,
        row_stride,
        cfg.batch,
        cfg.seq,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.kv_head_dim,
        cfg.kv_dim,
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
        "\n[KvRearrange batch={} seq={} heads={}/{} dim={}]",
        cfg.batch, cfg.seq, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
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
fn test_kv_rearrange_qwen05b_q() {
    // Qwen2.5-0.5B Q heads: n_heads=14, n_kv_heads=14, head_dim=64
    run_parity_test(TestConfig {
        batch: 1,
        seq: 8,
        n_heads: 14,
        n_kv_heads: 14,
        head_dim: 64,
        kv_dim: 896,
        kv_head_dim: 64,
    });
}

#[test]
#[serial]
fn test_kv_rearrange_qwen05b_kv() {
    // Qwen2.5-0.5B K/V heads: n_heads=14, n_kv_heads=2, head_dim=64
    run_parity_test(TestConfig {
        batch: 1,
        seq: 8,
        n_heads: 14,
        n_kv_heads: 2,
        head_dim: 64,
        kv_dim: 128,
        kv_head_dim: 64,
    });
}

#[test]
#[serial]
fn test_kv_rearrange_gqa_7b() {
    // Typical 7B model with GQA: 32 query heads, 4 KV heads
    run_parity_test(TestConfig {
        batch: 1,
        seq: 16,
        n_heads: 32,
        n_kv_heads: 4,
        head_dim: 128,
        kv_dim: 512,
        kv_head_dim: 128,
    });
}

#[test]
#[serial]
fn test_kv_rearrange_no_gqa() {
    // MHA: n_heads == n_kv_heads (no grouping)
    run_parity_test(TestConfig {
        batch: 1,
        seq: 8,
        n_heads: 8,
        n_kv_heads: 8,
        head_dim: 64,
        kv_dim: 512,
        kv_head_dim: 64,
    });
}

#[test]
#[serial]
fn test_kv_rearrange_single_token() {
    // Autoregressive: seq=1
    run_parity_test(TestConfig {
        batch: 1,
        seq: 1,
        n_heads: 14,
        n_kv_heads: 2,
        head_dim: 64,
        kv_dim: 128,
        kv_head_dim: 64,
    });
}

#[test]
#[serial]
fn test_kv_rearrange_batched() {
    // Batched inference
    run_parity_test(TestConfig {
        batch: 4,
        seq: 8,
        n_heads: 8,
        n_kv_heads: 2,
        head_dim: 64,
        kv_dim: 128,
        kv_head_dim: 64,
    });
}
