//! Comprehensive RoPE Test Suite for Foundry.
//!
//! Tests the RoPE kernel against CPU reference.

use half::f16;
use metallic_foundry::{
    Foundry, metals::rope::{Rope, RopeParamsResolved}, storage::Pooled, tensor::{F16 as FoundryF16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

// Or better: don't alias F16 locally to avoid confusion, use fully qualified or distinct names.

const TOLERANCE: f32 = 1e-4; // Small tolerance for f16 precision

// ============================================================================
// Test Configuration
// ============================================================================

struct TestConfig {
    dim: usize,      // Feature dimension (must be even)
    seq_len: usize,  // Sequence length
    num_rows: usize, // Total rows (heads * batch)
    position_offset: u32,
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================

fn cpu_rope(input: &[f16], cos: &[f16], sin: &[f16], dim: usize, seq_len: usize, position_offset: usize) -> Vec<f16> {
    let total_elements = input.len();
    let half_dim = dim / 2;
    let mut output = vec![f16::from_f32(0.0); total_elements];

    for gid in 0..total_elements {
        let feature_idx = gid % dim;
        let row_idx = gid / dim;
        let pos = (row_idx % seq_len) + position_offset;
        let pair = if feature_idx < half_dim {
            feature_idx
        } else {
            feature_idx - half_dim
        };

        let cosv = cos[pos * half_dim + pair].to_f32();
        let sinv = sin[pos * half_dim + pair].to_f32();

        if feature_idx < half_dim {
            let x_i = input[gid].to_f32();
            let x_j = input[row_idx * dim + feature_idx + half_dim].to_f32();
            let out_i = x_i * cosv - x_j * sinv;
            output[gid] = f16::from_f32(out_i);
        } else {
            let x_j = input[gid].to_f32();
            let x_i = input[row_idx * dim + (feature_idx - half_dim)].to_f32();
            let out_j = x_j * cosv + x_i * sinv;
            output[gid] = f16::from_f32(out_j);
        }
    }

    output
}

// ============================================================================
// Parity Test Helper
// ============================================================================

fn run_parity_test(cfg: TestConfig) {
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rng();

    let half_dim = cfg.dim / 2;
    let total_elements = cfg.num_rows * cfg.dim;
    let max_seq = cfg.seq_len + cfg.position_offset as usize;

    // Generate random input
    let input_data: Vec<f16> = (0..total_elements).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();

    // Generate cos/sin caches [max_seq, half_dim]
    let cos_data: Vec<f16> = (0..max_seq * half_dim)
        .map(|i| {
            let pos = i / half_dim;
            let pair = i % half_dim;
            let angle = pos as f32 * (1.0_f32 / (10000.0_f32.powf(2.0 * pair as f32 / cfg.dim as f32)));
            f16::from_f32(angle.cos())
        })
        .collect();
    let sin_data: Vec<f16> = (0..max_seq * half_dim)
        .map(|i| {
            let pos = i / half_dim;
            let pair = i % half_dim;
            let angle = pos as f32 * (1.0_f32 / (10000.0_f32.powf(2.0 * pair as f32 / cfg.dim as f32)));
            f16::from_f32(angle.sin())
        })
        .collect();

    // =========================================================================
    // Foundry Kernel
    // =========================================================================
    let input_foundry =
        FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![total_elements], TensorInit::CopyFrom(&input_data)).unwrap();
    let output_foundry = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![total_elements], TensorInit::Uninitialized).unwrap();
    let cos_foundry =
        FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![max_seq * half_dim], TensorInit::CopyFrom(&cos_data)).unwrap();
    let sin_foundry =
        FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![max_seq * half_dim], TensorInit::CopyFrom(&sin_data)).unwrap();

    let params = RopeParamsResolved {
        dim: cfg.dim as u32,
        seq_len: cfg.seq_len as u32,
        position_offset: cfg.position_offset,
        total_elements: total_elements as u32,
    };

    let input_arg = TensorArg::from_tensor(&input_foundry);
    let output_arg = TensorArg::from_tensor(&output_foundry);
    let cos_arg = TensorArg::from_tensor(&cos_foundry);
    let sin_arg = TensorArg::from_tensor(&sin_foundry);

    let kernel = Rope::new(&input_arg, &output_arg, &cos_arg, &sin_arg, params);
    foundry.run(&kernel).unwrap();

    let foundry_result = FoundryTensor::to_vec(&output_foundry, &foundry);

    // =========================================================================
    // CPU Reference
    // =========================================================================
    let cpu_result = cpu_rope(
        &input_data,
        &cos_data,
        &sin_data,
        cfg.dim,
        cfg.seq_len,
        cfg.position_offset as usize,
    );

    // =========================================================================
    // Comparison
    // =========================================================================
    let mut cpu_vs_foundry: f32 = 0.0;

    for i in 0..total_elements {
        cpu_vs_foundry = cpu_vs_foundry.max((cpu_result[i].to_f32() - foundry_result[i].to_f32()).abs());
    }

    println!(
        "\n[RoPE dim={} seq={} rows={} offset={}]",
        cfg.dim, cfg.seq_len, cfg.num_rows, cfg.position_offset
    );
    println!("  CPU vs Foundry max diff:    {:.6}", cpu_vs_foundry);

    assert!(cpu_vs_foundry <= TOLERANCE, "CPU vs Foundry mismatch: {}", cpu_vs_foundry);
}

// ============================================================================
// F16 Parity Tests
// ============================================================================

#[test]
#[serial]
fn test_rope_64x8_4() {
    run_parity_test(TestConfig {
        dim: 64,
        seq_len: 8,
        num_rows: 4,
        position_offset: 0,
    });
}

#[test]
#[serial]
fn test_rope_128x16_8() {
    run_parity_test(TestConfig {
        dim: 128,
        seq_len: 16,
        num_rows: 8,
        position_offset: 0,
    });
}

#[test]
#[serial]
fn test_rope_256x32_4() {
    run_parity_test(TestConfig {
        dim: 256,
        seq_len: 32,
        num_rows: 4,
        position_offset: 0,
    });
}

#[test]
#[serial]
fn test_rope_64x1_14() {
    // Single token, multiple heads (like autoregressive)
    run_parity_test(TestConfig {
        dim: 64,
        seq_len: 1,
        num_rows: 14,
        position_offset: 0,
    });
}

#[test]
#[serial]
fn test_rope_with_offset() {
    // Incremental decoding with position offset
    run_parity_test(TestConfig {
        dim: 64,
        seq_len: 1,
        num_rows: 4,
        position_offset: 10,
    });
}

#[test]
#[serial]
fn test_rope_896_heads() {
    // Qwen2.5-0.5B: d_model=896, n_heads=14, head_dim=64
    run_parity_test(TestConfig {
        dim: 64,
        seq_len: 8,
        num_rows: 14,
        position_offset: 0,
    });
}

// ============================================================================
// Sanity Edge Cases (Varied Shapes & Batches)
// ============================================================================

#[test]
#[serial]
fn test_rope_3x127x64() {
    run_parity_test(TestConfig {
        dim: 64,
        seq_len: 127,
        num_rows: 3,
        position_offset: 0,
    });
}

#[test]
#[serial]
fn test_rope_32x1x128_offset1024() {
    run_parity_test(TestConfig {
        dim: 128,
        seq_len: 1,
        num_rows: 32,
        position_offset: 1024,
    });
}
