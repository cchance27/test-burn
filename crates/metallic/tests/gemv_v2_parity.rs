//! GemvV2 Parity Test Suite
//!
//! Compares GemvV2 (Stage composition) against existing Foundry GEMV kernels:
//! - GemvColMajor (for NK layout)
//! - GemvRowMajor (for KN layout)
//!
//! Coverage:
//! - NK (Row-Major) and KN (Col-Major) layouts
//! - With and without bias
//! - Various shapes: square, K > N, N > K, large

use half::f16;
use metallic::{
    compound::stages::Layout, foundry::{Foundry, storage::Pooled, tensor::Tensor as FoundryTensor}, metals::gemv::{GemvStrategy, GemvV2Args, get_gemv_v2_kernel_f16, get_gemv_v2_kernel_q8, warp_dispatch_config}, tensor::{F16, TensorInit}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

// Tight tolerance replaced by dynamic tolerance in check_results

// ============================================================================
// Test Configuration
// ============================================================================

#[derive(Clone)]
struct V2TestConfig {
    k: usize,
    n: usize,
    with_bias: bool,
    layout: TestLayout,
    alpha: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum TestLayout {
    NK, // Row-Major: weights[row * K + k] -> uses GemvColMajor
    KN, // Col-Major: weights[k * N + row] -> uses GemvRowMajor
}

impl Default for V2TestConfig {
    fn default() -> Self {
        Self {
            k: 128,
            n: 128,
            with_bias: false,
            layout: TestLayout::NK,
            alpha: 1.0,
        }
    }
}

// ============================================================================
// CPU Reference Implementations
// ============================================================================

fn run_cpu_gemv_f16(k: usize, n: usize, layout: Layout, weights: &[f16], input: &[f16], bias: Option<&[f16]>, alpha: f32) -> Vec<f16> {
    let mut output = vec![f16::from_f32(0.0); n];
    for row in 0..n {
        let mut acc = 0.0f32;
        for ki in 0..k {
            let w_idx = match layout {
                Layout::RowMajor => row * k + ki,
                Layout::ColMajor => ki * n + row,
                Layout::Canonical => (ki % 32) + 32 * (row + (ki / 32) * n),
            };
            let w = weights[w_idx].to_f32();
            let x = input[ki].to_f32();
            acc += w * x;
        }
        let mut res = acc * alpha;
        if let Some(b) = bias {
            res += b[row].to_f32();
        }
        output[row] = f16::from_f32(res);
    }
    output
}

fn run_cpu_gemv_q8(
    k: usize,
    n: usize,
    layout: Layout,
    weights_i8: &[i8],
    scales: &[f16],
    weights_per_block: usize,
    input: &[f16],
    bias: Option<&[f16]>,
    alpha: f32,
) -> Vec<f16> {
    let mut output = vec![f16::from_f32(0.0); n];
    let blocks_per_k = (k + weights_per_block - 1) / weights_per_block;
    for row in 0..n {
        let mut acc = 0.0f32;
        let mut ki = 0;
        while ki < k {
            let chunk_size = if ki + 8 <= k { 8 } else { k - ki };
            let mut chunk_acc = 0.0f32;
            let block_idx = ki / weights_per_block;
            let scale = scales[row * blocks_per_k + block_idx].to_f32();

            for i in 0..chunk_size {
                let curr_k = ki + i;
                let w_idx = match layout {
                    Layout::RowMajor => row * k + curr_k,
                    Layout::ColMajor => curr_k * n + row,
                    Layout::Canonical => (curr_k % weights_per_block) + weights_per_block * (row + (curr_k / weights_per_block) * n),
                };
                let w = weights_i8[w_idx] as f32;
                let x = input[curr_k].to_f32();
                chunk_acc += w * x;
            }
            acc += chunk_acc * scale;
            ki += chunk_size;
        }
        let mut res = acc * alpha;
        if let Some(b) = bias {
            res += b[row].to_f32();
        }
        output[row] = f16::from_f32(res);
    }
    output
}

// ============================================================================
// Test Runner - V2 vs CPU Reference
// ============================================================================

fn run_gemv_v2_parity_test(cfg: V2TestConfig) {
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rng();

    // Generate random test data
    let weights_data: Vec<f32> = (0..cfg.k * cfg.n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let input_data: Vec<f32> = (0..cfg.k).map(|_| rng.random_range(-1.0..1.0)).collect();
    let bias_data: Vec<f32> = (0..cfg.n).map(|_| rng.random_range(-0.5..0.5)).collect();

    // Convert to f16
    let weights_half: Vec<f16> = weights_data.iter().map(|&x| f16::from_f32(x)).collect();
    let input_half: Vec<f16> = input_data.iter().map(|&x| f16::from_f32(x)).collect();
    let bias_half: Vec<f16> = bias_data.iter().map(|&x| f16::from_f32(x)).collect();

    // Create tensors for legacy
    let weights = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.k * cfg.n], TensorInit::CopyFrom(&weights_half)).unwrap();
    let input = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.k], TensorInit::CopyFrom(&input_half)).unwrap();
    //let output_legacy = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::Uninitialized).unwrap();
    let output_v2 = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::Uninitialized).unwrap();
    let bias = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::CopyFrom(&bias_half)).unwrap();

    // ========== Run CPU Reference ==========
    let cpu_layout = match cfg.layout {
        TestLayout::NK => Layout::RowMajor,
        TestLayout::KN => Layout::ColMajor,
    };
    let cpu_output_f16 = run_cpu_gemv_f16(
        cfg.k,
        cfg.n,
        cpu_layout,
        &weights_half,
        &input_half,
        if cfg.with_bias { Some(&bias_half) } else { None },
        cfg.alpha,
    );

    // ========== Run V2 GEMV ==========
    let args = GemvV2Args {
        weights: TensorArg::from_tensor(&weights),
        scale_bytes: TensorArg::from_tensor(&weights), // Dummy
        input: TensorArg::from_tensor(&input),
        output: TensorArg::from_tensor(&output_v2),
        bias: if cfg.with_bias {
            TensorArg::from_tensor(&bias)
        } else {
            TensorArg::from_tensor(&output_v2)
        },
        has_bias: if cfg.with_bias { 1 } else { 0 },
        k_dim: cfg.k as u32,
        n_dim: cfg.n as u32,
        weights_per_block: 32,
        alpha: cfg.alpha,
    };

    let layout = match cfg.layout {
        TestLayout::NK => Layout::RowMajor,
        TestLayout::KN => Layout::ColMajor,
    };
    let kernel = get_gemv_v2_kernel_f16(layout, GemvStrategy::Vectorized);
    let dispatch = warp_dispatch_config(cfg.n as u32);

    foundry.run(&kernel.bind(args, dispatch)).unwrap();

    // ========== Compare Results ==========
    let v2_f16 = FoundryTensor::to_vec(&output_v2, &foundry);

    let cpu_output: Vec<f32> = cpu_output_f16.iter().map(|x| x.to_f32()).collect();
    let v2_output: Vec<f32> = v2_f16.iter().map(|x| x.to_f32()).collect();

    println!("\n=== Test Config ===");
    println!("Layout: {:?}", cfg.layout);
    println!("K: {}, N: {}", cfg.k, cfg.n);
    println!("With bias: {}", cfg.with_bias);
    println!("CPU output (first 10): {:?}", &cpu_output[..10.min(cfg.n)]);
    println!("V2 output (first 10):  {:?}", &v2_output[..10.min(cfg.n)]);

    // Verify results with tolerance scaling with K
    // F16 precision decays with sqrt(K). Since V2 uses float4/8-way accumulation
    // while Legacy uses 4-way unrolling or single-lane, slight deviations are expected
    // in large kernels due to floating point associativity.
    let tolerance = 1e-3 * (cfg.k as f32).sqrt();

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;

    for (i, (l, v)) in cpu_output.iter().zip(v2_output.iter()).enumerate() {
        let diff = (l - v).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    println!("Max diff: {} at index {}", max_diff, max_diff_idx);
    println!("Tolerance: {}", tolerance);

    if max_diff > tolerance {
        panic!(
            "GemvV2 vs CPU parity failed: max_diff = {} at index {}, expected < {}",
            max_diff, max_diff_idx, tolerance
        );
    }
}

// ============================================================================
// NK Layout Tests (Row-Major) - Uses GemvColMajor
// ============================================================================

#[test]
#[serial]
fn test_gemv_v2_nk_128x128() {
    println!("\n=== GemvV2 NK 128x128 ===");
    run_gemv_v2_parity_test(V2TestConfig {
        k: 128,
        n: 128,
        layout: TestLayout::NK,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemv_v2_nk_256x256() {
    println!("\n=== GemvV2 NK 256x256 ===");
    run_gemv_v2_parity_test(V2TestConfig {
        k: 256,
        n: 256,
        layout: TestLayout::NK,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemv_v2_nk_512x128() {
    println!("\n=== GemvV2 NK 512x128 (K >> N) ===");
    run_gemv_v2_parity_test(V2TestConfig {
        k: 512,
        n: 128,
        layout: TestLayout::NK,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemv_v2_nk_alpha_256x256() {
    run_gemv_v2_parity_test(V2TestConfig {
        k: 256,
        n: 256,
        alpha: 0.5,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemv_v2_kn_alpha_128x128() {
    run_gemv_v2_parity_test(V2TestConfig {
        k: 128,
        n: 128,
        layout: TestLayout::KN,
        alpha: 2.0,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemv_v2_nk_128x512() {
    println!("\n=== GemvV2 NK 128x512 (N >> K) ===");
    run_gemv_v2_parity_test(V2TestConfig {
        k: 128,
        n: 512,
        layout: TestLayout::NK,
        ..Default::default()
    });
}

// ============================================================================
// NK Layout with Bias Tests
// ============================================================================

#[test]
#[serial]
fn test_gemv_v2_nk_bias_128x128() {
    println!("\n=== GemvV2 NK 128x128 with bias ===");
    run_gemv_v2_parity_test(V2TestConfig {
        k: 128,
        n: 128,
        layout: TestLayout::NK,
        with_bias: true,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemv_v2_nk_bias_256x256() {
    println!("\n=== GemvV2 NK 256x256 with bias ===");
    run_gemv_v2_parity_test(V2TestConfig {
        k: 256,
        n: 256,
        layout: TestLayout::NK,
        with_bias: true,
        ..Default::default()
    });
}

// ============================================================================
// KN Layout Tests (Col-Major) - Uses GemvRowMajor
// ============================================================================

#[test]
#[serial]
fn test_gemv_v2_kn_128x128() {
    println!("\n=== GemvV2 KN 128x128 ===");
    run_gemv_v2_parity_test(V2TestConfig {
        k: 128,
        n: 128,
        layout: TestLayout::KN,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemv_v2_kn_256x256() {
    println!("\n=== GemvV2 KN 256x256 ===");
    run_gemv_v2_parity_test(V2TestConfig {
        k: 256,
        n: 256,
        layout: TestLayout::KN,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemv_v2_kn_512x128() {
    println!("\n=== GemvV2 KN 512x128 (K >> N) ===");
    run_gemv_v2_parity_test(V2TestConfig {
        k: 512,
        n: 128,
        layout: TestLayout::KN,
        ..Default::default()
    });
}

// ============================================================================
// KN Layout with Bias Tests
// ============================================================================

#[test]
#[serial]
fn test_gemv_v2_kn_bias_128x128() {
    println!("\n=== GemvV2 KN 128x128 with bias ===");
    run_gemv_v2_parity_test(V2TestConfig {
        k: 128,
        n: 128,
        layout: TestLayout::KN,
        with_bias: true,
        ..Default::default()
    });
}

// ============================================================================
// Large Shape Tests
// ============================================================================

#[test]
#[serial]
fn test_gemv_v2_nk_4096x4096() {
    println!("\n=== GemvV2 NK 4096x4096 (large) ===");
    run_gemv_v2_parity_test(V2TestConfig {
        k: 4096,
        n: 4096,
        layout: TestLayout::NK,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemv_v2_nk_large_with_bias() {
    println!("\n=== GemvV2 NK 2048x1024 with bias ===");
    run_gemv_v2_parity_test(V2TestConfig {
        k: 2048,
        n: 1024,
        layout: TestLayout::NK,
        with_bias: true,
        ..Default::default()
    });
}

// ============================================================================
// Q8 Quantization Tests
// ============================================================================

fn run_gemv_v2_q8_parity_test(cfg: V2TestConfig) {
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rng();

    // Q8 Params
    let block_size = 32;
    let n_blocks = (cfg.k * cfg.n) / block_size;

    // Generate random Q8 data
    // Weights: int8 (interpreted as u8 for tensor)
    let weights_data: Vec<u8> = (0..cfg.k * cfg.n).map(|_| rng.random_range(0..255)).collect();
    // Scales: f16
    let scales_data: Vec<f16> = (0..n_blocks).map(|_| f16::from_f32(rng.random_range(0.1..2.0))).collect();
    // Input: f16
    let input_data: Vec<f16> = (0..cfg.k).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();
    // Bias: f16
    let bias_data: Vec<f16> = (0..cfg.n).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();

    // Create tensors
    // Use U8 for weights to simulate raw bytes
    use metallic::tensor::U8;
    let weights = FoundryTensor::<U8, Pooled>::new(&mut foundry, vec![cfg.k * cfg.n], TensorInit::CopyFrom(&weights_data)).unwrap();
    let scales = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_blocks], TensorInit::CopyFrom(&scales_data)).unwrap();
    let input = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.k], TensorInit::CopyFrom(&input_data)).unwrap();
    let _output_legacy = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::Uninitialized).unwrap();
    let output_v2 = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::Uninitialized).unwrap();
    let bias = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::CopyFrom(&bias_data)).unwrap();

    let args = GemvV2Args {
        weights: TensorArg::from_tensor(&weights),
        scale_bytes: TensorArg::from_tensor(&scales),
        input: TensorArg::from_tensor(&input),
        output: TensorArg::from_tensor(&output_v2),
        bias: if cfg.with_bias {
            TensorArg::from_tensor(&bias)
        } else {
            TensorArg::from_tensor(&output_v2)
        },
        has_bias: if cfg.with_bias { 1 } else { 0 },
        k_dim: cfg.k as u32,
        n_dim: cfg.n as u32,
        weights_per_block: 32,
        alpha: cfg.alpha,
    };

    let layout = match cfg.layout {
        TestLayout::NK => Layout::RowMajor,
        TestLayout::KN => Layout::ColMajor,
    };
    let kernel = get_gemv_v2_kernel_q8(layout, GemvStrategy::Vectorized);
    let dispatch = warp_dispatch_config(cfg.n as u32);

    foundry.run(&kernel.bind(args, dispatch)).unwrap();

    // ========== Run CPU Reference (Q8) ==========
    let weights_i8: Vec<i8> = weights_data.iter().map(|&x| x as i8).collect();
    let cpu_output_f16 = run_cpu_gemv_q8(
        cfg.k,
        cfg.n,
        layout,
        &weights_i8,
        &scales_data,
        32,
        &input_data,
        if cfg.with_bias { Some(&bias_data) } else { None },
        cfg.alpha,
    );
    let v2_f16 = FoundryTensor::to_vec(&output_v2, &foundry);

    let cpu_output: Vec<f32> = cpu_output_f16.iter().map(|x| x.to_f32()).collect();
    let v2_output: Vec<f32> = v2_f16.iter().map(|x| x.to_f32()).collect();

    let tolerance = 1e-2 * (cfg.k as f32).sqrt(); // Q8 has higher variance
    let mut max_diff = 0.0f32;
    for (l, v) in cpu_output.iter().zip(v2_output.iter()) {
        max_diff = max_diff.max((l - v).abs());
    }

    println!("Max diff (Q8): {}", max_diff);
    if max_diff > tolerance {
        panic!("Q8 Parity failed: max_diff {} > {}", max_diff, tolerance);
    }

    println!("V2 Q8 run successful.");
}

#[test]
#[serial]
fn test_gemv_v2_nk_q8_128x128() {
    println!("\n=== GemvV2 NK 128x128 Q8 (Sanity) ===");
    run_gemv_v2_q8_parity_test(V2TestConfig {
        k: 128,
        n: 128,
        layout: TestLayout::NK,
        ..Default::default()
    });
}
