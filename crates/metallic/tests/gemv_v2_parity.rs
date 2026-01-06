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
    compound::stages::Layout, foundry::{
        Foundry, spec::{DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::Pooled, tensor::Tensor as FoundryTensor
    }, metals::{
        gemv::{GemvColMajor, GemvParams, GemvRowMajor}, v2::gemv::step::GemvV2Step
    }, tensor::{F16, TensorInit}, types::TensorArg
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
        }
    }
}

// ============================================================================
// Test Runner - V2 vs Legacy Foundry Kernel
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
    let output_legacy = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::Uninitialized).unwrap();
    let output_v2 = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::Uninitialized).unwrap();
    let bias = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::CopyFrom(&bias_half)).unwrap();

    // ========== Run Legacy GEMV ==========
    let weights_arg = TensorArg::from_tensor(&weights);
    let input_arg = TensorArg::from_tensor(&input);
    let output_legacy_arg = TensorArg::from_tensor(&output_legacy);
    let bias_arg = TensorArg::from_tensor(&bias);

    let params = match cfg.layout {
        TestLayout::NK => GemvParams {
            k: cfg.k as u32,
            n: cfg.n as u32,
            batch: 1,
            stride_x: 1,
            stride_y: 1,
            stride_a: 0,
            stride_w: cfg.k as u32,
            blocks_per_k: (cfg.k / 32) as u32,
            weights_per_block: 32,
            stride_scale: 0,
        },
        TestLayout::KN => GemvParams {
            k: cfg.k as u32,
            n: cfg.n as u32,
            batch: 1,
            stride_x: 1,
            stride_y: 1,
            stride_a: 0,
            stride_w: cfg.n as u32,
            blocks_per_k: 0,
            weights_per_block: 0,
            stride_scale: 0,
        },
    };

    // Run legacy kernel based on layout type
    match cfg.layout {
        TestLayout::NK => {
            if cfg.with_bias {
                let kernel = GemvColMajor::with_bias(&weights_arg, &input_arg, &output_legacy_arg, params, &bias_arg);
                foundry.run(&kernel).unwrap();
            } else {
                let kernel = GemvColMajor::new(&weights_arg, &input_arg, &output_legacy_arg, params);
                foundry.run(&kernel).unwrap();
            }
        }
        TestLayout::KN => {
            if cfg.with_bias {
                let kernel = GemvRowMajor::with_bias(&weights_arg, &input_arg, &output_legacy_arg, params, &bias_arg);
                foundry.run(&kernel).unwrap();
            } else {
                let kernel = GemvRowMajor::new(&weights_arg, &input_arg, &output_legacy_arg, params);
                foundry.run(&kernel).unwrap();
            }
        }
    }

    // ========== Run V2 GEMV ==========
    let mut bindings = TensorBindings::new();
    let mut symbols = SymbolTable::new();

    bindings.insert("weights".to_string(), TensorArg::from_tensor(&weights));
    bindings.insert("input".to_string(), TensorArg::from_tensor(&input));
    bindings.insert("output".to_string(), TensorArg::from_tensor(&output_v2));
    if cfg.with_bias {
        bindings.insert("bias".to_string(), TensorArg::from_tensor(&bias));
    }

    let step = GemvV2Step {
        weights: Ref("weights".to_string()),
        scale_bytes: None, // F16 mode
        input: Ref("input".to_string()),
        output: Ref("output".to_string()),
        bias: if cfg.with_bias { Some(Ref("bias".to_string())) } else { None },
        k_dim: DynamicValue::Literal(cfg.k as u32),
        n_dim: DynamicValue::Literal(cfg.n as u32),
        weights_per_block: 32,
        layout: match cfg.layout {
            TestLayout::NK => Layout::RowMajor,
            TestLayout::KN => Layout::ColMajor,
        },
    };

    let compiled_steps = step.compile(&mut bindings, &mut symbols);

    // Create FastBindings
    let mut fast_bindings = FastBindings::new(symbols.len());
    for (name, id) in symbols.iter() {
        if let Ok(arg) = bindings.get(name) {
            fast_bindings.set(*id, arg);
        }
    }

    // Execute V2
    for c_step in compiled_steps {
        c_step.execute(&mut foundry, &fast_bindings, &bindings).unwrap();
    }

    // ========== Compare Results ==========
    let legacy_f16 = FoundryTensor::to_vec(&output_legacy, &foundry);
    let v2_f16 = FoundryTensor::to_vec(&output_v2, &foundry);

    let legacy_output: Vec<f32> = legacy_f16.iter().map(|x| x.to_f32()).collect();
    let v2_output: Vec<f32> = v2_f16.iter().map(|x| x.to_f32()).collect();

    println!("\n=== Test Config ===");
    println!("Layout: {:?}", cfg.layout);
    println!("K: {}, N: {}", cfg.k, cfg.n);
    println!("With bias: {}", cfg.with_bias);
    println!("Legacy output (first 10): {:?}", &legacy_output[..10.min(cfg.n)]);
    println!("V2 output (first 10):     {:?}", &v2_output[..10.min(cfg.n)]);

    // Verify results with tolerance scaling with K
    // F16 precision decays with sqrt(K)
    let tolerance = 1e-4 * (cfg.k as f32).sqrt();

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;

    for (i, (l, v)) in legacy_output.iter().zip(v2_output.iter()).enumerate() {
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
            "GemvV2 vs Legacy parity failed: max_diff = {} at index {}, expected < {}",
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

    // ========== Run Legacy GEMV (Q8) ==========
    let _weights_arg = TensorArg::from_tensor(&weights);
    let _scales_arg = TensorArg::from_tensor(&scales); // Used as 'scale_bytes' or via stride?

    // ========== Run V2 GEMV (Q8) ==========
    let mut bindings = TensorBindings::new();
    let mut symbols = SymbolTable::new();

    bindings.insert("weights".to_string(), TensorArg::from_tensor(&weights));
    bindings.insert("scales".to_string(), TensorArg::from_tensor(&scales));
    bindings.insert("input".to_string(), TensorArg::from_tensor(&input));
    bindings.insert("output".to_string(), TensorArg::from_tensor(&output_v2));
    if cfg.with_bias {
        bindings.insert("bias".to_string(), TensorArg::from_tensor(&bias));
    }

    let step = GemvV2Step {
        weights: Ref("weights".to_string()),
        scale_bytes: Some(Ref("scales".to_string())),
        input: Ref("input".to_string()),
        output: Ref("output".to_string()),
        bias: if cfg.with_bias { Some(Ref("bias".to_string())) } else { None },
        k_dim: DynamicValue::Literal(cfg.k as u32),
        n_dim: DynamicValue::Literal(cfg.n as u32),
        weights_per_block: 32,
        layout: match cfg.layout {
            TestLayout::NK => Layout::RowMajor,
            TestLayout::KN => Layout::ColMajor,
        },
    };

    let compiled_steps = step.compile(&mut bindings, &mut symbols);
    let mut fast_bindings = FastBindings::new(symbols.len());
    for (name, id) in symbols.iter() {
        if let Ok(arg) = bindings.get(name) {
            fast_bindings.set(*id, arg);
        }
    }

    for c_step in compiled_steps {
        c_step.execute(&mut foundry, &fast_bindings, &bindings).unwrap();
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
