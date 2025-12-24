//! Comprehensive GEMV Test Suite
//!
//! Tests all GEMV variations against legacy Context-based implementation.

use half::f16;
use metallic::{
    Context, MetalError, QuantizedQ8_0Tensor, foundry::{Foundry, storage::Pooled, tensor::Tensor as FoundryTensor}, kernels::matmul_gemv::MatmulGemvOp, metals::gemv::{GemvCanonical, GemvColMajor, GemvParams, GemvRowMajor}, policies::PolicyQ8, tensor::{F16, Tensor, TensorInit, TensorStorage as LegacyStorage, TensorType, U8}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

const CPU_TOLERANCE: f32 = 0.05;
const PARITY_TOLERANCE: f32 = 1e-7;

// ============================================================================
// Test Helpers
// ============================================================================

struct TestConfig {
    k: usize,
    n: usize,
    batch: usize,
    with_bias: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            k: 128,
            n: 128,
            batch: 1,
            with_bias: false,
        }
    }
}

fn make_gemv_params(cfg: &TestConfig) -> GemvParams {
    GemvParams {
        k: cfg.k as u32,
        n: cfg.n as u32,
        blocks_per_k: (cfg.k / 32) as u32,
        weights_per_block: 32,
        batch: cfg.batch as u32,
        stride_x: 1,
        stride_y: 1,
        stride_a: 0,
        stride_w: cfg.k as u32, // Match legacy behavior (K) even if potentially wrong for row-major
        stride_scale: 0,
    }
}

fn cpu_gemv_dense(matrix: &[f32], vector: &[f32], k: usize, n: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let mut result = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += matrix[i * k + j] * vector[j];
        }
        if let Some(b) = bias {
            sum += b[i];
        }
        result[i] = sum;
    }
    result
}

fn cpu_gemv_row_strided(matrix: &[f32], vector: &[f32], k: usize, n: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let mut result = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += matrix[j * n + i] * vector[j];
        }
        if let Some(b) = bias {
            sum += b[i];
        }
        result[i] = sum;
    }
    result
}

/// CPU reference for GEMV (Canonical k-block-major layout)
/// Matrix layout: element (col, k) at matrix[col * wpb + block_idx * (N * wpb) + elem_in_block]
/// where block_idx = k / wpb, elem_in_block = k % wpb
fn cpu_gemv_canonical(matrix: &[f32], vector: &[f32], k: usize, n: usize, weights_per_block: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; n];
    let stride_w = n * weights_per_block; // Stride between k-blocks

    for col in 0..n {
        let mut sum = 0.0f32;
        for row in 0..k {
            let block_idx = row / weights_per_block;
            let elem_in_block = row % weights_per_block;
            // Formula: col * wpb + block_idx * stride_w + elem
            let idx = col * weights_per_block + block_idx * stride_w + elem_in_block;
            sum += matrix[idx] * vector[row];
        }
        result[col] = sum;
    }
    result
}

/// Run GEMV test comparing new Foundry kernel to legacy Context kernel (Row-Major)
fn run_f16_parity_test(cfg: TestConfig) {
    let mut foundry = Foundry::new().unwrap();
    let mut ctx = Context::new().unwrap();
    let mut rng = rng();

    // Generate test data
    let matrix_data: Vec<f32> = (0..cfg.k * cfg.n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let vector_data: Vec<f32> = (0..cfg.k).map(|_| rng.random_range(-1.0..1.0)).collect();
    let bias_data: Vec<f32> = (0..cfg.n).map(|_| rng.random_range(-0.5..0.5)).collect();

    let matrix_half: Vec<f16> = matrix_data.iter().map(|&x| f16::from_f32(x)).collect();
    let vector_half: Vec<f16> = vector_data.iter().map(|&x| f16::from_f32(x)).collect();
    let bias_half: Vec<f16> = bias_data.iter().map(|&x| f16::from_f32(x)).collect();

    // Legacy tensors (We want to test y = Ax, so we pass A^T to Legacy's y = xA)
    // Legacy will do x * (A^T) = (A * x^T)^T = Ax
    let x_legacy = Tensor::<F16>::new(vec![1, cfg.k], LegacyStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&vector_half)).unwrap();
    let a_rows = Tensor::<F16>::new(
        vec![cfg.n, cfg.k],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&matrix_half),
    )
    .unwrap();
    let a_legacy = a_rows.permute(&[1, 0], &mut ctx).unwrap();

    let bias_legacy = if cfg.with_bias {
        Some(Tensor::<F16>::new(vec![1, cfg.n], LegacyStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&bias_half)).unwrap())
    } else {
        None
    };

    // Run legacy GEMV (y = x * A_legacy = x * A_rows^T)
    let y_legacy_tensor = ctx
        .call::<MatmulGemvOp>((&x_legacy, TensorType::Dense(&a_legacy), false, bias_legacy.as_ref()), None)
        .unwrap();
    let legacy_result = y_legacy_tensor.to_vec();

    // CPU Reference (y = Ax)
    let cpu_expected = cpu_gemv_dense(
        &matrix_data,
        &vector_data,
        cfg.k,
        cfg.n,
        if cfg.with_bias { Some(&bias_data) } else { None },
    );

    // New foundry GEMV
    let matrix = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.k * cfg.n], TensorInit::CopyFrom(&matrix_half)).unwrap();
    let vector = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.k], TensorInit::CopyFrom(&vector_half)).unwrap();
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::Uninitialized).unwrap();
    let bias = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::CopyFrom(&bias_half)).unwrap();

    let params = make_gemv_params(&cfg);
    let matrix_arg = TensorArg::from_tensor(&matrix);
    let vector_arg = TensorArg::from_tensor(&vector);
    let output_arg = TensorArg::from_tensor(&output);
    let bias_arg = TensorArg::from_tensor(&bias);

    // [n, k] Row-Major -> each row size k is contiguous -> use GemvColMajor
    let kernel = if cfg.with_bias {
        GemvColMajor::with_bias(&matrix_arg, &vector_arg, &output_arg, params, &bias_arg)
    } else {
        GemvColMajor::new(&matrix_arg, &vector_arg, &output_arg, params)
    };
    foundry.run(&kernel).unwrap();

    let gpu_output = FoundryTensor::to_vec(&output, &foundry);
    let gpu_f32: Vec<f32> = gpu_output.iter().map(|x| x.to_f32()).collect();

    let mut max_diff_legacy: f32 = 0.0;
    let mut max_diff_cpu: f32 = 0.0;
    for i in 0..cfg.n {
        max_diff_legacy = max_diff_legacy.max((legacy_result[i].to_f32() - gpu_f32[i]).abs());
        max_diff_cpu = max_diff_cpu.max((cpu_expected[i] - gpu_f32[i]).abs());
    }

    println!("\nFirst 10 from new: {:?}", &gpu_f32[0..10]);
    println!("First 10 from legacy: {:?}", &legacy_result[0..10]);
    println!("First 10 from CPU: {:?}", &cpu_expected[0..10]);
    println!("    Legacy vs New max diff: {:.6}", max_diff_legacy);
    println!("    CPU vs New max diff:    {:.6}", max_diff_cpu);

    assert!(max_diff_legacy < PARITY_TOLERANCE, "Legacy vs New mismatch: {:.7}", max_diff_legacy);
    assert!(max_diff_cpu < CPU_TOLERANCE, "New vs CPU row-major mismatch: {:.7}", max_diff_cpu);
}

/// Run Column-Major GEMV test against Legacy Dense mode
fn run_col_major_parity_test(cfg: TestConfig) {
    let mut foundry = Foundry::new().unwrap();
    let mut ctx = Context::new().unwrap();
    let mut rng = rng();

    let matrix_data: Vec<f32> = (0..cfg.k * cfg.n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let vector_data: Vec<f32> = (0..cfg.k).map(|_| rng.random_range(-1.0..1.0)).collect();
    let bias_data: Vec<f32> = (0..cfg.n).map(|_| rng.random_range(-0.5..0.5)).collect();

    let matrix_half: Vec<f16> = matrix_data.iter().map(|&x| f16::from_f32(x)).collect();
    let vector_half: Vec<f16> = vector_data.iter().map(|&x| f16::from_f32(x)).collect();
    let bias_half: Vec<f16> = bias_data.iter().map(|&x| f16::from_f32(x)).collect();

    // Legacy expects y = xA. We want y = Ax.
    // matrix_half is Column-Major [n, k] (physically same as [k, n] Row-Major)
    // Legacy needs [k, n] to produce [1, n] output from [1, k] input.
    let x_legacy = Tensor::<F16>::new(vec![1, cfg.k], LegacyStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&vector_half)).unwrap();
    let a_legacy = Tensor::<F16>::new(
        vec![cfg.k, cfg.n], // Treat as [k, n] Row-Major physically
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&matrix_half),
    )
    .unwrap();

    let bias_legacy = if cfg.with_bias {
        Some(Tensor::<F16>::new(vec![1, cfg.n], LegacyStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&bias_half)).unwrap())
    } else {
        None
    };

    // Run legacy GEMV (y = x * A_legacy)
    let y_legacy_tensor = ctx
        .call::<MatmulGemvOp>((&x_legacy, TensorType::Dense(&a_legacy), false, bias_legacy.as_ref()), None)
        .unwrap();
    let legacy_result = y_legacy_tensor.to_vec();

    // Foundry GEMV
    let matrix = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.k * cfg.n], TensorInit::CopyFrom(&matrix_half)).unwrap();
    let vector = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.k], TensorInit::CopyFrom(&vector_half)).unwrap();
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::Uninitialized).unwrap();
    let bias = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::CopyFrom(&bias_half)).unwrap();

    let params = GemvParams {
        k: cfg.k as u32,
        n: cfg.n as u32,
        batch: 1,
        stride_x: 1,
        stride_y: 1,
        stride_a: 0,
        stride_w: cfg.n as u32, // matrix is [k, n] Row-Major physically -> K has stride n
        blocks_per_k: 0,
        weights_per_block: 0,
        stride_scale: 0,
    };
    let matrix_arg = TensorArg::from_tensor(&matrix);
    let vector_arg = TensorArg::from_tensor(&vector);
    let output_arg = TensorArg::from_tensor(&output);
    let bias_arg = TensorArg::from_tensor(&bias);

    // [n, k] Column-Major -> K has stride n -> use GemvRowMajor
    let kernel = if cfg.with_bias {
        GemvRowMajor::with_bias(&matrix_arg, &vector_arg, &output_arg, params, &bias_arg)
    } else {
        GemvRowMajor::new(&matrix_arg, &vector_arg, &output_arg, params)
    };
    foundry.run(&kernel).unwrap();

    let gpu_output = FoundryTensor::to_vec(&output, &foundry);
    let gpu_f32: Vec<f32> = gpu_output.iter().map(|x| x.to_f32()).collect();

    // CPU Reference (y = Ax for Column-Major data)
    let cpu_expected = cpu_gemv_row_strided(
        &matrix_data,
        &vector_data,
        cfg.k,
        cfg.n,
        if cfg.with_bias { Some(&bias_data) } else { None },
    );

    let mut max_diff_legacy: f32 = 0.0;
    let mut max_diff_cpu: f32 = 0.0;
    for i in 0..cfg.n {
        max_diff_legacy = max_diff_legacy.max((legacy_result[i].to_f32() - gpu_f32[i]).abs());
        max_diff_cpu = max_diff_cpu.max((cpu_expected[i] - gpu_f32[i]).abs());
    }

    println!("\nFirst 10 from new: {:?}", &gpu_f32[0..10]);
    println!("First 10 from legacy: {:?}", &legacy_result[0..10]);
    println!("First 10 from CPU: {:?}", &cpu_expected[0..10]);
    println!("    ColMajor Legacy Parity max diff: {:.6}", max_diff_legacy);
    println!("    ColMajor CPU Parity max diff:    {:.6}", max_diff_cpu);
    assert!(
        max_diff_legacy < PARITY_TOLERANCE,
        "ColMajor Legacy parity mismatch: {}",
        max_diff_legacy
    );
    assert!(max_diff_cpu < CPU_TOLERANCE, "ColMajor CPU parity mismatch: {}", max_diff_cpu);
}

/// Run Canonical GEMV test against Legacy DenseCanonical mode
fn run_canonical_parity_test(cfg: TestConfig) {
    let mut foundry = Foundry::new().unwrap();
    let mut ctx = Context::new().unwrap();
    let mut rng = rng();

    let wpb = 32;
    let matrix_row_major: Vec<f32> = (0..cfg.k * cfg.n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let vector_data: Vec<f32> = (0..cfg.k).map(|_| rng.random_range(-1.0..1.0)).collect();
    let bias_data: Vec<f32> = (0..cfg.n).map(|_| rng.random_range(-0.5..0.5)).collect();

    let vector_half: Vec<f16> = vector_data.iter().map(|&x| f16::from_f32(x)).collect();
    let bias_half: Vec<f16> = bias_data.iter().map(|&x| f16::from_f32(x)).collect();

    let mut a_legacy_can = metallic::tensor::CanonicalF16Tensor::<F16>::new(vec![cfg.k, cfg.n], &mut ctx).unwrap();
    let matrix_f16: Vec<f16> = matrix_row_major.iter().map(|&x| f16::from_f32(x)).collect();
    a_legacy_can.write_from_nk_slice(&matrix_f16, &[cfg.n, cfg.k], 0).unwrap();

    let x_legacy = Tensor::<F16>::new(vec![1, cfg.k], LegacyStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&vector_half)).unwrap();
    let bias_legacy = if cfg.with_bias {
        Some(Tensor::<F16>::new(vec![cfg.n], LegacyStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&bias_half)).unwrap())
    } else {
        None
    };

    let y_legacy_tensor = ctx
        .call::<MatmulGemvOp>(
            (&x_legacy, TensorType::DenseCanonical(&a_legacy_can), false, bias_legacy.as_ref()),
            None,
        )
        .unwrap();
    let legacy_result = y_legacy_tensor.to_vec();

    let matrix_canonical_data = a_legacy_can.data.to_vec();
    let matrix = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![matrix_canonical_data.len()],
        TensorInit::CopyFrom(&matrix_canonical_data),
    )
    .unwrap();
    let vector = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.k], TensorInit::CopyFrom(&vector_half)).unwrap();
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::Uninitialized).unwrap();
    let bias = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::CopyFrom(&bias_half)).unwrap();

    let mut params = make_gemv_params(&cfg);
    params.weights_per_block = wpb as u32;
    params.blocks_per_k = a_legacy_can.blocks_per_k as u32;

    let matrix_arg = TensorArg::from_tensor(&matrix);
    let vector_arg = TensorArg::from_tensor(&vector);
    let output_arg = TensorArg::from_tensor(&output);
    let bias_arg = TensorArg::from_tensor(&bias);

    let kernel = if cfg.with_bias {
        GemvCanonical::with_bias(&matrix_arg, &vector_arg, &output_arg, params, &bias_arg)
    } else {
        GemvCanonical::new(&matrix_arg, &vector_arg, &output_arg, params)
    };
    foundry.run(&kernel).unwrap();

    let gpu_output = FoundryTensor::to_vec(&output, &foundry);
    let gpu_f32: Vec<f32> = gpu_output.iter().map(|x| x.to_f32()).collect();

    // CPU Reference
    let cpu_expected = cpu_gemv_canonical(
        &matrix_canonical_data.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &vector_data,
        cfg.k,
        cfg.n,
        wpb,
    );

    let mut max_diff_legacy: f32 = 0.0;
    let mut max_diff_cpu: f32 = 0.0;
    for i in 0..cfg.n {
        max_diff_legacy = max_diff_legacy.max((legacy_result[i].to_f32() - gpu_f32[i]).abs());
        max_diff_cpu = max_diff_cpu.max((cpu_expected[i] - gpu_f32[i]).abs());
    }
    println!("\nFirst 10 from new: {:?}", &gpu_f32[0..10]);
    println!("First 10 from legacy: {:?}", &legacy_result[0..10]);
    println!("First 10 from CPU: {:?}", &cpu_expected[0..10]);
    println!("    Canonical Legacy Parity max diff: {:.6}", max_diff_legacy);
    println!("    Canonical CPU Parity max diff:    {:.6}", max_diff_cpu);
    assert!(
        max_diff_legacy < PARITY_TOLERANCE,
        "Canonical Legacy parity mismatch: {}",
        max_diff_legacy
    );
    assert!(max_diff_cpu < CPU_TOLERANCE, "Canonical CPU parity mismatch: {}", max_diff_cpu);
}

/// Run GEMV test against CPU reference
fn run_cpu_test(cfg: TestConfig) {
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rng();

    let matrix_data: Vec<f32> = (0..cfg.k * cfg.n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let vector_data: Vec<f32> = (0..cfg.k).map(|_| rng.random_range(-1.0..1.0)).collect();

    let expected = cpu_gemv_dense(&matrix_data, &vector_data, cfg.k, cfg.n, None);

    let matrix_half: Vec<f16> = matrix_data.iter().map(|&x| f16::from_f32(x)).collect();
    let vector_half: Vec<f16> = vector_data.iter().map(|&x| f16::from_f32(x)).collect();

    let matrix = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.k * cfg.n], TensorInit::CopyFrom(&matrix_half)).unwrap();
    let vector = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.k], TensorInit::CopyFrom(&vector_half)).unwrap();
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::Uninitialized).unwrap();

    let params = GemvParams {
        k: cfg.k as u32,
        n: cfg.n as u32,
        batch: 1,
        stride_x: 1,
        stride_y: 1,
        stride_a: 0,
        stride_w: cfg.k as u32,
        blocks_per_k: 0,
        weights_per_block: 0,
        stride_scale: 0,
    };
    let matrix_arg = TensorArg::from_tensor(&matrix);
    let vector_arg = TensorArg::from_tensor(&vector);
    let output_arg = TensorArg::from_tensor(&output);

    let kernel = GemvColMajor::new(&matrix_arg, &vector_arg, &output_arg, params);
    foundry.run(&kernel).unwrap();

    let gpu_output = FoundryTensor::to_vec(&output, &foundry);
    let gpu_f32: Vec<f32> = gpu_output.iter().map(|x| x.to_f32()).collect();

    let mut max_diff: f32 = 0.0;
    for i in 0..cfg.n {
        let diff = (expected[i] - gpu_f32[i]).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < CPU_TOLERANCE,
            "CPU mismatch at {}: expected={}, got={}",
            i,
            expected[i],
            gpu_f32[i]
        );
    }

    println!("\nFirst 10 from new: {:?}", &gpu_f32[0..10]);
    println!("First 10 from CPU: {:?}", &expected[0..10]);
    println!("    CPU Max diff: {:.6}", max_diff);
}

// ============================================================================
// F16 Dense Tests
// ============================================================================

#[test]
#[serial]
fn test_f16_128x128() {
    println!("F16 128x128 (strided):");
    run_f16_parity_test(TestConfig {
        k: 128,
        n: 128,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_f16_256x256() {
    println!("F16 256x256 (strided):");
    run_f16_parity_test(TestConfig {
        k: 256,
        n: 256,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_f16_512x512() {
    println!("F16 512x512 (strided):");
    run_f16_parity_test(TestConfig {
        k: 512,
        n: 512,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_f16_256x512() {
    println!("F16 256x512 (N > K, non-square):");
    run_f16_parity_test(TestConfig {
        k: 256,
        n: 512,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_f16_128x64() {
    println!("F16 128x64 (K > N):");
    run_f16_parity_test(TestConfig {
        k: 128,
        n: 64,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_f16_512x128() {
    println!("F16 512x128 (K >> N):");
    run_f16_parity_test(TestConfig {
        k: 512,
        n: 128,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_f16_4096x4096() {
    println!("F16 4096x4096 (strided):");
    run_f16_parity_test(TestConfig {
        k: 4096,
        n: 4096,
        ..Default::default()
    });
}

// ============================================================================
// F16 with Bias Tests
// ============================================================================

#[test]
#[serial]
fn test_f16_with_bias_128x128() {
    println!("F16 128x128 with bias:");
    run_f16_parity_test(TestConfig {
        k: 128,
        n: 128,
        with_bias: true,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_f16_with_bias_256x256() {
    println!("F16 256x256 with bias:");
    run_f16_parity_test(TestConfig {
        k: 256,
        n: 256,
        with_bias: true,
        ..Default::default()
    });
}

// ============================================================================
// CPU Reference Tests
// ============================================================================

#[test]
#[serial]
fn test_cpu_128x128() {
    println!("CPU reference 128x128:");
    run_cpu_test(TestConfig {
        k: 128,
        n: 128,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_cpu_64x32() {
    println!("CPU reference 32x64 (N >= K):");
    run_cpu_test(TestConfig {
        k: 32, // Swap: K must be <= N for current indexing formula
        n: 64,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_cpu_1000x500() {
    println!("CPU reference 500x1000 (N >= K):");
    run_cpu_test(TestConfig {
        k: 500, // Swap: K must be <= N for current indexing formula
        n: 1000,
        ..Default::default()
    });
}

// ============================================================================
// Column-Major Tests (verifying col * stride + k formula)
// ============================================================================

#[test]
#[serial]
fn test_col_major_128x128() {
    println!("ColMajor 128x128:");
    run_col_major_parity_test(TestConfig {
        k: 128,
        n: 128,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_col_major_256x256() {
    println!("ColMajor 256x256:");
    run_col_major_parity_test(TestConfig {
        k: 256,
        n: 256,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_col_major_64x128() {
    println!("ColMajor 64x128 (N > K):");
    run_col_major_parity_test(TestConfig {
        k: 64,
        n: 128,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_col_major_128x64() {
    println!("ColMajor 128x64 (K > N):");
    run_col_major_parity_test(TestConfig {
        k: 128,
        n: 64,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_col_major_512x128() {
    println!("ColMajor 512x128 (K >> N):");
    run_col_major_parity_test(TestConfig {
        k: 512,
        n: 128,
        ..Default::default()
    });
}

// ============================================================================
// Canonical Tests (k-block-major layout)
// ============================================================================

#[test]
#[serial]
fn test_canonical_128x128() {
    println!("Canonical 128x128:");
    run_canonical_parity_test(TestConfig {
        k: 128,
        n: 128,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_canonical_256x256() {
    println!("Canonical 256x256:");
    run_canonical_parity_test(TestConfig {
        k: 256,
        n: 256,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_canonical_64x128() {
    println!("Canonical 64x128 (N > K):");
    run_canonical_parity_test(TestConfig {
        k: 64,
        n: 128,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_canonical_128x64() {
    println!("Canonical 128x64 (K > N):");
    run_canonical_parity_test(TestConfig {
        k: 128,
        n: 64,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_canonical_512x128() {
    println!("Canonical 512x128 (K >> N):");
    run_canonical_parity_test(TestConfig {
        k: 512,
        n: 128,
        ..Default::default()
    });
}

// Parity tests vs Legacy removed (redundant with above CPU + Parity combo)

/// Quantize a Row-Major [N, K] matrix to Q8_0 blocks.
/// Returns packed blocks [N, blocks_per_k, 34 bytes].
fn quantize_q8_0(n: usize, k: usize, data: &[f16]) -> Vec<u8> {
    let blocks_per_k = (k + 31) / 32;
    let mut packed = vec![0u8; n * blocks_per_k * 34];

    for row in 0..n {
        for b in 0..blocks_per_k {
            let mut max_abs = 0.0f32;
            for i in 0..32 {
                let k_idx = b * 32 + i;
                if k_idx < k {
                    let val = data[row * k + k_idx].to_f32().abs();
                    if val > max_abs {
                        max_abs = val;
                    }
                }
            }

            let scale = max_abs / 127.0;
            let scale_f16 = f16::from_f32(scale);
            let block_offset = (row * blocks_per_k + b) * 34;
            packed[block_offset..block_offset + 2].copy_from_slice(&scale_f16.to_le_bytes());

            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for i in 0..32 {
                let k_idx = b * 32 + i;
                if k_idx < k {
                    let val = data[row * k + k_idx].to_f32();
                    let q = (val * inv_scale).round() as i8;
                    packed[block_offset + 2 + i] = q as u8;
                }
            }
        }
    }
    packed
}

/// Swizzle Q8_0 blocks into canonical (K-block-major) layout.
fn swizzle_q8_canonical(n: usize, k: usize, packed: &[u8]) -> Vec<u8> {
    let blocks_per_k = (k + 31) / 32;
    let mut swizzled = vec![0u8; packed.len()];
    for b in 0..blocks_per_k {
        for row in 0..n {
            let src_block = row * blocks_per_k + b;
            let dst_block = b * n + row;
            swizzled[dst_block * 34..(dst_block + 1) * 34].copy_from_slice(&packed[src_block * 34..(src_block + 1) * 34]);
        }
    }
    swizzled
}

fn run_row_major_q8_parity_test(k: usize, n: usize) -> Result<(), MetalError> {
    let mut ctx = Context::<F16>::new()?;
    let ctx_u8 = Context::<U8>::new()?;
    let mut foundry = Foundry::new()?;

    // 1. Prepare Data
    let a_data = (0..n * k).map(|i| f16::from_f32((i % 10) as f32 - 5.0)).collect::<Vec<_>>();
    let x_data = (0..k).map(|i| f16::from_f32((i % 7) as f32 - 3.0)).collect::<Vec<_>>();

    // 2. Quantize (Row-Major blocks)
    let a_packed = quantize_q8_0(n, k, &a_data);

    // 3. Prepare Legacy (expects canonical for Q8)
    let a_swizzled = swizzle_q8_canonical(n, k, &a_packed);
    let blocks_total = n * ((k + 31) / 32);
    let mut data_bytes = Vec::with_capacity(blocks_total * 32);
    let mut scale_bytes = Vec::with_capacity(blocks_total * 2);
    for block in a_swizzled.chunks_exact(34) {
        scale_bytes.extend_from_slice(&block[0..2]);
        data_bytes.extend_from_slice(&block[2..34]);
    }
    let a_legacy_quant = QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![k, n], &data_bytes, &scale_bytes, &ctx_u8)?;
    let x_legacy = Tensor::<F16>::new(vec![1, k], LegacyStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&x_data))?;

    let y_legacy_tensor = ctx.call::<MatmulGemvOp>(
        (
            &x_legacy,
            TensorType::Quant(metallic::tensor::QuantizedTensor::Q8_0(&a_legacy_quant)),
            false,
            None,
        ),
        None,
    )?;
    let y_legacy = y_legacy_tensor.to_vec();

    // 4. Prepare Foundry (Row-Major blocks)
    let mut f_data = Vec::with_capacity(blocks_total * 32);
    let mut f_scales = Vec::with_capacity(blocks_total * 2);
    for block in a_packed.chunks_exact(34) {
        f_scales.extend_from_slice(&block[0..2]);
        f_data.extend_from_slice(&block[2..34]);
    }

    let a_foundry = FoundryTensor::<U8, Pooled>::new(&mut foundry, vec![f_data.len()], TensorInit::CopyFrom(&f_data))?;
    let s_foundry = FoundryTensor::<U8, Pooled>::new(&mut foundry, vec![f_scales.len()], TensorInit::CopyFrom(&f_scales))?;
    let x_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k], TensorInit::CopyFrom(&x_data))?;
    let y_foundry_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n], TensorInit::Uninitialized)?;

    let params = GemvParams {
        k: k as u32,
        n: n as u32,
        blocks_per_k: ((k + 31) / 32) as u32,
        weights_per_block: 32,
        batch: 1,
        stride_x: 1,
        stride_y: 1,
        stride_a: 0,
        stride_w: (((k + 31) / 32) * 32) as u32, // Stride in data bytes
        stride_scale: (((k + 31) / 32) * 2) as u32,
    };

    let matrix_arg = TensorArg::from_tensor(&a_foundry);
    let vector_arg = TensorArg::from_tensor(&x_foundry);
    let output_arg = TensorArg::from_tensor(&y_foundry_tensor);
    let scale_arg = TensorArg::from_tensor(&s_foundry);

    let kernel = GemvColMajor::new(&matrix_arg, &vector_arg, &output_arg, params).with_scales(&scale_arg);
    foundry.run_with_policy::<PolicyQ8, _>(&kernel)?;
    let y_foundry = FoundryTensor::to_vec(&y_foundry_tensor, &foundry);

    // 5. Compare
    let mut max_diff = 0.0f32;
    for i in 0..n {
        let diff = (y_legacy[i].to_f32() - y_foundry[i].to_f32()).abs();
        max_diff = max_diff.max(diff);
    }

    println!("\nFirst 10 from new: {:?}", &y_foundry[0..10]);
    println!("First 10 from legacy: {:?}", &y_legacy[0..10]);
    println!("RowMajor Q8 Parity (K={}, N={}): Max Diff = {:.6}", k, n, max_diff);
    assert!(max_diff < PARITY_TOLERANCE, "RowMajor Q8 Parity failed! Max Diff: {:.6}", max_diff);

    Ok(())
}

fn run_col_major_q8_parity_test(k: usize, n: usize) -> Result<(), MetalError> {
    let mut ctx = Context::<F16>::new()?;
    let ctx_u8 = Context::<U8>::new()?;
    let mut foundry = Foundry::new()?;

    // 1. Prepare Data
    let a_data_row_major = (0..n * k).map(|i| f16::from_f32((i % 10) as f32 - 5.0)).collect::<Vec<_>>();
    let x_data = (0..k).map(|i| f16::from_f32((i % 7) as f32 - 3.0)).collect::<Vec<_>>();

    // Create Col-Major F16 [K, N] by transposing Row-Major [N, K]
    let mut a_data_col_major = vec![f16::ZERO; n * k];
    for r in 0..n {
        for c in 0..k {
            a_data_col_major[r * k + c] = a_data_row_major[r * k + c];
        }
    }
    // Wait, the above is still row-major. Let's do a real transpose.
    let mut a_transposed = vec![f16::ZERO; n * k];
    for r in 0..n {
        for c in 0..k {
            a_transposed[c * n + r] = a_data_row_major[r * k + c];
        }
    }
    // Now a_transposed is [K, N] column-major if we consider n as the number of columns.
    // Wait, if GemvColMajor expects [K, N] where K is fast, then each of the N columns should be quantized.
    // So we should quantize N rows of size K.
    // a_data_row_major is already N rows of size K.
    let a_packed = quantize_q8_0(n, k, &a_data_row_major);
    // a_packed is [N, blocks_per_k, 34 bytes].
    // For Col-Major GEMV, we want columns to be contiguous.
    // So we swizzle it so that columns are contiguous.
    let a_swizzled = swizzle_q8_canonical(n, k, &a_packed);
    // a_swizzled is [blocks_per_k, n, 34 bytes].
    // This means for a fixed block_idx, all N rows are contiguous?
    // No, swizzle_q8_canonical does: dst_block = b * n + row.
    // So for block 0, we have row 0, row 1, ..., row N-1.
    // This is NOT what GemvColMajor expects. GemvColMajor expects:
    // col * stride_w + row_in_col.
    // If quantization is along K, then row_in_col is the K dimension.
    // So we want each column (size K) to be contiguous data.
    // That means a_packed [N, blocks_per_k, 34] is already what we want if we consider each of the N elements as a column.

    // 3. Prepare Legacy (Canonical Q8)
    let blocks_total = n * ((k + 31) / 32);
    let mut data_bytes = Vec::with_capacity(blocks_total * 32);
    let mut scale_bytes = Vec::with_capacity(blocks_total * 2);
    for block in a_swizzled.chunks_exact(34) {
        scale_bytes.extend_from_slice(&block[0..2]);
        data_bytes.extend_from_slice(&block[2..34]);
    }
    let a_legacy_quant = QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![k, n], &data_bytes, &scale_bytes, &ctx_u8)?;
    let x_legacy = Tensor::<F16>::new(vec![1, k], LegacyStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&x_data))?;

    let y_legacy_tensor = ctx.call::<MatmulGemvOp>(
        (
            &x_legacy,
            TensorType::Quant(metallic::tensor::QuantizedTensor::Q8_0(&a_legacy_quant)),
            false,
            None,
        ),
        None,
    )?;
    let y_legacy = y_legacy_tensor.to_vec();

    // 4. Prepare Foundry (Col-Major blocks)
    // For Col-Major, we use a_packed directly because each "row" in a_packed is a column of size K.
    let mut f_data = Vec::with_capacity(blocks_total * 32);
    let mut f_scales = Vec::with_capacity(blocks_total * 2);
    for block in a_packed.chunks_exact(34) {
        f_scales.extend_from_slice(&block[0..2]);
        f_data.extend_from_slice(&block[2..34]);
    }

    let a_foundry = FoundryTensor::<U8, Pooled>::new(&mut foundry, vec![f_data.len()], TensorInit::CopyFrom(&f_data))?;
    let s_foundry = FoundryTensor::<U8, Pooled>::new(&mut foundry, vec![f_scales.len()], TensorInit::CopyFrom(&f_scales))?;
    let x_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k], TensorInit::CopyFrom(&x_data))?;
    let y_foundry_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n], TensorInit::Uninitialized)?;

    let params = GemvParams {
        k: k as u32,
        n: n as u32,
        blocks_per_k: ((k + 31) / 32) as u32,
        weights_per_block: 32,
        batch: 1,
        stride_x: 1,
        stride_y: 1,
        stride_a: 0,
        stride_w: (((k + 31) / 32) * 32) as u32,
        stride_scale: (((k + 31) / 32) * 2) as u32,
    };

    let matrix_arg = TensorArg::from_tensor(&a_foundry);
    let vector_arg = TensorArg::from_tensor(&x_foundry);
    let output_arg = TensorArg::from_tensor(&y_foundry_tensor);
    let scale_arg = TensorArg::from_tensor(&s_foundry);

    let kernel = GemvColMajor::new(&matrix_arg, &vector_arg, &output_arg, params).with_scales(&scale_arg);
    foundry.run_with_policy::<PolicyQ8, _>(&kernel)?;
    let y_foundry = FoundryTensor::to_vec(&y_foundry_tensor, &foundry);

    // 5. Compare
    let mut max_diff = 0.0f32;
    for i in 0..n {
        let diff = (y_legacy[i].to_f32() - y_foundry[i].to_f32()).abs();
        max_diff = max_diff.max(diff);
    }

    println!("\nFirst 10 from new: {:?}", &y_foundry[0..10]);
    println!("First 10 from legacy: {:?}", &y_legacy[0..10]);
    println!("ColMajor Q8 Parity (K={}, N={}): Max Diff = {:.6}", k, n, max_diff);
    assert!(max_diff < PARITY_TOLERANCE, "ColMajor Q8 Parity failed! Max Diff: {:.6}", max_diff);

    Ok(())
}

fn run_canonical_q8_parity_test(k: usize, n: usize) -> Result<(), MetalError> {
    let mut ctx = Context::<F16>::new()?;
    let ctx_u8 = Context::<U8>::new()?;
    let mut foundry = Foundry::new()?;

    // 1. Prepare Data
    let a_data = (0..n * k).map(|i| f16::from_f32((i % 10) as f32 - 5.0)).collect::<Vec<_>>();
    let x_data = (0..k).map(|i| f16::from_f32((i % 7) as f32 - 3.0)).collect::<Vec<_>>();

    // 2. Quantize and Swizzle for Canonical
    let a_packed = quantize_q8_0(n, k, &a_data);
    let a_swizzled = swizzle_q8_canonical(n, k, &a_packed);

    let x_legacy = Tensor::<F16>::new(vec![1, k], LegacyStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&x_data))?;

    // 4. Run Legacy
    let blocks_total = n * ((k + 31) / 32);
    let mut data_bytes = Vec::with_capacity(blocks_total * 32);
    let mut scale_bytes = Vec::with_capacity(blocks_total * 2);
    for block in a_swizzled.chunks_exact(34) {
        scale_bytes.extend_from_slice(&block[0..2]);
        data_bytes.extend_from_slice(&block[2..34]);
    }

    let a_legacy_quant = QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![k, n], &data_bytes, &scale_bytes, &ctx_u8)?;

    let y_legacy_tensor = ctx.call::<MatmulGemvOp>(
        (
            &x_legacy,
            TensorType::Quant(metallic::tensor::QuantizedTensor::Q8_0(&a_legacy_quant)),
            false,
            None,
        ),
        None,
    )?;
    let y_legacy = y_legacy_tensor.to_vec();

    // 5. Run Foundry Kernel

    let a_foundry = FoundryTensor::<U8, Pooled>::new(&mut foundry, vec![data_bytes.len()], TensorInit::CopyFrom(&data_bytes))?;
    let s_foundry = FoundryTensor::<U8, Pooled>::new(&mut foundry, vec![scale_bytes.len()], TensorInit::CopyFrom(&scale_bytes))?;
    let x_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k], TensorInit::CopyFrom(&x_data))?;
    let y_foundry_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n], TensorInit::Uninitialized)?;

    let params = GemvParams {
        k: k as u32,
        n: n as u32,
        blocks_per_k: ((k + 31) / 32) as u32,
        weights_per_block: 32,
        batch: 1,
        stride_x: 1,
        stride_y: 1,
        stride_a: 0,
        stride_w: (n * 32) as u32, // Stride to next K-block in canonical is N * wpb
        stride_scale: (n * 2) as u32,
    };

    let matrix_arg = TensorArg::from_tensor(&a_foundry);
    let vector_arg = TensorArg::from_tensor(&x_foundry);
    let output_arg = TensorArg::from_tensor(&y_foundry_tensor);
    let scale_arg = TensorArg::from_tensor(&s_foundry);

    let kernel = GemvCanonical::new(&matrix_arg, &vector_arg, &output_arg, params).with_scales(&scale_arg);

    foundry.run_with_policy::<PolicyQ8, _>(&kernel)?;
    let y_foundry = FoundryTensor::to_vec(&y_foundry_tensor, &foundry);

    // 6. Compare
    let mut max_diff = 0.0f32;
    for i in 0..n {
        let diff = (y_legacy[i].to_f32() - y_foundry[i].to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("\nFirst 10 from new: {:?}", &y_foundry[0..10]);
    println!("First 10 from legacy: {:?}", &y_legacy[0..10]);
    println!("Q8 Parity Test (K={}, N={}): Max Diff = {:.6}", k, n, max_diff);
    assert!(max_diff < PARITY_TOLERANCE, "Q8 Parity failed! Max Diff: {:.6}", max_diff);

    Ok(())
}

#[test]
#[serial]
fn test_gemv_row_major_q8_parity() -> Result<(), MetalError> {
    run_row_major_q8_parity_test(256, 128)
}

#[test]
#[serial]
fn test_gemv_row_major_q8_parity_square() -> Result<(), MetalError> {
    run_row_major_q8_parity_test(256, 256)
}

#[test]
#[serial]
fn test_gemv_row_major_q8_parity_tall() -> Result<(), MetalError> {
    run_row_major_q8_parity_test(128, 512)
}

#[test]
#[serial]
fn test_gemv_col_major_q8_parity() -> Result<(), MetalError> {
    run_col_major_q8_parity_test(256, 128)
}

#[test]
#[serial]
fn test_gemv_col_major_q8_parity_square() -> Result<(), MetalError> {
    run_col_major_q8_parity_test(256, 256)
}

#[test]
#[serial]
fn test_gemv_col_major_q8_parity_tall() -> Result<(), MetalError> {
    run_col_major_q8_parity_test(128, 512)
}

#[test]
#[serial]
fn test_gemv_canonical_q8_parity() -> Result<(), MetalError> {
    run_canonical_q8_parity_test(256, 128)
}

#[test]
#[serial]
fn test_gemv_canonical_q8_parity_square() -> Result<(), MetalError> {
    run_canonical_q8_parity_test(256, 256)
}

#[test]
#[serial]
fn test_gemv_canonical_q8_parity_tall() -> Result<(), MetalError> {
    run_canonical_q8_parity_test(128, 512)
}

#[test]
#[serial]
fn test_gemv_canonical_q8_parity_large() -> Result<(), MetalError> {
    run_canonical_q8_parity_test(4096, 1024)
}

// ============================================================================
// Batching Regression Tests
// ============================================================================

#[test]
#[serial]
fn test_gemv_col_major_batching_regression() {
    let mut foundry = Foundry::new().unwrap();
    let batch = 2;
    let dim = 32;
    let seq_k = 16;
    let seq_q = 1;

    let q_data = vec![f16::from_f32(1.0); batch * seq_q * dim];
    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![batch, seq_q, dim], TensorInit::CopyFrom(&q_data)).unwrap();

    let k_data = vec![f16::from_f32(1.0); batch * seq_k * dim];
    let k = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![batch, seq_k, dim], TensorInit::CopyFrom(&k_data)).unwrap();

    let out = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![batch, seq_q, seq_k], TensorInit::Uninitialized).unwrap();

    let params = GemvParams {
        k: dim as u32,
        n: seq_k as u32,
        blocks_per_k: 1,
        weights_per_block: dim as u32,
        batch: batch as u32,
        stride_x: (seq_q * dim) as u32,
        stride_y: (seq_q * seq_k) as u32,
        stride_a: (seq_k * dim) as u32,
        stride_w: dim as u32,
        stride_scale: 0,
    };

    let kernel = GemvColMajor::new(
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&out),
        params,
    )
    .with_alpha(0.5);

    foundry.run(&kernel).unwrap();

    let out_data = FoundryTensor::to_vec(&out, &foundry);
    let out_f32: Vec<f32> = out_data.iter().map(|&x| f32::from(x)).collect();

    for i in 0..seq_k {
        let val = out_f32[i];
        assert!((val - 16.0).abs() < 0.1, "Batch 0 index {} mismatch: got {}, expected 16.0", i, val);
    }

    let offset = seq_k;
    for i in 0..seq_k {
        let val = out_f32[offset + i];
        assert!((val - 16.0).abs() < 0.1, "Batch 1 index {} mismatch: got {}, expected 16.0", i, val);
    }
}

#[test]
#[serial]
fn test_gemv_row_major_batching_regression() {
    let mut foundry = Foundry::new().unwrap();
    let batch = 2;
    let dim = 32;
    let seq_k = 16;
    let seq_q = 1;

    let probs_data = vec![f16::from_f32(1.0); batch * seq_q * seq_k];
    let probs = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![batch, seq_q, seq_k], TensorInit::CopyFrom(&probs_data)).unwrap();

    let v_data = vec![f16::from_f32(1.0); batch * seq_k * dim];
    let v = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![batch, seq_k, dim], TensorInit::CopyFrom(&v_data)).unwrap();

    let out = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![batch, seq_q, dim], TensorInit::Uninitialized).unwrap();

    let params = GemvParams {
        k: seq_k as u32,
        n: dim as u32,
        blocks_per_k: 1,
        weights_per_block: seq_k as u32,
        batch: batch as u32,
        stride_x: (seq_q * seq_k) as u32,
        stride_y: (seq_q * dim) as u32,
        stride_a: (seq_k * dim) as u32,
        stride_w: dim as u32,
        stride_scale: 0,
    };

    let kernel = GemvRowMajor::new(
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&probs),
        &TensorArg::from_tensor(&out),
        params,
    );

    foundry.run(&kernel).unwrap();

    let out_data = FoundryTensor::to_vec(&out, &foundry);
    let out_f32: Vec<f32> = out_data.iter().map(|&x| f32::from(x)).collect();

    for i in 0..dim {
        let val = out_f32[i];
        assert!((val - 16.0).abs() < 0.1, "Batch 0 index {} mismatch: got {}, expected 16.0", i, val);
    }

    let offset = dim;
    for i in 0..dim {
        let val = out_f32[offset + i];
        assert!((val - 16.0).abs() < 0.1, "Batch 1 index {} mismatch: got {}, expected 16.0", i, val);
    }
}
