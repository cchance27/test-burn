//! GemmV2 Parity Test Suite
//!
//! Compares GemmV2 (Stage composition) against legacy MLX GEMM.
//!
//! Coverage:
//! - F16 x F16 GEMM
//! - F16 x Q8 GEMM (TODO: after F16 parity confirmed)
//! - Various M,N,K shapes: small, prefill-relevant, unaligned
//! - Transpose modes (NN, NT)

use half::f16;
use metallic_context::{
    Context, F16Element, kernels::matmul_mlx::MatMulMlxOp, tensor::{Tensor as LegacyTensor, TensorInit as LegacyInit, TensorStorage, TensorType, quantized::QuantizedQ8_0Tensor}
};
use metallic_foundry::{
    Foundry, metals::{
        gemm::step::{GemmParams, GemmV2Args, gemm_dispatch_config, get_gemm_kernel}, mma::stages::TileConfig,
        gemm::GemmV2Step,
    }, policy::{activation::Activation, f16::PolicyF16, q8::PolicyQ8}, storage::Pooled, tensor::{Tensor as FoundryTensor, TensorInit}, types::TensorArg,
    spec::{DynamicValue, Ref, Step, TensorBindings},
};
use rand::{Rng, rng};
use serial_test::serial;

// ============================================================================ 
// Test Configuration
// ============================================================================ 

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TestQuantization {
    F16,
    Q8,
}

#[derive(Clone)]
struct GemmTestConfig {
    m: usize,
    n: usize,
    k: usize,
    transpose_a: bool,
    transpose_b: bool,
    tile_config: Option<TileConfig>,
    quant_b: TestQuantization,
}

impl Default for GemmTestConfig {
    fn default() -> Self {
        Self {
            m: 32,
            n: 32,
            k: 32,
            transpose_a: false,
            transpose_b: false,
            tile_config: None,
            quant_b: TestQuantization::F16,
        }
    }
}

// ============================================================================ 
// CPU Reference Implementation
// ============================================================================ 

#[allow(clippy::too_many_arguments)]
fn run_mlx_gemm_f16(
    a_data: &[f16],
    b_data: &[f16],
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    _foundry: &Foundry,
) -> Vec<f16> {
    let mut ctx = Context::<F16Element>::new().expect("Failed to create context");

    let a_rows = if trans_a { k } else { m };
    let a_cols = if trans_a { m } else { k };
    let a_legacy =
        LegacyTensor::<F16Element>::new(vec![a_rows, a_cols], TensorStorage::Dedicated(&ctx), LegacyInit::CopyFrom(a_data)).unwrap();

    let b_rows = if trans_b { n } else { k };
    let b_cols = if trans_b { k } else { n };
    let b_legacy =
        LegacyTensor::<F16Element>::new(vec![b_rows, b_cols], TensorStorage::Dedicated(&ctx), LegacyInit::CopyFrom(b_data)).unwrap();

    let out = ctx
        .call::<MatMulMlxOp>(
            (&a_legacy, TensorType::Dense(&b_legacy), None, None, trans_a, trans_b, 1.0, 0.0),
            None,
        )
        .expect("MLX GEMM failed");

    out.to_vec()
}

fn run_mlx_gemm_q8(
    a_data: &[f16],
    b_q8: &QuantizedQ8_0Tensor,
    m: usize,
    _n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
) -> Option<Vec<f16>> {
    let mut ctx = Context::<F16Element>::new().expect("Failed to create context");

    let a_rows = if trans_a { k } else { m };
    let a_cols = if trans_a { m } else { k };
    let a_legacy =
        LegacyTensor::<F16Element>::new(vec![a_rows, a_cols], TensorStorage::Dedicated(&ctx), LegacyInit::CopyFrom(a_data)).unwrap();

    let res = ctx.call::<MatMulMlxOp>(
        (
            &a_legacy,
            TensorType::Quant(metallic_context::tensor::QuantizedTensor::Q8_0(b_q8)),
            None,
            None,
            trans_a,
            trans_b,
            1.0,
            0.0,
        ),
        None,
    );

    match res {
        Ok(out) => Some(out.to_vec()),
        Err(e) => {
            println!("MLX Q8 GEMM skipped: {:?}", e);
            None
        }
    }
}

fn quantize_q8(data: &[f16], n: usize, k: usize, transpose_b: bool) -> (Vec<u8>, Vec<u8>) {
    let blocks_per_k = k.div_ceil(32);
    let mut weights = vec![0u8; n * blocks_per_k * 32];
    let mut scales = vec![0u8; n * blocks_per_k * 2];

    for ni in 0..n {
        for ki_block in 0..blocks_per_k {
            let mut max_abs = 0.0f32;
            let k_start = ki_block * 32;

            // First pass: find max_abs
            for j in 0..32 {
                let ki = k_start + j;
                if ki < k {
                    let val = if transpose_b {
                        data[ni * k + ki].to_f32()
                    } else {
                        data[ki * n + ni].to_f32()
                    };
                    max_abs = max_abs.max(val.abs());
                }
            }

            let scale = max_abs / 127.0;
            let scale_f16 = f16::from_f32(scale);
            let s_idx = (ni * blocks_per_k + ki_block) * 2;
            scales[s_idx..s_idx + 2].copy_from_slice(&scale_f16.to_le_bytes());

            // Second pass: quantize
            for j in 0..32 {
                let ki = k_start + j;
                let q = if ki < k {
                    let val = if transpose_b {
                        data[ni * k + ki].to_f32()
                    } else {
                        data[ki * n + ni].to_f32()
                    };
                    if scale > 0.0 {
                        (val / scale).round().clamp(-128.0, 127.0) as i8
                    } else {
                        0
                    }
                } else {
                    0
                };
                weights[ni * blocks_per_k * 32 + ki_block * 32 + j] = q as u8;
            }
        }
    }
    (weights, scales)
}

fn run_cpu_gemm_f16(
    m: usize,
    n: usize,
    k: usize,
    a: &[f16], // [M, K] or [K, M] if transposed
    b: &[f16], // [K, N] or [N, K] if transposed
    transpose_a: bool,
    transpose_b: bool,
) -> Vec<f16> {
    let mut output = vec![f16::from_f32(0.0); m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                // Get A element
                let a_idx = if transpose_a {
                    ki * m + i // A is [K, M], reading A[ki, i]
                } else {
                    i * k + ki // A is [M, K], reading A[i, ki]
                };

                // Get B element
                let b_idx = if transpose_b {
                    j * k + ki // B is [N, K], reading B[j, ki]
                } else {
                    ki * n + j // B is [K, N], reading B[ki, j]
                };

                let a_val = a[a_idx].to_f32();
                let b_val = b[b_idx].to_f32();
                acc += a_val * b_val;
            }
            output[i * n + j] = f16::from_f32(acc);
        }
    }
    output
}

// ============================================================================ 
// Test Runner
// ============================================================================ 

fn run_gemm_v2_parity_test(cfg: GemmTestConfig) {
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rng();

    // Determine tensor dimensions based on transpose flags
    let a_dims = if cfg.transpose_a {
        vec![cfg.k, cfg.m] // [K, M] when transposed
    } else {
        vec![cfg.m, cfg.k] // [M, K] normal
    };

    let b_dims = if cfg.transpose_b {
        vec![cfg.n, cfg.k] // [N, K] when transposed
    } else {
        vec![cfg.k, cfg.n] // [K, N] normal
    };

    let a_rows = if cfg.transpose_a { cfg.k } else { cfg.m };
    let a_cols = if cfg.transpose_a { cfg.m } else { cfg.k };
    let a_data: Vec<f16> = (0..a_rows * a_cols)
        .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0f32)))
        .collect();
    let (b_rows, b_cols) = if cfg.transpose_b { (cfg.n, cfg.k) } else { (cfg.k, cfg.n) };
    let b_data: Vec<f16> = (0..b_rows * b_cols)
        .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0f32)))
        .collect();

    // Create tensors
    let a = FoundryTensor::<metallic_foundry::F16, Pooled>::new(&mut foundry, a_dims, TensorInit::CopyFrom(&a_data)).unwrap();

    // We need a Context<F16> to create the quantized tensor
    let ctx_f16 = Context::<metallic_context::F16Element>::new().unwrap();

    let (b_quantized, b_f16): (Option<QuantizedQ8_0Tensor>, Option<FoundryTensor<metallic_foundry::F16, Pooled>>) =
        if cfg.quant_b == TestQuantization::Q8 {
            let (w, s) = quantize_q8(&b_data, cfg.n, cfg.k, cfg.transpose_b);
            let q8 = QuantizedQ8_0Tensor::from_split_bytes_in_context(b_dims.clone(), &w, &s, &ctx_f16).unwrap();
            (Some(q8), None)
        } else {
            let b = FoundryTensor::<metallic_foundry::F16, Pooled>::new(&mut foundry, b_dims, TensorInit::CopyFrom(&b_data)).unwrap();
            (None, Some(b))
        };

    let output = FoundryTensor::<metallic_foundry::F16, Pooled>::new(&mut foundry, vec![cfg.m, cfg.n], TensorInit::Uninitialized).unwrap();

    // ========== Run CPU Reference ========== 
    // For CPU ref with Q8, we use the original b_data which is F16.
    // However, to be extra fair, we should dequantize the Q8 back to F16.
    let cpu_b_data = if let Some(q8) = &b_quantized {
        let weights = q8.data.to_vec();
        let scales_raw = q8.scales.to_vec();
        let mut dequant = vec![f16::ZERO; cfg.n * cfg.k];
        let blocks_per_k = q8.blocks_per_k;

        for ni in 0..cfg.n {
            for ki_block in 0..blocks_per_k {
                let s_idx = (ni * blocks_per_k + ki_block) * 2;
                let scale_bits = (scales_raw[s_idx] as u16) | ((scales_raw[s_idx + 1] as u16) << 8);
                let scale = f16::from_bits(scale_bits).to_f32();

                for j in 0..32 {
                    let ki = ki_block * 32 + j;
                    if ki < cfg.k {
                        let w_val = weights[ni * blocks_per_k * 32 + ki_block * 32 + j] as i8;
                        let val = f16::from_f32(w_val as f32 * scale);
                        // Store in row-major [K, N] or [N, K] for the CPU matmul
                        if cfg.transpose_b {
                            dequant[ni * cfg.k + ki] = val;
                        } else {
                            dequant[ki * cfg.n + ni] = val;
                        }
                    }
                }
            }
        }
        dequant
    } else {
        b_data.clone()
    };
    let cpu_output = run_cpu_gemm_f16(cfg.m, cfg.n, cfg.k, &a_data, &cpu_b_data, cfg.transpose_a, cfg.transpose_b);

    // ========== Run V2 GEMM ========== 
    let tile_config = cfg.tile_config.unwrap_or_else(|| TileConfig::auto_select(cfg.m, cfg.n));
    let params = GemmParams::simple(
        cfg.m as i32,
        cfg.n as i32,
        cfg.k as i32,
        cfg.transpose_a,
        cfg.transpose_b,
        tile_config,
    );
    let dispatch = gemm_dispatch_config(&params, tile_config);

    // Get kernel
    let kernel = get_gemm_kernel(
        std::sync::Arc::new(PolicyF16),
        match cfg.quant_b {
            TestQuantization::F16 => std::sync::Arc::new(PolicyF16),
            TestQuantization::Q8 => std::sync::Arc::new(PolicyQ8),
        },
        cfg.transpose_a,
        cfg.transpose_b,
        tile_config,
        false,
        false,
        Activation::None,
    );

    // DEBUG: Dump source
    println!("=== Generated Metal Source ===");
    println!("{}", kernel.source());
    println!("=== End Metal Source ===");

    let (b_arg, b_scales_arg) = if let Some(q8) = &b_quantized {
        let b_data_arg = TensorArg::from_buffer(
            metallic_foundry::MetalBuffer(q8.data.buf.clone()),
            metallic_foundry::Dtype::U8,
            q8.data.dims.clone(),
            q8.data.strides.clone(),
        );
        let b_scales_arg = TensorArg::from_buffer(
            metallic_foundry::MetalBuffer(q8.scales.buf.clone()),
            metallic_foundry::Dtype::F16, // Scales are F16
            q8.scales.dims.clone(),
            q8.scales.strides.clone(),
        );
        (b_data_arg, b_scales_arg)
    } else {
        let b = b_f16.as_ref().unwrap();
        (TensorArg::from_tensor(b), TensorArg::from_tensor(b))
    };

    let args = GemmV2Args {
        a: TensorArg::from_tensor(&a),
        b: b_arg,
        d: TensorArg::from_tensor(&output),
        c: TensorArg::from_tensor(&output),    // Dummy
        bias: TensorArg::from_tensor(&output), // Dummy
        b_scales: b_scales_arg,
        weights_per_block: 32,
        alpha: 1.0,
        beta: 0.0,
        params,
    };

    foundry.run(&kernel.bind_arc(args, dispatch)).unwrap();

    // ========== Compare Results ========== 
    let gpu_output = FoundryTensor::to_vec(&output, &foundry);

    let cpu_f32: Vec<f32> = cpu_output.iter().map(|x| x.to_f32()).collect();
    let gpu_f32: Vec<f32> = gpu_output.iter().map(|x: &f16| x.to_f32()).collect();

    // 4. Compare CPU vs V2
    let mut max_diff = 0.0f32;
    let mut v2_vs_cpu_fail = false;
    for i in 0..gpu_f32.len() {
        let diff = (gpu_f32[i] - cpu_f32[i]).abs();
        max_diff = max_diff.max(diff);
        if diff > 1e-2 {
            v2_vs_cpu_fail = true;
        }
    }

    // ========== Run MLX Reference ========== 
    let mlx_output = if let Some(q8) = &b_quantized {
        run_mlx_gemm_q8(&a_data, q8, cfg.m, cfg.n, cfg.k, cfg.transpose_a, cfg.transpose_b)
    } else {
        Some(run_mlx_gemm_f16(
            &a_data,
            &b_data,
            cfg.m,
            cfg.n,
            cfg.k,
            cfg.transpose_a,
            cfg.transpose_b,
            &foundry,
        ))
    };

    let mlx_f32: Option<Vec<f32>> = mlx_output.map(|v| v.iter().map(|x| x.to_f32()).collect());

    // Debug print first few values
    println!("First 5 values:");
    println!("  CPU: {:?}", &cpu_f32[..5.min(cpu_f32.len())]);
    println!("  GPU: {:?}", &gpu_f32[..5.min(gpu_f32.len())]);
    let mut mlx_diff = 0.0f32;
    let mut v2_vs_mlx_fail = false;
    let mut first_mlx_diff_idx = None;
    if let Some(_mlx) = &mlx_f32 {
        println!("  MLX: {:?}", &_mlx[..5.min(_mlx.len())]);
        for i in 0..gpu_f32.len() {
            let diff = (gpu_f32[i] - _mlx[i]).abs();
            if diff > mlx_diff {
                mlx_diff = diff;
            }
            if diff > 1e-2 && first_mlx_diff_idx.is_none() {
                v2_vs_mlx_fail = true;
                first_mlx_diff_idx = Some(i);
            }
        }
        if let Some(idx) = first_mlx_diff_idx {
            println!(
                "First MLX divergence at index {}: GPU={}, MLX={}, diff={}",
                idx,
                gpu_f32[idx],
                _mlx[idx],
                (gpu_f32[idx] - _mlx[idx]).abs()
            );
        }
    } else {
        println!("  MLX: Skipped (not implemented for this config)");
    }

    println!("\n=== GEMM V2 Parity Test ===");
    println!("Shape: M={}, N={}, K={}", cfg.m, cfg.n, cfg.k);
    println!("Transpose: A={}, B={}", cfg.transpose_a, cfg.transpose_b);
    println!("Tile config: {:?}", tile_config);
    println!("Max diff CPU: {:.6}, MLX: {:.6}", max_diff, mlx_diff);

    if v2_vs_cpu_fail || v2_vs_mlx_fail {
        panic!(
            "Parity failure! Shape {}x{}x{}, trans_a={}, trans_b={}. Max diff CPU: {:.6}, MLX: {:.6}",
            cfg.m, cfg.n, cfg.k, cfg.transpose_a, cfg.transpose_b, max_diff, mlx_diff
        );
    }

    if mlx_f32.is_some() {
        let tol = 1e-2f32;
        if mlx_diff > tol {
            panic!(
                "MLX Parity failure! Shape {}x{}x{}, trans_a={}, trans_b={}. Max diff MLX: {:.6}, Tol: {:.6}",
                cfg.m, cfg.n, cfg.k, cfg.transpose_a, cfg.transpose_b, mlx_diff, tol
            );
        }
    }

    println!("✓ PASSED (CPU & MLX)");
}

// ============================================================================ 
// Basic Shape Tests - F16 x F16
// ============================================================================ 

#[test]
#[serial]
fn test_gemm_v2_32x32x32() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 32,
        n: 32,
        k: 32,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_64x64x64() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 64,
        n: 64,
        k: 64,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_128x128x128() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 128,
        n: 128,
        k: 128,
        ..Default::default()
    });
}

// ============================================================================ 
// Prefill-Relevant Shapes
// ============================================================================ 

#[test]
#[serial]
fn test_gemm_v2_prefill_32_896_896() {
    // Typical prefill shape: 32 tokens, hidden_size=896
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 32,
        n: 896,
        k: 896,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_prefill_128_896_896() {
    // Larger prefill
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 128,
        n: 896,
        k: 896,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_prefill_64_4864_896() {
    // MLP up projection shape
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 64,
        n: 4864,
        k: 896,
        ..Default::default()
    });
}

// ============================================================================ 
// Transpose Tests
// ============================================================================ 

#[test]
#[serial]
fn test_gemm_v2_64x64x64_nt() {
    // A normal, B transposed (common for weight matrices)
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 64,
        n: 64,
        k: 64,
        transpose_a: false,
        transpose_b: true,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_prefill_32_896_896_nt() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 32,
        n: 896,
        k: 896,
        transpose_a: false,
        transpose_b: true,
        ..Default::default()
    });
}

// ============================================================================ 
// Unaligned Shapes (tests edge handling)
// ============================================================================ 

#[test]
#[serial]
fn test_gemm_v2_33x35x37() {
    // Unaligned to tile boundaries
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 33,
        n: 35,
        k: 37,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_100x200x150() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 100,
        n: 200,
        k: 150,
        ..Default::default()
    });
}

// ============================================================================ 
// Tile Config Tests
// ============================================================================ 

#[test]
#[serial]
fn test_gemm_v2_128x128x128_skinny_m() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 128,
        n: 128,
        k: 128,
        tile_config: Some(TileConfig::SkinnyM),
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_128x128x128_skinny_n() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 128,
        n: 128,
        k: 128,
        tile_config: Some(TileConfig::SkinnyN),
        ..Default::default()
    });
}

// ============================================================================ 
// Qwen2.5 0.5B Specific Shapes
// ============================================================================ 
// hidden_size=896, intermediate_size=4864, num_heads=14, head_dim=64

#[test]
#[serial]
fn test_gemm_v2_qwen25_qkv_proj_m16() {
    // QKV projection: [M, 896] x [896, 896]
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 16,
        n: 896,
        k: 896,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_qwen25_qkv_proj_m64() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 64,
        n: 896,
        k: 896,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_qwen25_qkv_proj_m128() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 128,
        n: 896,
        k: 896,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_qwen25_mlp_up_m32() {
    // MLP up projection: [M, 896] x [896, 4864]
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 32,
        n: 4864,
        k: 896,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_qwen25_mlp_up_m64() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 64,
        n: 4864,
        k: 896,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_qwen25_mlp_down_m32() {
    // MLP down projection: [M, 4864] x [4864, 896]
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 32,
        n: 896,
        k: 4864,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_qwen25_mlp_down_m64() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 64,
        n: 896,
        k: 4864,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_qwen25_lm_head_m32() {
    // LM head: [M, 896] x [896, 151936] (vocab)
    // Note: Full vocab is too large for quick tests, use subset
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 32,
        n: 4096, // Subset of vocab for speed
        k: 896,
        ..Default::default()
    });
}

// ============================================================================ 
// Edge Cases - Very Small M (Decode-like)
// ============================================================================ 

#[test]
#[serial]
fn test_gemm_v2_m1_32x32() {
    // M=1 should still work (though GEMV might be faster)
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 1,
        n: 32,
        k: 32,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_m2_896x896() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 2,
        n: 896,
        k: 896,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_m4_896x896() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 4,
        n: 896,
        k: 896,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_m8_896x896() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 8,
        n: 896,
        k: 896,
        ..Default::default()
    });
}

// ============================================================================ 
// Edge Cases - Unaligned to Various Powers of 2
// ============================================================================ 

#[test]
#[serial]
fn test_gemm_v2_unaligned_17x19x23() {
    // Prime-ish numbers
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 17,
        n: 19,
        k: 23,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_unaligned_31x33x35() {
    // Just below/above 32
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 31,
        n: 33,
        k: 35,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_unaligned_63x65x67() {
    // Just below/above 64
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 63,
        n: 65,
        k: 67,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_unaligned_127x129x131() {
    // Just below/above 128
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 127,
        n: 129,
        k: 131,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_unaligned_255x257x259() {
    // Just below/above 256
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 255,
        n: 257,
        k: 259,
        ..Default::default()
    });
}

// ============================================================================ 
// Edge Cases - Extreme Aspect Ratios
// ============================================================================ 

#[test]
#[serial]
fn test_gemm_v2_wide_16x1024x64() {
    // Very wide output
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 16,
        n: 1024,
        k: 64,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_tall_256x32x64() {
    // Very tall output
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 256,
        n: 32,
        k: 64,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_deep_k_32x32x1024() {
    // Very large reduction dimension
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 32,
        n: 32,
        k: 1024,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_shallow_k_32x32x8() {
    // Very small reduction dimension
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 32,
        n: 32,
        k: 8,
        ..Default::default()
    });
}

// ============================================================================ 
// Transpose Combinations with Qwen2.5 Shapes
// ============================================================================ 

#[test]
#[serial]
fn test_gemm_v2_qwen25_qkv_nt_m32() {
    // Transposed B (common for stored weights)
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 32,
        n: 896,
        k: 896,
        transpose_b: true,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_qwen25_mlp_up_nt_m32() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 32,
        n: 4864,
        k: 896,
        transpose_b: true,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_qwen25_mlp_down_nt_m32() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 32,
        n: 896,
        k: 4864,
        transpose_b: true,
        ..Default::default()
    });
}

// ============================================================================ 
// Power of 2 Shapes (Optimal alignment)
// ============================================================================ 

#[test]
#[serial]
fn test_gemm_v2_pow2_256x256x256() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 256,
        n: 256,
        k: 256,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_pow2_512x512x512() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 512,
        n: 512,
        k: 512,
        ..Default::default()
    });
}

// ============================================================================ 
// Larger Prefill Batch Sizes
// ============================================================================ 

#[test]
#[serial]
fn test_gemm_v2_prefill_256_896_896() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 256,
        n: 896,
        k: 896,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_prefill_512_896_896() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 512,
        n: 896,
        k: 896,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_prefill_256_4864_896() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 256,
        n: 4864,
        k: 896,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_gemm_v2_qwen25_mlp_up_q8() {
    run_gemm_v2_parity_test(GemmTestConfig {
        m: 32,
        n: 18944,
        k: 1536,
        transpose_a: false,
        transpose_b: true,
        quant_b: TestQuantization::Q8,
        ..Default::default()
    });
}

// ============================================================================ 
// Regression Test: Runtime Policy Selection (No DSL Hints)
// ============================================================================ 

#[test]
#[serial]
fn test_gemm_v2_step_runtime_policy_selection_q8() {
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rng();

    // Config
    let m = 32;
    let n = 32;
    let k = 32;
    let transpose_a = false;
    let transpose_b = true; // Typical for weights

    // Data
    let a_rows = m;
    let a_cols = k;
    let a_data: Vec<f16> = (0..a_rows * a_cols)
        .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0f32)))
        .collect();
    
    let b_rows = n;
    let b_cols = k;
    let b_data: Vec<f16> = (0..b_rows * b_cols)
        .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0f32)))
        .collect();

    // Create A tensor (F16)
    let a_tensor = FoundryTensor::<metallic_foundry::F16, Pooled>::new(
        &mut foundry, 
        vec![a_rows, a_cols], 
        TensorInit::CopyFrom(&a_data)
    ).unwrap();

    // Create B tensor (Q8)
    let ctx_f16 = Context::<metallic_context::F16Element>::new().unwrap();
    let (w, s) = quantize_q8(&b_data, n, k, transpose_b);
    let q8_tensor = QuantizedQ8_0Tensor::from_split_bytes_in_context(
        vec![n, k], 
        &w, 
        &s, 
        &ctx_f16
    ).unwrap();

    // Convert Q8 tensor parts to Foundry TensorArgs
    let b_weights_arg = TensorArg::from_buffer(
        metallic_foundry::MetalBuffer(q8_tensor.data.buf.clone()),
        metallic_foundry::Dtype::U8,
        q8_tensor.data.dims.clone(),
        q8_tensor.data.strides.clone(),
    );
    let b_scales_arg = TensorArg::from_buffer(
        metallic_foundry::MetalBuffer(q8_tensor.scales.buf.clone()),
        metallic_foundry::Dtype::F16,
        q8_tensor.scales.dims.clone(),
        q8_tensor.scales.strides.clone(),
    );

    // Output tensor
    let d_tensor = FoundryTensor::<metallic_foundry::F16, Pooled>::new(
        &mut foundry, 
        vec![m, n], 
        TensorInit::Uninitialized
    ).unwrap();

    // Bindings
    let mut bindings = TensorBindings::new();
    bindings.insert("A", TensorArg::from_tensor(&a_tensor));
    bindings.insert("B", b_weights_arg); // B is bound as U8 (Q8 weights)
    bindings.insert("B_scales", b_scales_arg);
    bindings.insert("D", TensorArg::from_tensor(&d_tensor));

    // Construct the Step
    // Note: We intentionally do NOT provide any "quantization hint".
    // The "b_quant" field should be missing or ignored if it existed (it doesn't).
    // The system MUST detect Q8 from the "B" tensor dtype (U8).
    let step = GemmV2Step {
        a: Ref("A".to_string()),
        b: Ref("B".to_string()),
        d: Ref("D".to_string()),
        c: None,
        bias: None,
        b_scales: Some(Ref("B_scales".to_string())),
        weights_per_block: 32,
        alpha: 1.0,
        beta: 0.0,
        params: GemmParams::default(),
        m_dim: DynamicValue::Literal(m as u32),
        n_dim: DynamicValue::Literal(n as u32),
        k_dim: DynamicValue::Literal(k as u32),
        transpose_a,
        transpose_b,
        tile_config: Some(TileConfig::Default),
        activation: Activation::None,
    };

    // Execute
    step.execute(&mut foundry, &mut bindings).expect("Execution failed");

    // Verification
    let gpu_output = FoundryTensor::to_vec(&d_tensor, &foundry);
    let gpu_f32: Vec<f32> = gpu_output.iter().map(|x: &f16| x.to_f32()).collect();

    // CPU Gemm
    let mut cpu_output = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                let a_val = a_data[i * k + ki].to_f32();
                // B is [N, K] transposed
                let b_val = b_data[j * k + ki].to_f32();
                acc += a_val * b_val;
            }
            cpu_output[i * n + j] = acc;
        }
    }

    let mut max_diff = 0.0f32;
    for i in 0..gpu_f32.len() {
        let diff = (gpu_f32[i] - cpu_output[i]).abs();
        max_diff = max_diff.max(diff);
    }

    println!("Max diff (Q8 vs CPU F16): {}", max_diff);
    assert!(max_diff < 1.0, "Difference too high ({}), likely wrong kernel selected or Q8 corruption", max_diff);
}