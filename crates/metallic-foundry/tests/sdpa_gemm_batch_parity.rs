// Test SDPA with batched queries (M>1) using GEMM path.
// Compares GPU GEMM-based SDPA against CPU reference for correctness.

use half::f16;
use metallic_foundry::{
    Foundry, MetalError, metals::{
        gemm::step::{GemmParams, GemmV2Args, gemm_dispatch_config, get_gemm_kernel}, mma::stages::TileConfig, softmax::{SoftmaxV2Args, get_softmax_v2_kernel}
    }, policy::f16::PolicyF16, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::{
        TensorArg, dispatch::{DispatchConfig, GridSize, ThreadgroupSize}
    }
};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Test SDPA GEMM path: Q @ K^T (GEMM), Softmax, Probs @ V (GEMM)
/// This tests the M>1 prefill path in SdpaMaterializedStep
#[test]
fn test_sdpa_gemm_batched() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;
    let mut rng = StdRng::seed_from_u64(123);

    // Config - small for verification
    let m = 4; // Number of query tokens (M>1 for GEMM path)
    let kv_seq_len = 8; // Key/Value sequence length
    let head_dim = 64;
    let n_heads = 2;
    let scale = 1.0 / (head_dim as f32).sqrt();

    println!("\n=== Testing SDPA GEMM Path ===");
    println!("M (query tokens): {}", m);
    println!("KV seq len: {}", kv_seq_len);
    println!("Head dim: {}", head_dim);
    println!("Num heads: {}", n_heads);

    // Generate test data
    // Q: [n_heads, m, head_dim] flattened
    // K: [n_heads, kv_seq_len, head_dim] flattened
    // V: [n_heads, kv_seq_len, head_dim] flattened
    let q_data: Vec<f16> = (0..n_heads * m * head_dim)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let k_data: Vec<f16> = (0..n_heads * kv_seq_len * head_dim)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let v_data: Vec<f16> = (0..n_heads * kv_seq_len * head_dim)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();

    // Upload to GPU
    let q_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * m * head_dim], TensorInit::CopyFrom(&q_data))?;
    let k_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * kv_seq_len * head_dim], TensorInit::CopyFrom(&k_data))?;
    let v_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * kv_seq_len * head_dim], TensorInit::CopyFrom(&v_data))?;

    // Output: [n_heads, m, head_dim]
    let output_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * m * head_dim], TensorInit::Uninitialized)?;

    // Scratch: [n_heads, m, kv_seq_len] for scores/probs
    let scores_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * m * kv_seq_len], TensorInit::Uninitialized)?;
    let probs_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * m * kv_seq_len], TensorInit::Uninitialized)?;

    // Scale tensor for softmax
    let scale_buf: Vec<f16> = vec![f16::from_f32(1.0)];
    let scale_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(&scale_buf))?;

    // Get base TensorArgs
    let q_arg = TensorArg::from_tensor(&q_tensor);
    let k_arg = TensorArg::from_tensor(&k_tensor);
    let v_arg = TensorArg::from_tensor(&v_tensor);
    let out_arg = TensorArg::from_tensor(&output_tensor);
    let scores_arg = TensorArg::from_tensor(&scores_tensor);
    let probs_arg = TensorArg::from_tensor(&probs_tensor);
    let scale_arg = TensorArg::from_tensor(&scale_tensor);

    let elem_size = std::mem::size_of::<f16>();
    let tile_config = TileConfig::default();

    // GEMM kernels
    // Q @ K^T: [m, head_dim] @ [head_dim, kv_seq_len] = [m, kv_seq_len]
    // K is [kv_seq_len, head_dim] row-major, so transpose_b=true
    let qk_gemm_kernel = get_gemm_kernel(
        std::sync::Arc::new(PolicyF16),
        std::sync::Arc::new(PolicyF16),
        false,
        true, // transpose_a=false, transpose_b=true
        tile_config,
        true,  // has_alpha_beta for scaling
        false, // has_bias
    );

    // Probs @ V: [m, kv_seq_len] @ [kv_seq_len, head_dim] = [m, head_dim]
    let av_gemm_kernel = get_gemm_kernel(
        std::sync::Arc::new(PolicyF16),
        std::sync::Arc::new(PolicyF16),
        false,
        false, // No transpose
        tile_config,
        false, // has_alpha_beta
        false, // has_bias
    );

    let softmax_kernel = get_softmax_v2_kernel();

    // Per-head strides (in bytes)
    let q_head_stride = m * head_dim * elem_size;
    let k_head_stride = kv_seq_len * head_dim * elem_size;
    let v_head_stride = kv_seq_len * head_dim * elem_size;
    let out_head_stride = m * head_dim * elem_size;
    let scratch_head_stride = m * kv_seq_len * elem_size;

    for h in 0..n_heads {
        let h_idx = h;

        // Slice tensors per head
        let mut q_h = q_arg.clone();
        q_h.offset += h_idx * q_head_stride;

        let mut k_h = k_arg.clone();
        k_h.offset += h_idx * k_head_stride;

        let mut v_h = v_arg.clone();
        v_h.offset += h_idx * v_head_stride;

        let mut out_h = out_arg.clone();
        out_h.offset += h_idx * out_head_stride;

        let mut scores_h = scores_arg.clone();
        scores_h.offset += h_idx * scratch_head_stride;

        let mut probs_h = probs_arg.clone();
        probs_h.offset += h_idx * scratch_head_stride;

        // 1. GEMM: Q @ K^T -> Scores [m, kv_seq_len]
        let qk_params = GemmParams::simple(
            m as i32,
            kv_seq_len as i32,
            head_dim as i32,
            false,
            true, // transpose_b
            tile_config,
        );
        let qk_dispatch = gemm_dispatch_config(&qk_params, tile_config);

        let qk_args = GemmV2Args {
            a: q_h.clone(),
            b: k_h,
            d: scores_h.clone(),
            c: scores_h.clone(),        // Dummy
            bias: scores_h.clone(),     // Dummy
            b_scales: scores_h.clone(), // Dummy
            weights_per_block: 32,
            params: qk_params,
            alpha: scale,
            beta: 0.0,
        };
        foundry.run(&qk_gemm_kernel.bind(qk_args, qk_dispatch))?;

        // 2. Softmax per row (each query has its own row)
        for row in 0..m {
            let row_offset = row * kv_seq_len * elem_size;
            let mut scores_row = scores_h.clone();
            scores_row.offset += row_offset;

            let mut probs_row = probs_h.clone();
            probs_row.offset += row_offset;

            let softmax_dispatch = DispatchConfig {
                grid: GridSize::d2(1, 1),
                group: ThreadgroupSize::d1(256),
            };

            let softmax_args = SoftmaxV2Args {
                input: scores_row,
                scale: scale_arg.clone(),
                output: probs_row,
                seq_k: kv_seq_len as u32,
                causal: 0, // Non-causal for simple test
                query_offset: 0,
            };
            foundry.run(&softmax_kernel.bind(softmax_args, softmax_dispatch))?;
        }

        // 3. GEMM: Probs @ V -> Output [m, head_dim]
        let av_params = GemmParams::simple(m as i32, head_dim as i32, kv_seq_len as i32, false, false, tile_config);
        let av_dispatch = gemm_dispatch_config(&av_params, tile_config);

        let av_args = GemmV2Args {
            a: probs_h,
            b: v_h,
            d: out_h.clone(),
            c: out_h.clone(),    // Dummy
            bias: out_h.clone(), // Dummy
            b_scales: out_h,     // Dummy
            weights_per_block: 32,
            params: av_params,
            alpha: 1.0,
            beta: 0.0,
        };
        foundry.run(&av_gemm_kernel.bind(av_args, av_dispatch))?;
    }

    // Read back result
    let gpu_out: Vec<f16> = output_tensor.to_vec(&foundry);
    let gpu_out_f32: Vec<f32> = gpu_out.iter().map(|x| x.to_f32()).collect();

    // CPU reference
    let q_f32: Vec<f32> = q_data.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k_data.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v_data.iter().map(|x| x.to_f32()).collect();

    let mut cpu_out = vec![0.0f32; n_heads * m * head_dim];

    for h in 0..n_heads {
        let q_off = h * m * head_dim;
        let k_off = h * kv_seq_len * head_dim;
        let v_off = h * kv_seq_len * head_dim;
        let out_off = h * m * head_dim;

        for q_idx in 0..m {
            // 1. Q[q_idx] @ K^T -> Scores[q_idx]
            // K is [kv_seq_len, head_dim] row-major
            let mut scores = vec![0.0f32; kv_seq_len];
            for s in 0..kv_seq_len {
                let mut sum = 0.0;
                for d in 0..head_dim {
                    // Q[q_idx, d] * K[s, d]
                    sum += q_f32[q_off + q_idx * head_dim + d] * k_f32[k_off + s * head_dim + d];
                }
                scores[s] = sum * scale;
            }

            // 2. Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let probs: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

            // 3. Probs @ V -> Output[q_idx]
            // V is [kv_seq_len, head_dim] row-major
            for d in 0..head_dim {
                let mut sum = 0.0;
                for s in 0..kv_seq_len {
                    sum += probs[s] * v_f32[v_off + s * head_dim + d];
                }
                cpu_out[out_off + q_idx * head_dim + d] = sum;
            }
        }
    }

    // Compare
    println!("\n=== Validation ===");
    let mut all_passed = true;
    for h in 0..n_heads {
        let off = h * m * head_dim;
        let gpu_slice = &gpu_out_f32[off..off + m * head_dim];
        let cpu_slice = &cpu_out[off..off + m * head_dim];

        let max_diff = gpu_slice
            .iter()
            .zip(cpu_slice.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0, f32::max);

        let avg_diff: f32 = gpu_slice.iter().zip(cpu_slice.iter()).map(|(g, c)| (g - c).abs()).sum::<f32>() / (m * head_dim) as f32;

        println!("Head {}: Max Diff = {:.6}, Avg Diff = {:.6}", h, max_diff, avg_diff);

        if max_diff >= 0.1 {
            println!("  FAIL: Max diff exceeds threshold!");
            // Print first few mismatches for debugging
            for i in 0..5.min(m * head_dim) {
                println!(
                    "    [{}] GPU: {:.4}, CPU: {:.4}, Diff: {:.4}",
                    i,
                    gpu_slice[i],
                    cpu_slice[i],
                    (gpu_slice[i] - cpu_slice[i]).abs()
                );
            }
            all_passed = false;
        }
    }

    assert!(all_passed, "SDPA GEMM parity test failed!");
    println!("\n✓ All heads passed!");

    Ok(())
}
