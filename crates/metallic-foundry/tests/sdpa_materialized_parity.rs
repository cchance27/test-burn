// Test SdpaMaterialized parity using FoundryTensor for correct TensorArg handling.
// This directly tests the Gemv+Softmax+Gemv sequence that SdpaMaterializedStep uses.

use std::sync::Arc;

use half::f16;
use metallic_foundry::{
    Foundry, MetalError, compound::Layout, metals::{
        gemv::step::{GemvStrategy, GemvV2Args, get_gemv_v2_kernel, warp_dispatch_config}, softmax::step::{SoftmaxV2Args, get_softmax_v2_kernel}
    }, policy::{activation::Activation, f16::PolicyF16}, storage::Pooled, tensor::{Tensor as FoundryTensor, TensorInit, dtypes::F16}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[test]
fn test_sdpa_materialized_parity() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;
    let mut rng = StdRng::seed_from_u64(42);

    // Config
    let seq_len = 8;
    let head_dim = 64;
    let n_heads = 2;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Generate data (flat f16 for simplicity matching kernel expectations)
    // Q: [n_heads * head_dim] - flattened
    // K: [n_heads * seq_len * head_dim] - flattened
    // V: [n_heads * seq_len * head_dim] - flattened
    let q_data: Vec<f16> = (0..n_heads * head_dim)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let k_data: Vec<f16> = (0..n_heads * seq_len * head_dim)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let v_data: Vec<f16> = (0..n_heads * seq_len * head_dim)
        .map(|_| f16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();

    // Upload to Foundry (1D flat tensors)
    let q_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * head_dim], TensorInit::CopyFrom(q_data.as_slice()))?;
    let k_tensor = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads * seq_len * head_dim],
        TensorInit::CopyFrom(k_data.as_slice()),
    )?;
    let v_tensor = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads * seq_len * head_dim],
        TensorInit::CopyFrom(v_data.as_slice()),
    )?;
    let output_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * head_dim], TensorInit::Uninitialized)?;
    let scores_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * seq_len], TensorInit::Uninitialized)?;
    let probs_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads * seq_len], TensorInit::Uninitialized)?;

    // Scale tensor (1.0 because Gemv applies scale via alpha)
    let scale_buf: Vec<f16> = vec![f16::from_f32(1.0)];
    let scale_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(scale_buf.as_slice()))?;

    // Kernels
    let qk_kernel = get_gemv_v2_kernel(Arc::new(PolicyF16), Layout::RowMajor, GemvStrategy::Vectorized, Activation::None);
    let qk_dispatch = warp_dispatch_config(seq_len as u32);

    let softmax_kernel = get_softmax_v2_kernel();
    let softmax_dispatch = DispatchConfig {
        grid: GridSize::d2(1, 1),
        group: ThreadgroupSize::d1(256),
    };

    let av_kernel = get_gemv_v2_kernel(Arc::new(PolicyF16), Layout::ColMajor, GemvStrategy::Vectorized, Activation::None);
    let av_dispatch = warp_dispatch_config(head_dim as u32);

    // Get base TensorArgs
    let q_arg = TensorArg::from_tensor(&q_tensor);
    let k_arg = TensorArg::from_tensor(&k_tensor);
    let v_arg = TensorArg::from_tensor(&v_tensor);
    let out_arg = TensorArg::from_tensor(&output_tensor);
    let scores_arg = TensorArg::from_tensor(&scores_tensor);
    let probs_arg = TensorArg::from_tensor(&probs_tensor);
    let scale_arg = TensorArg::from_tensor(&scale_tensor);

    let elem_size = std::mem::size_of::<f16>();

    // Strides (in bytes)
    let q_head_stride = head_dim * elem_size;
    let k_head_stride = seq_len * head_dim * elem_size;
    let v_head_stride = seq_len * head_dim * elem_size;
    let out_head_stride = head_dim * elem_size;
    let scratch_head_stride = seq_len * elem_size;

    for h in 0..n_heads {
        // Offset each tensor to simulate per-head slicing
        let mut q_h = q_arg.clone();
        q_h.offset += h * q_head_stride;

        let mut k_h = k_arg.clone();
        k_h.offset += h * k_head_stride;

        let mut v_h = v_arg.clone();
        v_h.offset += h * v_head_stride;

        let mut out_h = out_arg.clone();
        out_h.offset += h * out_head_stride;

        let mut scores_h = scores_arg.clone();
        scores_h.offset += h * scratch_head_stride;

        let mut probs_h = probs_arg.clone();
        probs_h.offset += h * scratch_head_stride;

        // 1. QK^T: Q @ K^T -> Scores (RowMajor GEMV)
        // K is [seq, head_dim]. Q is [head_dim]. Result is [seq].
        // RowMajor: weights[row * K + k]. row=seq, k=head_dim.
        let qk_args = GemvV2Args {
            weights: k_h.clone(),
            scale_bytes: k_h.clone(), // dummy
            input: q_h,
            output: scores_h.clone(),
            bias: scores_h.clone(), // dummy
            has_bias: 0,
            k_dim: head_dim as u32,
            n_dim: seq_len as u32,
            weights_per_block: 32,
            alpha: scale,
            residual: scores_h.clone(),
            has_residual: 0,
            beta: 0.0,
        };
        foundry.run(&qk_kernel.clone().bind_arc(qk_args, qk_dispatch))?;

        // 2. Softmax: Scores -> Probs
        let softmax_args = SoftmaxV2Args {
            input: scores_h,
            scale: scale_arg.clone(),
            output: probs_h.clone(),
            seq_k: seq_len as u32,
            causal: 0, // non-causal for simplicity
            query_offset: 0,
        };
        foundry.run(&softmax_kernel.clone().bind_arc(softmax_args, softmax_dispatch))?;

        // 3. Probs @ V -> Output (ColMajor GEMV)
        // V is [seq, head_dim]. Probs is [seq]. Result is [head_dim].
        // ColMajor: weights[k * N + n]. k=seq, n=head_dim.
        let av_args = GemvV2Args {
            weights: v_h.clone(),
            scale_bytes: v_h.clone(), // dummy
            input: probs_h,
            output: out_h.clone(),
            bias: out_h.clone(), // dummy
            has_bias: 0,
            k_dim: seq_len as u32,
            n_dim: head_dim as u32,
            weights_per_block: 32,
            alpha: 1.0,
            residual: out_h.clone(),
            has_residual: 0,
            beta: 0.0,
        };
        foundry.run(&av_kernel.clone().bind_arc(av_args, av_dispatch))?;
    }

    // Read back result
    let gpu_out: Vec<f16> = output_tensor.to_vec(&foundry);
    let gpu_out_f32: Vec<f32> = gpu_out.iter().map(|x| x.to_f32()).collect();

    // CPU Reference
    let q_f32: Vec<f32> = q_data.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k_data.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v_data.iter().map(|x| x.to_f32()).collect();

    let mut cpu_out = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        // Per-head offsets
        let q_off = h * head_dim;
        let k_off = h * seq_len * head_dim;
        let v_off = h * seq_len * head_dim;
        let out_off = h * head_dim;

        // 1. QK^T (row-major: K[s * head_dim + d])
        let mut scores = vec![0.0f32; seq_len];
        for s in 0..seq_len {
            let mut sum = 0.0;
            for d in 0..head_dim {
                sum += k_f32[k_off + s * head_dim + d] * q_f32[q_off + d];
            }
            scores[s] = sum * scale;
        }

        // 2. Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let probs: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        // 3. Probs @ V (col-major for kernel: V[s * head_dim + d])
        // CPU ref: output[d] = sum_s probs[s] * V[s, d]
        for d in 0..head_dim {
            let mut sum = 0.0;
            for s in 0..seq_len {
                sum += probs[s] * v_f32[v_off + s * head_dim + d];
            }
            cpu_out[out_off + d] = sum;
        }
    }

    // Compare
    println!("Validating results...");
    for h in 0..n_heads {
        let off = h * head_dim;
        let gpu_slice = &gpu_out_f32[off..off + head_dim];
        let cpu_slice = &cpu_out[off..off + head_dim];
        let max_diff = gpu_slice
            .iter()
            .zip(cpu_slice.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0, f32::max);
        println!("Head {}: Max Diff = {}", h, max_diff);
        assert!(max_diff < 0.1, "Head {} mismatch! Max diff {}", h, max_diff);
    }

    Ok(())
}
