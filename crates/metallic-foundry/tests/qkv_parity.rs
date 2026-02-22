use std::{sync::Arc, time::Instant};

use half::f16;
use metallic_foundry::{
    Foundry, compound::{CompoundKernel, Layout, stages::WarpLayoutStage}, metals::{
        gemv::{
            qkv_stages::{MultiWarpReduceStage, MultiWriteOutputStage, ParallelProjectStage}, qkv_step::FusedQkvArgs
        }, rmsnorm::stages::RmsNormComputeStage
    }, policy::q8::PolicyQ8, storage::Pooled, tensor::{F16, Q8_0, Tensor, TensorInit}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};
use rand::Rng;
use serial_test::serial;

fn run_cpu_rmsnorm(input: &[f16], gamma: &[f16]) -> Vec<f16> {
    let k = input.len();
    let mut sum_sq = 0.0f32;
    for &x in input {
        let x_f = x.to_f32();
        sum_sq += x_f * x_f;
    }
    let inv_rms = 1.0 / (sum_sq / k as f32 + 1e-6).sqrt();
    input
        .iter()
        .zip(gamma.iter())
        .map(|(&x, &g)| f16::from_f32(x.to_f32() * inv_rms * g.to_f32()))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_cpu_qkv(
    input_norm: &[f16],
    w_q: &[f16],
    w_k: &[f16],
    w_v: &[f16],
    b_q: &[f16],
    b_k: &[f16],
    b_v: &[f16],
    k_dim: usize,
    n_dim: usize,
    n_kv: usize,
) -> (Vec<f16>, Vec<f16>, Vec<f16>) {
    let mut out_q = vec![f16::from_f32(0.0); n_dim];
    let mut out_k = vec![f16::from_f32(0.0); n_kv];
    let mut out_v = vec![f16::from_f32(0.0); n_kv];

    for i in 0..n_dim {
        let mut sum = 0.0f32;
        for j in 0..k_dim {
            sum += input_norm[j].to_f32() * w_q[i * k_dim + j].to_f32();
        }
        out_q[i] = f16::from_f32(sum + b_q[i].to_f32());
    }

    for i in 0..n_kv {
        let mut sum_k = 0.0f32;
        let mut sum_v = 0.0f32;
        for j in 0..k_dim {
            sum_k += input_norm[j].to_f32() * w_k[i * k_dim + j].to_f32();
            sum_v += input_norm[j].to_f32() * w_v[i * k_dim + j].to_f32();
        }
        out_k[i] = f16::from_f32(sum_k + b_k[i].to_f32());
        out_v[i] = f16::from_f32(sum_v + b_v[i].to_f32());
    }

    (out_q, out_k, out_v)
}

fn dequantize_q8(data: &[u8], scales: &[u8], k: usize, n: usize) -> Vec<f16> {
    let mut out = vec![f16::from_f32(0.0); k * n];
    let w_per_blk = 32;
    let blocks_per_k = k.div_ceil(w_per_blk);
    for r in 0..n {
        for b in 0..blocks_per_k {
            let s_off = r * blocks_per_k * 2 + b * 2;
            let scale_bits = u16::from_le_bytes([scales[s_off], scales[s_off + 1]]);
            let scale = f16::from_bits(scale_bits);
            for i in 0..w_per_blk {
                let k_idx = b * w_per_blk + i;
                if k_idx < k {
                    let q = data[r * k + k_idx] as i8;
                    out[r * k + k_idx] = f16::from_f32(q as f32 * scale.to_f32());
                }
            }
        }
    }
    out
}

fn quantize_q8_0(k: usize, n: usize, weights: &[f16]) -> (Vec<u8>, Vec<u8>) {
    let weights_per_block = 32;
    let scale_bytes_per_block = 2;
    let blocks_per_k = k.div_ceil(weights_per_block);
    let total_blocks = blocks_per_k * n;
    let mut data = vec![0u8; total_blocks * weights_per_block];
    let mut scales = vec![0u8; total_blocks * scale_bytes_per_block];

    for col in 0..n {
        for bk in 0..blocks_per_k {
            let base_k = bk * weights_per_block;
            let blk_len = weights_per_block.min(k - base_k);
            let mut max_abs = 0f32;
            for i in 0..blk_len {
                max_abs = max_abs.max(weights[col * k + base_k + i].to_f32().abs());
            }
            let d = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            let d_bits = f16::from_f32(d).to_bits().to_le_bytes();
            let _sb = (bk * n + col) * scale_bytes_per_block; // Note: layout should match WEIGHT_INDEX
            // Actually WEIGHT_INDEX is (row * k_dim + k)
            // So for a Q8 it is (row * blocks_per_k + bk)
            let sb_actual = (col * blocks_per_k + bk) * scale_bytes_per_block;
            scales[sb_actual..sb_actual + 2].copy_from_slice(&d_bits);

            let base = (col * blocks_per_k + bk) * weights_per_block;
            for i in 0..weights_per_block {
                let val = if i < blk_len { weights[col * k + base_k + i].to_f32() } else { 0.0 };
                let q = if d > 0.0 { (val / d).round().clamp(-127.0, 127.0) as i8 } else { 0 };
                data[base + i] = q as u8;
            }
        }
    }
    (data, scales)
}

fn assert_close(a: &[f16], b: &[f16], name: &str) {
    assert_eq!(a.len(), b.len());
    let mut max_diff = 0.0f32;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x.to_f32() - y.to_f32()).abs();
        max_diff = max_diff.max(diff);
        assert!(diff < 0.01, "{} disparity at index {}: CPU={} GPU={} diff={}", name, i, x, y, diff);
    }
    println!("{} max diff: {}", name, max_diff);
}

fn swizzle_q8_weights_nk(rows_n: usize, cols_k: usize, raw_weights: &[u8]) -> Vec<u8> {
    let weights_per_block = 32;
    let blocks_per_row = cols_k.div_ceil(weights_per_block);
    let expected_bytes = rows_n * blocks_per_row * weights_per_block;
    assert_eq!(
        raw_weights.len(),
        expected_bytes,
        "swizzle_q8_weights_nk: expected {} bytes for rows_n={} cols_k={} (blocks_per_row={}), got {}",
        expected_bytes,
        rows_n,
        cols_k,
        blocks_per_row,
        raw_weights.len()
    );

    let mut swizzled = vec![0u8; raw_weights.len()];
    for k_block in 0..blocks_per_row {
        for row in 0..rows_n {
            let src_block = row * blocks_per_row + k_block;
            let dst_block = k_block * rows_n + row;
            let src = src_block * weights_per_block;
            let dst = dst_block * weights_per_block;
            swizzled[dst..dst + weights_per_block].copy_from_slice(&raw_weights[src..src + weights_per_block]);
        }
    }
    swizzled
}

#[test]
#[serial]
#[allow(clippy::too_many_arguments)]
fn test_qkv_parity() {
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rand::rng();

    let k_dim = 1024;
    let n_dim = 1024;
    let n_kv = 128; // GQA

    let mut x_data = vec![f16::from_f32(0.0); k_dim];
    let mut gamma_data = vec![f16::from_f32(0.0); k_dim];
    let mut w_q_data = vec![f16::from_f32(0.0); n_dim * k_dim];
    let mut w_k_data = vec![f16::from_f32(0.0); n_kv * k_dim];
    let mut w_v_data = vec![f16::from_f32(0.0); n_kv * k_dim];
    let mut b_q_data = vec![f16::from_f32(0.0); n_dim];
    let mut b_k_data = vec![f16::from_f32(0.0); n_kv];
    let mut b_v_data = vec![f16::from_f32(0.0); n_kv];

    for (x, g) in x_data.iter_mut().zip(gamma_data.iter_mut()) {
        *x = f16::from_f32(rng.random_range(-1.0..1.0));
        *g = f16::from_f32(rng.random_range(0.5..1.5));
    }
    for x in &mut w_q_data {
        *x = f16::from_f32(rng.random_range(-0.1..0.1));
    }
    for (wk, wv) in w_k_data.iter_mut().zip(w_v_data.iter_mut()) {
        let val_k = f16::from_f32(rng.random_range(-0.1..0.1));
        let val_v = f16::from_f32(rng.random_range(-0.1..0.1));
        *wk = val_k;
        *wv = val_v;
    }
    for x in &mut b_q_data {
        *x = f16::from_f32(rng.random_range(-0.1..0.1));
    }
    for (bk, bv) in b_k_data.iter_mut().zip(b_v_data.iter_mut()) {
        let val_k = f16::from_f32(rng.random_range(-0.1..0.1));
        let val_v = f16::from_f32(rng.random_range(-0.1..0.1));
        *bk = val_k;
        *bv = val_v;
    }

    // Quantize
    let (wq_bytes, sq_bytes) = quantize_q8_0(k_dim, n_dim, &w_q_data);
    let (wk_bytes, sk_bytes) = quantize_q8_0(k_dim, n_kv, &w_k_data);
    let (wv_bytes, sv_bytes) = quantize_q8_0(k_dim, n_kv, &w_v_data);

    // CPU Reference with dequantized weights (for bit-exact parity)
    let wq_deq = dequantize_q8(&wq_bytes, &sq_bytes, k_dim, n_dim);
    let wk_deq = dequantize_q8(&wk_bytes, &sk_bytes, k_dim, n_kv);
    let wv_deq = dequantize_q8(&wv_bytes, &sv_bytes, k_dim, n_kv);

    let x_norm = run_cpu_rmsnorm(&x_data, &gamma_data);
    let (ref_q, ref_k, ref_v) = run_cpu_qkv(
        &x_norm, &wq_deq, &wk_deq, &wv_deq, &b_q_data, &b_k_data, &b_v_data, k_dim, n_dim, n_kv,
    );

    // GPU Setup
    let x_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![k_dim], TensorInit::CopyFrom(&x_data)).unwrap();
    let gamma_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![k_dim], TensorInit::CopyFrom(&gamma_data)).unwrap();

    // QKV kernels use a K-block-major (Canonical) weight layout for streaming efficiency.
    // Scales remain row-major (per-row block scales are indexed separately).
    let wq_swizzled = swizzle_q8_weights_nk(n_dim, k_dim, &wq_bytes);
    let wk_swizzled = swizzle_q8_weights_nk(n_kv, k_dim, &wk_bytes);
    let wv_swizzled = swizzle_q8_weights_nk(n_kv, k_dim, &wv_bytes);

    // Q8_0 weights + scales are packed bytes on GPU (`device uchar*`), so upload as raw Q8_0 tensors.
    let wq_tensor = Tensor::<Q8_0, Pooled>::new(&mut foundry, vec![wq_swizzled.len()], TensorInit::CopyFrom(&wq_swizzled)).unwrap();
    let sq_tensor = Tensor::<Q8_0, Pooled>::new(&mut foundry, vec![sq_bytes.len()], TensorInit::CopyFrom(&sq_bytes)).unwrap();
    let wk_tensor = Tensor::<Q8_0, Pooled>::new(&mut foundry, vec![wk_swizzled.len()], TensorInit::CopyFrom(&wk_swizzled)).unwrap();
    let sk_tensor = Tensor::<Q8_0, Pooled>::new(&mut foundry, vec![sk_bytes.len()], TensorInit::CopyFrom(&sk_bytes)).unwrap();
    let wv_tensor = Tensor::<Q8_0, Pooled>::new(&mut foundry, vec![wv_swizzled.len()], TensorInit::CopyFrom(&wv_swizzled)).unwrap();
    let sv_tensor = Tensor::<Q8_0, Pooled>::new(&mut foundry, vec![sv_bytes.len()], TensorInit::CopyFrom(&sv_bytes)).unwrap();

    let b_q_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_dim], TensorInit::CopyFrom(&b_q_data)).unwrap();
    let b_k_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv], TensorInit::CopyFrom(&b_k_data)).unwrap();
    let b_v_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv], TensorInit::CopyFrom(&b_v_data)).unwrap();

    let out_q_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_dim], TensorInit::Uninitialized).unwrap();
    let out_k_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv], TensorInit::Uninitialized).unwrap();
    let out_v_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv], TensorInit::Uninitialized).unwrap();

    // Compile Kernel
    let kernel = Arc::new(
        CompoundKernel::new("fused_qkv_rmsnorm_test")
            .with_manual_output(true)
            .prologue(
                WarpLayoutStage::new(Layout::Canonical {
                    expected_k: 0,
                    expected_n: 0,
                })
                .with_warps(8),
            )
            // FusedQkvArgs buffer indices:
            // input=6, k_dim=7, gamma=18, epsilon=19
            .prologue(RmsNormComputeStage::new(6, 7, 19))
            .main(ParallelProjectStage::new(Arc::new(PolicyQ8)).with_norm("inv_rms"))
            .epilogue(MultiWarpReduceStage)
            .epilogue(MultiWriteOutputStage::new())
            .compile(),
    );

    let args = FusedQkvArgs {
        w_q: TensorArg::from_tensor(&wq_tensor),
        s_q: Some(TensorArg::from_tensor(&sq_tensor)),
        w_k: TensorArg::from_tensor(&wk_tensor),
        s_k: Some(TensorArg::from_tensor(&sk_tensor)),
        w_v: TensorArg::from_tensor(&wv_tensor),
        s_v: Some(TensorArg::from_tensor(&sv_tensor)),
        input: TensorArg::from_tensor(&x_tensor),
        k_dim: k_dim as u32,
        n_dim: n_dim as u32,
        n_kv: n_kv as u32,
        weights_per_block: 32,
        out_q: TensorArg::from_tensor(&out_q_tensor),
        out_k: TensorArg::from_tensor(&out_k_tensor),
        out_v: TensorArg::from_tensor(&out_v_tensor),
        b_q: TensorArg::from_tensor(&b_q_tensor),
        b_k: TensorArg::from_tensor(&b_k_tensor),
        b_v: TensorArg::from_tensor(&b_v_tensor),
        has_bias: 1,
        gamma: TensorArg::from_tensor(&gamma_tensor),
        epsilon: 1e-6,
    };

    let warps_per_tg = 8;
    let num_groups = n_dim.max(n_kv).div_ceil(warps_per_tg);
    let dispatch = DispatchConfig {
        grid: GridSize::d1(num_groups),
        group: ThreadgroupSize::d1(warps_per_tg * 32),
    };

    foundry.run(&kernel.clone().bind_arc(args.clone(), dispatch)).unwrap();
    let gpu_q = out_q_tensor.to_vec(&foundry);
    let gpu_k = out_k_tensor.to_vec(&foundry);
    let gpu_v = out_v_tensor.to_vec(&foundry);

    println!("--- New Kernel Parity ---");
    assert_close(&ref_q, &gpu_q, "Q Output");
    assert_close(&ref_k, &gpu_k, "K Output");
    assert_close(&ref_v, &gpu_v, "V Output");

    // Benchmark
    let iters = 1000;
    println!("Warming up...");
    for _ in 0..10 {
        foundry.run(&kernel.clone().bind_arc(args.clone(), dispatch)).unwrap();
    }

    let t0 = Instant::now();
    foundry.start_capture().unwrap();
    for _ in 0..iters {
        foundry.run(&kernel.clone().bind_arc(args.clone(), dispatch)).unwrap();
    }
    let buf = foundry.end_capture().unwrap();
    buf.wait_until_completed();
    let elapsed = t0.elapsed();
    println!("QKV Fused (new) 1000 iterations: {:?} (avg {:?})", elapsed, elapsed / iters);
}
