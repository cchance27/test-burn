//! SDPA V2 vs Context RoPE+SDPA Parity Test
//!
//! Compares the V2 FusedMhaStep (RoPE+SDPA fused) against
//! Context's RoPE + SDPA composed operations.

use half::f16;
use metallic_context::{
    Context,
    kernels::rope::RoPEOp,
    tensor::{F16 as LegacyF16, F16Element, Tensor as LegacyTensor, TensorInit as LegacyInit, TensorStorage as LegacyStorage}, // Legacy types
};
use metallic_foundry::{
    Foundry, MetalError, metals::{
        rope::RopeParamsResolved, sdpa::{stages::SdpaParamsResolved, step::FusedMhaStep}
    }, storage::Pooled, tensor::{Tensor as FoundryTensor, TensorInit, dtypes::F16 as F16Type}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

const TOLERANCE: f32 = 0.05; // FP16 accumulation differences

fn generate_random_f16(size: usize) -> Vec<f16> {
    let mut rng = rng();
    (0..size).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect()
}

/// CPU RoPE implementation for K (matches Context's RoPEOp behavior)
fn cpu_rope_k(k_data: &[f16], cos_data: &[f16], sin_data: &[f16], batch: usize, kv_len: usize, head_dim: usize) -> Vec<f16> {
    let mut k_roped = k_data.to_vec();
    let dim_half = head_dim / 2;

    for b in 0..batch {
        for t in 0..kv_len {
            let k_offset = b * kv_len * head_dim + t * head_dim;
            for i in 0..dim_half {
                let cos_v = cos_data[t * dim_half + i].to_f32();
                let sin_v = sin_data[t * dim_half + i].to_f32();

                let x_i = k_data[k_offset + i].to_f32();
                let x_j = k_data[k_offset + dim_half + i].to_f32();

                // Match Context kernel: out_i = x_i * cos - x_j * sin; out_j = x_j * cos + x_i * sin
                k_roped[k_offset + i] = f16::from_f32(x_i * cos_v - x_j * sin_v);
                k_roped[k_offset + dim_half + i] = f16::from_f32(x_j * cos_v + x_i * sin_v);
            }
        }
    }
    k_roped
}

fn compare_tensors(a: &[f16], b: &[f16], name: &str, tolerance: f32) {
    assert_eq!(a.len(), b.len(), "Tensor {} size mismatch: {} vs {}", name, a.len(), b.len());

    let mut max_diff = 0.0f32;
    let mut max_idx = 0;

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x.to_f32() - y.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("{}: max_diff={:.6} at idx {}", name, max_diff, max_idx);

    assert!(
        max_diff < tolerance,
        "{} mismatch: max_diff={} at idx {} (V2={}, Ctx={})",
        name,
        max_diff,
        max_idx,
        a[max_idx].to_f32(),
        b[max_idx].to_f32()
    );
}

/// Test FusedMhaStep (V2) vs Context's RoPE + SDPA composition
#[test]
#[serial]
fn test_sdpa_v2_vs_context_rope_sdpa() -> Result<(), MetalError> {
    // Configuration for decode case
    let batch = 1;
    let heads = 4;
    let q_len = 1;
    let kv_len = 64;
    let head_dim = 64;
    let total_batch = batch * heads;
    let dim_half = head_dim / 2;

    // Generate random input data
    let q_data = generate_random_f16(total_batch * q_len * head_dim);
    let k_data = generate_random_f16(total_batch * kv_len * head_dim);
    let v_data = generate_random_f16(total_batch * kv_len * head_dim);

    // Use IDENTITY RoPE: cos=1, sin=0 (no rotation)
    // This isolates the SDPA algorithm difference from RoPE
    let cos_data: Vec<f16> = vec![f16::ONE; kv_len * dim_half];
    let sin_data: Vec<f16> = vec![f16::ZERO; kv_len * dim_half];

    // Pre-rope K for V2 (FusedMhaStep assumes K is already roped in KV cache)
    let k_roped_data = cpu_rope_k(&k_data, &cos_data, &sin_data, total_batch, kv_len, head_dim);

    // =========================================================================
    // V2 FusedMhaStep (RoPE Q + SDPA with pre-roped K)
    // =========================================================================
    let mut foundry = Foundry::new()?;

    let q_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![batch, heads, q_len, head_dim], TensorInit::CopyFrom(&q_data))?;
    // Use pre-roped K for V2
    let k_v2 = FoundryTensor::<F16Type, Pooled>::new(
        &mut foundry,
        vec![batch, heads, kv_len, head_dim],
        TensorInit::CopyFrom(&k_roped_data),
    )?;
    let v_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![batch, heads, kv_len, head_dim], TensorInit::CopyFrom(&v_data))?;
    let cos_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![kv_len, dim_half], TensorInit::CopyFrom(&cos_data))?;
    let sin_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![kv_len, dim_half], TensorInit::CopyFrom(&sin_data))?;
    let out_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![batch, heads, q_len, head_dim], TensorInit::Uninitialized)?;

    let rope_params = RopeParamsResolved {
        dim: head_dim as u32,
        seq_len: kv_len as u32,
        position_offset: (kv_len - 1) as u32, // Q position
        total_elements: (total_batch * head_dim) as u32,
    };
    let sdpa_params = SdpaParamsResolved {
        kv_len: kv_len as u32,
        head_dim: head_dim as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
        stride_k_s: k_v2.strides()[2] as u32,
        stride_v_s: v_v2.strides()[2] as u32,
    };

    let q_strides = (q_v2.strides()[0] as u32, q_v2.strides()[1] as u32);
    let k_strides = (k_v2.strides()[0] as u32, k_v2.strides()[1] as u32);
    let v_strides = (v_v2.strides()[0] as u32, v_v2.strides()[1] as u32);
    let out_strides = (out_v2.strides()[0] as u32, out_v2.strides()[1] as u32);

    let v2_step = FusedMhaStep::compile(
        &mut foundry,
        &TensorArg::from_tensor(&q_v2),
        &TensorArg::from_tensor(&k_v2),
        &TensorArg::from_tensor(&v_v2),
        &TensorArg::from_tensor(&cos_v2),
        &TensorArg::from_tensor(&sin_v2),
        &TensorArg::from_tensor(&out_v2),
        rope_params,
        sdpa_params,
        batch as u32, // batch (not total_batch)
        heads as u32, // heads (not 1)
        head_dim as u32,
        q_strides,
        k_strides,
        v_strides,
        out_strides,
    )?;

    use metallic_foundry::spec::{CompiledStep, FastBindings, SymbolTable, TensorBindings};
    v2_step.execute(
        &mut foundry,
        &FastBindings::default(),
        &TensorBindings::default(),
        &SymbolTable::new(),
    )?;

    let res_v2: Vec<f16> = out_v2.to_vec(&foundry);

    // =========================================================================
    // Context: RoPE(Q) + RoPE(K) + SDPA
    // =========================================================================
    let mut ctx = Context::<F16Element>::new().unwrap();

    // Create input tensors
    let q_ctx = LegacyTensor::<LegacyF16>::new(
        vec![total_batch, q_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&q_data),
    )
    .unwrap();
    let k_ctx = LegacyTensor::<LegacyF16>::new(
        vec![total_batch, kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&k_data),
    )
    .unwrap();
    let v_ctx = LegacyTensor::<LegacyF16>::new(
        vec![total_batch, kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&v_data),
    )
    .unwrap();
    let cos_ctx = LegacyTensor::<LegacyF16>::new(
        vec![kv_len, dim_half],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&cos_data),
    )
    .unwrap();
    let sin_ctx = LegacyTensor::<LegacyF16>::new(
        vec![kv_len, dim_half],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&sin_data),
    )
    .unwrap();

    // Apply RoPE to Q (position = kv_len - 1, seq_len = 1)
    let q_roped = ctx
        .call::<RoPEOp>(
            (q_ctx, cos_ctx.clone(), sin_ctx.clone(), head_dim as u32, 1, (kv_len - 1) as u32),
            None,
        )
        .unwrap();

    // Apply RoPE to K (position = 0..kv_len)
    let k_roped = ctx
        .call::<RoPEOp>((k_ctx, cos_ctx, sin_ctx, head_dim as u32, kv_len as u32, 0), None)
        .unwrap();

    ctx.synchronize();

    // Debug: Compare the pre-roped K values
    let k_roped_ctx = k_roped.try_to_vec().unwrap();
    let mut k_diff = 0.0f32;
    for (i, (a, b)) in k_roped_data.iter().zip(k_roped_ctx.iter()).enumerate() {
        let d = (a.to_f32() - b.to_f32()).abs();
        if d > k_diff {
            k_diff = d;
            if d > 0.01 {
                println!("K RoPE diff at {}: cpu={} ctx={}", i, a.to_f32(), b.to_f32());
            }
        }
    }
    println!("K RoPE max diff (CPU vs Context): {}", k_diff);

    // Run SDPA on the roped Q and K
    let out_ctx = ctx
        .scaled_dot_product_attention_with_offset(&q_roped, &k_roped, &v_ctx, true, kv_len - 1)
        .unwrap();

    ctx.synchronize();
    let res_ctx = out_ctx.try_to_vec().unwrap();

    // Compare V2 vs Context
    compare_tensors(&res_v2, &res_ctx, "FusedMhaStep vs Context(RoPE+SDPA)", TOLERANCE);

    Ok(())
}

/// Test with multiple batch/head combinations
#[test]
#[serial]
fn test_sdpa_v2_vs_context_larger() -> Result<(), MetalError> {
    let batch = 1;
    let heads = 14; // Match realistic head count
    let q_len = 1;
    let kv_len = 256;
    let head_dim = 64;
    let total_batch = batch * heads;
    let dim_half = head_dim / 2;

    let q_data = generate_random_f16(total_batch * q_len * head_dim);
    let k_data = generate_random_f16(total_batch * kv_len * head_dim);
    let v_data = generate_random_f16(total_batch * kv_len * head_dim);
    let cos_data = generate_random_f16(kv_len * dim_half);
    let sin_data = generate_random_f16(kv_len * dim_half);

    // Pre-rope K for V2
    let k_roped_data = cpu_rope_k(&k_data, &cos_data, &sin_data, total_batch, kv_len, head_dim);

    // V2 FusedMhaStep
    let mut foundry = Foundry::new()?;

    let q_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![batch, heads, q_len, head_dim], TensorInit::CopyFrom(&q_data))?;
    let k_v2 = FoundryTensor::<F16Type, Pooled>::new(
        &mut foundry,
        vec![batch, heads, kv_len, head_dim],
        TensorInit::CopyFrom(&k_roped_data),
    )?;
    let v_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![batch, heads, kv_len, head_dim], TensorInit::CopyFrom(&v_data))?;
    let cos_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![kv_len, dim_half], TensorInit::CopyFrom(&cos_data))?;
    let sin_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![kv_len, dim_half], TensorInit::CopyFrom(&sin_data))?;
    let out_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![batch, heads, q_len, head_dim], TensorInit::Uninitialized)?;

    let v2_step = FusedMhaStep::compile(
        &mut foundry,
        &TensorArg::from_tensor(&q_v2),
        &TensorArg::from_tensor(&k_v2),
        &TensorArg::from_tensor(&v_v2),
        &TensorArg::from_tensor(&cos_v2),
        &TensorArg::from_tensor(&sin_v2),
        &TensorArg::from_tensor(&out_v2),
        RopeParamsResolved {
            dim: head_dim as u32,
            seq_len: kv_len as u32,
            position_offset: (kv_len - 1) as u32,
            total_elements: (total_batch * head_dim) as u32,
        },
        SdpaParamsResolved {
            kv_len: kv_len as u32,
            head_dim: head_dim as u32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            stride_k_s: k_v2.strides()[2] as u32,
            stride_v_s: v_v2.strides()[2] as u32,
        },
        batch as u32, // batch (not total_batch)
        heads as u32, // heads (not 1)
        head_dim as u32,
        (q_v2.strides()[0] as u32, q_v2.strides()[1] as u32),
        (k_v2.strides()[0] as u32, k_v2.strides()[1] as u32),
        (v_v2.strides()[0] as u32, v_v2.strides()[1] as u32),
        (out_v2.strides()[0] as u32, out_v2.strides()[1] as u32),
    )?;

    use metallic_foundry::spec::{CompiledStep, FastBindings, SymbolTable, TensorBindings};
    v2_step.execute(
        &mut foundry,
        &FastBindings::default(),
        &TensorBindings::default(),
        &SymbolTable::new(),
    )?;
    let res_v2 = out_v2.to_vec(&foundry);

    // Context: RoPE + SDPA
    let mut ctx = Context::<F16Element>::new().unwrap();
    let q_ctx = LegacyTensor::<LegacyF16>::new(
        vec![total_batch, q_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&q_data),
    )
    .unwrap();
    let k_ctx = LegacyTensor::<LegacyF16>::new(
        vec![total_batch, kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&k_data),
    )
    .unwrap();
    let v_ctx = LegacyTensor::<LegacyF16>::new(
        vec![total_batch, kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&v_data),
    )
    .unwrap();
    let cos_ctx = LegacyTensor::<LegacyF16>::new(
        vec![kv_len, dim_half],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&cos_data),
    )
    .unwrap();
    let sin_ctx = LegacyTensor::<LegacyF16>::new(
        vec![kv_len, dim_half],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&sin_data),
    )
    .unwrap();

    let q_roped = ctx
        .call::<RoPEOp>(
            (q_ctx, cos_ctx.clone(), sin_ctx.clone(), head_dim as u32, 1, (kv_len - 1) as u32),
            None,
        )
        .unwrap();
    let k_roped = ctx
        .call::<RoPEOp>((k_ctx, cos_ctx, sin_ctx, head_dim as u32, kv_len as u32, 0), None)
        .unwrap();
    let out_ctx = ctx
        .scaled_dot_product_attention_with_offset(&q_roped, &k_roped, &v_ctx, true, kv_len - 1)
        .unwrap();

    ctx.synchronize();
    let res_ctx = out_ctx.try_to_vec().unwrap();

    compare_tensors(&res_v2, &res_ctx, "FusedMhaStep vs Context (14 heads, 256 kv)", TOLERANCE);

    Ok(())
}
