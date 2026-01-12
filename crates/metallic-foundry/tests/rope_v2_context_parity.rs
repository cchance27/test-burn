//! RoPE V2 vs Context Parity Test
//!
//! Isolates RoPE implementation comparison:
//! - V2 Foundry Rope step (metals/rope/mod.rs)
//! - Context RoPEOp (kernels/rope/mod.rs)

use half::f16;
use metallic_context::{
    Context,
    F16Element,
    kernels::rope::RoPEOp,
    tensor::TensorInit as LegacyInit, // Explicit alias
    tensor::{F16 as LegacyF16, Tensor as LegacyTensor, TensorStorage as LegacyStorage},
};
use metallic_foundry::{
    Foundry, MetalError, metals::rope::{Rope, RopeParamsResolved}, storage::Pooled, tensor::{F16 as FoundryF16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

const TOLERANCE: f32 = 0.001; // Tight tolerance for RoPE

fn generate_random_f16(size: usize) -> Vec<f16> {
    let mut rng = rng();
    (0..size).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect()
}

fn map_legacy_err(e: metallic_context::MetalError) -> MetalError {
    MetalError::OperationFailed(format!("Legacy Error: {:?}", e))
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

    if max_diff >= tolerance {
        println!("First 10 elements comparison:");
        for i in 0..10.min(a.len()) {
            println!(
                "  [{}] V2={:.6} Context={:.6} diff={:.6}",
                i,
                a[i].to_f32(),
                b[i].to_f32(),
                (a[i].to_f32() - b[i].to_f32()).abs()
            );
        }
    }

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

/// Test V2 Rope step vs Context RoPEOp for a single position (Q-style)
#[test]
#[serial]
fn test_rope_v2_vs_context_single_position() -> Result<(), MetalError> {
    let batch = 4;
    let seq_len = 1;
    let head_dim = 64;
    let kv_len = 128;
    let dim_half = head_dim / 2;
    let position = kv_len - 1; // Last position

    // Generate random input and cos/sin caches
    let input_data = generate_random_f16(batch * seq_len * head_dim);
    let cos_data = generate_random_f16(kv_len * dim_half);
    let sin_data = generate_random_f16(kv_len * dim_half);

    // =========================================================================
    // V2 Foundry Rope Step
    // =========================================================================
    let mut foundry = Foundry::new()?;

    let input_v2 =
        FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![batch * seq_len, head_dim], TensorInit::CopyFrom(&input_data))?;
    let output_v2 = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![batch * seq_len, head_dim], TensorInit::Uninitialized)?;
    let cos_v2 = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![kv_len, dim_half], TensorInit::CopyFrom(&cos_data))?;
    let sin_v2 = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![kv_len, dim_half], TensorInit::CopyFrom(&sin_data))?;

    let rope_params = RopeParamsResolved {
        dim: head_dim as u32,
        seq_len: seq_len as u32,
        position_offset: position as u32,
        total_elements: (batch * seq_len * head_dim) as u32,
    };

    let rope_step = Rope::new(
        &TensorArg::from_tensor(&input_v2),
        &TensorArg::from_tensor(&output_v2),
        &TensorArg::from_tensor(&cos_v2),
        &TensorArg::from_tensor(&sin_v2),
        rope_params,
    );
    foundry.run(&rope_step)?;

    let res_v2 = output_v2.to_vec(&foundry);

    // =========================================================================
    // Context RoPEOp
    // =========================================================================
    let mut ctx = Context::<F16Element>::new().unwrap();

    let input_ctx = LegacyTensor::<LegacyF16>::new(
        vec![batch * seq_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&input_data),
    )
    .map_err(map_legacy_err)?;
    let cos_ctx = LegacyTensor::<LegacyF16>::new(
        vec![kv_len, dim_half],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&cos_data),
    )
    .map_err(map_legacy_err)?;
    let sin_ctx = LegacyTensor::<LegacyF16>::new(
        vec![kv_len, dim_half],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&sin_data),
    )
    .map_err(map_legacy_err)?;

    let output_ctx = ctx
        .call::<RoPEOp>(
            (input_ctx, cos_ctx, sin_ctx, head_dim as u32, seq_len as u32, position as u32),
            None,
        )
        .map_err(map_legacy_err)?;

    ctx.synchronize();
    let res_ctx = output_ctx.try_to_vec().map_err(map_legacy_err)?;

    // Compare
    compare_tensors(&res_v2, &res_ctx, "RoPE V2 vs Context (single position)", TOLERANCE);

    Ok(())
}

/// Test V2 Rope step vs Context RoPEOp for multiple positions (K-style)
#[test]
#[serial]
fn test_rope_v2_vs_context_full_sequence() -> Result<(), MetalError> {
    let batch = 4;
    let kv_len = 64;
    let head_dim = 64;
    let dim_half = head_dim / 2;

    // Generate random input and cos/sin caches
    let input_data = generate_random_f16(batch * kv_len * head_dim);
    let cos_data = generate_random_f16(kv_len * dim_half);
    let sin_data = generate_random_f16(kv_len * dim_half);

    // =========================================================================
    // V2 Foundry Rope Step
    // =========================================================================
    let mut foundry = Foundry::new()?;

    let input_v2 =
        FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![batch * kv_len, head_dim], TensorInit::CopyFrom(&input_data))?;
    let output_v2 = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![batch * kv_len, head_dim], TensorInit::Uninitialized)?;
    let cos_v2 = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![kv_len, dim_half], TensorInit::CopyFrom(&cos_data))?;
    let sin_v2 = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![kv_len, dim_half], TensorInit::CopyFrom(&sin_data))?;

    let rope_params = RopeParamsResolved {
        dim: head_dim as u32,
        seq_len: kv_len as u32,
        position_offset: 0,
        total_elements: (batch * kv_len * head_dim) as u32,
    };

    let rope_step = Rope::new(
        &TensorArg::from_tensor(&input_v2),
        &TensorArg::from_tensor(&output_v2),
        &TensorArg::from_tensor(&cos_v2),
        &TensorArg::from_tensor(&sin_v2),
        rope_params,
    );
    foundry.run(&rope_step)?;

    let res_v2 = output_v2.to_vec(&foundry);

    // =========================================================================
    // Context RoPEOp
    // =========================================================================
    let mut ctx = Context::<F16Element>::new().unwrap();

    let input_ctx = LegacyTensor::<LegacyF16>::new(
        vec![batch * kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&input_data),
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

    let output_ctx = ctx
        .call::<RoPEOp>((input_ctx, cos_ctx, sin_ctx, head_dim as u32, kv_len as u32, 0), None)
        .unwrap();

    ctx.synchronize();
    let res_ctx = output_ctx.try_to_vec().unwrap();

    // Compare
    compare_tensors(&res_v2, &res_ctx, "RoPE V2 vs Context (full sequence)", TOLERANCE);

    Ok(())
}
