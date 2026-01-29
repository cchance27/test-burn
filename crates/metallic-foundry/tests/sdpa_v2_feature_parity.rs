use half::f16;
use metallic_context::{
    Context, tensor::{F16 as LegacyF16, F16Element, Tensor as LegacyTensor, TensorInit as LegacyInit, TensorStorage as LegacyStorage}
};
use metallic_foundry::{
    self, Foundry, MetalError, metals::{
        flashattention::{stages::SdpaParamsResolved, step::RopeFlashDecodeStep}, rope::RopeParamsResolved
    }, storage::Pooled, tensor::{F16 as FoundryF16, Tensor as FoundryTensor, TensorInit}
};
use rand::{Rng, rng};
use serial_test::serial;

fn map_legacy_err(e: metallic_context::MetalError) -> MetalError {
    MetalError::OperationFailed(format!("Legacy Error: {:?}", e))
}

/// Tolerance for SDPA output comparison.
/// Legacy Context might use MPS or optimized kernels with different rounding.
/// We expect reasonable alignment.
const TOLERANCE: f32 = 0.05;

fn generate_random_f16(size: usize) -> Vec<f16> {
    let mut rng = rng();
    (0..size).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect()
}

fn compare_tensors(a: &[f16], b: &[f16], name: &str, tolerance: f32) {
    if a.len() != b.len() {
        panic!("Tensor {} size mismatch: {} vs {}", name, a.len(), b.len());
    }
    let mut max_diff = 0.0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x.to_f32() - y.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tolerance {
            panic!(
                "Mismatch in {} at index {}: V2={} vs Context={} (diff={})",
                name,
                i,
                x.to_f32(),
                y.to_f32(),
                diff
            );
        }
    }
    println!("Max diff for {}: {}", name, max_diff);
}

#[test]
#[serial]
fn test_sdpa_v2_context_parity() -> Result<(), MetalError> {
    // Configuration
    // Decode case: Batch=1, Heads=4, Seq=1 (Q), KV_Len=128
    let batch = 1;
    let heads = 4;
    let q_len = 1;
    let kv_len = 128;
    let head_dim = 64;

    let total_batch = batch * heads;

    // Generate Data
    let q_data = generate_random_f16(batch * heads * q_len * head_dim);
    let k_data = generate_random_f16(batch * heads * kv_len * head_dim);
    let v_data = generate_random_f16(batch * heads * kv_len * head_dim);
    // This test isolates SDPA behavior (not RoPE).
    // We use identity RoPE tables (cos=1, sin=0) so both implementations effectively see the same
    // already-roped Q/K values without depending on RoPE kernel/caching behavior.

    // --- Foundry Setup (V2) ---
    let mut foundry = Foundry::new()?;
    let q_f = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![batch, heads, q_len, head_dim], TensorInit::CopyFrom(&q_data))?;
    let k_f = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![batch, heads, kv_len, head_dim], TensorInit::CopyFrom(&k_data))?;
    let v_f = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![batch, heads, kv_len, head_dim], TensorInit::CopyFrom(&v_data))?;
    let out_f = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![batch, heads, q_len, head_dim], TensorInit::Uninitialized)?;

    let cos_data = vec![f16::ONE; kv_len * head_dim / 2]; // Identity rotation
    let sin_data = vec![f16::ZERO; kv_len * head_dim / 2];

    let cos_f = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![kv_len, head_dim / 2], TensorInit::CopyFrom(&cos_data))?;
    let sin_f = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![kv_len, head_dim / 2], TensorInit::CopyFrom(&sin_data))?;

    let rope_params = RopeParamsResolved {
        dim: head_dim as u32,
        seq_len: kv_len as u32,
        position_offset: (kv_len - 1) as u32,
        total_elements: (total_batch * head_dim) as u32,
    };
    let sdpa_params = SdpaParamsResolved {
        kv_len: kv_len as u32,
        head_dim: head_dim as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
        stride_k_s: k_f.strides()[2] as u32,
        stride_v_s: v_f.strides()[2] as u32,
    };

    let q_strides = (q_f.strides()[0] as u32, q_f.strides()[1] as u32);
    let k_strides = (k_f.strides()[0] as u32, k_f.strides()[1] as u32);
    let v_strides = (v_f.strides()[0] as u32, v_f.strides()[1] as u32);
    let out_strides = (out_f.strides()[0] as u32, out_f.strides()[1] as u32);

    let v2_step = RopeFlashDecodeStep::compile(
        &mut foundry,
        &metallic_foundry::TensorArg::from_tensor(&q_f),
        &metallic_foundry::TensorArg::from_tensor(&k_f), // K is passed as-is
        &metallic_foundry::TensorArg::from_tensor(&v_f),
        &metallic_foundry::TensorArg::from_tensor(&cos_f),
        &metallic_foundry::TensorArg::from_tensor(&sin_f),
        &metallic_foundry::TensorArg::from_tensor(&out_f),
        rope_params,
        sdpa_params,
        batch as u32,
        heads as u32,
        head_dim as u32,
        q_strides,
        k_strides,
        v_strides,
        out_strides,
    )?;

    // Execute Foundry
    use metallic_foundry::spec::{CompiledStep, FastBindings, SymbolTable, TensorBindings};
    v2_step.execute(
        &mut foundry,
        &FastBindings::default(),
        &TensorBindings::default(),
        &SymbolTable::new(),
    )?;

    let res_f: Vec<f16> = out_f.to_vec(&foundry);

    // --- Context Setup (Legacy) ---
    let mut ctx = Context::<F16Element>::new().unwrap();

    let q_leg = LegacyTensor::<LegacyF16>::new(
        vec![total_batch, q_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&q_data),
    )
    .map_err(map_legacy_err)?;

    let k_leg = LegacyTensor::<LegacyF16>::new(
        vec![total_batch, kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&k_data),
    )
    .map_err(map_legacy_err)?;

    let v_leg = LegacyTensor::<LegacyF16>::new(
        vec![total_batch, kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        LegacyInit::CopyFrom(&v_data),
    )
    .map_err(map_legacy_err)?;

    let out_ctx = ctx
        .scaled_dot_product_attention_with_offset(&q_leg, &k_leg, &v_leg, true, kv_len - 1)
        .unwrap();

    let res_c_vec = out_ctx.try_to_vec().unwrap();

    // Compare
    compare_tensors(&res_f, &res_c_vec, "Foundry vs Context", TOLERANCE);

    Ok(())
}
