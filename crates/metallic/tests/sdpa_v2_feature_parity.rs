use half::f16;
use metallic::{
    Context, F16Element, MetalError, foundry::{Foundry, storage::Pooled, tensor::Tensor as FoundryTensor}, metals::{
        rope::RopeParamsResolved, sdpa::{stages::SdpaParamsResolved, step::FusedMhaStep}
    }, tensor::{F16, Tensor, TensorInit, TensorStorage as LegacyStorage, dtypes::F16 as F16Type}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

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
    // Context needs separate RoPE cache maybe?
    // Or we can pre-rope inputs for simplicity if Context allows raw SDPA.
    // Context::scaled_dot_product_attention assumes inputs are already ROPED usually?
    // Let's assume passed Q and K are ROPED.
    // So we don't need to involve RoPE implementation differences here, just SDPA core.

    // --- Foundry Setup (V2) ---
    let mut foundry = Foundry::new()?;
    let q_f = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![total_batch, q_len, head_dim], TensorInit::CopyFrom(&q_data))?;
    let k_f = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![total_batch, kv_len, head_dim], TensorInit::CopyFrom(&k_data))?;
    let v_f = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![total_batch, kv_len, head_dim], TensorInit::CopyFrom(&v_data))?;
    let out_f = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![total_batch, q_len, head_dim], TensorInit::Uninitialized)?;

    let cos_data = vec![f16::ONE; kv_len * head_dim / 2]; // Identity rotation
    let sin_data = vec![f16::ZERO; kv_len * head_dim / 2];

    let cos_f = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![kv_len, head_dim / 2], TensorInit::CopyFrom(&cos_data))?;
    let sin_f = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![kv_len, head_dim / 2], TensorInit::CopyFrom(&sin_data))?;

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
        stride_k_s: k_f.strides()[1] as u32,
        stride_v_s: v_f.strides()[1] as u32,
    };

    let q_strides = (q_f.strides()[0] as u32, q_f.strides()[1] as u32);
    let k_strides = (k_f.strides()[0] as u32, k_f.strides()[1] as u32);
    let v_strides = (v_f.strides()[0] as u32, v_f.strides()[1] as u32);
    let out_strides = (out_f.strides()[0] as u32, out_f.strides()[1] as u32);

    let v2_step = FusedMhaStep::compile(
        &mut foundry,
        &TensorArg::from_tensor(&q_f),
        &TensorArg::from_tensor(&k_f), // K is passed as-is
        &TensorArg::from_tensor(&v_f),
        &TensorArg::from_tensor(&cos_f),
        &TensorArg::from_tensor(&sin_f),
        &TensorArg::from_tensor(&out_f),
        rope_params,
        sdpa_params,
        total_batch as u32,
        1,
        head_dim as u32,
        q_strides,
        k_strides,
        v_strides,
        out_strides,
    )?;

    // Execute Foundry
    use metallic::foundry::spec::{CompiledStep, FastBindings, TensorBindings};
    v2_step.execute(&mut foundry, &FastBindings::default(), &TensorBindings::default())?;

    let res_f = out_f.to_vec(&foundry);

    // --- Context Setup (Legacy) ---
    let mut ctx = Context::<F16Element>::new()?;

    let q_ctx = Tensor::<F16>::new(
        vec![total_batch, q_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&q_data),
    )?;
    let k_ctx = Tensor::<F16>::new(
        vec![total_batch, kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&k_data),
    )?;
    let v_ctx = Tensor::<F16>::new(
        vec![total_batch, kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&v_data),
    )?;

    let out_ctx = ctx.scaled_dot_product_attention(&q_ctx, &k_ctx, &v_ctx, true)?;

    let res_c_vec = out_ctx.try_to_vec()?;

    // Compare
    compare_tensors(&res_f, &res_c_vec, "Foundry vs Context", TOLERANCE);

    Ok(())
}
