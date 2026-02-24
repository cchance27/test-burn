use half::f16;
use metallic_foundry::{
    Foundry, metals::swiglu::step::FusedSwigluStep, spec::{Step, TensorBindings}, storage::Pooled, tensor::{F16, Q8_0, Tensor as FoundryTensor, TensorInit}
};

#[test]
fn test_fused_swiglu_mixed_policy_fails_fast() -> Result<(), Box<dyn std::error::Error>> {
    let mut foundry = Foundry::new()?;
    let mut bindings = TensorBindings::new();

    let m: usize = 1;
    let k_dim: usize = 32;
    let n_dim: usize = 32;
    bindings.set_int_global("m", m);

    let input_f16 = vec![f16::from_f32(0.5); m * k_dim];
    let gamma_f16 = vec![f16::from_f32(1.0); k_dim];
    let b_gate_f16 = vec![f16::from_f32(0.0); n_dim];
    let b_up_f16 = vec![f16::from_f32(0.0); n_dim];

    // Gate is quantized Q8, Up is dense F16 -> mixed-policy should hard-fail.
    let w_gate_q8 = vec![0u8; n_dim * k_dim];
    let s_gate_bytes = vec![0u8; n_dim * 2];
    let w_up_f16 = vec![f16::from_f32(0.1); n_dim * k_dim];

    let input = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, k_dim], TensorInit::CopyFrom(&input_f16))?;
    let gamma = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k_dim], TensorInit::CopyFrom(&gamma_f16))?;
    let w_gate = FoundryTensor::<Q8_0, Pooled>::new(&mut foundry, vec![n_dim, k_dim], TensorInit::CopyFrom(&w_gate_q8))?;
    let s_gate = FoundryTensor::<Q8_0, Pooled>::new(&mut foundry, vec![n_dim * 2], TensorInit::CopyFrom(&s_gate_bytes))?;
    let w_up = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_dim, k_dim], TensorInit::CopyFrom(&w_up_f16))?;
    let b_gate = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_dim], TensorInit::CopyFrom(&b_gate_f16))?;
    let b_up = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_dim], TensorInit::CopyFrom(&b_up_f16))?;
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, n_dim], TensorInit::Uninitialized)?;

    bindings.insert("input".to_string(), metallic_foundry::types::TensorArg::from_tensor(&input));
    bindings.insert("gamma".to_string(), metallic_foundry::types::TensorArg::from_tensor(&gamma));
    bindings.insert("wg".to_string(), metallic_foundry::types::TensorArg::from_tensor(&w_gate));
    bindings.insert("wg_scales".to_string(), metallic_foundry::types::TensorArg::from_tensor(&s_gate));
    bindings.insert("wu".to_string(), metallic_foundry::types::TensorArg::from_tensor(&w_up));
    bindings.insert("bg".to_string(), metallic_foundry::types::TensorArg::from_tensor(&b_gate));
    bindings.insert("bu".to_string(), metallic_foundry::types::TensorArg::from_tensor(&b_up));
    bindings.insert("out".to_string(), metallic_foundry::types::TensorArg::from_tensor(&output));

    let fused = FusedSwigluStep {
        input: "input".into(),
        gamma: "gamma".into(),
        wg: "wg".into(),
        wu: "wu".into(),
        bg: Some("bg".into()),
        bu: Some("bu".into()),
        out: "out".into(),
        epsilon: 1e-6,
        weights_per_block: 32,
    };

    let err = fused
        .execute(&mut foundry, &mut bindings)
        .expect_err("mixed-policy fused SwiGLU should fail fast");
    let msg = err.to_string();
    assert!(msg.contains("mixed-policy is unsupported"), "unexpected error: {msg}");
    Ok(())
}
