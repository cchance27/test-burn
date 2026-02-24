use metallic_env::{EnvVarGuard, FoundryEnvVar, SDPA_DEBUG_ONLINE_COMPARE_MIN_KV};
use metallic_foundry::{
    Foundry, MetalError, metals::sdpa::step::FlashAttentionStep, spec::{DynamicValue, Ref, Step, TensorBindings}, storage::Pooled, tensor::{F32, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};

#[test]
fn flashattention_debug_compare_accepts_f32_decode_views() -> Result<(), MetalError> {
    let _compare_guard = EnvVarGuard::set(FoundryEnvVar::SdpaDebugOnlineCompare, "1");
    let _min_kv_guard = SDPA_DEBUG_ONLINE_COMPARE_MIN_KV.set_guard(0).expect("set min-kv env");

    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 2;
    let head_dim: u32 = 64;
    let kv_len: u32 = 8;
    let d_model = (n_heads * head_dim) as usize;

    let q_host = vec![0.0f32; d_model];
    let kv_host = vec![0.0f32; (n_heads as usize) * (kv_len as usize) * (head_dim as usize)];
    let out_host = vec![0.0f32; d_model];

    let q = FoundryTensor::<F32, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::CopyFrom(&q_host))?;
    let k = FoundryTensor::<F32, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::CopyFrom(&kv_host),
    )?;
    let v = FoundryTensor::<F32, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::CopyFrom(&kv_host),
    )?;
    let output = FoundryTensor::<F32, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::CopyFrom(&out_host))?;

    let mut bindings = TensorBindings::new();
    bindings.insert("q".to_string(), TensorArg::from_tensor(&q));
    bindings.insert("k".to_string(), TensorArg::from_tensor(&k));
    bindings.insert("v".to_string(), TensorArg::from_tensor(&v));
    bindings.insert("o".to_string(), TensorArg::from_tensor(&output));

    let step = FlashAttentionStep {
        q: Ref("q".into()),
        k: Ref("k".into()),
        v: Ref("v".into()),
        output: Ref("o".into()),
        causal: true,
        query_offset: DynamicValue::Literal(kv_len - 1),
        n_heads: DynamicValue::Literal(n_heads),
        head_dim: DynamicValue::Literal(head_dim),
        kv_seq_len: DynamicValue::Literal(kv_len),
        m: DynamicValue::Literal(1),
        kv_head_major: true,
    };

    step.execute(&mut foundry, &mut bindings)?;
    Ok(())
}
