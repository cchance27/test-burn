use half::f16;
use metallic_foundry::{
    Foundry, metals::embedding::EmbeddingStep, spec::{DynamicValue, Step, TensorBindings}, storage::Pooled, tensor::{F16, Q8_0, Tensor as FoundryTensor, TensorInit, U32}, types::TensorArg
};

#[test]
fn test_embedding_q8_smoke() -> Result<(), Box<dyn std::error::Error>> {
    let mut foundry = Foundry::new()?;
    let mut bindings = TensorBindings::new();

    let vocab: usize = 2;
    let d_model: usize = 32;
    let tokens: usize = 3;
    let blocks_per_k = d_model / 32;
    assert_eq!(blocks_per_k, 1);

    // Table: [vocab, d_model] as int8 bytes.
    let mut table: Vec<u8> = vec![0u8; vocab * d_model];
    for feat in 0..d_model {
        table[feat] = feat as i8 as u8; // tok0: 0..31
        table[d_model + feat] = (2 * feat) as i8 as u8; // tok1: 0..62
    }

    // Scales: one fp16 scale per block per row (2 bytes).
    let mut scales: Vec<u8> = vec![0u8; vocab * blocks_per_k * 2];
    let s0 = f16::from_f32(0.5);
    let s1 = f16::from_f32(1.0);
    scales[0..2].copy_from_slice(&s0.to_bits().to_le_bytes());
    scales[2..4].copy_from_slice(&s1.to_bits().to_le_bytes());

    let ids: Vec<u32> = vec![0, 1, 1];

    let table_t = FoundryTensor::<Q8_0, Pooled>::new(&mut foundry, vec![vocab, d_model], TensorInit::CopyFrom(&table))?;
    let scales_t = FoundryTensor::<Q8_0, Pooled>::new(&mut foundry, vec![scales.len()], TensorInit::CopyFrom(&scales))?;
    let ids_t = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![tokens], TensorInit::CopyFrom(&ids))?;
    let out_t = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![tokens * d_model], TensorInit::Uninitialized)?;

    bindings.insert("embedding".to_string(), TensorArg::from_tensor(&table_t));
    bindings.insert("embedding_scales".to_string(), TensorArg::from_tensor(&scales_t));
    bindings.insert("input_ids".to_string(), TensorArg::from_tensor(&ids_t));
    bindings.insert("out".to_string(), TensorArg::from_tensor(&out_t));

    let step = EmbeddingStep {
        table: "embedding".into(),
        indices: "input_ids".into(),
        output: "out".into(),
        params: metallic_foundry::metals::embedding::EmbeddingParams {
            d_model: DynamicValue::Literal(d_model as u32),
            total_elements: DynamicValue::Literal((tokens * d_model) as u32),
            vocab_size: DynamicValue::Literal(vocab as u32),
        },
    };

    step.execute(&mut foundry, &mut bindings)?;
    foundry.synchronize()?;

    let out = out_t.to_vec(&foundry);
    for pos in 0..tokens {
        let tok = ids[pos] as usize;
        let scale = if tok == 0 { 0.5f32 } else { 1.0f32 };
        for feat in 0..d_model {
            let got = out[pos * d_model + feat].to_f32();
            let expected = (if tok == 0 { feat as f32 } else { (2 * feat) as f32 }) * scale;
            let diff = (got - expected).abs();
            assert!(diff < 0.01, "pos={} feat={} got={} expected={}", pos, feat, got, expected);
        }
    }

    Ok(())
}
