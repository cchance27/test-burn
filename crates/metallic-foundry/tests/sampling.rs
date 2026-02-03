//! Test suite for SampleTopKTopP kernel.

use half::f16;
use metallic_foundry::{
    Foundry, metals::sampling::SampleTopK, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit, dtypes::U32}, types::TensorArg
};
use serial_test::serial;

#[test]
#[serial]
fn test_sample_topk_foundry_argmax() {
    let mut foundry = Foundry::new().unwrap();

    let vocab_size = 32000;
    let k = 1;
    let top_p = 1.0;
    let temperature = 1.0;
    let seed = 42;

    // Argmax check: 100.0 vs 0.0
    let mut logits_data: Vec<f16> = vec![f16::from_f32(0.0); vocab_size];
    let expected_token = 12345;
    logits_data[expected_token] = f16::from_f32(100.0);

    let logits_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![vocab_size], TensorInit::CopyFrom(&logits_data)).unwrap();

    let output_foundry = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![1], TensorInit::Uninitialized).unwrap();

    let logits_arg = TensorArg::from_tensor(&logits_foundry);
    let output_arg = TensorArg::from_tensor(&output_foundry);

    let min_p = 0.0;
    let kernel = SampleTopK::new(&logits_arg, &output_arg, vocab_size as u32, k, top_p, min_p, temperature, seed);

    foundry.run(&kernel).unwrap();

    let foundry_token = FoundryTensor::to_vec(&output_foundry, &foundry)[0];
    println!("Foundry Token: {}, Expected: {}", foundry_token, expected_token);
    assert_eq!(foundry_token, expected_token as u32, "Foundry failed Argmax");
}

#[test]
#[serial]
fn test_sample_topk_fused_determinism() {
    let mut foundry = Foundry::new().unwrap();

    let vocab_size = 10000;
    let k = 40;
    let top_p = 0.95;
    let min_p = 0.0;
    let temperature = 1.0;
    let seed = 12345;

    let logits_data: Vec<f16> = (0..vocab_size).map(|i| f16::from_f32((i % 100) as f32)).collect();

    let logits_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![vocab_size], TensorInit::CopyFrom(&logits_data)).unwrap();

    let output_foundry_1 = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![1], TensorInit::Uninitialized).unwrap();

    let output_foundry_2 = FoundryTensor::<U32, Pooled>::new(&mut foundry, vec![1], TensorInit::Uninitialized).unwrap();

    let kernel1 = SampleTopK::new(
        &TensorArg::from_tensor(&logits_foundry),
        &TensorArg::from_tensor(&output_foundry_1),
        vocab_size as u32,
        k,
        top_p,
        min_p,
        temperature,
        seed,
    );
    foundry.run(&kernel1).unwrap();

    let kernel2 = SampleTopK::new(
        &TensorArg::from_tensor(&logits_foundry),
        &TensorArg::from_tensor(&output_foundry_2),
        vocab_size as u32,
        k,
        top_p,
        min_p,
        temperature,
        seed,
    );
    foundry.run(&kernel2).unwrap();

    let token1 = FoundryTensor::to_vec(&output_foundry_1, &foundry)[0];
    let token2 = FoundryTensor::to_vec(&output_foundry_2, &foundry)[0];

    assert_eq!(token1, token2, "Determinism failed");
}
