//! Test suite for utility kernels: ElemwiseAdd, Arange, Ones, RandomUniform.

use half::f16;
use metallic::{
    Context, F16Element, foundry::{Foundry, storage::Pooled, tensor::Tensor as FoundryTensor}, kernels::{
        elemwise_add::BroadcastElemwiseAddInplaceOp, tensors::{ArangeOp, OnesOp, RandomUniformOp}
    }, metals::{
        elemwise::{ElemwiseAdd, ElemwiseAddParams}, tensor::{Arange, Ones, RandomUniform}
    }, tensor::{F16, Tensor, TensorInit, TensorStorage as LegacyStorage}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

const TOLERANCE: f32 = 1e-3; // Small tolerance for f16

// ============================================================================
// ElemwiseAdd Tests
// ============================================================================

fn cpu_broadcast_add(a: &[f16], b: &[f16]) -> Vec<f16> {
    let b_len = b.len();
    a.iter()
        .enumerate()
        .map(|(i, &av)| f16::from_f32(av.to_f32() + b[i % b_len].to_f32()))
        .collect()
}

#[test]
#[serial]
fn test_elemwise_add_broadcast() {
    let mut ctx = Context::<F16Element>::new().unwrap();
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rng();

    let rows = 8;
    let cols = 64;
    let total = rows * cols;

    let a_data: Vec<f16> = (0..total).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();
    let b_data: Vec<f16> = (0..cols).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();

    // Legacy
    let a_legacy = Tensor::<F16>::new(vec![rows, cols], LegacyStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&a_data)).unwrap();
    let b_legacy = Tensor::<F16>::new(vec![cols], LegacyStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&b_data)).unwrap();
    let _ = ctx
        .call::<BroadcastElemwiseAddInplaceOp>((a_legacy.clone(), b_legacy), None)
        .unwrap();
    let legacy_result = a_legacy.to_vec();

    // Foundry
    let a_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![total], TensorInit::CopyFrom(&a_data)).unwrap();
    let b_foundry = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cols], TensorInit::CopyFrom(&b_data)).unwrap();

    let a_arg = TensorArg::from_tensor(&a_foundry);
    let b_arg = TensorArg::from_tensor(&b_foundry);
    let params = ElemwiseAddParams {
        total_elements: total as u32,
        b_len: cols as u32,
    };
    let kernel = ElemwiseAdd::new_inplace(&a_arg, &b_arg, params.total_elements, params.b_len);
    foundry.run(&kernel).unwrap();
    let foundry_result = FoundryTensor::to_vec(&a_foundry, &foundry);

    // CPU
    let cpu_result = cpu_broadcast_add(&a_data, &b_data);

    let mut max_diff_legacy = 0.0f32;
    let mut max_diff_cpu = 0.0f32;
    for i in 0..total {
        max_diff_legacy = max_diff_legacy.max((legacy_result[i].to_f32() - foundry_result[i].to_f32()).abs());
        max_diff_cpu = max_diff_cpu.max((cpu_result[i].to_f32() - foundry_result[i].to_f32()).abs());
    }

    println!(
        "[ElemwiseAdd] Legacy vs Foundry: {:.6}, CPU vs Foundry: {:.6}",
        max_diff_legacy, max_diff_cpu
    );
    assert!(max_diff_legacy <= TOLERANCE && max_diff_cpu <= TOLERANCE);
}

// ============================================================================
// Arange Tests
// ============================================================================

fn cpu_arange(length: usize) -> Vec<f16> {
    (0..length).map(|i| f16::from_f32(i as f32)).collect()
}

#[test]
#[serial]
fn test_arange() {
    let mut ctx = Context::<F16Element>::new().unwrap();
    let mut foundry = Foundry::new().unwrap();

    let length = 128;

    // Legacy
    let legacy_result = ctx.call::<ArangeOp>(length, None).unwrap().to_vec();

    // Foundry
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![length], TensorInit::Uninitialized).unwrap();
    let output_arg = TensorArg::from_tensor(&output);
    let kernel = Arange::new(&output_arg, length);
    foundry.run(&kernel).unwrap();
    let foundry_result = FoundryTensor::to_vec(&output, &foundry);

    // CPU
    let cpu_result = cpu_arange(length);

    let mut max_diff_legacy = 0.0f32;
    let mut max_diff_cpu = 0.0f32;
    for i in 0..length {
        max_diff_legacy = max_diff_legacy.max((legacy_result[i].to_f32() - foundry_result[i].to_f32()).abs());
        max_diff_cpu = max_diff_cpu.max((cpu_result[i].to_f32() - foundry_result[i].to_f32()).abs());
    }

    println!(
        "[Arange] Legacy vs Foundry: {:.6}, CPU vs Foundry: {:.6}",
        max_diff_legacy, max_diff_cpu
    );
    assert!(max_diff_legacy <= TOLERANCE && max_diff_cpu <= TOLERANCE);
}

// ============================================================================
// Ones Tests
// ============================================================================

fn cpu_ones(length: usize) -> Vec<f16> {
    vec![f16::from_f32(1.0); length]
}

#[test]
#[serial]
fn test_ones() {
    let mut ctx = Context::<F16Element>::new().unwrap();
    let mut foundry = Foundry::new().unwrap();

    let length = 256;

    // Legacy
    let legacy_result = ctx.call::<OnesOp>(vec![length], None).unwrap().to_vec();

    // Foundry
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![length], TensorInit::Uninitialized).unwrap();
    let output_arg = TensorArg::from_tensor(&output);
    let kernel = Ones::new(&output_arg, length as u32);
    foundry.run(&kernel).unwrap();
    let foundry_result = FoundryTensor::to_vec(&output, &foundry);

    // CPU
    let cpu_result = cpu_ones(length);

    let mut max_diff_legacy = 0.0f32;
    let mut max_diff_cpu = 0.0f32;
    for i in 0..length {
        max_diff_legacy = max_diff_legacy.max((legacy_result[i].to_f32() - foundry_result[i].to_f32()).abs());
        max_diff_cpu = max_diff_cpu.max((cpu_result[i].to_f32() - foundry_result[i].to_f32()).abs());
    }

    println!(
        "[Ones] Legacy vs Foundry: {:.6}, CPU vs Foundry: {:.6}",
        max_diff_legacy, max_diff_cpu
    );
    assert!(max_diff_legacy <= TOLERANCE && max_diff_cpu <= TOLERANCE);
}

// ============================================================================
// RandomUniform Tests
// ============================================================================

#[test]
#[serial]
fn test_random_uniform() {
    let mut ctx = Context::<F16Element>::new().unwrap();
    let mut foundry = Foundry::new().unwrap();

    let length = 512;
    let min_val = 0.0f32;
    let max_val = 1.0f32;
    let seed = 42u32;

    // Legacy
    let legacy_result = ctx
        .call::<RandomUniformOp>((vec![length], min_val, max_val, Some(seed)), None)
        .unwrap()
        .to_vec();

    // Foundry
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![length], TensorInit::Uninitialized).unwrap();
    let output_arg = TensorArg::from_tensor(&output);
    let kernel = RandomUniform::new(&output_arg, length, seed, min_val, max_val);
    foundry.run(&kernel).unwrap();
    let foundry_result = FoundryTensor::to_vec(&output, &foundry);

    // Both should produce same values with same seed
    let mut max_diff = 0.0f32;
    for i in 0..length {
        max_diff = max_diff.max((legacy_result[i].to_f32() - foundry_result[i].to_f32()).abs());
        // Check range
        assert!(
            (min_val..=max_val).contains(&foundry_result[i].to_f32()),
            "Value {} out of range",
            foundry_result[i].to_f32()
        );
    }

    println!("[RandomUniform] Legacy vs Foundry: {:.6}", max_diff);
    assert!(max_diff <= TOLERANCE);
}
