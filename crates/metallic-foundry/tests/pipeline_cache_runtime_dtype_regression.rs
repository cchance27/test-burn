use half::f16;
use metallic_foundry::{
    Foundry, metals::elemwise::{ElemwiseAdd, ElemwiseAddParamsResolved}, storage::Pooled, tensor::{F16, F32, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use serial_test::serial;

#[test]
#[serial]
fn test_pipeline_cache_runtime_dtype_specialization_no_collision_elemwise_add() -> Result<(), Box<dyn std::error::Error>> {
    let mut foundry = Foundry::new()?;

    let total_elements = 128usize;
    let b_len = 16usize;
    let params = ElemwiseAddParamsResolved {
        total_elements: total_elements as u32,
        b_len: b_len as u32,
    };

    // First run compiles and caches the F16-specialized pipeline.
    let a16_data: Vec<f16> = (0..total_elements).map(|i| f16::from_f32((i as f32) * 0.03125 - 1.5)).collect();
    let b16_data: Vec<f16> = (0..b_len).map(|i| f16::from_f32((i as f32) * 0.125 - 0.5)).collect();
    let a16 = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![total_elements], TensorInit::CopyFrom(&a16_data))?;
    let b16 = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![b_len], TensorInit::CopyFrom(&b16_data))?;
    let out16 = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![total_elements], TensorInit::Uninitialized)?;

    let add16 = ElemwiseAdd::new(
        &TensorArg::from_tensor(&a16),
        &TensorArg::from_tensor(&b16),
        &TensorArg::from_tensor(&out16),
        params.clone(),
    );
    foundry.run(&add16)?;

    // Second run uses the same kernel function with F32 bindings.
    // Before the cache-key fix this could incorrectly reuse the F16 pipeline.
    let a32_data: Vec<f32> = (0..total_elements).map(|i| ((i as f32) * 0.01731) - 0.73).collect();
    let b32_data: Vec<f32> = (0..b_len).map(|i| ((i as f32) * -0.04321) + 0.91).collect();
    let expected: Vec<f32> = (0..total_elements).map(|i| a32_data[i] + b32_data[i % b_len]).collect();

    let a32 = FoundryTensor::<F32, Pooled>::new(&mut foundry, vec![total_elements], TensorInit::CopyFrom(&a32_data))?;
    let b32 = FoundryTensor::<F32, Pooled>::new(&mut foundry, vec![b_len], TensorInit::CopyFrom(&b32_data))?;
    let out32 = FoundryTensor::<F32, Pooled>::new(&mut foundry, vec![total_elements], TensorInit::Uninitialized)?;

    let add32 = ElemwiseAdd::new(
        &TensorArg::from_tensor(&a32),
        &TensorArg::from_tensor(&b32),
        &TensorArg::from_tensor(&out32),
        params,
    );
    foundry.run(&add32)?;

    let got = FoundryTensor::to_vec(&out32, &foundry);
    for i in 0..total_elements {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 1e-5,
            "F32 run mismatch at index {}: got={}, expected={}, diff={}",
            i,
            got[i],
            expected[i],
            diff
        );
    }

    Ok(())
}
