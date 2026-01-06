use half::f16;
use metallic::{
    foundry::{
        Foundry, spec::{DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings, compiled::CompiledStep}, storage::Pooled, tensor::Tensor as FoundryTensor
    }, metals::{
        softmax::SoftmaxVec,              // Legacy
        v2::softmax::step::SoftmaxV2Step, // V2
    }, tensor::{TensorInit, dtypes::F16 as F16Dtype}, types::TensorArg
};

#[test]
fn test_softmax_v2_parity_f16() {
    let mut foundry = Foundry::new().unwrap();

    let rows = 4;
    let seq_k = 256;
    let seq_q = 4;

    // 1. Create Data
    let input_f32: Vec<f32> = (0..(rows * seq_k)).map(|i| (i % 50) as f32 * 0.1 - 2.5).collect();
    let input_data: Vec<f16> = input_f32.iter().map(|&x| f16::from_f32(x)).collect();

    // 2. Create Tensors
    let input = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![rows, seq_k], TensorInit::CopyFrom(&input_data)).unwrap();
    let output_v2 = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![rows, seq_k], TensorInit::Uninitialized).unwrap();
    let output_legacy = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![rows, seq_k], TensorInit::Uninitialized).unwrap();
    let scale = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(&[f16::ONE])).unwrap();

    // 3. Run Legacy
    let legacy_kernel = SoftmaxVec::new(
        &TensorArg::from_tensor(&input),
        &TensorArg::from_tensor(&output_legacy),
        rows as u32,
        seq_q as u32,
        seq_k as u32,
        false, // causal
        0,     // query_offset
    );
    foundry.run(&legacy_kernel).unwrap();

    // 4. Run V2
    // Setup compilation environment
    let mut bindings = TensorBindings::new();
    let mut symbols = SymbolTable::new();

    // Manually register symbols and bindings
    // V2 step uses Refs "input", "output", "scale"
    bindings.insert("input".to_string(), TensorArg::from_tensor(&input));
    bindings.insert("output".to_string(), TensorArg::from_tensor(&output_v2));
    bindings.insert("scale".to_string(), TensorArg::from_tensor(&scale));

    // FastBindings (needed for execution)
    // We must manually interpolate Ref names to fast indices
    // The `compile` step does: symbol_table.get_or_create(...)
    // So we should run compile first.

    let step = SoftmaxV2Step {
        input: Ref("input".to_string()),
        output: Ref("output".to_string()),
        scale: Ref("scale".to_string()),
        causal: false,
        query_offset: DynamicValue::Literal(0),
    };

    let compiled_steps = step.compile(&mut bindings, &mut symbols);

    // Now create FastBindings based on symbols
    let mut fast_bindings = FastBindings::new(symbols.len());

    // Populate fast bindings manually (Step::compile creates indices, but Executor::execute populates tensor values)
    // We mimic Executor behavior here:
    for (name, id) in symbols.iter() {
        if let Ok(arg) = bindings.get(name) {
            fast_bindings.set(*id, arg);
        }
    }

    // Execute V2
    for c_step in compiled_steps {
        c_step.execute(&mut foundry, &fast_bindings, &bindings).unwrap();
    }

    // 5. Compare Results
    let res_legacy = FoundryTensor::to_vec(&output_legacy, &foundry);
    let res_v2 = FoundryTensor::to_vec(&output_v2, &foundry);

    let mut max_diff = 0.0f32;
    for i in 0..res_legacy.len() {
        let diff = (res_legacy[i].to_f32() - res_v2[i].to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Max Diff V2 vs Legacy: {}", max_diff);
    assert!(max_diff < 0.001, "Softmax V2 vs Legacy parity mismatch");
}
