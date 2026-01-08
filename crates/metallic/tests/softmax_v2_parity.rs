use half::f16;
use metallic::{
    foundry::{
        Foundry, spec::{DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::Pooled, tensor::Tensor as FoundryTensor
    },
    metals::softmax::SoftmaxV2Step, // V2 moved to main
    tensor::{TensorInit, dtypes::F16 as F16Dtype},
    types::TensorArg,
};

// CPU Reference implementation
fn cpu_softmax(input: &[f16], rows: usize, cols: usize) -> Vec<f16> {
    let mut output = vec![f16::ZERO; input.len()];

    for r in 0..rows {
        let row_offset = r * cols;
        let row_slice = &input[row_offset..row_offset + cols];

        let max_val = row_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.to_f32()));

        let mut sum_exp = 0.0f32;
        let mut exps = Vec::with_capacity(cols);

        for &val in row_slice {
            let e = (val.to_f32() - max_val).exp();
            sum_exp += e;
            exps.push(e);
        }

        for c in 0..cols {
            output[row_offset + c] = f16::from_f32(exps[c] / sum_exp);
        }
    }
    output
}

fn run_softmax_v2_test(rows: usize, seq_k: usize) {
    let mut foundry = Foundry::new().unwrap();

    // 1. Create Data
    let input_f32: Vec<f32> = (0..(rows * seq_k)).map(|i| (i % 50) as f32 * 0.1 - 2.5).collect();
    let input_data: Vec<f16> = input_f32.iter().map(|&x| f16::from_f32(x)).collect();

    // 2. Create Tensors
    let input = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![rows, seq_k], TensorInit::CopyFrom(&input_data)).unwrap();
    let output_v2 = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![rows, seq_k], TensorInit::Uninitialized).unwrap();
    let scale = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(&[f16::ONE])).unwrap();

    // 3. Run V2
    let mut bindings = TensorBindings::new();
    let mut symbols = SymbolTable::new();

    bindings.insert("input".to_string(), TensorArg::from_tensor(&input));
    bindings.insert("output".to_string(), TensorArg::from_tensor(&output_v2));
    bindings.insert("scale".to_string(), TensorArg::from_tensor(&scale));

    let step = SoftmaxV2Step {
        input: Ref("input".to_string()),
        output: Ref("output".to_string()),
        scale: Ref("scale".to_string()),
        causal: false,
        query_offset: DynamicValue::Literal(0),
    };

    let compiled_steps = step.compile(&mut bindings, &mut symbols);
    let mut fast_bindings = FastBindings::new(symbols.len());
    for (name, id) in symbols.iter() {
        if let Ok(arg) = bindings.get(name) {
            fast_bindings.set(*id, arg);
        }
    }

    for c_step in compiled_steps {
        c_step.execute(&mut foundry, &fast_bindings, &bindings).unwrap();
    }

    // 4. Compare Results with CPU Ref
    let res_v2 = FoundryTensor::to_vec(&output_v2, &foundry);
    let res_cpu = cpu_softmax(&input_data, rows, seq_k);

    let mut max_diff = 0.0f32;
    for i in 0..res_cpu.len() {
        let diff = (res_cpu[i].to_f32() - res_v2[i].to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Rows={} K={} -> Max Diff vs CPU: {}", rows, seq_k, max_diff);
    // Softmax can have slight precision diffs due to fp16/fp32 mixing in Metal vs CPU
    assert!(max_diff < 0.01, "Softmax V2 vs CPU mismatch (K={})", seq_k);
}

#[test]
fn test_softmax_v2_vec() {
    run_softmax_v2_test(4, 256);
}

#[test]
fn test_softmax_v2_block() {
    run_softmax_v2_test(1, 4096);
}
