//! Softmax V2 vs Context<T> Parity Test
//!
//! Compares the new V2 Stage-based Softmax implementation against
//! Context<T>'s SoftmaxKernelOp to ensure numerical consistency.

use half::f16;
use metallic::{
    Context, F16Element, MetalError, foundry::{
        Foundry, spec::{DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::Pooled, tensor::Tensor as FoundryTensor
    }, kernels::softmax_kernel::SoftmaxKernelOp, metals::softmax::SoftmaxV2Step, tensor::{Tensor, TensorInit, TensorStorage, dtypes::F16 as F16Dtype}, types::TensorArg
};
use serial_test::serial;

const TOLERANCE: f32 = 0.001; // Tight tolerance for softmax

fn compare_results(v2: &[f16], context: &[f16], name: &str) {
    assert_eq!(v2.len(), context.len(), "Size mismatch in {}", name);

    let mut max_diff = 0.0f32;
    let mut max_idx = 0;

    for (i, (a, b)) in v2.iter().zip(context.iter()).enumerate() {
        let diff = (a.to_f32() - b.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("{}: max_diff={:.6} at idx {}", name, max_diff, max_idx);

    assert!(
        max_diff < TOLERANCE,
        "{} mismatch: max_diff={} at idx {} (V2={}, Ctx={})",
        name,
        max_diff,
        max_idx,
        v2[max_idx].to_f32(),
        context[max_idx].to_f32()
    );
}

fn run_softmax_parity_test(rows: usize, cols: usize, causal: bool) -> Result<(), MetalError> {
    // Generate deterministic test data
    let input_data: Vec<f16> = (0..(rows * cols))
        .map(|i| f16::from_f32((i as f32 - (rows * cols) as f32 / 2.0) / 100.0))
        .collect();

    // --- V2 Softmax (Foundry) ---
    let mut foundry = Foundry::new()?;

    let input_v2 = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![rows, cols], TensorInit::CopyFrom(&input_data))?;
    let output_v2 = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![rows, cols], TensorInit::Uninitialized)?;
    let scale = FoundryTensor::<F16Dtype, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(&[f16::ONE]))?;

    let mut bindings = TensorBindings::new();
    let mut symbols = SymbolTable::new();

    bindings.insert("input".to_string(), TensorArg::from_tensor(&input_v2));
    bindings.insert("output".to_string(), TensorArg::from_tensor(&output_v2));
    bindings.insert("scale".to_string(), TensorArg::from_tensor(&scale));

    let step = SoftmaxV2Step {
        input: Ref("input".to_string()),
        output: Ref("output".to_string()),
        scale: Ref("scale".to_string()),
        causal,
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
        c_step.execute(&mut foundry, &fast_bindings, &bindings, &symbols)?;
    }

    let res_v2 = output_v2.to_vec(&foundry);

    // --- Context Softmax ---
    let mut ctx = Context::<F16Element>::new()?;

    let input_ctx = Tensor::new(
        vec![1, rows, cols], // SoftmaxKernelOp expects 3D: [batch, rows, cols]
        TensorStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&input_data),
    )?;

    let output_ctx = ctx.call::<SoftmaxKernelOp>(
        (
            &input_ctx,
            rows as u32, // rows_total
            rows as u32, // seq_q
            cols as u32, // seq_k
            causal as u32,
            0u32, // query_offset
        ),
        None,
    )?;

    ctx.synchronize();
    let res_ctx = output_ctx.try_to_vec()?;

    // Compare
    let label = format!("Softmax {}x{} causal={}", rows, cols, causal);
    compare_results(&res_v2, &res_ctx, &label);

    Ok(())
}

#[test]
#[serial]
fn test_softmax_v2_context_parity_small() -> Result<(), MetalError> {
    run_softmax_parity_test(4, 64, false)?;
    run_softmax_parity_test(4, 64, true)?;
    Ok(())
}

#[test]
#[serial]
fn test_softmax_v2_context_parity_medium() -> Result<(), MetalError> {
    run_softmax_parity_test(16, 256, false)?;
    run_softmax_parity_test(16, 256, true)?;
    Ok(())
}

#[test]
#[serial]
fn test_softmax_v2_context_parity_large() -> Result<(), MetalError> {
    run_softmax_parity_test(64, 1024, false)?;
    Ok(())
}
