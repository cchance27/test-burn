use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::{
    Foundry, MetalError, spec::{CompiledStep, FastBindings, SymbolTable, TensorBindings}, types::TensorArg
};

/// A simple, compiled element-wise add step: Out = A + B.
///
/// Unlike `ElemwiseAdd` (which supports broadcasting params), this step
/// runtime-infers dimensions from the input tensors and performs a direct
/// index-to-index addition. It assumes A, B, and Out have the same total element count.
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "elemwise/add.metal",
    function = "broadcast_add_kernel_f16",
    args = SimpleElemwiseAddParams,
    dispatch = per_element,
    step = true,
    execute = false
)]
pub struct SimpleElemwiseAdd {
    pub a: TensorArg,
    pub b: TensorArg,
    #[arg(output)]
    pub out: TensorArg,
    #[arg(skip)]
    pub params: SimpleElemwiseAddParams,
}

#[derive(MetalStruct, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct SimpleElemwiseAddParams {
    pub total_elements: u32,
    pub b_len: u32,
}

impl CompiledStep for CompiledSimpleElemwiseAddStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        _bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let a = fast_bindings.get(self.a).ok_or(MetalError::InputNotFound("Add: a".into()))?;
        let b = fast_bindings.get(self.b).ok_or(MetalError::InputNotFound("Add: b".into()))?;
        let out = fast_bindings.get(self.out).ok_or(MetalError::InputNotFound("Add: out".into()))?;

        let total = a.dims.iter().product::<usize>() as u32;
        let b_len = b.dims.iter().product::<usize>() as u32;

        let kernel = SimpleElemwiseAdd {
            a: TensorArg::from_tensor(a),
            b: TensorArg::from_tensor(b),
            out: TensorArg::from_tensor(out),
            params: SimpleElemwiseAddParams {
                total_elements: total,
                b_len,
            },
        };

        foundry.run(&kernel)?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "SimpleElemwiseAdd"
    }
}
