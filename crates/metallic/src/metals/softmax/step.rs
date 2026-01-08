use std::sync::OnceLock;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::stages::{SoftmaxMaxStage, SoftmaxNormStage, SoftmaxSumStage};
use crate::{
    MetalError, compound::{CompiledCompoundKernel, CompoundKernel, stages::LayoutStage}, foundry::{
        Foundry, spec::{CompiledStep, DynamicValue, Ref, Step, SymbolTable, TensorBindings}
    }, types::{DispatchConfig, GridSize, KernelArg, TensorArg, ThreadgroupSize}
};

#[derive(Debug, Serialize, Deserialize)]
pub struct SoftmaxV2Step {
    pub input: Ref,
    pub output: Ref,
    #[serde(default)]
    pub scale: Ref,
    pub causal: bool,
    #[serde(default)]
    pub query_offset: DynamicValue<u32>,
}

// Static cache for the kernel template
pub fn get_softmax_v2_kernel() -> &'static CompiledCompoundKernel {
    use crate::compound::stages::SimdStage;

    static KERNEL: OnceLock<CompiledCompoundKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        CompoundKernel::new("softmax_v2_fused")
            // Layout: emit row/col indices
            .prologue(LayoutStage::row_major())
            // Phase 1: Compute local max, then reduce
            .prologue(SoftmaxMaxStage::new("matrix"))
            .prologue(SimdStage::reduce_max("local_max", "row_max"))
            // Phase 2: Compute local sum, then reduce
            .prologue(SoftmaxSumStage::new("row_max"))
            .prologue(SimdStage::reduce_sum("local_sum", "row_sum"))
            // Phase 3: Normalize and write
            .main(SoftmaxNormStage::new("row_max", "row_sum"))
            .with_manual_output(true)
            .compile()
    })
}

#[derive(Debug, Clone)]
pub struct CompiledSoftmaxV2Step {
    pub input_idx: usize,
    pub output_idx: usize,
    pub scale_idx: usize,
    pub causal: bool,
    pub query_offset: DynamicValue<u32>,
    pub seq_k: DynamicValue<u32>,
}

/// Arguments for SoftmaxV2 kernel dispatch.
///
/// Uses `#[derive(KernelArgs)]` to auto-generate safe binding code.
#[derive(Debug, KernelArgs)]
pub struct SoftmaxV2Args {
    #[arg(buffer = 0)]
    pub input: TensorArg,
    #[arg(buffer = 1)]
    pub scale: TensorArg,
    #[arg(buffer = 2, output)]
    pub output: TensorArg,
    #[arg(buffer = 3)]
    pub seq_k: u32,
    #[arg(buffer = 4)]
    pub causal: u32,
    #[arg(buffer = 5)]
    pub query_offset: u32,
}

#[typetag::serde(name = "SoftmaxV2")]
impl Step for SoftmaxV2Step {
    fn name(&self) -> &'static str {
        "SoftmaxV2"
    }

    fn execute(&self, _foundry: &mut Foundry, _bindings: &mut TensorBindings) -> Result<(), MetalError> {
        Err(MetalError::OperationNotSupported("SoftmaxV2 only supports compile()".into()))
    }

    fn compile(&self, resolver: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let input_idx = symbols.get_or_create(resolver.interpolate(self.input.0.clone()));
        let output_idx = symbols.get_or_create(resolver.interpolate(self.output.0.clone()));
        let scale_idx = symbols.get_or_create(resolver.interpolate(self.scale.0.clone()));

        get_softmax_v2_kernel();

        vec![Box::new(CompiledSoftmaxV2Step {
            input_idx,
            output_idx,
            scale_idx,
            causal: self.causal,
            query_offset: self.query_offset.clone(),
            seq_k: DynamicValue::Literal(0),
        })]
    }
}

impl CompiledStep for CompiledSoftmaxV2Step {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &crate::foundry::spec::FastBindings,
        bindings: &TensorBindings,
    ) -> Result<(), MetalError> {
        let input = fast_bindings
            .get(self.input_idx)
            .ok_or_else(|| MetalError::InputNotFound(format!("Input {}", self.input_idx)))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or_else(|| MetalError::InputNotFound(format!("Output {}", self.output_idx)))?;
        let scale = fast_bindings
            .get(self.scale_idx)
            .ok_or_else(|| MetalError::InputNotFound(format!("Scale {}", self.scale_idx)))?;

        let dims = input.dims();
        let batch = dims[0] as u32;
        let seq_q = dims[1] as u32;
        let seq_k = dims.last().copied().unwrap_or(1) as u32;

        let q_offset = self.query_offset.resolve(bindings);

        let args = SoftmaxV2Args {
            input: TensorArg::from_tensor(input),
            scale: TensorArg::from_tensor(scale),
            output: TensorArg::from_tensor(output),
            seq_k,
            causal: if self.causal { 1 } else { 0 },
            query_offset: q_offset,
        };

        let dispatch = DispatchConfig {
            grid: GridSize::d2(batch as usize, seq_q as usize),
            group: ThreadgroupSize::d1(256),
        };

        let bound_kernel = get_softmax_v2_kernel().bind(args, dispatch);

        foundry.run(&bound_kernel)
    }

    fn name(&self) -> &'static str {
        "SoftmaxV2"
    }
}
