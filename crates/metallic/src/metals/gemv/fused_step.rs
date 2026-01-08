use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::{
    stages::{VectorizedDotStage, WarpWriteOutputStage}, step::{GemvStrategy, warp_dispatch_config}
};
use crate::{
    MetalError, compound::{
        CompiledCompoundKernel, CompoundKernel, stages::{Layout, Quantization, WarpLayoutStage, WarpReduceStage}
    }, foundry::{
        Foundry, spec::{CompiledStep, DynamicValue, Ref, Step, SymbolTable, TensorBindings}
    }, metals::rmsnorm::stages::RmsNormComputeStage, types::TensorArg
};

/// Fused GEMV Step: RMSNorm(Input) -> GEMV(Input_Norm, Weights) -> Output
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FusedGemvStep {
    pub input: Ref,
    pub weights: Ref,
    pub scales: Ref,
    pub output: Ref,
    pub gamma: Ref, // RMSNorm gamma
    pub bias: Option<Ref>,
    pub k_dim: DynamicValue<u32>,
    pub n_dim: DynamicValue<u32>,
    pub weights_per_block: DynamicValue<u32>,
    pub strategy: GemvStrategy,
}

#[derive(Debug, Clone)]
pub struct CompiledFusedGemvStep {
    pub step: FusedGemvStep,
    pub input_idx: usize,
    pub weights_idx: usize,
    pub scales_idx: usize,
    pub output_idx: usize,
    pub gamma_idx: usize,
    pub bias_idx: Option<usize>,
}

#[derive(Debug, KernelArgs)]
pub struct FusedGemvArgs {
    #[arg(buffer = 0)]
    pub weights: TensorArg,
    #[arg(buffer = 1)]
    pub scales: TensorArg,
    #[arg(buffer = 2)]
    pub input: TensorArg,
    #[arg(buffer = 3)]
    pub output: TensorArg,
    #[arg(buffer = 4)]
    pub k_dim: u32,
    #[arg(buffer = 5)]
    pub n_dim: u32,
    #[arg(buffer = 6)]
    pub weights_per_block: u32,
    #[arg(buffer = 7)]
    pub bias: TensorArg,
    #[arg(buffer = 8)]
    pub has_bias: u32,
    #[arg(buffer = 9)]
    pub alpha: f32,
    #[arg(buffer = 10)]
    pub gamma: TensorArg,
}

#[typetag::serde(name = "FusedGemv")]
impl Step for FusedGemvStep {
    fn name(&self) -> &'static str {
        "FusedGemv"
    }

    fn execute(&self, _foundry: &mut Foundry, _bindings: &mut TensorBindings) -> Result<(), MetalError> {
        Err(MetalError::OperationNotSupported("FusedGemv only supports compile()".into()))
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let input_idx = symbols.get_or_create(bindings.interpolate(self.input.0.clone()));
        let weights_idx = symbols.get_or_create(bindings.interpolate(self.weights.0.clone()));
        let scales_idx = symbols.get_or_create(bindings.interpolate(self.scales.0.clone()));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));
        let gamma_idx = symbols.get_or_create(bindings.interpolate(self.gamma.0.clone()));
        let bias_idx = self.bias.as_ref().map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));

        vec![Box::new(CompiledFusedGemvStep {
            step: self.clone(),
            input_idx,
            weights_idx,
            scales_idx,
            output_idx,
            gamma_idx,
            bias_idx,
        })]
    }
}

impl CompiledStep for CompiledFusedGemvStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &crate::foundry::spec::FastBindings,
        bindings: &TensorBindings,
    ) -> Result<(), MetalError> {
        let input = fast_bindings.get(self.input_idx).ok_or(MetalError::InputNotFound("input".into()))?;
        let weights = fast_bindings
            .get(self.weights_idx)
            .ok_or(MetalError::InputNotFound("weights".into()))?;
        let scales = fast_bindings
            .get(self.scales_idx)
            .ok_or(MetalError::InputNotFound("scales".into()))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or(MetalError::InputNotFound("output".into()))?;
        let gamma = fast_bindings.get(self.gamma_idx).ok_or(MetalError::InputNotFound("gamma".into()))?;

        let (bias, has_bias) = if let Some(idx) = self.bias_idx {
            let b = fast_bindings.get(idx).ok_or(MetalError::InputNotFound("bias".into()))?;
            (TensorArg::from_tensor(b), 1)
        } else {
            (TensorArg::from_tensor(output), 0)
        };

        let k_dim = self.step.k_dim.resolve(bindings);
        let n_dim = self.step.n_dim.resolve(bindings);
        let weights_per_block = self.step.weights_per_block.resolve(bindings);

        let args = FusedGemvArgs {
            weights: TensorArg::from_tensor(weights),
            scales: TensorArg::from_tensor(scales),
            input: TensorArg::from_tensor(input),
            output: TensorArg::from_tensor(output),
            k_dim,
            n_dim,
            weights_per_block,
            bias,
            has_bias,
            alpha: 1.0,
            gamma: TensorArg::from_tensor(gamma),
        };

        let kernel = get_fused_gemv_kernel(self.step.strategy);
        let dispatch = warp_dispatch_config(n_dim);

        foundry.run(&kernel.bind(args, dispatch))
    }

    fn name(&self) -> &'static str {
        "FusedGemv"
    }
}

fn get_fused_gemv_kernel(_strategy: GemvStrategy) -> &'static CompiledCompoundKernel {
    Box::leak(Box::new(
        CompoundKernel::new("fused_gemv_rmsnorm_v2")
            .with_manual_output(true)
            .prologue(WarpLayoutStage::new(Layout::RowMajor)) // Defines row_idx, lane_id
            .prologue(RmsNormComputeStage::new(2, 4))
            .main(VectorizedDotStage::new(Quantization::Q8).with_norm(10, "inv_rms"))
            .epilogue(WarpReduceStage::sum("partial_dot", "row_sum"))
            .epilogue(WarpWriteOutputStage::new())
            .compile(),
    ))
}
