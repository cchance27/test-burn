use std::sync::Arc;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::{
    stages::{VectorizedDotStage, WarpWriteOutputNoResidualStage}, step::GemvStrategy
};
use crate::{
    Foundry, MetalError, compound::{CompiledCompoundKernel, stages::WarpReduceStage}, fusion::MetalPolicy, metals::{
        common::{cache::get_or_build_policy_compound_kernel, composition::manual_output_row_major}, rmsnorm::stages::RmsNormComputeStage
    }, spec::{CompiledStep, DynamicValue, Ref, ResolvedSymbols, Step, SymbolTable, TensorBindings}, types::{DispatchConfig, TensorArg}
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
    pub weights_resolved: ResolvedSymbols,
    pub input_idx: usize,
    pub output_idx: usize,
    pub gamma_idx: usize,
    pub bias_idx: Option<usize>,
}

#[derive(Debug, KernelArgs)]
pub struct FusedGemvArgs {
    #[arg(buffer = 0, metal_type = "const device uchar*")]
    pub weights: TensorArg,
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    pub scales: Option<TensorArg>,
    #[arg(buffer = 2, metal_type = "const device InputStorageT*")]
    pub input: TensorArg,
    #[arg(buffer = 3, output, metal_type = "device OutputStorageT*")]
    pub output: TensorArg,
    #[arg(buffer = 4)]
    pub k_dim: u32,
    #[arg(buffer = 5)]
    pub n_dim: u32,
    #[arg(buffer = 6)]
    pub weights_per_block: u32,
    #[arg(buffer = 7, metal_type = "const device BiasStorageT*")]
    pub bias: TensorArg,
    #[arg(buffer = 8)]
    pub has_bias: u32,
    #[arg(buffer = 9)]
    pub alpha: f32,
    #[arg(buffer = 10, metal_type = "const device GammaStorageT*")]
    pub gamma: TensorArg,
    #[arg(buffer = 11)]
    pub epsilon: f32,
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
        let weights_name = bindings.interpolate(self.weights.0.clone());
        let weights_idx = symbols.get_or_create(weights_name.clone());
        let weights_scales_idx = symbols.get_or_create(format!("{weights_name}_scales"));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));
        let gamma_idx = symbols.get_or_create(bindings.interpolate(self.gamma.0.clone()));
        let bias_idx = self.bias.as_ref().map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));

        vec![Box::new(CompiledFusedGemvStep {
            step: self.clone(),
            weights_resolved: crate::spec::ResolvedSymbols {
                weights: weights_idx,
                scales: weights_scales_idx.into(),
                bias: None,
            },
            input_idx,
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
        fast_bindings: &crate::spec::FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let input = fast_bindings.get(self.input_idx).ok_or(MetalError::InputNotFound("input".into()))?;
        let weights_tensor = fast_bindings
            .get(self.weights_resolved.weights)
            .ok_or(MetalError::InputNotFound("weights".into()))?;

        // Centralized Quantization Binding
        let policy = crate::policy::resolve_policy(weights_tensor.dtype);
        let loader = policy.loader_stage();

        let weights_args = loader.bind(fast_bindings, &self.weights_resolved);

        let weights = weights_args[0].clone();
        let scales = if weights_args.len() > 1 {
            Some(weights_args[1].clone())
        } else {
            None
        };
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
        let weights_per_block = if policy.has_scale() {
            policy.meta().weights_per_block as u32
        } else {
            self.step.weights_per_block.resolve(bindings)
        };

        // Fast fused path is F16-only by design.
        if !matches!(input.dtype, crate::tensor::Dtype::F16) {
            return Err(MetalError::OperationNotSupported(format!(
                "FusedGemv supports only F16 input dtype; got {:?}. Use explicit RMSNorm + GEMV steps for non-F16 paths.",
                input.dtype
            )));
        }

        let args = FusedGemvArgs {
            weights,
            scales,
            input: TensorArg::from_tensor(input),
            output: TensorArg::from_tensor(output),
            k_dim,
            n_dim,
            weights_per_block,
            bias,
            has_bias,
            alpha: 1.0,
            gamma: TensorArg::from_tensor(gamma),
            epsilon: bindings.get_f32_var_or("rms_eps", 1e-6),
        };

        let kernel = get_fused_gemv_kernel(self.step.strategy, policy);
        let dispatch = DispatchConfig::warp_per_row(n_dim, 1);

        if crate::instrument::emit_cb_timing_metrics() {
            // Fused GEMV (matmul + RMSNorm) - still useful to classify as matmul for comparisons.
            let mut data = rustc_hash::FxHashMap::default();
            data.insert("op".to_string(), "matmul".to_string());
            data.insert("backend".to_string(), "gemv".to_string());
            data.insert("batch".to_string(), "1".to_string());
            data.insert("m".to_string(), "1".to_string());
            data.insert("n".to_string(), n_dim.to_string());
            data.insert("k".to_string(), k_dim.to_string());
            data.insert("tA".to_string(), "0".to_string());
            data.insert("tB".to_string(), "1".to_string());
            foundry.run(&kernel.clone().bind_arc_with_metrics(args, dispatch, data))
        } else {
            foundry.run(&kernel.bind_arc(args, dispatch))
        }
    }

    fn name(&self) -> &'static str {
        "FusedGemv"
    }
}

fn get_fused_gemv_kernel(_strategy: GemvStrategy, policy: Arc<dyn MetalPolicy>) -> Arc<CompiledCompoundKernel> {
    get_or_build_policy_compound_kernel("fused_gemv_rmsnorm", policy, move |policy| {
        let kernel_name = format!("fused_gemv_rmsnorm_{}", policy.short_name());

        manual_output_row_major(&kernel_name, 8)
            .prologue(RmsNormComputeStage::new(2, 4, 11))
            .main(VectorizedDotStage::new(policy.clone()).with_norm("inv_rms"))
            .epilogue(WarpReduceStage::sum("partial_dot", "row_sum"))
            .epilogue(WarpWriteOutputNoResidualStage::new())
            .compile()
    })
}
