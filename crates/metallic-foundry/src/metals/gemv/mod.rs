use metallic_macros::{Kernel, KernelArgs, MetalStruct};
use serde::{Deserialize, Serialize};

mod config;
pub mod fused_step;
pub mod stages;
pub mod step;

pub use fused_step::*;
pub use step::{GemvStrategy, GemvV2Args, get_gemv_v2_kernel};

use crate::{
    Foundry, MetalError, compound::Layout, policy::activation::Activation, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, types::TensorArg
};

fn resolve_gemv_strategy_for_input(
    requested_strategy: GemvStrategy,
    input_dtype: crate::tensor::Dtype,
    layout: Layout,
    n_dim: u32,
) -> Result<GemvStrategy, MetalError> {
    const COL_MAJOR_AUTO_SCALAR_MIN_N: u32 = 4096;

    if matches!(requested_strategy, GemvStrategy::Vectorized) && !matches!(input_dtype, crate::tensor::Dtype::F16) {
        return Err(MetalError::OperationNotSupported(format!(
            "GemvV2 Vectorized strategy requires F16 input dtype, got {:?}. Use Auto/Canonical for non-F16 inputs.",
            input_dtype
        )));
    }
    if matches!(requested_strategy, GemvStrategy::Scalar) && !matches!(layout, Layout::ColMajor) {
        return Err(MetalError::OperationNotSupported(format!(
            "GemvV2 Scalar strategy only supports ColMajor layout, got {layout:?}."
        )));
    }

    Ok(match (requested_strategy, input_dtype, layout) {
        (GemvStrategy::Auto, crate::tensor::Dtype::F16, Layout::ColMajor) if n_dim >= COL_MAJOR_AUTO_SCALAR_MIN_N => GemvStrategy::Scalar,
        (GemvStrategy::Auto, crate::tensor::Dtype::F16, _) => GemvStrategy::Auto,
        (GemvStrategy::Auto, _, _) => GemvStrategy::Canonical,
        (other, _, _) => other,
    })
}

#[derive(MetalStruct, Clone, Debug, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct GemvV2Params {
    pub k_dim: DynamicValue<u32>,
    pub n_dim: DynamicValue<u32>,
    #[serde(default = "default_wpb")]
    pub weights_per_block: u32,
    #[serde(default = "default_batch")]
    pub batch: DynamicValue<u32>,
}

fn default_wpb() -> u32 {
    32
}
fn default_batch() -> DynamicValue<u32> {
    DynamicValue::Literal(1)
}

#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "gemv/gemv.metal",
    function = "gemv_kernel",
    include = [
        "gemv/common.metal",
        "gemv/dot.metal",
        "gemv/vectorized_stage.metal",
        "gemv/scalar_output.metal"
    ],
    args = GemvV2Params,
    step = false,
    execute = false
)]
pub struct GemvV2 {
    #[arg(buffer = 0, metal_type = "const device uchar*")]
    pub weights: TensorArg,
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    pub scale_bytes: Option<TensorArg>,
    #[arg(meta, scale_for = "weights")]
    pub derived_scales: TensorArg,
    #[arg(buffer = 2, metal_type = "const device InputStorageT*")]
    pub input: TensorArg,
    #[arg(buffer = 3, output, metal_type = "device OutputStorageT*")]
    pub output: TensorArg,
    #[arg(buffer = 7, metal_type = "const device BiasStorageT*")]
    pub bias: Option<TensorArg>,
    #[arg(buffer = 10, metal_type = "const device ResidualStorageT*")]
    pub residual: Option<TensorArg>,
    #[arg(meta)]
    pub layout: Layout,
    #[arg(meta)]
    pub strategy: Option<GemvStrategy>,
    #[arg(meta)]
    pub activation: Activation,
    pub alpha: f32,
    pub beta: f32,
    pub has_bias: u32,
    pub has_residual: u32,
    pub params: GemvV2ParamsResolved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemvV2UnifiedExecutionStep {
    pub weights: Ref,
    pub input: Ref,
    pub output: Ref,
    pub bias: Option<Ref>,
    pub residual: Option<Ref>,
    pub scale_bytes: Option<Ref>,
    pub params: GemvV2Params,
    pub layout: Layout,
    pub strategy: Option<GemvStrategy>,
    pub activation: Activation,
    pub alpha: f32,
    pub beta: f32,
    pub has_bias: u32,
    pub has_residual: u32,
}

#[derive(Debug)]
pub struct CompiledGemvV2UnifiedExecutionStep {
    pub weights: usize,
    pub input: usize,
    pub output: usize,
    pub bias: Option<usize>,
    pub residual: Option<usize>,
    pub scale_bytes: Option<usize>,
    pub params: GemvV2Params,
    pub layout: Layout,
    pub strategy: Option<GemvStrategy>,
    pub activation: Activation,
    pub alpha: f32,
    pub beta: f32,
    pub has_bias: u32,
    pub has_residual: u32,
}

#[typetag::serde(name = "GemvV2Unified")]
impl Step for GemvV2UnifiedExecutionStep {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = FastBindings::new(symbols.len());
        for (name, symbol_id) in symbols.iter() {
            if let Ok(tensor) = bindings.get(name) {
                fast_bindings.set(*symbol_id, tensor);
            }
        }
        for step in compiled {
            step.execute(foundry, &fast_bindings, bindings, &symbols)?;
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "GemvV2Unified"
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let weights_name = bindings.interpolate(self.weights.0.clone());
        vec![Box::new(CompiledGemvV2UnifiedExecutionStep {
            weights: symbols.get_or_create(weights_name.clone()),
            input: symbols.get_or_create(bindings.interpolate(self.input.0.clone())),
            output: symbols.get_or_create(bindings.interpolate(self.output.0.clone())),
            bias: self.bias.as_ref().map(|r| symbols.get_or_create(bindings.interpolate(r.0.clone()))),
            residual: self
                .residual
                .as_ref()
                .map(|r| symbols.get_or_create(bindings.interpolate(r.0.clone()))),
            // If the caller doesn't provide an explicit scales ref, derive it from weights by convention.
            // This matches the `scale_for` macro behavior and avoids fragile spec wiring.
            scale_bytes: self
                .scale_bytes
                .as_ref()
                .map(|r| symbols.get_or_create(bindings.interpolate(r.0.clone())))
                .or_else(|| Some(symbols.get_or_create(format!("{weights_name}_scales")))),
            params: self.params.clone(),
            layout: self.layout,
            strategy: self.strategy,
            activation: self.activation,
            alpha: self.alpha,
            beta: self.beta,
            has_bias: self.has_bias,
            has_residual: self.has_residual,
        })]
    }
}

impl CompiledStep for CompiledGemvV2UnifiedExecutionStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        use crate::types::DispatchConfig;

        let weights = fast_bindings
            .get(self.weights)
            .ok_or_else(|| MetalError::InputNotFound("weights".into()))?;
        let input = fast_bindings
            .get(self.input)
            .ok_or_else(|| MetalError::InputNotFound("input".into()))?;
        let output = fast_bindings
            .get(self.output)
            .ok_or_else(|| MetalError::InputNotFound("output".into()))?;

        let k_dim = self.params.k_dim.resolve(bindings);
        let n_dim = self.params.n_dim.resolve(bindings);
        let batch = self.params.batch.resolve(bindings);

        // Resolve policy purely from the bound weight dtype.
        // Quantization is intentionally "invisible" to the DSL/spec layer; bindings decide runtime policy.
        let policy = crate::policy::resolve_policy(weights.dtype);
        let weights_per_block = if policy.has_scale() {
            policy.meta().weights_per_block as u32
        } else {
            self.params.weights_per_block
        };

        let requested_strategy = self.strategy.unwrap_or(GemvStrategy::Auto);
        let selected_strategy = resolve_gemv_strategy_for_input(requested_strategy, input.dtype, self.layout, n_dim)?;
        if matches!(requested_strategy, GemvStrategy::Auto)
            && (matches!(selected_strategy, GemvStrategy::Canonical) || matches!(selected_strategy, GemvStrategy::Scalar))
        {
            tracing::debug!(
                target: "metallic_foundry::metals::gemv",
                input_dtype = ?input.dtype,
                requested_strategy = ?requested_strategy,
                selected_strategy = ?selected_strategy,
                layout = ?self.layout,
                n_dim,
                "GemvV2 Auto strategy resolved to a non-default path"
            );
        }

        let kernel = get_gemv_v2_kernel(
            policy.clone() as std::sync::Arc<dyn crate::fusion::MetalPolicy>,
            self.layout,
            selected_strategy,
            self.activation,
        )?;

        let dispatch = DispatchConfig::warp_per_row(n_dim, batch);

        let bias = if self.has_bias != 0 {
            let idx = self.bias.ok_or_else(|| MetalError::InputNotFound("bias field missing".into()))?;
            fast_bindings
                .get(idx)
                .ok_or_else(|| MetalError::InputNotFound("bias tensor not found".into()))?
        } else {
            output
        };

        let residual = if self.has_residual != 0 {
            let idx = self
                .residual
                .ok_or_else(|| MetalError::InputNotFound("residual field missing".into()))?;
            fast_bindings
                .get(idx)
                .ok_or_else(|| MetalError::InputNotFound("residual tensor not found".into()))?
        } else {
            output
        };

        let args = GemvV2Args {
            weights: TensorArg::from_tensor(weights),
            scale_bytes: if policy.has_scale() {
                let idx = self
                    .scale_bytes
                    .ok_or_else(|| MetalError::InputNotFound("scale field missing".into()))?;
                let scales = fast_bindings
                    .get(idx)
                    .ok_or_else(|| MetalError::InputNotFound("scale tensor not found".into()))?;
                TensorArg::from_tensor(scales)
            } else {
                TensorArg::from_tensor(weights)
            },
            input: TensorArg::from_tensor(input),
            output: TensorArg::from_tensor(output),
            k_dim,
            n_dim,
            weights_per_block,
            bias: TensorArg::from_tensor(bias),
            has_bias: self.has_bias,
            alpha: self.alpha,
            residual: TensorArg::from_tensor(residual),
            has_residual: self.has_residual,
            beta: self.beta,
        };

        foundry.run(&kernel.bind_arc(args, dispatch))?;
        Ok(())
    }
}

#[path = "mod.test.rs"]
mod tests;
