use std::sync::OnceLock;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::{SwigluParamsResolved, stages::SwigluStage};
use crate::{
    MetalError, compound::{CompiledCompoundKernel, CompoundKernel}, foundry::{
        Foundry, spec::{CompiledStep, DynamicValue, Ref, Step, SymbolTable, TensorBindings}
    }, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

#[derive(Debug, Serialize, Deserialize)]
pub struct SwigluStep {
    pub gate: Ref,
    pub up_inout: Ref,
    pub gate_bias: Ref,
    pub up_bias: Ref,
    pub bias_len: DynamicValue<u32>,
    pub gate_leading_stride: DynamicValue<u32>,
    pub up_leading_stride: DynamicValue<u32>,
}

#[derive(Debug, Clone)]
pub struct CompiledSwigluStep {
    pub gate: usize,
    pub up_inout: usize,
    pub gate_bias: usize,
    pub up_bias: usize,
    pub bias_len: DynamicValue<u32>,
    pub gate_leading_stride: DynamicValue<u32>,
    pub up_leading_stride: DynamicValue<u32>,
}

#[derive(Debug, KernelArgs)]
pub struct SwigluArgs {
    #[arg(buffer = 0)]
    pub gate: TensorArg,
    #[arg(buffer = 1, output)]
    pub up_inout: TensorArg,
    #[arg(buffer = 2)]
    pub gate_bias: TensorArg,
    #[arg(buffer = 3)]
    pub up_bias: TensorArg,
    #[arg(buffer = 4)]
    pub params: SwigluParamsResolved,
}

#[typetag::serde(name = "SwigluV2")]
impl Step for SwigluStep {
    fn name(&self) -> &'static str {
        "Swiglu"
    }

    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = crate::foundry::spec::SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = crate::foundry::spec::FastBindings::new(symbols.len());

        // Bind all symbols found in the table
        for (name, symbol_id) in symbols.iter() {
            if let Ok(tensor) = bindings.get(name) {
                fast_bindings.set(*symbol_id, tensor);
            }
        }

        for step in compiled {
            step.execute(foundry, &fast_bindings, bindings)?;
        }

        Ok(())
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let gate = symbols.get_or_create(bindings.interpolate(self.gate.0.clone()));
        let up_inout = symbols.get_or_create(bindings.interpolate(self.up_inout.0.clone()));
        let gate_bias = symbols.get_or_create(bindings.interpolate(self.gate_bias.0.clone()));
        let up_bias = symbols.get_or_create(bindings.interpolate(self.up_bias.0.clone()));

        vec![Box::new(CompiledSwigluStep {
            gate,
            up_inout,
            gate_bias,
            up_bias,
            bias_len: self.bias_len.clone(),
            gate_leading_stride: self.gate_leading_stride.clone(),
            up_leading_stride: self.up_leading_stride.clone(),
        })]
    }
}

impl CompiledStep for CompiledSwigluStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &crate::foundry::spec::FastBindings,
        bindings: &TensorBindings,
    ) -> Result<(), MetalError> {
        let gate = fast_bindings.get(self.gate).ok_or(MetalError::InputNotFound("gate".into()))?;
        let up = fast_bindings.get(self.up_inout).ok_or(MetalError::InputNotFound("up".into()))?;
        let gate_bias = fast_bindings
            .get(self.gate_bias)
            .ok_or(MetalError::InputNotFound("gate_bias".into()))?;
        let up_bias = fast_bindings.get(self.up_bias).ok_or(MetalError::InputNotFound("up_bias".into()))?;

        let total_elements = up.dims.iter().product::<usize>() as u32;
        let bias_len = self.bias_len.resolve(bindings);
        let gate_stride = self.gate_leading_stride.resolve(bindings);
        let up_stride = self.up_leading_stride.resolve(bindings);

        // Auto-detect vector width
        let vector_width = if bias_len % 4 == 0 { 4 } else { 1 };

        let params = SwigluParamsResolved {
            total_elements,
            bias_len,
            vector_width,
            gate_leading_stride: gate_stride,
            up_leading_stride: up_stride,
        };

        let args = SwigluArgs {
            gate: gate.clone(),
            up_inout: up.clone(),
            gate_bias: gate_bias.clone(),
            up_bias: up_bias.clone(),
            params,
        };

        // Determine dispatch
        let vector_width_usize = std::cmp::max(vector_width as usize, 1);
        let base_threads = 256;
        let threads_per_group = std::cmp::max(base_threads / vector_width_usize, 1);

        let total_threads = if vector_width > 1 {
            let vectorized = total_elements / vector_width;
            let remainder = total_elements % vector_width;
            (vectorized + remainder) as usize
        } else {
            total_elements as usize
        };

        let num_groups = (total_threads + threads_per_group - 1) / threads_per_group;

        let dispatch = DispatchConfig {
            grid: GridSize::d1(num_groups),
            group: ThreadgroupSize::d1(threads_per_group),
        };

        // JIT Compile Kernel
        // We use a key based on vector width to allow specializing if needed
        // Here we just use a single compiled kernel since the code handles both paths
        static KERNEL: OnceLock<CompiledCompoundKernel> = OnceLock::new();
        let kernel = KERNEL.get_or_init(|| {
            // Need a dummy params for init?
            // Actually CompoundKernel relies on Stage emitting code.
            // SwigluStage code expects params to be bound.
            let dummy_params = SwigluParamsResolved::default();
            CompoundKernel::new("swiglu_v2")
                .main_dyn(Box::new(SwigluStage::new(dummy_params)))
                .with_manual_output(true) // Swiglu writes its own output
                .compile()
        });

        foundry.run(&kernel.bind(args, dispatch))
    }

    fn name(&self) -> &'static str {
        "Swiglu"
    }
}

// =============================================================================
// FusedSwigluStep - RMSNorm → Gate/Up GEMVs → SwiGLU activation
// Maps to legacy "SwiGluF16CanonicalFusedRmsnorm" op
// =============================================================================

/// Fused SwiGLU Step: RMSNorm(Input) → Gate/Up GEMVs → SwiGLU
///
/// This maps to the legacy "SwiGluF16CanonicalFusedRmsnorm" op in model specs.
/// Currently implemented as a placeholder that decomposes into separate steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedSwigluStep {
    pub input: Ref,
    pub gamma: Ref,      // RMSNorm weights
    pub wg: Ref,         // Gate weight matrix (alias for w_gate)
    pub wu: Ref,         // Up weight matrix (alias for w_up)
    pub bg: Option<Ref>, // Gate bias
    pub bu: Option<Ref>, // Up bias
    pub out: Ref,        // Output
    #[serde(default = "default_epsilon")]
    pub epsilon: f32,
    #[serde(default = "default_wpb")]
    pub weights_per_block: u32,
}

fn default_epsilon() -> f32 {
    1e-6
}
fn default_wpb() -> u32 {
    32
}

#[derive(Debug, Clone)]
pub struct CompiledFusedSwigluStep {
    pub step: FusedSwigluStep,
    pub input_idx: usize,
    pub gamma_idx: usize,
    pub wg_idx: usize,
    pub wu_idx: usize,
    pub bg_idx: Option<usize>,
    pub bu_idx: Option<usize>,
    pub out_idx: usize,
}

#[typetag::serde(name = "SwiGluF16CanonicalFusedRmsnorm")]
impl Step for FusedSwigluStep {
    fn name(&self) -> &'static str {
        "SwiGluF16CanonicalFusedRmsnorm"
    }

    fn execute(&self, _foundry: &mut Foundry, _bindings: &mut TensorBindings) -> Result<(), MetalError> {
        Err(MetalError::OperationNotSupported("FusedSwiglu only supports compile()".into()))
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let input_idx = symbols.get_or_create(bindings.interpolate(self.input.0.clone()));
        let gamma_idx = symbols.get_or_create(bindings.interpolate(self.gamma.0.clone()));
        let wg_idx = symbols.get_or_create(bindings.interpolate(self.wg.0.clone()));
        let wu_idx = symbols.get_or_create(bindings.interpolate(self.wu.0.clone()));
        let bg_idx = self.bg.as_ref().map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));
        let bu_idx = self.bu.as_ref().map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));
        let out_idx = symbols.get_or_create(bindings.interpolate(self.out.0.clone()));

        vec![Box::new(CompiledFusedSwigluStep {
            step: FusedSwigluStep {
                input: self.input.clone(),
                gamma: self.gamma.clone(),
                wg: self.wg.clone(),
                wu: self.wu.clone(),
                bg: self.bg.clone(),
                bu: self.bu.clone(),
                out: self.out.clone(),
                epsilon: self.epsilon,
                weights_per_block: self.weights_per_block,
            },
            input_idx,
            gamma_idx,
            wg_idx,
            wu_idx,
            bg_idx,
            bu_idx,
            out_idx,
        })]
    }
}

impl CompiledStep for CompiledFusedSwigluStep {
    fn execute(
        &self,
        _foundry: &mut Foundry,
        _fast_bindings: &crate::foundry::spec::FastBindings,
        _bindings: &TensorBindings,
    ) -> Result<(), MetalError> {
        // TODO: Implement proper fused kernel
        // For now, this is a stub that will need implementation
        // The legacy kernel fused: RMSNorm → Gate/Up GEMVs → SwiGLU
        Err(MetalError::OperationNotSupported(
            "FusedSwiglu execution not yet implemented - use legacy kernel".into(),
        ))
    }

    fn name(&self) -> &'static str {
        "FusedSwiglu"
    }
}
