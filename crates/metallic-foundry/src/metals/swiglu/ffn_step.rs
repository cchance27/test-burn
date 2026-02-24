use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::{
    config::{effective_weights_per_block, resolve_rms_eps}, kernels::get_fused_ffn_kernel
};
use crate::{
    Foundry, MetalError, metals::common::{
        policy_slots::bind_dual_policy_slots, runtime::{require_non_empty_io, require_vector_len, resolve_batch, tail_dim}
    }, spec::{CompiledStep, FastBindings, Ref, ResolvedSymbols, Step, SymbolTable, TensorBindings}, types::{DispatchConfig, TensorArg}
};

fn default_wpb() -> u32 {
    32
}

/// Fused FFN Step: RMSNorm(Input) → Gate/Up GEMVs → SwiGLU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedFfnSwiGluRmsNormStep {
    pub input: Ref,
    pub gamma: Ref,
    pub w_gate: Ref,
    pub w_up: Ref,
    pub b_gate: Option<Ref>,
    pub b_up: Option<Ref>,
    pub output: Ref,
    #[serde(default = "default_wpb")]
    pub weights_per_block: u32,
    pub epsilon: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct CompiledFusedFfnSwiGluRmsNormStep {
    pub input_idx: usize,
    pub gamma_idx: usize,
    pub w_gate_name: String,
    pub w_gate_resolved: ResolvedSymbols,
    pub w_up_name: String,
    pub w_up_resolved: ResolvedSymbols,
    pub b_gate_idx: Option<usize>,
    pub b_up_idx: Option<usize>,
    pub output_idx: usize,
    pub weights_per_block: u32,
    pub epsilon: Option<f32>,
}

#[derive(Debug, KernelArgs)]
pub struct FusedFfnArgs {
    #[arg(buffer = 0, metal_type = "const device uchar*")]
    pub w_gate: TensorArg,
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    pub s_gate: Option<TensorArg>,
    #[arg(buffer = 2, metal_type = "const device uchar*")]
    pub w_up: TensorArg,
    #[arg(buffer = 3, metal_type = "const device uchar*")]
    pub s_up: Option<TensorArg>,
    #[arg(buffer = 4, metal_type = "const device InputStorageT*")]
    pub input: TensorArg,
    #[arg(buffer = 5, output, metal_type = "device OutputStorageT*")]
    pub output: TensorArg,
    #[arg(buffer = 6)]
    pub k_dim: u32,
    #[arg(buffer = 7)]
    pub n_dim: u32,
    #[arg(buffer = 8)]
    pub weights_per_block_gate: u32,
    #[arg(buffer = 9)]
    pub weights_per_block_up: u32,
    #[arg(buffer = 10, metal_type = "const device GammaStorageT*")]
    pub gamma: TensorArg,
    #[arg(buffer = 11, metal_type = "const device BiasStorageT*")]
    pub b_gate: TensorArg,
    #[arg(buffer = 12, metal_type = "const device BiasStorageT*")]
    pub b_up: TensorArg,
    #[arg(buffer = 13)]
    pub has_b_gate: u32,
    #[arg(buffer = 14)]
    pub has_b_up: u32,
    #[arg(buffer = 15)]
    pub epsilon: f32,
}

#[typetag::serde(name = "FusedFfnSwiGluRmsNorm")]
impl Step for FusedFfnSwiGluRmsNormStep {
    fn name(&self) -> &'static str {
        "FusedFfnSwiGluRmsNorm"
    }

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

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let input_idx = symbols.get_or_create(bindings.interpolate(self.input.0.clone()));
        let gamma_idx = symbols.get_or_create(bindings.interpolate(self.gamma.0.clone()));
        let w_gate_name = bindings.interpolate(self.w_gate.0.clone());
        let w_up_name = bindings.interpolate(self.w_up.0.clone());
        let w_gate_idx = symbols.get_or_create(w_gate_name.clone());
        let w_gate_scales_idx = symbols.get_or_create(format!("{w_gate_name}_scales"));
        let w_up_idx = symbols.get_or_create(w_up_name.clone());
        let w_up_scales_idx = symbols.get_or_create(format!("{w_up_name}_scales"));
        let b_gate_idx = self
            .b_gate
            .as_ref()
            .map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));
        let b_up_idx = self.b_up.as_ref().map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));

        vec![Box::new(CompiledFusedFfnSwiGluRmsNormStep {
            input_idx,
            gamma_idx,
            w_gate_name: w_gate_name.clone(),
            w_gate_resolved: ResolvedSymbols {
                weights: w_gate_idx,
                scales: w_gate_scales_idx.into(),
                bias: None,
            },
            w_up_name: w_up_name.clone(),
            w_up_resolved: ResolvedSymbols {
                weights: w_up_idx,
                scales: w_up_scales_idx.into(),
                bias: None,
            },
            b_gate_idx,
            b_up_idx,
            output_idx,
            weights_per_block: self.weights_per_block,
            epsilon: self.epsilon,
        })]
    }
}

impl CompiledStep for CompiledFusedFfnSwiGluRmsNormStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let get = |idx| fast_bindings.get(idx).ok_or(MetalError::InputNotFound("".into()));

        let input = get(self.input_idx)?;
        let gamma = get(self.gamma_idx)?;
        let w_gate = get(self.w_gate_resolved.weights)?;
        let w_up = get(self.w_up_resolved.weights)?;
        let output = get(self.output_idx)?;

        let policy_slots = bind_dual_policy_slots(fast_bindings, w_gate.dtype, &self.w_gate_resolved, w_up.dtype, &self.w_up_resolved);
        let same_policy = policy_slots.same_policy;
        let policy_gate = policy_slots.a.policy.clone();
        let policy_up = policy_slots.b.policy.clone();

        let (b_gate, has_b_gate) = if let Some(idx) = self.b_gate_idx {
            (TensorArg::from_tensor(get(idx)?), 1u32)
        } else {
            (TensorArg::from_tensor(output), 0u32)
        };

        let (b_up, has_b_up) = if let Some(idx) = self.b_up_idx {
            (TensorArg::from_tensor(get(idx)?), 1u32)
        } else {
            (TensorArg::from_tensor(output), 0u32)
        };

        require_non_empty_io("FusedFfnSwiGluRmsNorm", input, output)?;
        let k_dim = tail_dim(input)?;
        let n_dim = tail_dim(output)?;
        let batch = resolve_batch(bindings);

        let w_gate_arg = policy_slots.a.weight();
        let s_gate = policy_slots.a.scale();
        let w_up_arg = policy_slots.b.weight();
        let s_up = policy_slots.b.scale();

        require_vector_len("gamma", &TensorArg::from_tensor(gamma), k_dim)?;
        if has_b_gate != 0 {
            require_vector_len("b_gate", &b_gate, n_dim)?;
        }
        if has_b_up != 0 {
            require_vector_len("b_up", &b_up, n_dim)?;
        }

        let weights_per_block_gate = effective_weights_per_block(policy_gate.as_ref(), self.weights_per_block);
        let weights_per_block_up = if same_policy {
            weights_per_block_gate
        } else {
            effective_weights_per_block(policy_up.as_ref(), self.weights_per_block)
        };

        let args = FusedFfnArgs {
            w_gate: w_gate_arg,
            s_gate,
            w_up: w_up_arg,
            s_up,
            input: TensorArg::from_tensor(input),
            output: TensorArg::from_tensor(output),
            k_dim,
            n_dim,
            weights_per_block_gate,
            weights_per_block_up,
            gamma: TensorArg::from_tensor(gamma),
            b_gate,
            b_up,
            has_b_gate,
            has_b_up,
            epsilon: resolve_rms_eps(bindings, self.epsilon),
        };

        let kernel = get_fused_ffn_kernel(policy_gate, policy_up);
        let dispatch = DispatchConfig::warp_per_row(n_dim, batch);

        foundry.run(&kernel.clone().bind_arc(args, dispatch))
    }

    fn name(&self) -> &'static str {
        "FusedFfnSwiGluRmsNorm"
    }
}
