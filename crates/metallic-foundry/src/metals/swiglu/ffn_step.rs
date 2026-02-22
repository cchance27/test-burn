use std::sync::Once;

use half::f16;
use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::{
    config::{effective_weights_per_block, resolve_rms_eps}, kernels::get_fused_ffn_kernel, runtime::{allocate_zero_bias, run_canonical_projection}
};
use crate::{
    Foundry, MetalError, metals::common::runtime::{require_non_empty_io, require_vector_len, resolve_batch, tail_dim}, spec::{CompiledStep, FastBindings, Ref, ResolvedSymbols, Step, SymbolTable, TensorBindings}, types::{DispatchConfig, MetalResourceOptions, TensorArg}
};

static MIXED_FFN_SWIGLU_FALLBACK_WARN_ONCE: Once = Once::new();

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
    #[arg(buffer = 0)]
    pub w_gate: TensorArg,
    #[arg(buffer = 1)]
    pub s_gate: Option<TensorArg>,
    #[arg(buffer = 2)]
    pub w_up: TensorArg,
    #[arg(buffer = 3)]
    pub s_up: Option<TensorArg>,
    #[arg(buffer = 4)]
    pub input: TensorArg,
    #[arg(buffer = 5, output)]
    pub output: TensorArg,
    #[arg(buffer = 6)]
    pub k_dim: u32,
    #[arg(buffer = 7)]
    pub n_dim: u32,
    #[arg(buffer = 8)]
    pub weights_per_block: u32,
    #[arg(buffer = 9)]
    pub gamma: TensorArg,
    #[arg(buffer = 10)]
    pub b_gate: TensorArg,
    #[arg(buffer = 11)]
    pub b_up: TensorArg,
    #[arg(buffer = 12)]
    pub has_b_gate: u32,
    #[arg(buffer = 13)]
    pub has_b_up: u32,
    #[arg(buffer = 14)]
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

        let policy_gate = crate::policy::resolve_policy(w_gate.dtype);
        let loader_gate = policy_gate.loader_stage();
        let args_gate = loader_gate.bind(fast_bindings, &self.w_gate_resolved);

        let policy_up = crate::policy::resolve_policy(w_up.dtype);
        let loader_up = policy_up.loader_stage();
        let args_up = loader_up.bind(fast_bindings, &self.w_up_resolved);

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

        if w_gate.dtype != w_up.dtype {
            MIXED_FFN_SWIGLU_FALLBACK_WARN_ONCE.call_once(|| {
                tracing::warn!(
                    gate_dtype = ?w_gate.dtype,
                    up_dtype = ?w_up.dtype,
                    batch,
                    k_dim,
                    n_dim,
                    "FusedFfnSwiGluRmsNorm mixed-policy fallback engaged; performance may regress until mixed-policy fused kernel support is added"
                );
            });
            let input_elems = input.dims.iter().product::<usize>();
            let normalized_buf = foundry
                .device
                .new_buffer(input_elems * std::mem::size_of::<f16>(), MetalResourceOptions::StorageModeShared)
                .ok_or_else(|| MetalError::OperationFailed("Failed to allocate mixed-ffn normalized input".into()))?;
            let normalized = TensorArg::from_buffer(
                normalized_buf,
                crate::tensor::Dtype::F16,
                input.dims.to_vec(),
                input.strides.to_vec(),
            );
            let epsilon = resolve_rms_eps(bindings, self.epsilon);
            crate::metals::rmsnorm::step::run_rmsnorm(
                foundry,
                &TensorArg::from_tensor(input),
                None,
                &normalized,
                &TensorArg::from_tensor(gamma),
                crate::metals::rmsnorm::RmsNormParamsResolved {
                    feature_dim: k_dim,
                    total_elements: input_elems as u32,
                    epsilon,
                },
            )?;

            let output_elems = output.dims.iter().product::<usize>();
            let gate_out_buf = foundry
                .device
                .new_buffer(output_elems * std::mem::size_of::<f16>(), MetalResourceOptions::StorageModeShared)
                .ok_or_else(|| MetalError::OperationFailed("Failed to allocate mixed-ffn gate output".into()))?;
            let gate_out = TensorArg::from_buffer(
                gate_out_buf,
                crate::tensor::Dtype::F16,
                output.dims.to_vec(),
                output.strides.to_vec(),
            );
            let up_out = TensorArg::from_tensor(output);

            run_canonical_projection(
                foundry,
                fast_bindings,
                policy_gate.clone(),
                &self.w_gate_resolved,
                &normalized,
                gate_out.clone(),
                k_dim,
                n_dim,
                batch,
                self.weights_per_block,
            )?;
            run_canonical_projection(
                foundry,
                fast_bindings,
                policy_up.clone(),
                &self.w_up_resolved,
                &normalized,
                up_out.clone(),
                k_dim,
                n_dim,
                batch,
                self.weights_per_block,
            )?;

            let zero_bias_tensor = allocate_zero_bias(foundry, n_dim)?;
            let gate_bias_tensor = if has_b_gate != 0 {
                b_gate.clone()
            } else {
                TensorArg::from_tensor(&zero_bias_tensor)
            };
            let up_bias_tensor = if has_b_up != 0 {
                b_up.clone()
            } else {
                TensorArg::from_tensor(&zero_bias_tensor)
            };

            let swiglu = crate::metals::swiglu::Swiglu::new_auto_vectorized(
                &gate_out,
                &up_out,
                &gate_bias_tensor,
                &up_bias_tensor,
                output_elems as u32,
                n_dim,
                n_dim,
                n_dim,
            );
            foundry.run(&swiglu)?;
            return Ok(());
        }

        let w_gate_arg = args_gate[0].clone();
        let s_gate = if args_gate.len() > 1 { Some(args_gate[1].clone()) } else { None };
        let w_up_arg = args_up[0].clone();
        let s_up = if args_up.len() > 1 { Some(args_up[1].clone()) } else { None };

        require_vector_len("gamma", &TensorArg::from_tensor(gamma), k_dim)?;
        if has_b_gate != 0 {
            require_vector_len("b_gate", &b_gate, n_dim)?;
        }
        if has_b_up != 0 {
            require_vector_len("b_up", &b_up, n_dim)?;
        }

        let weights_per_block = effective_weights_per_block(policy_gate.as_ref(), self.weights_per_block);

        let args = FusedFfnArgs {
            w_gate: w_gate_arg,
            s_gate,
            w_up: w_up_arg,
            s_up,
            input: TensorArg::from_tensor(input),
            output: TensorArg::from_tensor(output),
            k_dim,
            n_dim,
            weights_per_block,
            gamma: TensorArg::from_tensor(gamma),
            b_gate,
            b_up,
            has_b_gate,
            has_b_up,
            epsilon: resolve_rms_eps(bindings, self.epsilon),
        };

        let kernel = get_fused_ffn_kernel(policy_gate);
        let dispatch = DispatchConfig::warp_per_row(n_dim, batch);

        foundry.run(&kernel.clone().bind_arc(args, dispatch))
    }

    fn name(&self) -> &'static str {
        "FusedFfnSwiGluRmsNorm"
    }
}
