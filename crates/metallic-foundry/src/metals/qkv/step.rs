use std::sync::Once;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use crate::{
    Foundry, MetalError, metals::gemv::{GemvStrategy, GemvV2Args, get_gemv_v2_kernel}, spec::{CompiledStep, DynamicValue, FastBindings, Ref, ResolvedSymbols, Step, SymbolTable, TensorBindings}, types::TensorArg
};

/// Fused QKV Step: RMSNorm(Input) -> Parallel QKV Projection -> 3 Outputs
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FusedQkvStep {
    pub input: Ref,
    pub w_q: Ref,
    pub s_q: Option<Ref>,
    pub w_k: Ref,
    pub s_k: Option<Ref>,
    pub w_v: Ref,
    pub s_v: Option<Ref>,
    pub out_q: Ref,
    pub out_k: Ref,
    pub out_v: Ref,
    pub gamma: Option<Ref>,
    pub bias_q: Option<Ref>,
    pub bias_k: Option<Ref>,
    pub bias_v: Option<Ref>,
    pub k_dim: DynamicValue<u32>,
    pub n_dim: DynamicValue<u32>,
    pub n_kv: DynamicValue<u32>,
    pub m: DynamicValue<u32>,
    pub weights_per_block: DynamicValue<u32>,
    pub strategy: GemvStrategy,
}

#[derive(Debug, Clone)]
pub struct CompiledFusedQkvStep {
    pub step: FusedQkvStep,
    pub input_idx: usize,
    pub w_q_name: String,
    pub w_q_resolved: ResolvedSymbols,
    pub w_k_name: String,
    pub w_k_resolved: ResolvedSymbols,
    pub w_v_name: String,
    pub w_v_resolved: ResolvedSymbols,
    pub out_q_idx: usize,
    pub out_k_idx: usize,
    pub out_v_idx: usize,
    pub gamma_idx: Option<usize>,
    pub bias_q_idx: Option<usize>,
    pub bias_k_idx: Option<usize>,
    pub bias_v_idx: Option<usize>,
}

#[derive(Debug, KernelArgs, Clone)]
pub struct FusedQkvArgs {
    #[arg(buffer = 0)]
    pub w_q: TensorArg,
    #[arg(buffer = 1)]
    pub s_q: Option<TensorArg>,
    #[arg(buffer = 2)]
    pub w_k: TensorArg,
    #[arg(buffer = 3)]
    pub s_k: Option<TensorArg>,
    #[arg(buffer = 4)]
    pub w_v: TensorArg,
    #[arg(buffer = 5)]
    pub s_v: Option<TensorArg>,
    #[arg(buffer = 6)]
    pub input: TensorArg,
    #[arg(buffer = 7)]
    pub k_dim: u32,
    #[arg(buffer = 8)]
    pub n_dim: u32,
    #[arg(buffer = 9)]
    pub n_kv: u32,
    #[arg(buffer = 10)]
    pub weights_per_block: u32,
    #[arg(buffer = 11)]
    pub out_q: TensorArg,
    #[arg(buffer = 12)]
    pub out_k: TensorArg,
    #[arg(buffer = 13)]
    pub out_v: TensorArg,
    #[arg(buffer = 14)]
    pub b_q: TensorArg,
    #[arg(buffer = 15)]
    pub b_k: TensorArg,
    #[arg(buffer = 16)]
    pub b_v: TensorArg,
    #[arg(buffer = 17)]
    pub has_bias: u32,
    #[arg(buffer = 18)]
    pub gamma: TensorArg,
    #[arg(buffer = 19)]
    pub epsilon: f32,
}

static MIXED_QKV_FALLBACK_WARN_ONCE: Once = Once::new();

#[typetag::serde(name = "FusedQkv")]
impl Step for FusedQkvStep {
    fn name(&self) -> &'static str {
        "FusedQkv"
    }

    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = FastBindings::new(symbols.len());

        // Bind all symbols found in the table
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
        let w_q_name = bindings.interpolate(self.w_q.0.clone());
        let w_q_idx = symbols.get_or_create(w_q_name.clone());
        let w_q_scales_idx = symbols.get_or_create(format!("{w_q_name}_scales"));

        let w_k_name = bindings.interpolate(self.w_k.0.clone());
        let w_k_idx = symbols.get_or_create(w_k_name.clone());
        let w_k_scales_idx = symbols.get_or_create(format!("{w_k_name}_scales"));

        let w_v_name = bindings.interpolate(self.w_v.0.clone());
        let w_v_idx = symbols.get_or_create(w_v_name.clone());
        let w_v_scales_idx = symbols.get_or_create(format!("{w_v_name}_scales"));

        let out_q_idx = symbols.get_or_create(bindings.interpolate(self.out_q.0.clone()));
        let out_k_idx = symbols.get_or_create(bindings.interpolate(self.out_k.0.clone()));
        let out_v_idx = symbols.get_or_create(bindings.interpolate(self.out_v.0.clone()));
        let gamma_idx = self
            .gamma
            .as_ref()
            .map(|g| symbols.get_or_create(bindings.interpolate(g.0.clone())));
        let bias_q_idx = self
            .bias_q
            .as_ref()
            .map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));
        let bias_k_idx = self
            .bias_k
            .as_ref()
            .map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));
        let bias_v_idx = self
            .bias_v
            .as_ref()
            .map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));

        vec![Box::new(CompiledFusedQkvStep {
            step: self.clone(),
            input_idx,
            w_q_name,
            w_q_resolved: ResolvedSymbols {
                weights: w_q_idx,
                scales: w_q_scales_idx.into(),
                bias: None,
            },
            w_k_name,
            w_k_resolved: ResolvedSymbols {
                weights: w_k_idx,
                scales: w_k_scales_idx.into(),
                bias: None,
            },
            w_v_name,
            w_v_resolved: ResolvedSymbols {
                weights: w_v_idx,
                scales: w_v_scales_idx.into(),
                bias: None,
            },
            out_q_idx,
            out_k_idx,
            out_v_idx,
            gamma_idx,
            bias_q_idx,
            bias_k_idx,
            bias_v_idx,
        })]
    }
}

impl CompiledStep for CompiledFusedQkvStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let get = |idx| fast_bindings.get(idx).ok_or(MetalError::InputNotFound("".into()));
        let input = get(self.input_idx)?;
        let w_q_tensor = get(self.w_q_resolved.weights)?;
        let w_k_tensor = get(self.w_k_resolved.weights)?;
        let w_v_tensor = get(self.w_v_resolved.weights)?;

        let out_q = get(self.out_q_idx)?;
        let out_k = get(self.out_k_idx)?;
        let out_v = get(self.out_v_idx)?;

        let k_dim = self.step.k_dim.resolve(bindings);
        let n_dim = self.step.n_dim.resolve(bindings);
        let n_kv = self.step.n_kv.resolve(bindings);
        let m = self.step.m.resolve(bindings).max(1);

        let q_policy = crate::policy::resolve_policy(w_q_tensor.dtype);
        let k_policy = crate::policy::resolve_policy(w_k_tensor.dtype);
        let v_policy = crate::policy::resolve_policy(w_v_tensor.dtype);
        let is_mixed_policy = w_q_tensor.dtype != w_k_tensor.dtype || w_q_tensor.dtype != w_v_tensor.dtype;

        // Mixed quantized Q/K/V projections cannot share one policy-specialized fused kernel.
        // Fall back to per-projection GEMV using each tensor's own runtime policy.
        if is_mixed_policy {
            MIXED_QKV_FALLBACK_WARN_ONCE.call_once(|| {
                tracing::warn!(
                    q_dtype = ?w_q_tensor.dtype,
                    k_dtype = ?w_k_tensor.dtype,
                    v_dtype = ?w_v_tensor.dtype,
                    m,
                    k_dim,
                    n_dim,
                    n_kv,
                    "FusedQkv mixed-policy fallback engaged; performance may regress until mixed-policy fused kernel support is added"
                );
            });
            let input_arg = TensorArg::from_tensor(input);
            let proj_input = if let Some(gamma_idx) = self.gamma_idx {
                let gamma = get(gamma_idx)?;
                let input_elems = input.dims().iter().product::<usize>();
                let normalized_buf = foundry
                    .device
                    .new_buffer(
                        input_elems * std::mem::size_of::<half::f16>(),
                        crate::types::MetalResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| MetalError::OperationFailed("Failed to allocate fused_qkv normalized input".into()))?;
                let normalized = TensorArg::from_buffer(
                    normalized_buf,
                    crate::tensor::Dtype::F16,
                    input.dims().to_vec(),
                    input.strides().to_vec(),
                );
                crate::metals::rmsnorm::step::run_rmsnorm(
                    foundry,
                    &input_arg,
                    None,
                    &normalized,
                    &TensorArg::from_tensor(gamma),
                    crate::metals::rmsnorm::RmsNormParamsResolved {
                        feature_dim: k_dim,
                        total_elements: input_elems as u32,
                        epsilon: bindings.get_f32_var_or("rms_eps", 1e-6),
                    },
                )?;
                normalized
            } else {
                input_arg
            };

            let run_projection = |foundry: &mut Foundry,
                                  policy: std::sync::Arc<dyn crate::policy::MetalPolicyRuntime>,
                                  resolved: &ResolvedSymbols,
                                  out_arg: TensorArg,
                                  n_out: u32,
                                  bias_idx: Option<usize>|
             -> Result<(), MetalError> {
                let loader = policy.loader_stage();
                let args = loader.bind(fast_bindings, resolved);
                let weights = args[0].clone();
                let scale_bytes = if args.len() > 1 { args[1].clone() } else { weights.clone() };
                let weights_per_block = if policy.has_scale() {
                    policy.meta().weights_per_block as u32
                } else {
                    self.step.weights_per_block.resolve(bindings)
                };

                let bias = if let Some(idx) = bias_idx {
                    TensorArg::from_tensor(get(idx)?)
                } else {
                    out_arg.clone()
                };
                let output = out_arg;

                let kernel = get_gemv_v2_kernel(
                    policy,
                    crate::compound::Layout::Canonical {
                        expected_k: k_dim as usize,
                        expected_n: n_out as usize,
                    },
                    self.step.strategy,
                    crate::policy::activation::Activation::None,
                );
                let gemv_args = GemvV2Args {
                    weights,
                    scale_bytes,
                    input: proj_input.clone(),
                    output: output.clone(),
                    k_dim,
                    n_dim: n_out,
                    weights_per_block,
                    bias,
                    has_bias: u32::from(bias_idx.is_some()),
                    alpha: 1.0,
                    residual: output,
                    has_residual: 0,
                    beta: 0.0,
                };
                let dispatch = crate::types::DispatchConfig::warp_per_row(n_out, m);
                foundry.run(&kernel.bind_arc(gemv_args, dispatch))
            };

            run_projection(
                foundry,
                q_policy,
                &self.w_q_resolved,
                TensorArg::from_tensor(out_q),
                n_dim,
                self.bias_q_idx,
            )?;
            run_projection(
                foundry,
                k_policy,
                &self.w_k_resolved,
                TensorArg::from_tensor(out_k),
                n_kv,
                self.bias_k_idx,
            )?;
            run_projection(
                foundry,
                v_policy,
                &self.w_v_resolved,
                TensorArg::from_tensor(out_v),
                n_kv,
                self.bias_v_idx,
            )?;
            return Ok(());
        }

        // Centralized Quantization Binding (uniform policy path).
        let policy = q_policy;
        let loader = policy.loader_stage();

        let q_args = loader.bind(fast_bindings, &self.w_q_resolved);
        let k_args = loader.bind(fast_bindings, &self.w_k_resolved);
        let v_args = loader.bind(fast_bindings, &self.w_v_resolved);

        let w_q = q_args[0].clone();
        let s_q = if q_args.len() > 1 { Some(q_args[1].clone()) } else { None };
        let w_k = k_args[0].clone();
        let s_k = if k_args.len() > 1 { Some(k_args[1].clone()) } else { None };
        let w_v = v_args[0].clone();
        let s_v = if v_args.len() > 1 { Some(v_args[1].clone()) } else { None };

        // Use input buffer as dummy for missing optional args.
        let dummy_arg = TensorArg::from_tensor(input);

        let gamma = if let Some(idx) = self.gamma_idx {
            TensorArg::from_tensor(get(idx)?)
        } else {
            dummy_arg.clone()
        };

        let (b_q, has_bias) = if let Some(idx) = self.bias_q_idx {
            (TensorArg::from_tensor(get(idx)?), 1)
        } else {
            (dummy_arg.clone(), 0)
        };
        let b_k = self
            .bias_k_idx
            .map(|idx| TensorArg::from_tensor(get(idx).expect("bias_k missing")))
            .unwrap_or_else(|| dummy_arg.clone());
        let b_v = self
            .bias_v_idx
            .map(|idx| TensorArg::from_tensor(get(idx).expect("bias_v missing")))
            .unwrap_or_else(|| dummy_arg.clone());

        let b_q_saved = b_q.clone();
        let weights_per_block = if policy.has_scale() {
            policy.meta().weights_per_block as u32
        } else {
            self.step.weights_per_block.resolve(bindings)
        };

        let args = FusedQkvArgs {
            w_q,
            s_q,
            w_k,
            s_k,
            w_v,
            s_v,
            input: TensorArg::from_tensor(input),
            k_dim,
            n_dim,
            n_kv,
            weights_per_block,
            out_q: TensorArg::from_tensor(out_q),
            out_k: TensorArg::from_tensor(out_k),
            out_v: TensorArg::from_tensor(out_v),
            b_q: b_q_saved,
            b_k,
            b_v,
            has_bias,
            gamma,
            epsilon: bindings.get_f32_var_or("rms_eps", 1e-6),
        };

        let kernel = super::kernels::get_fused_qkv_kernel(self.step.strategy, policy, self.gamma_idx.is_some());

        // Manual dispatch to support M dimension (batch)
        const WARPS_PER_TG: usize = 8;
        const SIMD_WIDTH: usize = 32;
        const TG_WIDTH: usize = WARPS_PER_TG * SIMD_WIDTH; // 256
        let num_tgs = (n_dim.max(n_kv) as usize).div_ceil(WARPS_PER_TG);

        let dispatch = crate::types::dispatch::DispatchConfig {
            grid: crate::types::dispatch::GridSize::new(num_tgs, m as usize, 1),
            group: crate::types::dispatch::ThreadgroupSize::new(TG_WIDTH, 1, 1),
        };

        foundry.run(&kernel.clone().bind_arc(args, dispatch))?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "FusedQkv"
    }
}
