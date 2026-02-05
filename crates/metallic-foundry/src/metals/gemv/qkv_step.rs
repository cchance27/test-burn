use std::sync::Arc;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::step::GemvStrategy;
use crate::{
    Foundry, MetalError, compound::{CompiledCompoundKernel, CompoundKernel, stages::WarpLayoutStage}, fusion::MetalPolicy, metals::{
        gemv::qkv_stages::{MultiWarpReduceStage, MultiWriteOutputStage, ParallelProjectStage}, rmsnorm::stages::RmsNormComputeStage
    }, spec::{CompiledStep, DynamicValue, FastBindings, Ref, ResolvedSymbols, Step, SymbolTable, TensorBindings}, types::TensorArg
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
        let _w_q_scales_idx = symbols.get_or_create(format!("{w_q_name}_scales"));

        let w_k_name = bindings.interpolate(self.w_k.0.clone());
        let w_k_idx = symbols.get_or_create(w_k_name.clone());
        let _w_k_scales_idx = symbols.get_or_create(format!("{w_k_name}_scales"));

        let w_v_name = bindings.interpolate(self.w_v.0.clone());
        let w_v_idx = symbols.get_or_create(w_v_name.clone());
        let _w_v_scales_idx = symbols.get_or_create(format!("{w_v_name}_scales"));

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
                scales: _w_q_scales_idx.into(),
                bias: None,
            },
            w_k_name,
            w_k_resolved: ResolvedSymbols {
                weights: w_k_idx,
                scales: _w_k_scales_idx.into(),
                bias: None,
            },
            w_v_name,
            w_v_resolved: ResolvedSymbols {
                weights: w_v_idx,
                scales: _w_v_scales_idx.into(),
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

        // Centralized Quantization Binding
        let policy = crate::policy::resolve_policy(w_q_tensor.dtype);
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
        let out_q = get(self.out_q_idx)?;
        let out_k = get(self.out_k_idx)?;
        let out_v = get(self.out_v_idx)?;
        // Use input buffer as dummy for missing optional args to avoid panic, this seems like asking for confusion but temporarily.
        let dummy_arg = TensorArg::from_tensor(input);

        let gamma = if let Some(idx) = self.gamma_idx {
            TensorArg::from_tensor(get(idx)?)
        } else {
            dummy_arg.clone()
        };

        let (b_q, has_bias) = if let Some(idx) = self.bias_q_idx {
            (TensorArg::from_tensor(fast_bindings.get(idx).unwrap()), 1)
        } else {
            (dummy_arg.clone(), 0)
        };
        let b_k = self
            .bias_k_idx
            .map(|idx| TensorArg::from_tensor(fast_bindings.get(idx).unwrap()))
            .unwrap_or_else(|| dummy_arg.clone());
        let b_v = self
            .bias_v_idx
            .map(|idx| TensorArg::from_tensor(fast_bindings.get(idx).unwrap()))
            .unwrap_or_else(|| dummy_arg.clone());

        let b_q_saved = b_q.clone();

        let k_dim = self.step.k_dim.resolve(bindings);
        let n_dim = self.step.n_dim.resolve(bindings);
        let n_kv = self.step.n_kv.resolve(bindings);
        let weights_per_block = self.step.weights_per_block.resolve(bindings);
        let m = self.step.m.resolve(bindings).max(1);

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
            epsilon: bindings.get_var("rms_eps").and_then(|v| v.parse::<f32>().ok()).unwrap_or(1e-6),
        };

        let kernel = get_fused_qkv_kernel(self.step.strategy, policy, self.gamma_idx.is_some());

        // Manual dispatch to support M dimension (batch)
        const WARPS_PER_TG: usize = 8;
        const SIMD_WIDTH: usize = 32;
        const TG_WIDTH: usize = WARPS_PER_TG * SIMD_WIDTH; // 256
        let num_tgs = (n_dim.max(n_kv) as usize).div_ceil(WARPS_PER_TG);

        let dispatch = crate::types::dispatch::DispatchConfig {
            grid: crate::types::dispatch::GridSize::new(num_tgs, m as usize, 1),
            group: crate::types::dispatch::ThreadgroupSize::new(TG_WIDTH, 1, 1),
        };

        if bindings.get_var("i").map(|v| v == "0").unwrap_or(false) {
            eprintln!("DEBUG FusedQkv Source:\n{}", kernel.source());
            eprintln!(
                "DEBUG FusedQkv Layer 0: k_dim={} n_dim={} n_kv={} weights_per_block={} | q_off={} k_off={} v_off={} input_off={}",
                k_dim, n_dim, n_kv, weights_per_block, out_q.offset, out_k.offset, out_v.offset, input.offset
            );
        }

        foundry.run(&kernel.clone().bind_arc(args, dispatch))?;

        if bindings.get_var("i").map(|v| v == "0").unwrap_or(false) {
            foundry.synchronize()?;
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "FusedQkv"
    }
}

fn get_fused_qkv_kernel(strategy: GemvStrategy, policy: Arc<dyn MetalPolicy>, has_norm: bool) -> Arc<CompiledCompoundKernel> {
    use crate::kernel_registry::{KernelCacheKey, kernel_registry};

    let variant = format!("{:?}_{}_{}", strategy, policy.short_name(), has_norm).to_lowercase();
    let key = KernelCacheKey::new("fused_qkv", variant);

    let policy_clone = policy.clone();
    kernel_registry().get_or_build(key, move || {
        let norm_suffix = if has_norm { "_rmsnorm" } else { "" };
        let kernel_name = format!("fused_qkv{}_{}", norm_suffix, policy_clone.short_name());
        let vec_width = policy_clone.optimization_hints().vector_load_size;

        // Configure layout with correct stride
        let mut compound = CompoundKernel::new(&kernel_name)
            .with_manual_output(true)
            .prologue(WarpLayoutStage::canonical().with_warps(8).with_elems_per_thread(vec_width as u32))
            .prologue(RmsNormComputeStage::new(6, 7, 19)); // Stage 1: RMSNorm compute

        // Stage 2: QKV Projection
        let vw = match vec_width {
            4 => super::stages::VectorWidth::Vec4,
            8 => super::stages::VectorWidth::Vec8,
            _ => panic!("Unsupported vector width: {}", vec_width),
        };

        let mut proj = ParallelProjectStage::new(policy_clone.clone()).with_vector_width(vw);
        if has_norm {
            proj = proj.with_norm(18, "inv_rms");
        }
        compound = compound.main(proj);

        compound
            .epilogue(MultiWarpReduceStage) // Stage 3: Reduction
            .epilogue(MultiWriteOutputStage) // Stage 4: Write Out
            .compile()
    })
}
