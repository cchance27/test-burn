use std::sync::OnceLock;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::step::GemvStrategy;
use crate::{
    MetalError, compound::{
        CompiledCompoundKernel, CompoundKernel, stages::{Quantization, WarpLayoutStage}
    }, foundry::{
        Foundry, spec::{CompiledStep, DynamicValue, Ref, Step, SymbolTable, TensorBindings}
    }, metals::{
        gemv::qkv_stages::{MultiWarpReduceStage, MultiWriteOutputStage, ParallelProjectStage}, rmsnorm::stages::RmsNormComputeStage
    }, types::TensorArg
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
    pub w_q_idx: usize,
    pub s_q_idx: Option<usize>,
    pub w_k_idx: usize,
    pub s_k_idx: Option<usize>,
    pub w_v_idx: usize,
    pub s_v_idx: Option<usize>,
    pub is_q8: bool,
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
    pub s_q: TensorArg,
    #[arg(buffer = 2)]
    pub w_k: TensorArg,
    #[arg(buffer = 3)]
    pub s_k: TensorArg,
    #[arg(buffer = 4)]
    pub w_v: TensorArg,
    #[arg(buffer = 5)]
    pub s_v: TensorArg,
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
}

#[typetag::serde(name = "FusedQkv")]
impl Step for FusedQkvStep {
    fn name(&self) -> &'static str {
        "FusedQkv"
    }

    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = SymbolTable::new();
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
        let input_idx = symbols.get_or_create(bindings.interpolate(self.input.0.clone()));
        let w_q_idx = symbols.get_or_create(bindings.interpolate(self.w_q.0.clone()));
        let s_q_idx = self.s_q.as_ref().map(|s| symbols.get_or_create(bindings.interpolate(s.0.clone())));
        let w_k_idx = symbols.get_or_create(bindings.interpolate(self.w_k.0.clone()));
        let s_k_idx = self.s_k.as_ref().map(|s| symbols.get_or_create(bindings.interpolate(s.0.clone())));
        let w_v_idx = symbols.get_or_create(bindings.interpolate(self.w_v.0.clone()));
        let s_v_idx = self.s_v.as_ref().map(|s| symbols.get_or_create(bindings.interpolate(s.0.clone())));
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

        let is_q8 = s_q_idx.is_some();
        let quant = if is_q8 { Quantization::Q8 } else { Quantization::F16 };

        // Ensure kernel is compiled
        get_fused_qkv_kernel(self.strategy, quant, gamma_idx.is_some());

        vec![Box::new(CompiledFusedQkvStep {
            step: self.clone(),
            input_idx,
            w_q_idx,
            s_q_idx,
            w_k_idx,
            s_k_idx,
            w_v_idx,
            s_v_idx,
            out_q_idx,
            out_k_idx,
            out_v_idx,
            gamma_idx,
            bias_q_idx,
            bias_k_idx,
            bias_v_idx,
            is_q8,
        })]
    }
}

impl CompiledStep for CompiledFusedQkvStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &crate::foundry::spec::FastBindings,
        bindings: &TensorBindings,
    ) -> Result<(), MetalError> {
        let get = |idx| fast_bindings.get(idx).ok_or(MetalError::InputNotFound("".into()));
        let input = get(self.input_idx)?;
        let w_q = get(self.w_q_idx)?;
        let s_q = if let Some(idx) = self.s_q_idx { get(idx)? } else { w_q };
        let w_k = get(self.w_k_idx)?;
        let s_k = if let Some(idx) = self.s_k_idx { get(idx)? } else { w_k };
        let w_v = get(self.w_v_idx)?;
        let s_v = if let Some(idx) = self.s_v_idx { get(idx)? } else { w_v };
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
            w_q: TensorArg::from_tensor(w_q),
            s_q: TensorArg::from_tensor(s_q),
            w_k: TensorArg::from_tensor(w_k),
            s_k: TensorArg::from_tensor(s_k),
            w_v: TensorArg::from_tensor(w_v),
            s_v: TensorArg::from_tensor(s_v),
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
        };

        let kernel = get_fused_qkv_kernel(
            self.step.strategy,
            if self.is_q8 { Quantization::Q8 } else { Quantization::F16 },
            self.gamma_idx.is_some(),
        );

        // Manual dispatch to support M dimension (batch)
        const WARPS_PER_TG: usize = 8;
        const SIMD_WIDTH: usize = 32;
        const TG_WIDTH: usize = WARPS_PER_TG * SIMD_WIDTH; // 256
        let num_tgs = (n_dim.max(n_kv) as usize + WARPS_PER_TG - 1) / WARPS_PER_TG;

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

        foundry.run(&kernel.bind(args, dispatch))?;

        if bindings.get_var("i").map(|v| v == "0").unwrap_or(false) {
            foundry.synchronize()?;
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "FusedQkv"
    }
}

fn get_fused_qkv_kernel(strategy: GemvStrategy, quant: Quantization, has_norm: bool) -> &'static CompiledCompoundKernel {
    static KERNELS: OnceLock<
        std::sync::Mutex<std::collections::HashMap<(GemvStrategy, Quantization, bool), &'static CompiledCompoundKernel>>,
    > = OnceLock::new();
    let cache = KERNELS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut cache = cache.lock().unwrap();

    let key = (strategy, quant, has_norm);
    if let Some(kernel) = cache.get(&key) {
        return *kernel;
    }

    let policy_name = quant.policy_name();
    let norm_suffix = if has_norm { "_rmsnorm" } else { "" };
    let kernel_name = format!("fused_qkv{}_{}", norm_suffix, policy_name.to_lowercase().replace("policy", ""));

    let mut compound = CompoundKernel::new(&kernel_name)
        .with_manual_output(true)
        .prologue(WarpLayoutStage::canonical().with_warps(8))
        .prologue(RmsNormComputeStage::new(6, 7)); // Stage 1: RMSNorm compute

    // Stage 2: QKV Projection
    let mut proj = ParallelProjectStage::new(quant);
    if has_norm {
        proj = proj.with_norm(18, "inv_rms");
    }
    compound = compound.main(proj);

    compound = compound
        .epilogue(MultiWarpReduceStage::default()) // Stage 3: Reduction
        .epilogue(MultiWriteOutputStage); // Stage 4: Write Out

    let compiled = Box::new(compound.compile());
    let static_ref = Box::leak(compiled);
    cache.insert(key, static_ref);
    static_ref
}
