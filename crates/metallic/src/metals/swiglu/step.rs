use std::sync::OnceLock;

use metallic_macros::KernelArgs;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::{
    SwigluParamsResolved, ffn_stages::{FfnDualProjectStage, FfnSwigluWriteStage, FfnWarpReduceStage}, stages::SwigluStage
};
use crate::{
    MetalError, compound::{
        CompiledCompoundKernel, CompoundKernel, stages::{Quantization, WarpLayoutStage}
    }, foundry::{
        Foundry, spec::{CompiledStep, DynamicValue, FastBindings, Ref, ResolvedSymbols, Step, SymbolTable, TensorBindings}
    }, metals::{gemv::step::warp_dispatch_config_2d, rmsnorm::stages::RmsNormComputeStage}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
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
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
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
    pub wg_name: String,
    pub wg_resolved: ResolvedSymbols,
    pub wu_name: String,
    pub wu_resolved: ResolvedSymbols,
    pub bg_idx: Option<usize>,
    pub bu_idx: Option<usize>,
    pub out_idx: usize,
}

#[typetag::serde(name = "SwiGluF16CanonicalFusedRmsnorm")]
impl Step for FusedSwigluStep {
    fn name(&self) -> &'static str {
        "SwiGluF16CanonicalFusedRmsnorm"
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
        let wg_name = bindings.interpolate(self.wg.0.clone());
        let wu_name = bindings.interpolate(self.wu.0.clone());
        let wg_idx = symbols.get_or_create(wg_name.clone());
        let _wg_scales_idx = symbols.get_or_create(format!("{wg_name}_scales"));
        let wu_idx = symbols.get_or_create(wu_name.clone());
        let _wu_scales_idx = symbols.get_or_create(format!("{wu_name}_scales"));
        let bg_idx = self.bg.as_ref().map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));
        let bu_idx = self.bu.as_ref().map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));
        let out_idx = symbols.get_or_create(bindings.interpolate(self.out.0.clone()));

        vec![Box::new(CompiledFusedSwigluStep {
            step: self.clone(),
            input_idx,
            gamma_idx,
            wg_name: wg_name.clone(),
            wg_resolved: ResolvedSymbols {
                weights: wg_idx,
                scales: _wg_scales_idx.into(),
                bias: None,
            },
            wu_name: wu_name.clone(),
            wu_resolved: ResolvedSymbols {
                weights: wu_idx,
                scales: _wu_scales_idx.into(),
                bias: None,
            },
            bg_idx,
            bu_idx,
            out_idx,
        })]
    }
}

impl CompiledStep for CompiledFusedSwigluStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        _fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let get = |idx| _fast_bindings.get(idx).ok_or(MetalError::InputNotFound("".into()));

        let input = get(self.input_idx)?;
        let gamma = get(self.gamma_idx)?;
        let w_gate = get(self.wg_resolved.weights)?;
        let w_up = get(self.wu_resolved.weights)?;
        let output = get(self.out_idx)?;

        let (b_gate, has_b_gate) = if let Some(idx) = self.bg_idx {
            (TensorArg::from_tensor(get(idx)?), 1u32)
        } else {
            (TensorArg::from_tensor(output), 0u32)
        };

        let (b_up, has_b_up) = if let Some(idx) = self.bu_idx {
            (TensorArg::from_tensor(get(idx)?), 1u32)
        } else {
            (TensorArg::from_tensor(output), 0u32)
        };

        let input_shape = input.dims.as_slice();
        let output_shape = output.dims.as_slice();
        if input_shape.is_empty() || output_shape.is_empty() {
            return Err(MetalError::InvalidShape("FusedSwiglu expects non-empty input/output shapes".into()));
        }

        let k_dim = *input_shape.last().unwrap() as u32;
        let n_dim = *output_shape.last().unwrap() as u32;

        if gamma.dims.len() != 1 || gamma.dims[0] as u32 != k_dim {
            return Err(MetalError::DimensionMismatch {
                expected: k_dim as usize,
                actual: gamma.dims.iter().product(),
            });
        }
        if has_b_gate != 0 && (b_gate.dims.len() != 1 || b_gate.dims[0] as u32 != n_dim) {
            return Err(MetalError::DimensionMismatch {
                expected: n_dim as usize,
                actual: b_gate.dims.iter().product(),
            });
        }
        if has_b_up != 0 && (b_up.dims.len() != 1 || b_up.dims[0] as u32 != n_dim) {
            return Err(MetalError::DimensionMismatch {
                expected: n_dim as usize,
                actual: b_up.dims.iter().product(),
            });
        }

        let weights_per_block = self.step.weights_per_block;
        let batch = bindings.get_int_global("m").unwrap_or(1).max(1) as u32;

        let policy_gate = crate::foundry::policy::resolve_policy(w_gate.dtype.into());
        let loader_gate = policy_gate.loader_stage();
        let args_gate = loader_gate.bind(_fast_bindings, &self.wg_resolved);

        let policy_up = crate::foundry::policy::resolve_policy(w_up.dtype.into());
        let loader_up = policy_up.loader_stage();
        let args_up = loader_up.bind(_fast_bindings, &self.wu_resolved);

        let (w_gate_arg, s_gate) = (args_gate[0].clone(), args_gate[1].clone());
        let (w_up_arg, s_up) = (args_up[0].clone(), args_up[1].clone());
        let quantization = loader_gate.quantization_type();

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
        };

        let kernel = get_fused_ffn_kernel(quantization);
        let dispatch = warp_dispatch_config_2d(n_dim, batch);

        foundry.run(&kernel.bind(args, dispatch))
    }

    fn name(&self) -> &'static str {
        "FusedSwiglu"
    }
}

// =============================================================================
// FusedFfnSwiGluRmsNorm Step (FoundryV2)
// =============================================================================

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
}

#[derive(Debug, KernelArgs)]
pub struct FusedFfnArgs {
    #[arg(buffer = 0)]
    pub w_gate: TensorArg,
    #[arg(buffer = 1)]
    pub s_gate: TensorArg,
    #[arg(buffer = 2)]
    pub w_up: TensorArg,
    #[arg(buffer = 3)]
    pub s_up: TensorArg,
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
        let _w_gate_scales_idx = symbols.get_or_create(format!("{w_gate_name}_scales"));
        let w_up_idx = symbols.get_or_create(w_up_name.clone());
        let _w_up_scales_idx = symbols.get_or_create(format!("{w_up_name}_scales"));
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
                scales: _w_gate_scales_idx.into(),
                bias: None,
            },
            w_up_name: w_up_name.clone(),
            w_up_resolved: ResolvedSymbols {
                weights: w_up_idx,
                scales: _w_up_scales_idx.into(),
                bias: None,
            },
            b_gate_idx,
            b_up_idx,
            output_idx,
            weights_per_block: self.weights_per_block,
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

        let policy_gate = crate::foundry::policy::resolve_policy(w_gate.dtype.into());
        let loader_gate = policy_gate.loader_stage();
        let args_gate = loader_gate.bind(fast_bindings, &self.w_gate_resolved);

        let policy_up = crate::foundry::policy::resolve_policy(w_up.dtype.into());
        let loader_up = policy_up.loader_stage();
        let args_up = loader_up.bind(fast_bindings, &self.w_up_resolved);

        let (w_gate_arg, s_gate) = (args_gate[0].clone(), args_gate[1].clone());
        let (w_up_arg, s_up) = (args_up[0].clone(), args_up[1].clone());
        let is_q8 = loader_gate.quantization_type() == Quantization::Q8;

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

        let input_shape = input.dims.as_slice();
        let output_shape = output.dims.as_slice();
        if input_shape.is_empty() || output_shape.is_empty() {
            return Err(MetalError::InvalidShape(
                "FusedFfnSwiGluRmsNorm expects non-empty input/output shapes".into(),
            ));
        }

        let k_dim = *input_shape.last().unwrap() as u32;
        let n_dim = *output_shape.last().unwrap() as u32;

        if gamma.dims.len() != 1 || gamma.dims[0] as u32 != k_dim {
            return Err(MetalError::DimensionMismatch {
                expected: k_dim as usize,
                actual: gamma.dims.iter().product(),
            });
        }
        if has_b_gate != 0 && (b_gate.dims.len() != 1 || b_gate.dims[0] as u32 != n_dim) {
            return Err(MetalError::DimensionMismatch {
                expected: n_dim as usize,
                actual: b_gate.dims.iter().product(),
            });
        }
        if has_b_up != 0 && (b_up.dims.len() != 1 || b_up.dims[0] as u32 != n_dim) {
            return Err(MetalError::DimensionMismatch {
                expected: n_dim as usize,
                actual: b_up.dims.iter().product(),
            });
        }

        let batch = bindings.get_int_global("m").unwrap_or(1).max(1) as u32;

        let args = FusedFfnArgs {
            w_gate: w_gate_arg,
            s_gate,
            w_up: w_up_arg,
            s_up,
            input: TensorArg::from_tensor(input),
            output: TensorArg::from_tensor(output),
            k_dim,
            n_dim,
            weights_per_block: self.weights_per_block,
            gamma: TensorArg::from_tensor(gamma),
            b_gate,
            b_up,
            has_b_gate,
            has_b_up,
        };

        let kernel = get_fused_ffn_kernel(if is_q8 { Quantization::Q8 } else { Quantization::F16 });
        let dispatch = warp_dispatch_config_2d(n_dim, batch);

        foundry.run(&kernel.bind(args, dispatch))
    }

    fn name(&self) -> &'static str {
        "FusedFfnSwiGluRmsNorm"
    }
}

fn get_fused_ffn_kernel(quant: Quantization) -> &'static CompiledCompoundKernel {
    static KERNELS: OnceLock<std::sync::Mutex<FxHashMap<Quantization, &'static CompiledCompoundKernel>>> = OnceLock::new();
    let cache = KERNELS.get_or_init(|| std::sync::Mutex::new(FxHashMap::default()));
    let mut cache = cache.lock().unwrap();

    if let Some(kernel) = cache.get(&quant) {
        return *kernel;
    }

    let policy_name = quant.policy_name();
    let kernel_name = format!("fused_ffn_swiglu_rmsnorm_{}", policy_name.to_lowercase().replace("policy", ""));

    let compiled = Box::new(
        CompoundKernel::new(&kernel_name)
            .with_manual_output(true)
            .prologue(WarpLayoutStage::row_major().with_warps(8))
            // Activation input is always F16; quantization only applies to weight loading stages.
            .prologue(RmsNormComputeStage::new(4, 6))
            .main(FfnDualProjectStage::new(quant).with_norm(9, "inv_rms"))
            .epilogue(FfnWarpReduceStage)
            .epilogue(FfnSwigluWriteStage)
            .compile(),
    );

    let static_ref = Box::leak(compiled);
    cache.insert(quant, static_ref);
    static_ref
}
