use std::sync::OnceLock;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::{
    qkv_stages::{MultiWarpReduceStage, MultiWriteOutputStage, ParallelProjectStage}, step::{GemvStrategy, warp_dispatch_config}
};
use crate::{
    MetalError, compound::{
        CompiledCompoundKernel, CompoundKernel, stages::{Layout, Quantization, WarpLayoutStage}
    }, foundry::{
        Foundry, spec::{CompiledStep, DynamicValue, Ref, Step, SymbolTable, TensorBindings}
    }, metals::rmsnorm::stages::RmsNormComputeStage, types::TensorArg
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
    pub gamma: Ref,
    pub bias_q: Option<Ref>,
    pub bias_k: Option<Ref>,
    pub bias_v: Option<Ref>,
    pub k_dim: DynamicValue<u32>,
    pub n_dim: DynamicValue<u32>,
    pub n_kv: DynamicValue<u32>,
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
    pub gamma_idx: usize,
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
        let gamma_idx = symbols.get_or_create(bindings.interpolate(self.gamma.0.clone()));
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
        get_fused_qkv_kernel(self.strategy, quant);

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

fn read_f16_buffer(arg: &crate::types::TensorArg) -> Vec<half::f16> {
    use objc2_metal::MTLBuffer;
    let buffer = arg.buffer.as_ref().expect("No buffer in TensorArg");
    let size = arg.dims.iter().product::<usize>();
    let mut data = vec![half::f16::from_f32(0.0); size];
    unsafe {
        let ptr_base = buffer.contents().as_ptr() as *const u8;
        let ptr = ptr_base.add(arg.offset);
        eprintln!(
            "DEBUG read_f16_buffer: ptr={:?} offset={} size={} product={}",
            ptr_base,
            arg.offset,
            size,
            size * 2
        );
        std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr() as *mut u8, size * 2);
    }
    data
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
        let gamma = get(self.gamma_idx)?;

        let (b_q, has_bias) = if let Some(idx) = self.bias_q_idx {
            (TensorArg::from_tensor(fast_bindings.get(idx).unwrap()), 1)
        } else {
            (TensorArg::from_tensor(input), 0)
        };
        let b_k = self
            .bias_k_idx
            .map(|idx| TensorArg::from_tensor(fast_bindings.get(idx).unwrap()))
            .unwrap_or_else(|| TensorArg::from_tensor(input));
        let b_v = self
            .bias_v_idx
            .map(|idx| TensorArg::from_tensor(fast_bindings.get(idx).unwrap()))
            .unwrap_or_else(|| TensorArg::from_tensor(input));

        let b_q_saved = b_q.clone();

        let k_dim = self.step.k_dim.resolve(bindings);
        let n_dim = self.step.n_dim.resolve(bindings);
        let n_kv = self.step.n_kv.resolve(bindings);
        let weights_per_block = self.step.weights_per_block.resolve(bindings);

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
            b_q,
            b_k,
            b_v,
            has_bias,
            gamma: TensorArg::from_tensor(gamma),
        };

        let kernel = get_fused_qkv_kernel(self.step.strategy, if self.is_q8 { Quantization::Q8 } else { Quantization::F16 });
        let dispatch = warp_dispatch_config(n_dim.max(n_kv));

        if bindings.get_var("i").map(|v| v == "0").unwrap_or(false) {
            eprintln!("DEBUG FusedQkv Source:\n{}", kernel.source());
            eprintln!(
                "DEBUG FusedQkv Layer 0: k_dim={} n_dim={} n_kv={} weights_per_block={} | q_off={} k_off={} v_off={} input_off={}",
                k_dim, n_dim, n_kv, weights_per_block, out_q.offset, out_k.offset, out_v.offset, input.offset
            );
            let w_q_data = read_f16_buffer(&TensorArg::from_tensor(w_q));
            eprintln!(
                "DEBUG FusedQkv Weights (w_q) sample (first 5): {:?}",
                &w_q_data[..5.min(w_q_data.len())]
            );
            eprintln!(
                "DEBUG FusedQkv Weights (w_q) non-zeros: {} / {}",
                w_q_data.iter().filter(|v| v.to_f32().abs() > 1e-6).count(),
                w_q_data.len()
            );
        }
        if bindings.get_var("i").map(|v| v == "0").unwrap_or(false) {
            use objc2_metal::MTLBuffer;
            let out_q_ptr = args
                .out_q
                .buffer
                .as_ref()
                .map(|b| b.contents().as_ptr())
                .unwrap_or(std::ptr::null_mut());
            eprintln!("DEBUG FusedQkv out_q buffer ptr BEFORE run: {:?}", out_q_ptr);
        }
        foundry.run(&kernel.bind(args, dispatch))?;

        if bindings.get_var("i").map(|v| v == "0").unwrap_or(false) {
            foundry.synchronize()?;
            let input_data = read_f16_buffer(&input);
            eprintln!(
                "DEBUG FusedQkv Input (hidden) sample (first 5): {:?}",
                &input_data[..5.min(input_data.len())]
            );
            eprintln!(
                "DEBUG FusedQkv Input (hidden) non-zeros: {} / {}",
                input_data.iter().filter(|v| v.to_f32().abs() > 1e-6).count(),
                input_data.len()
            );

            let q_data = read_f16_buffer(&out_q);
            eprintln!("DEBUG FusedQkv OutQ sample (first 5): {:?}", &q_data[..5.min(q_data.len())]);
            if q_data.len() > 666 {
                eprintln!("DEBUG FusedQkv OutQ index 666: {:?}", q_data[666]);
            }
            if has_bias > 0 {
                let bq_data = read_f16_buffer(&b_q_saved);
                if bq_data.len() > 666 {
                    eprintln!("DEBUG FusedQkv BiasQ index 666: {:?}", bq_data[666]);
                }
            }
            eprintln!(
                "DEBUG FusedQkv OutQ non-zeros: {} / {}",
                q_data.iter().filter(|v| v.to_f32().abs() > 1e-6).count(),
                q_data.len()
            );
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "FusedQkv"
    }
}

fn get_fused_qkv_kernel(_strategy: GemvStrategy, quant: Quantization) -> &'static CompiledCompoundKernel {
    match quant {
        Quantization::Q8 => {
            static QKV_Q8: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            QKV_Q8.get_or_init(|| {
                CompoundKernel::new("fused_qkv_rmsnorm_q8")
                    .with_manual_output(true)
                    .prologue(WarpLayoutStage::new(Layout::Canonical).with_warps(8))
                    .prologue(RmsNormComputeStage::new(6, 7))
                    .main(ParallelProjectStage::new(Quantization::Q8).with_norm(18, "inv_rms"))
                    .epilogue(MultiWarpReduceStage::default())
                    .epilogue(MultiWriteOutputStage)
                    .compile()
            })
        }
        Quantization::F16 => {
            static QKV_F16: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            QKV_F16.get_or_init(|| {
                CompoundKernel::new("fused_qkv_rmsnorm_f16")
                    .with_manual_output(true)
                    .prologue(WarpLayoutStage::new(Layout::Canonical).with_warps(8))
                    .prologue(RmsNormComputeStage::new(6, 7))
                    .main(ParallelProjectStage::new(Quantization::F16).with_norm(18, "inv_rms"))
                    .epilogue(MultiWarpReduceStage::default())
                    .epilogue(MultiWriteOutputStage)
                    .compile()
            })
        }
    }
}
