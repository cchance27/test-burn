use std::sync::{Arc, OnceLock};

use metallic_macros::KernelArgs;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;
use serde::{Deserialize, Serialize};

use crate::{
    MetalError, compound::{CompiledCompoundKernel, CompoundKernel}, foundry::{
        Foundry, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}
    }, metals::{
        rope::{RopeParams, RopeParamsResolved, stage::RopeStage}, sdpa::stages::{HeadLayoutStage, SdpaOnlineStage, SdpaParams, SdpaParamsResolved}
    }, types::{
        TensorArg, dispatch::{DispatchConfig, GridSize, ThreadgroupSize}
    }
};

/// Get the static compiled kernel template.
fn get_fused_mha_kernel() -> &'static CompiledCompoundKernel {
    static KERNEL: OnceLock<CompiledCompoundKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        let name = "fused_mha_rope_decode_v2";

        // Use dummy values for template definition - only types/structure matter here
        let dummy_tensor = TensorArg::default();
        let dummy_layout = HeadLayoutStage::new(
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        );
        let dummy_rope = RopeStage::new(dummy_tensor.clone(), dummy_tensor.clone(), RopeParams::default());
        let dummy_core = SdpaOnlineStage::new(SdpaParams::default());

        let kernel = CompoundKernel::new(name)
            .prologue(dummy_layout)
            .prologue(dummy_rope)
            .main(dummy_core)
            .with_manual_output(true)
            .compile();

        let source_str = match crate::foundry::Kernel::source(&kernel) {
            crate::foundry::KernelSource::String(s) => s,
            _ => "N/A".to_string(),
        };
        println!(
            "\n=== FUSED MHA KERNEL SOURCE ===\n{}\n==============================\n",
            source_str
        );
        kernel
    })
}

/// Fused Multi-Head Attention + RoPE Kernel Step (Decode).
///
/// Manually compiled step that bypasses the DSL Step trait for now,
/// directly returning a CompiledFusedMhaStep.
pub struct FusedMhaStep;

impl FusedMhaStep {
    /// Get the generated Metal source code for this kernel.
    pub fn source() -> String {
        let kernel = get_fused_mha_kernel();
        match crate::foundry::Kernel::source(kernel) {
            crate::foundry::KernelSource::String(s) => s,
            _ => "Source not available (Binary/Other)".to_string(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        _foundry: &mut Foundry,
        q: &TensorArg,
        k: &TensorArg,
        v: &TensorArg,
        cos: &TensorArg,
        sin: &TensorArg,
        output: &TensorArg,
        rope_params: RopeParamsResolved,
        sdpa_params: SdpaParamsResolved,
        batch: u32,
        heads: u32,
        head_dim: u32,
        q_strides: (u32, u32),
        k_strides: (u32, u32),
        v_strides: (u32, u32),
        out_strides: (u32, u32),
    ) -> Result<CompiledFusedMhaStep, MetalError> {
        // Ensure kernel is initialized
        let kernel = get_fused_mha_kernel();
        let source = match crate::foundry::Kernel::source(kernel) {
            crate::foundry::KernelSource::String(s) => s,
            _ => "N/A".to_string(),
        };

        Ok(CompiledFusedMhaStep {
            source,
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            cos: cos.clone(),
            sin: sin.clone(),
            output: output.clone(),
            params_rope: rope_params,
            sdpa_params,
            batch,
            heads,
            head_dim,
            q_strides,
            k_strides,
            v_strides,
            out_strides,
        })
    }
}

/// Arguments for the Fused MHA Kernel.
/// Checks strict order matching the Stages: Layout -> Rope -> Core.
#[derive(KernelArgs)]
struct FusedMhaArgs {
    // Layout Stage
    q: TensorArg,
    k: TensorArg,
    v: TensorArg,
    output: TensorArg,
    q_stride_b: u32,
    q_stride_h: u32,
    k_stride_b: u32,
    k_stride_h: u32,
    v_stride_b: u32,
    v_stride_h: u32,
    out_stride_b: u32,
    out_stride_h: u32,

    // Rope Stage
    cos: TensorArg,
    sin: TensorArg,
    params_rope: RopeParamsResolved,

    // Core Stage
    sdpa_params: SdpaParamsResolved,
}

#[derive(Debug, Default)]
pub struct CompiledFusedMhaStep {
    pub source: String,
    q: TensorArg,
    k: TensorArg,
    v: TensorArg,
    cos: TensorArg,
    sin: TensorArg,
    output: TensorArg,

    params_rope: RopeParamsResolved,
    sdpa_params: SdpaParamsResolved,

    q_strides: (u32, u32),
    k_strides: (u32, u32),
    v_strides: (u32, u32),
    out_strides: (u32, u32),

    batch: u32,
    heads: u32,
    head_dim: u32,
}

impl CompiledFusedMhaStep {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q: TensorArg,
        k: TensorArg,
        v: TensorArg,
        cos: TensorArg,
        sin: TensorArg,
        output: TensorArg,
        params_rope: RopeParamsResolved,
        sdpa_params: SdpaParamsResolved,
        batch: u32,
        heads: u32,
        head_dim: u32,
        q_strides: (u32, u32),
        k_strides: (u32, u32),
        v_strides: (u32, u32),
        out_strides: (u32, u32),
        source: String,
    ) -> Self {
        Self {
            source,
            q,
            k,
            v,
            cos,
            sin,
            output,
            params_rope,
            sdpa_params,
            batch,
            heads,
            head_dim,
            q_strides,
            k_strides,
            v_strides,
            out_strides,
        }
    }
}

impl CompiledStep for CompiledFusedMhaStep {
    fn execute(&self, foundry: &mut Foundry, _fast_bindings: &FastBindings, _globals: &TensorBindings) -> Result<(), MetalError> {
        // Construct Args
        let args = FusedMhaArgs {
            q: self.q.clone(),
            k: self.k.clone(),
            v: self.v.clone(),
            output: self.output.clone(),
            q_stride_b: self.q_strides.0,
            q_stride_h: self.q_strides.1,
            k_stride_b: self.k_strides.0,
            k_stride_h: self.k_strides.1,
            v_stride_b: self.v_strides.0,
            v_stride_h: self.v_strides.1,
            out_stride_b: self.out_strides.0,
            out_stride_h: self.out_strides.1,
            cos: self.cos.clone(),
            sin: self.sin.clone(),
            params_rope: self.params_rope,
            sdpa_params: self.sdpa_params,
        };
        // Dispatch Configuration
        // Grid: (1, Heads, Batch) -> one threadgroup per head per batch
        // Group: (HeadDim/4, 1, 1) -> vec_dim threads handle one head vectorized
        let vec_dim = self.head_dim / 4;
        let grid = GridSize::new(1, self.heads as usize, self.batch as usize);
        let group = ThreadgroupSize::d1(vec_dim as usize);
        let config = DispatchConfig::new(grid, group);

        let kernel = get_fused_mha_kernel();
        let bound = kernel.bind(args, config);
        foundry.run(&bound)
    }

    fn name(&self) -> &'static str {
        "FusedMha"
    }
}

// =============================================================================
// DSL-Compatible SdpaStep (legacy "Sdpa" op mapping)
// =============================================================================

/// SDPA Step for DSL compatibility.
/// Maps to the legacy "Sdpa" op in model specs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdpaStep {
    pub q: Ref,
    pub k: Ref,
    pub v: Ref,
    pub output: Ref,
    #[serde(default)]
    pub causal: bool,
    pub n_heads: DynamicValue<u32>,
    pub head_dim: DynamicValue<u32>,
    pub kv_seq_len: DynamicValue<u32>,
    #[serde(default)]
    pub query_offset: DynamicValue<u32>,
    #[serde(default)]
    pub kv_head_major: bool,
}

#[derive(Debug, Clone)]
pub struct CompiledSdpaStep {
    pub step: SdpaStep,
    pub q_idx: usize,
    pub k_idx: usize,
    pub v_idx: usize,
    pub output_idx: usize,
    pub pipeline: Arc<OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

#[typetag::serde(name = "Sdpa")]
impl Step for SdpaStep {
    fn name(&self) -> &'static str {
        "Sdpa"
    }

    fn execute(&self, _foundry: &mut Foundry, _bindings: &mut TensorBindings) -> Result<(), MetalError> {
        Err(MetalError::OperationNotSupported("Sdpa only supports compile()".into()))
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let q_idx = symbols.get_or_create(bindings.interpolate(self.q.0.clone()));
        let k_idx = symbols.get_or_create(bindings.interpolate(self.k.0.clone()));
        let v_idx = symbols.get_or_create(bindings.interpolate(self.v.0.clone()));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));

        vec![Box::new(CompiledSdpaStep {
            step: self.clone(),
            q_idx,
            k_idx,
            v_idx,
            output_idx,
            pipeline: Arc::new(OnceLock::new()),
        })]
    }
}

/// Get or create static SDPA kernel (standalone, no RoPE fusion)
fn get_sdpa_standalone_kernel() -> &'static CompiledCompoundKernel {
    static KERNEL: OnceLock<CompiledCompoundKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        use crate::metals::sdpa::stages::SdpaStandaloneStage;

        let dummy_tensor = TensorArg::default();
        let dummy_layout = HeadLayoutStage::new(
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        );
        // Use SdpaStandaloneStage which loads Q from buffer directly
        let dummy_core = SdpaStandaloneStage::new(SdpaParams::default());

        CompoundKernel::new("sdpa_standalone_v2")
            .prologue(dummy_layout)
            .main(dummy_core)
            .with_manual_output(true)
            .compile()
    })
}

/// Arguments for standalone SDPA (no RoPE)
#[derive(Debug, KernelArgs)]
struct SdpaStandaloneArgs {
    q: TensorArg,
    k: TensorArg,
    v: TensorArg,
    #[arg(output)]
    output: TensorArg,
    q_stride_b: u32,
    q_stride_h: u32,
    k_stride_b: u32,
    k_stride_h: u32,
    v_stride_b: u32,
    v_stride_h: u32,
    out_stride_b: u32,
    out_stride_h: u32,
    sdpa_params: SdpaParamsResolved,
}

impl CompiledStep for CompiledSdpaStep {
    fn execute(&self, foundry: &mut Foundry, fast_bindings: &FastBindings, bindings: &TensorBindings) -> Result<(), MetalError> {
        let q = fast_bindings.get(self.q_idx).ok_or(MetalError::InputNotFound("q".into()))?;
        let k = fast_bindings.get(self.k_idx).ok_or(MetalError::InputNotFound("k".into()))?;
        let v = fast_bindings.get(self.v_idx).ok_or(MetalError::InputNotFound("v".into()))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or(MetalError::InputNotFound("output".into()))?;

        let n_heads = self.step.n_heads.resolve(bindings);
        let head_dim = self.step.head_dim.resolve(bindings);
        let kv_seq_len = self.step.kv_seq_len.resolve(bindings);

        // Compute strides based on head-major layout
        let batch = 1u32;
        let (q_stride_b, q_stride_h) = if self.step.kv_head_major {
            (n_heads * head_dim, head_dim)
        } else {
            (head_dim, n_heads * head_dim)
        };
        let (k_stride_b, k_stride_h) = (n_heads * kv_seq_len * head_dim, kv_seq_len * head_dim);
        let (v_stride_b, v_stride_h) = (n_heads * kv_seq_len * head_dim, kv_seq_len * head_dim);
        let (out_stride_b, out_stride_h) = (n_heads * head_dim, head_dim);

        let scale = 1.0 / (head_dim as f32).sqrt();

        let sdpa_params = SdpaParamsResolved {
            kv_len: kv_seq_len,
            head_dim,
            scale,
            stride_k_s: head_dim, // Stride between sequence positions in K
            stride_v_s: head_dim, // Stride between sequence positions in V
        };

        let args = SdpaStandaloneArgs {
            q: TensorArg::from_tensor(q),
            k: TensorArg::from_tensor(k),
            v: TensorArg::from_tensor(v),
            output: TensorArg::from_tensor(output),
            q_stride_b,
            q_stride_h,
            k_stride_b,
            k_stride_h,
            v_stride_b,
            v_stride_h,
            out_stride_b,
            out_stride_h,
            sdpa_params,
        };

        // Dispatch: Grid (1, Heads, Batch) with (HeadDim/4) threads per group
        let vec_dim = head_dim / 4;
        let grid = GridSize::new(1, n_heads as usize, batch as usize);
        let group = ThreadgroupSize::d1(vec_dim as usize);
        let config = DispatchConfig::new(grid, group);

        let kernel = get_sdpa_standalone_kernel();

        // Optimized dispatch
        let pipeline = if let Some(p) = self.pipeline.get() {
            p
        } else {
            let p = foundry.load_kernel(kernel)?;
            let _ = self.pipeline.set(p);
            self.pipeline.get().unwrap()
        };

        let bound = kernel.bind(args, config);
        foundry.dispatch_pipeline(pipeline, &bound, config)
    }

    fn name(&self) -> &'static str {
        "Sdpa"
    }
}

// =============================================================================
// SdpaMaterializedStep (Gemv + Softmax + Gemv)
// =============================================================================

use half::f16;

use crate::{
    compound::stages::Layout, foundry::constants, metals::{
        gemm::step::{GemmParams, GemmV2Args, gemm_dispatch_config, get_gemm_kernel}, gemv::{GemvStrategy, GemvV2Args, get_gemv_v2_kernel_f16, warp_dispatch_config}, mma::stages::TileConfig, softmax::{SoftmaxV2Args, SoftmaxV2SdpaBatchedArgs, get_softmax_v2_kernel, get_softmax_v2_sdpa_batched_kernel}
    }
};

/// Materialized SDPA (Gemv/Gemm + Softmax + Gemv/Gemm).
/// Uses GEMV for M=1 (decode) and GEMM for M>1 (prefill).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdpaMaterializedStep {
    pub q: Ref,
    pub k: Ref, // Assumed (SeqK, HeadDim)
    pub v: Ref, // Assumed (SeqK, HeadDim)
    pub output: Ref,
    #[serde(default)]
    pub causal: bool,
    pub query_offset: DynamicValue<u32>,
    pub n_heads: DynamicValue<u32>,
    pub head_dim: DynamicValue<u32>,
    pub kv_seq_len: DynamicValue<u32>,
    /// Query sequence length (M=1 for decode, M>1 for prefill)
    #[serde(default)]
    pub m: DynamicValue<u32>,
    #[serde(default)]
    pub kv_head_major: bool,
}

#[derive(Debug, Clone)]
pub struct CompiledSdpaMaterializedStep {
    pub step: SdpaMaterializedStep,
    pub q_idx: usize,
    pub k_idx: usize,
    pub v_idx: usize,
    pub output_idx: usize,
}

#[typetag::serde(name = "SdpaMaterialized")]
impl Step for SdpaMaterializedStep {
    fn name(&self) -> &'static str {
        "SdpaMaterialized"
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
        let q_idx = symbols.get_or_create(bindings.interpolate(self.q.0.clone()));
        let k_idx = symbols.get_or_create(bindings.interpolate(self.k.0.clone()));
        let v_idx = symbols.get_or_create(bindings.interpolate(self.v.0.clone()));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));

        vec![Box::new(CompiledSdpaMaterializedStep {
            step: self.clone(),
            q_idx,
            k_idx,
            v_idx,
            output_idx,
        })]
    }
}

impl CompiledStep for CompiledSdpaMaterializedStep {
    fn execute(&self, foundry: &mut Foundry, fast_bindings: &FastBindings, bindings: &TensorBindings) -> Result<(), MetalError> {
        // 1. Resolve Inputs
        let q = fast_bindings.get(self.q_idx).ok_or(MetalError::InputNotFound("q".into()))?;
        let k = fast_bindings.get(self.k_idx).ok_or(MetalError::InputNotFound("k".into()))?;
        let v = fast_bindings.get(self.v_idx).ok_or(MetalError::InputNotFound("v".into()))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or(MetalError::InputNotFound("output".into()))?;

        let head_dim = self.step.head_dim.resolve(bindings);
        let kv_seq_len = self.step.kv_seq_len.resolve(bindings);
        let n_heads = self.step.n_heads.resolve(bindings);
        let q_offset_val = self.step.query_offset.resolve(bindings);
        let m = self.step.m.resolve(bindings).max(1); // Query sequence length (M=1 for decode, M>1 for prefill)

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Bytes per element
        let elem_size = std::mem::size_of::<f16>();

        // 2. Allocate Temps (scores and probs) -> Size = n_heads * m * kv_seq_len
        // For M>1 prefill, each query produces kv_seq_len scores
        let (scores_all, probs_all) = crate::metals::sdpa::scratch::get_sdpa_scratch_f16(foundry, n_heads, m, kv_seq_len)?;

        // Softmax scaling is applied in the QK matmul (alpha=scale). For softmax itself, we use 1.0.
        // IMPORTANT: this must not allocate+upload per call; it kills decode throughput.
        let scale_arg = constants::f16_scalar(foundry, f16::ONE)?;

        // Strides
        // Q: Head-major [n_heads, M, head_dim]
        // K/V: Head-major [n_heads, Seq, head_dim]
        // Output: token-major [M, d_model] (so subsequent MatMul sees [M, K])

        // Offset (in bytes) to move from Head H to Head H+1.
        // NOTE: Q is produced by KvRearrange and is head-major; for M>1, each head occupies M*head_dim elements.
        let q_head_stride = (m as usize) * (head_dim as usize) * elem_size;

        // K/V may be either:
        // - tightly packed history (e.g. RepeatKvHeads output): [n_heads, kv_seq_len, head_dim]
        // - full cache view: [n_heads, max_seq_len, head_dim] (stride uses max_seq_len)
        let k_seq_stride = k.dims.get(1).copied().unwrap_or(kv_seq_len as usize);
        let v_seq_stride = v.dims.get(1).copied().unwrap_or(kv_seq_len as usize);
        let k_head_stride = k_seq_stride * (head_dim as usize) * elem_size;
        let v_head_stride = v_seq_stride * (head_dim as usize) * elem_size;

        let out_head_stride = (head_dim as usize) * elem_size;
        let scratch_head_stride = (m as usize) * (kv_seq_len as usize) * elem_size;

        let d_model_dim = (n_heads as usize) * (head_dim as usize);

        // Get softmax kernel (same for both paths)
        let softmax_kernel = get_softmax_v2_kernel();

        let force_gemv_decode = std::env::var("METALLIC_SDPA_DECODE_GEMV").is_ok();

        if m == 1 && force_gemv_decode {
            // =========== DECODE PATH (M=1): Legacy per-head GEMV ===========
            // NOTE: This is significantly slower due to per-head dispatch overhead. The default path
            // uses batched GEMM even for M=1.
            let qk_kernel = get_gemv_v2_kernel_f16(Layout::RowMajor, GemvStrategy::Vectorized);
            let qk_dispatch = warp_dispatch_config(kv_seq_len);

            let softmax_dispatch = DispatchConfig {
                grid: GridSize::d2(1, 1),
                group: ThreadgroupSize::d1(256),
            };

            let av_kernel = get_gemv_v2_kernel_f16(Layout::ColMajor, GemvStrategy::Vectorized);
            let av_dispatch = warp_dispatch_config(head_dim);

            for h in 0..n_heads {
                let h_idx = h as usize;

                // Offset Inputs
                let mut q_h = TensorArg::from_tensor(q);
                q_h.offset += h_idx * q_head_stride;

                let mut k_h = TensorArg::from_tensor(k);
                k_h.offset += h_idx * k_head_stride;

                let mut v_h = TensorArg::from_tensor(v);
                v_h.offset += h_idx * v_head_stride;

                let mut out_h = TensorArg::from_tensor(output);
                out_h.offset += h_idx * out_head_stride;

                // Offset Scratch
                let mut scores_h = scores_all.clone();
                scores_h.offset += h_idx * scratch_head_stride;

                let mut probs_h = probs_all.clone();
                probs_h.offset += h_idx * scratch_head_stride;

                // GEMV 1: Q @ K^T -> Scores
                let qk_args = GemvV2Args {
                    weights: k_h.clone(),
                    scale_bytes: k_h.clone(),
                    input: q_h,
                    output: scores_h.clone(),
                    bias: scores_h.clone(),
                    has_bias: 0,
                    k_dim: head_dim,
                    n_dim: kv_seq_len,
                    weights_per_block: 32,
                    alpha: scale,
                    residual: scores_h.clone(),
                    has_residual: 0,
                    beta: 0.0,
                };
                foundry.run(&qk_kernel.bind(qk_args, qk_dispatch))?;

                // Softmax: Scores -> Probs
                let softmax_args = SoftmaxV2Args {
                    input: scores_h,
                    scale: scale_arg.clone(),
                    output: probs_h.clone(),
                    seq_k: kv_seq_len,
                    causal: if self.step.causal { 1 } else { 0 },
                    query_offset: q_offset_val,
                };
                foundry.run(&softmax_kernel.bind(softmax_args, softmax_dispatch))?;

                // GEMV 2: Probs @ V -> Output
                let out_h_arg = out_h.clone();
                let av_args = GemvV2Args {
                    weights: v_h.clone(),
                    scale_bytes: v_h,
                    input: probs_h,
                    output: out_h_arg.clone(),
                    bias: out_h_arg.clone(),
                    has_bias: 0,
                    k_dim: kv_seq_len,
                    n_dim: head_dim,
                    weights_per_block: 32,
                    alpha: 1.0,
                    residual: out_h_arg,
                    has_residual: 0,
                    beta: 0.0,
                };
                foundry.run(&av_kernel.bind(av_args, av_dispatch))?;
            }
        } else {
            // =========== DEFAULT PATH: Batched GEMM over heads (M>=1) ===========
            use crate::compound::stages::Quantization;

            let tile_config = TileConfig::default();

            // Get GEMM kernels for Q@K^T and Probs@V
            // Q@K^T: [M, head_dim] @ [head_dim, kv_seq_len] = [M, kv_seq_len]
            // K is stored as [kv_seq_len, head_dim] row-major, so K^T is ColMajor read
            let qk_gemm_kernel = get_gemm_kernel(
                Quantization::F16,
                Quantization::F16,
                false,
                true, // transpose_a=false, transpose_b=true (K^T)
                tile_config,
                true,  // has_alpha_beta (needed for scale)
                false, // has_bias
            );

            // Probs@V: [M, kv_seq_len] @ [kv_seq_len, head_dim] = [M, head_dim]
            let av_gemm_kernel = get_gemm_kernel(
                Quantization::F16,
                Quantization::F16,
                false,
                false, // No transpose
                tile_config,
                false, // has_alpha_beta (alpha=1.0)
                false, // has_bias
            );

            // Batch heads over GEMM's gid.z dimension to avoid per-head dispatch overhead.
            // Q: [n_heads, M, head_dim] contiguous per head
            // K/V: [n_heads, kv_seq_len, head_dim] contiguous per head
            // Scores/Probs: [n_heads, M, kv_seq_len] contiguous per head
            // Output: token-major [M, d_model]; each head writes to a column slice, so
            //         the per-head batch stride is head_dim elements (column offset).

            // GEMM 1: Q @ K^T -> Scores [M, kv_seq_len]
            let mut qk_params = GemmParams::simple(
                m as i32,
                kv_seq_len as i32,
                head_dim as i32,
                false,
                true, // transpose_b=true for K^T (K stored as [kv_seq_len, head_dim])
                tile_config,
            );
            qk_params.batch_stride_a = (m as i64) * (head_dim as i64);
            qk_params.batch_stride_b = (k_seq_stride as i64) * (head_dim as i64);
            qk_params.batch_stride_c = (m as i64) * (kv_seq_len as i64);
            qk_params.batch_stride_d = (m as i64) * (kv_seq_len as i64);

            let qk_dispatch = {
                let base = gemm_dispatch_config(&qk_params, tile_config);
                DispatchConfig {
                    grid: GridSize::new(qk_params.tiles_n as usize, qk_params.tiles_m as usize, n_heads as usize),
                    group: base.group,
                }
            };

            let q_base = TensorArg::from_tensor(q);
            let k_base = TensorArg::from_tensor(k);

            let qk_args = GemmV2Args {
                a: q_base,
                b: k_base,
                d: scores_all.clone(),
                c: scores_all.clone(),        // Dummy - no residual
                bias: scores_all.clone(),     // Dummy - no bias
                b_scales: scores_all.clone(), // Dummy - F16 weights
                weights_per_block: 32,
                params: qk_params,
                alpha: scale,
                beta: 0.0,
            };
            foundry.run(&qk_gemm_kernel.bind(qk_args, qk_dispatch))?;

            // Softmax: flatten heads into the row dimension to dispatch once.
            // Row index is (head * M + row), but causal masking must use (row % M).
            let softmax_sdpa_kernel = get_softmax_v2_sdpa_batched_kernel();
            let softmax_dispatch = DispatchConfig {
                grid: GridSize::d1((n_heads as usize) * (m as usize)),
                group: ThreadgroupSize::d1(256),
            };

            let softmax_args = SoftmaxV2SdpaBatchedArgs {
                input: scores_all.clone(),
                scale: scale_arg.clone(),
                output: probs_all.clone(),
                seq_k: kv_seq_len,
                causal: if self.step.causal { 1 } else { 0 },
                query_offset: q_offset_val,
                rows_per_batch: m,
            };
            foundry.run(&softmax_sdpa_kernel.bind(softmax_args, softmax_dispatch))?;

            // GEMM 2: Probs @ V -> Output [M, head_dim] written into token-major [M, d_model].
            let mut av_params = GemmParams::simple(m as i32, head_dim as i32, kv_seq_len as i32, false, false, tile_config);
            av_params.ldd = d_model_dim as i32;
            av_params.ldc = d_model_dim as i32;
            av_params.batch_stride_a = (m as i64) * (kv_seq_len as i64);
            av_params.batch_stride_b = (v_seq_stride as i64) * (head_dim as i64);
            av_params.batch_stride_c = head_dim as i64;
            av_params.batch_stride_d = head_dim as i64;

            let av_dispatch = {
                let base = gemm_dispatch_config(&av_params, tile_config);
                DispatchConfig {
                    grid: GridSize::new(av_params.tiles_n as usize, av_params.tiles_m as usize, n_heads as usize),
                    group: base.group,
                }
            };

            let v_base = TensorArg::from_tensor(v);
            let out_base = TensorArg::from_tensor(output);

            let av_args = GemmV2Args {
                a: probs_all,
                b: v_base,
                d: out_base.clone(),
                c: out_base.clone(),    // Dummy - no residual
                bias: out_base.clone(), // Dummy - no bias
                b_scales: out_base,     // Dummy - F16 weights
                weights_per_block: 32,
                params: av_params,
                alpha: 1.0,
                beta: 0.0,
            };
            foundry.run(&av_gemm_kernel.bind(av_args, av_dispatch))?;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "SdpaMaterialized"
    }
}
