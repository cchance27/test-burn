use std::sync::Arc;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use crate::{
    Foundry, MetalError, compound::{CompiledCompoundKernel, CompoundKernel}, metals::{
        rope::{RopeParams, RopeParamsResolved, stage::RopeStage}, sdpa::stages::{HeadLayoutStage, SdpaOnlineStage, SdpaParams, SdpaParamsResolved}
    }, policy::activation::Activation, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, types::{
        TensorArg, dispatch::{DispatchConfig, GridSize, ThreadgroupSize}
    }
};

/// Get the static compiled kernel template.
/// Get the compiled kernel template.
fn get_fused_mha_kernel() -> Arc<CompiledCompoundKernel> {
    use crate::kernel_registry::{KernelCacheKey, kernel_registry};
    let key = KernelCacheKey::new("fused_mha", "rope_decode_v2");

    kernel_registry().get_or_build(key, || {
        let name = "fused_mha_rope_decode_v2";
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

        CompoundKernel::new(name)
            .prologue(dummy_layout)
            .prologue(dummy_rope)
            .main(dummy_core)
            .with_manual_output(true)
            .compile()
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
        match crate::Kernel::source(&*kernel) {
            crate::KernelSource::String(s) => s,
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
        let source = match crate::Kernel::source(&*kernel) {
            crate::KernelSource::String(s) => s,
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
    fn execute(
        &self,
        foundry: &mut Foundry,
        _fast_bindings: &FastBindings,
        _globals: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
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
        // Ensure at least one full SIMD group so `simd_sum` reductions are well-defined.
        let group = ThreadgroupSize::d1((vec_dim.max(32)) as usize);
        let config = DispatchConfig::new(grid, group);

        let kernel = get_fused_mha_kernel();
        let bound = kernel.clone().bind_arc(args, config);
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
        })]
    }
}

/// Get or create static SDPA kernel (standalone, no RoPE fusion)
/// Get or create SDPA kernel (standalone, no RoPE fusion)
fn get_sdpa_standalone_kernel() -> Arc<CompiledCompoundKernel> {
    use crate::kernel_registry::{KernelCacheKey, kernel_registry};
    let key = KernelCacheKey::new("sdpa_standalone", "v2");

    kernel_registry().get_or_build(key, || {
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
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let q = fast_bindings.get(self.q_idx).ok_or(MetalError::InputNotFound("q".into()))?;
        let k = fast_bindings.get(self.k_idx).ok_or(MetalError::InputNotFound("k".into()))?;
        let v = fast_bindings.get(self.v_idx).ok_or(MetalError::InputNotFound("v".into()))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or(MetalError::InputNotFound("output".into()))?;

        let n_heads = self.step.n_heads.resolve(bindings);
        let head_dim = self.step.head_dim.resolve(bindings);
        let kv_seq_len = self.step.kv_seq_len.resolve(bindings);

        // Compute strides based on head-major layout [n_heads, allocated_capacity, head_dim]
        let batch = 1u32;
        let capacity = k.dims.get(1).copied().unwrap_or(kv_seq_len as usize) as u32;

        let (q_stride_b, q_stride_h) = if self.step.kv_head_major {
            (n_heads * head_dim, head_dim)
        } else {
            (head_dim, n_heads * head_dim)
        };
        // Use capacity for head stride to prevent head overlap/corruption
        let (k_stride_b, k_stride_h) = (n_heads * capacity * head_dim, capacity * head_dim);
        let (v_stride_b, v_stride_h) = (n_heads * capacity * head_dim, capacity * head_dim);
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

        let bound = kernel.clone().bind_arc(args, config);
        foundry.run(&bound)
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
    constants, metals::{
        gemm::step::{GemmParams, GemmV2Args, gemm_dispatch_config, get_gemm_kernel}, mma::stages::TileConfig, softmax::{SoftmaxV2SdpaBatchedArgs, get_softmax_v2_sdpa_batched_kernel}
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
        let mut symbols = crate::spec::SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = crate::spec::FastBindings::new(symbols.len());

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
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
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
        let _elem_size = std::mem::size_of::<f16>();

        // 2. Allocate Temps (scores and probs) -> Size = n_heads * m * kv_seq_len
        // For M>1 prefill, each query produces kv_seq_len scores
        // Use default tile config (32x32) - auto_select was causing regression.
        //
        // NOTE: We pad the scratch M-stride for M>1 to keep head slices isolated even if a kernel
        // writes full tiles. For M=1 (decode) we avoid extra work to preserve throughput; the GEMM
        // epilogue uses safe stores for edge tiles so decode remains correct without padding.
        let tile_config = TileConfig::default();
        let bm = tile_config.tile_sizes().0;

        // Use centralized TiledLayout for scratch and output padding
        let scratch_layout = crate::compound::layout::TiledLayout::sdpa_scratch(n_heads, m, kv_seq_len, bm);

        // Re-allocate Temps if logic changed (scratch cache handles resizing)
        let (scores_all, probs_all) = crate::metals::sdpa::scratch::get_sdpa_scratch_f16(foundry, scratch_layout)?;

        // Softmax scaling is applied in the QK matmul (alpha=scale). For softmax itself, we use 1.0.
        // IMPORTANT: this must not allocate+upload per call; it kills decode throughput.
        let scale_arg = constants::f16_scalar(foundry, f16::ONE)?;

        // K/V may be either:
        // - tightly packed history (e.g. RepeatKvHeads output): [n_heads, kv_seq_len, head_dim]
        // - full cache view: [n_heads, max_seq_len, head_dim] (stride uses max_seq_len)
        let k_seq_stride = k.dims.get(1).copied().unwrap_or(kv_seq_len as usize);
        let v_seq_stride = v.dims.get(1).copied().unwrap_or(kv_seq_len as usize);

        let d_model_dim = (n_heads as usize) * (head_dim as usize);

        {
            // =========== Batched GEMM over heads (M>=1) ===========
            // tile_config is already selected above

            // Get GEMM kernels for Q@K^T and Probs@V
            // Q@K^T: [M, head_dim] @ [head_dim, kv_seq_len] = [M, kv_seq_len]
            // K is stored as [kv_seq_len, head_dim] row-major, so K^T is ColMajor read
            let qk_gemm_kernel = get_gemm_kernel(
                Arc::new(crate::policy::f16::PolicyF16),
                Arc::new(crate::policy::f16::PolicyF16),
                false,
                true, // transpose_a=false, transpose_b=true (K^T)
                tile_config,
                true,  // has_alpha_beta (needed for scale)
                false, // has_bias
                Activation::None,
            );

            // Probs@V: [M, kv_seq_len] @ [kv_seq_len, head_dim] = [M, head_dim]
            let av_gemm_kernel = get_gemm_kernel(
                Arc::new(crate::policy::f16::PolicyF16),
                Arc::new(crate::policy::f16::PolicyF16),
                false,
                false, // No transpose
                tile_config,
                false, // has_alpha_beta (alpha=1.0)
                false, // has_bias
                Activation::None,
            );

            // Batch heads over GEMM's gid.z dimension to avoid per-head dispatch overhead.
            // Q: [n_heads, M, head_dim] contiguous per head
            // K/V: [n_heads, kv_seq_len, head_dim] contiguous per head
            // Scores/Probs: [n_heads, M, kv_seq_len] contiguous per head -- STRIDED NOW
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

            // Output C (Scores) is written to Scratch [H, PaddedM, SeqLen] (Head-Major).
            // So we stride by head_stride between heads.
            qk_params.batch_stride_c = scratch_layout.head_stride as i64;
            qk_params.batch_stride_d = scratch_layout.head_stride as i64;

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
            foundry.run(&qk_gemm_kernel.clone().bind_arc(qk_args, qk_dispatch))?;

            // Softmax: flatten heads into the row dimension to dispatch once.
            // Row index is (head * M + row), but causal masking must use (row % M).
            // PADDING: Dispatch over padded_m to match stride.
            let softmax_sdpa_kernel = get_softmax_v2_sdpa_batched_kernel();
            let softmax_dispatch = DispatchConfig {
                grid: GridSize::d1((n_heads as usize) * (scratch_layout.padded_m as usize)),
                group: ThreadgroupSize::d1(256),
            };

            let softmax_args = SoftmaxV2SdpaBatchedArgs {
                input: scores_all.clone(),
                scale: scale_arg.clone(),
                output: probs_all.clone(),
                seq_k: kv_seq_len,
                causal: if self.step.causal { 1 } else { 0 },
                query_offset: q_offset_val,
                rows_per_batch: scratch_layout.padded_m,
            };
            foundry.run(&softmax_sdpa_kernel.clone().bind_arc(softmax_args, softmax_dispatch))?;

            // GEMM 2: Probs @ V -> Output [M, head_dim] written into token-major [M, d_model].
            // NOTE: Output is actually [H, M, D] in scratch/temp buffer usually?
            // Wait, "Output [M, head_dim] written into token-major [M, d_model]" comment suggests D is Interleaved.
            // But if Q was Head-Major, V used to be Head-Major?
            let mut av_params = GemmParams::simple(m as i32, head_dim as i32, kv_seq_len as i32, false, false, tile_config);
            av_params.ldd = d_model_dim as i32;
            av_params.ldc = d_model_dim as i32;

            // Output D: If we want Interleaved [M, H, D] output for next layers?
            // The original code passed 'output' (which is passed from outside).
            // And originally: av_params.batch_stride_c = head_dim as i64;
            // Wait. Original code had:
            // av_params.batch_stride_c = head_dim as i64;
            // av_params.batch_stride_d = head_dim as i64;
            // This implies Output D IS Interleaved?

            // Input A (Probs) is Padded Head-Major.
            av_params.batch_stride_a = scratch_layout.head_stride as i64;

            // Input B (V Cache). Originally: (v_seq_stride as i64) * (head_dim as i64). (Head-Major).
            av_params.batch_stride_b = (v_seq_stride as i64) * (head_dim as i64);

            // Output strides (Interleaved):
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
            foundry.run(&av_gemm_kernel.clone().bind_arc(av_args, av_dispatch))?;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "SdpaMaterialized"
    }
}
