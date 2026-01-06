use std::sync::OnceLock;

use metallic_macros::KernelArgs;

use crate::{
    MetalError, compound::{CompiledCompoundKernel, CompoundKernel}, foundry::{
        Foundry, spec::{CompiledStep, FastBindings, TensorBindings}
    }, metals::{
        rope::{RopeParams, RopeParamsResolved, stage::RopeStage}, v2::attention::stages::{HeadLayoutStage, SdpaCoreStage, SdpaParams, SdpaParamsResolved}
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
        let dummy_core = SdpaCoreStage::new(SdpaParams::default());

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
        // Grid: (HeadDim/4, Heads, Batch) -> (x, y, z) - Vectorized
        // Group: (HeadDim/4, 1, 1) -> (x, y, z)
        let vec_dim = self.head_dim / 4;
        let grid = GridSize::new(vec_dim as usize, self.heads as usize, self.batch as usize);
        let group = ThreadgroupSize::d1(vec_dim as usize);
        let config = DispatchConfig::new(grid, group);

        let kernel = get_fused_mha_kernel();
        let bound = kernel.bind(args, config);
        foundry.run(&bound)
    }
}
