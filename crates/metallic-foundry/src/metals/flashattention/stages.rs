use std::sync::Arc;

use metallic_macros::{KernelArgs, MetalStruct, Stage};

use super::variants::{FlashDecodeScalar, FlashDecodeTgOut, FlashDecodeVariant};
use crate::{compound::BufferArg, fusion::MetalPolicy, types::TensorArg};

/// Handles layout and indexing for Multi-Head Attention (Decode).
///
/// Computes base pointers for Q, K, V, Out for a specific Batch and Head.
/// Assumes Dispatch Grid: (1, Heads, Batch) or similar where gid.y/z map to Head/Batch.
/// Intended for Single-Token Decode (Seq=1).
#[derive(KernelArgs, Clone, Debug, Stage)]
#[stage(
    includes("flashattention/flash_decode.metal"),
    policy_field = "policy",
    buffer_args_fn = "stage_buffer_args",
    template_bindings(
        q_t = "self.policy_elem_type(\"q\", \"half\")",
        k_t = "self.policy_elem_type(\"k\", \"half\")",
        v_t = "self.policy_elem_type(\"v\", \"half\")",
        out_t = "self.policy_elem_type(\"output\", \"half\")"
    ),
    emit = r#"
    // HeadLayoutStage (Policy: {policy_short})
    auto layout = run_flash_head_layout_stage<{q_t}, {k_t}, {v_t}, {out_t}>(
        (const device {q_t}*)q,
        (const device {k_t}*)k,
        (const device {v_t}*)v,
        (device {out_t}*)output,
        q_stride_b,
        q_stride_h,
        k_stride_b,
        k_stride_h,
        v_stride_b,
        v_stride_h,
        out_stride_b,
        out_stride_h,
        gid
    );
    const device {q_t}* q_ptr = layout.q_ptr;
    const device {k_t}* k_ptr = layout.k_ptr;
    const device {v_t}* v_ptr = layout.v_ptr;
    device {out_t}* output_ptr = layout.output_ptr;
"#,
    out_var = "output_ptr"
)]
pub struct HeadLayoutStage {
    pub q: TensorArg,
    pub k: TensorArg,
    pub v: TensorArg,
    #[arg(output)]
    pub output: TensorArg,

    pub q_stride_b: u32,
    pub q_stride_h: u32,
    pub k_stride_b: u32,
    pub k_stride_h: u32,
    pub v_stride_b: u32,
    pub v_stride_h: u32,
    pub out_stride_b: u32,
    pub out_stride_h: u32,

    #[arg(skip, stage_skip)]
    pub policy: Arc<dyn MetalPolicy>,
}

impl HeadLayoutStage {
    fn stage_buffer_args(&self) -> Vec<BufferArg> {
        let buffers = vec![
            ("q", 0, "const device half*"),
            ("k", 1, "const device half*"),
            ("v", 2, "const device half*"),
            ("output", 3, "device half*"),
        ];

        let mut args = Vec::new();
        // Resolve types from policy
        let policy_types = self.policy.buffer_types();

        for (name, idx, default_type) in buffers {
            let type_str = policy_types
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, t)| *t)
                .unwrap_or(default_type);

            args.push(BufferArg {
                name,
                metal_type: type_str,
                buffer_index: idx,
            });
        }

        // Add scalar strides
        let scalars = [
            ("q_stride_b", 4),
            ("q_stride_h", 5),
            ("k_stride_b", 6),
            ("k_stride_h", 7),
            ("v_stride_b", 8),
            ("v_stride_h", 9),
            ("out_stride_b", 10),
            ("out_stride_h", 11),
        ];

        for (name, idx) in scalars {
            args.push(BufferArg {
                name,
                metal_type: "constant uint&",
                buffer_index: idx,
            });
        }

        args
    }

    fn policy_elem_type(&self, name: &str, def_elem: &str) -> String {
        let policy_types = self.policy.buffer_types();
        policy_types
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, t)| *t)
            .unwrap_or(def_elem)
            .trim_end_matches('*')
            .trim_start_matches("const device ")
            .trim_start_matches("device ")
            .trim()
            .to_string()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q: TensorArg,
        k: TensorArg,
        v: TensorArg,
        output: TensorArg,
        q_strides: (u32, u32),
        k_strides: (u32, u32),
        v_strides: (u32, u32),
        out_strides: (u32, u32),
        policy: Arc<dyn MetalPolicy>,
    ) -> Self {
        Self {
            q,
            k,
            v,
            output,
            q_stride_b: q_strides.0,
            q_stride_h: q_strides.1,
            k_stride_b: k_strides.0,
            k_stride_h: k_strides.1,
            v_stride_b: v_strides.0,
            v_stride_h: v_strides.1,
            out_stride_b: out_strides.0,
            out_stride_h: out_strides.1,
            policy,
        }
    }
}

// RopeStage imported from metals::rope::stage::RopeStage

/// Core Flash FlashDecodeFused Loop (Flash/Decode) with Q coming from RoPE `q_shared`.
///
/// We keep two variants to stay within the 32KB threadgroup-memory limit while maintaining
/// good numerical behavior:
/// - D=64 uses the half2 kernel + float2 accumulator (lower TG memory, larger KV block)
/// - D=128 uses the half4 kernel + float4 accumulator (needed for D=128)
///
/// - D=128 uses the half4 kernel + float4 accumulator (needed for D=128)
#[derive(KernelArgs, Clone, Debug, Stage)]
#[stage(
    includes("simd.metal", "flashattention/flash_decode.metal", "softmax/streaming.metal"),
    policy_field = "policy",
    struct_defs = "SdpaParams",
    template_bindings(emit_call = "self.fused_emit_call()"),
    emit = r#"
    {emit_call}
"#,
    out_var = "output_ptr"
)]
pub struct FlashDecodeFusedStage<const HEAD_DIM: usize> {
    #[arg(buffer = 15, metal_type = "constant SdpaParams&")]
    pub sdpa_params: SdpaParams,
    #[arg(skip, stage_skip)]
    pub variant: FlashDecodeVariant,
    #[arg(skip, stage_skip)]
    pub policy: Arc<dyn MetalPolicy>,
}

impl<const HEAD_DIM: usize> FlashDecodeFusedStage<HEAD_DIM> {
    pub fn new(sdpa_params: SdpaParams, variant: FlashDecodeVariant, policy: Arc<dyn MetalPolicy>) -> Self {
        Self {
            sdpa_params,
            variant,
            policy,
        }
    }

    fn fused_emit_call(&self) -> String {
        let head_dim = HEAD_DIM as u32;
        self.variant
            .validate_for_head_dim(head_dim)
            .unwrap_or_else(|e| panic!("Invalid FlashDecodeFusedStage variant for head_dim={head_dim}: {e:?}"));

        let tg_out_half = match self.variant.tg_out {
            FlashDecodeTgOut::Float => 0,
            FlashDecodeTgOut::Half => 1,
        };

        match self.variant.scalar {
            FlashDecodeScalar::Half2 => format!(
                r#"
    uint warp = (uint)__simd_group_id;
    uint lane = (uint)__simd_lane_id;
    FLASH_DECODE_DECLARE_REDUCE_SHARED_HALF2({warps}, {tg_out_half}, shared_warp_max, shared_warp_sums, shared_warp_out);
    const threadgroup half2* q_vec = (const threadgroup half2*)q_shared;
    run_flash_decode_fused_half2_stage<{warps}, {keys_per_warp}, {tg_out_half}>(
        q_vec, k_ptr, v_ptr, output_ptr, sdpa_params, warp, lane, shared_warp_max, shared_warp_sums, shared_warp_out
    );
"#,
                warps = self.variant.warps,
                keys_per_warp = self.variant.keys_per_warp,
                tg_out_half = tg_out_half
            ),
            FlashDecodeScalar::Half4 => format!(
                r#"
    uint warp = (uint)__simd_group_id;
    uint lane = (uint)__simd_lane_id;
    FLASH_DECODE_DECLARE_REDUCE_SHARED_HALF4({warps}, {tg_out_half}, shared_warp_max, shared_warp_sums, shared_warp_out);
    const threadgroup half4* q_vec = (const threadgroup half4*)q_shared;
    run_flash_decode_fused_half4_stage<{warps}, {keys_per_warp}, {tg_out_half}>(
        q_vec, k_ptr, v_ptr, output_ptr, sdpa_params, warp, lane, shared_warp_max, shared_warp_sums, shared_warp_out
    );
"#,
                warps = self.variant.warps,
                keys_per_warp = self.variant.keys_per_warp,
                tg_out_half = tg_out_half
            ),
        }
    }
}

/// Standalone FlashDecode Stage - loads Q directly from buffer (no RoPE fusion).
///
/// Standalone FlashDecode Stage - loads Q directly from buffer (no RoPE fusion).
#[derive(KernelArgs, Clone, Debug, Stage)]
#[stage(
    includes("simd.metal", "flashattention/flash_decode.metal", "softmax/streaming.metal"),
    policy_field = "policy",
    struct_defs = "SdpaParams",
    template_bindings(emit_call = "self.standalone_emit_call()"),
    emit = r#"
    {emit_call}
"#,
    out_var = "output_ptr"
)]
pub struct FlashDecodeStage<const HEAD_DIM: usize> {
    #[arg(buffer = 12, metal_type = "constant SdpaParams&")]
    pub sdpa_params: SdpaParams,
    #[arg(skip, stage_skip)]
    pub variant: FlashDecodeVariant,
    #[arg(skip, stage_skip)]
    pub policy: Arc<dyn MetalPolicy>,
}

impl<const HEAD_DIM: usize> FlashDecodeStage<HEAD_DIM> {
    pub fn new(sdpa_params: SdpaParams, variant: FlashDecodeVariant, policy: Arc<dyn MetalPolicy>) -> Self {
        Self {
            sdpa_params,
            variant,
            policy,
        }
    }

    fn standalone_emit_call(&self) -> String {
        let head_dim = HEAD_DIM as u32;
        self.variant
            .validate_for_head_dim(head_dim)
            .unwrap_or_else(|e| panic!("Invalid FlashDecodeStage variant for head_dim={head_dim}: {e:?}"));

        let tg_out_half = match self.variant.tg_out {
            FlashDecodeTgOut::Float => 0,
            FlashDecodeTgOut::Half => 1,
        };

        match self.variant.scalar {
            FlashDecodeScalar::Half2 => format!(
                r#"
    uint warp = (uint)__simd_group_id;
    uint lane = (uint)__simd_lane_id;
    FLASH_DECODE_DECLARE_Q_SHARED_HALF2(q_shared);
    FLASH_DECODE_DECLARE_REDUCE_SHARED_HALF2({warps}, {tg_out_half}, shared_warp_max, shared_warp_sums, shared_warp_out);
    run_flash_decode_standalone_half2_stage<{warps}, {keys_per_warp}, {tg_out_half}>(
        q_ptr, k_ptr, v_ptr, output_ptr, sdpa_params, warp, lane, q_shared, shared_warp_max, shared_warp_sums, shared_warp_out
    );
"#,
                warps = self.variant.warps,
                keys_per_warp = self.variant.keys_per_warp,
                tg_out_half = tg_out_half
            ),
            FlashDecodeScalar::Half4 => format!(
                r#"
    uint warp = (uint)__simd_group_id;
    uint lane = (uint)__simd_lane_id;
    FLASH_DECODE_DECLARE_Q_SHARED_HALF4(q_shared);
    FLASH_DECODE_DECLARE_REDUCE_SHARED_HALF4({warps}, {tg_out_half}, shared_warp_max, shared_warp_sums, shared_warp_out);
    run_flash_decode_standalone_half4_stage<{warps}, {keys_per_warp}, {tg_out_half}, {vec4}>(
        q_ptr, k_ptr, v_ptr, output_ptr, sdpa_params, warp, lane, q_shared, shared_warp_max, shared_warp_sums, shared_warp_out
    );
"#,
                warps = self.variant.warps,
                keys_per_warp = self.variant.keys_per_warp,
                tg_out_half = tg_out_half,
                vec4 = HEAD_DIM / 4
            ),
        }
    }
}

// Backward compatibility alias (mostly for tests/imports)
pub type SdpaStandaloneStageD64 = FlashDecodeStage<64>;
pub type SdpaStandaloneStageD128 = FlashDecodeStage<128>;
pub type SdpaStandaloneStage = FlashDecodeStage<64>;

pub type SdpaOnlineStageD64 = FlashDecodeFusedStage<64>;
pub type SdpaOnlineStageD128 = FlashDecodeFusedStage<128>;

#[derive(MetalStruct, Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct SdpaPrefillParams {
    pub kv_len: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub stride_k_s: u32,
    pub stride_v_s: u32,
    pub query_offset: u32,

    // Strides for manual calculation
    pub q_stride_b: u32,
    pub q_stride_h: u32,
    pub k_stride_b: u32,
    pub k_stride_h: u32,
    pub v_stride_b: u32,
    pub v_stride_h: u32,
    pub out_stride_b: u32,
    pub out_stride_h: u32,

    // Stride for Q sequence dim (e.g. D vs n_heads*D)
    pub q_stride_m: u32,

    // Stride for Output sequence dim (n_heads*D)
    pub out_stride_m: u32,

    // GQA Group Size (n_heads / n_kv_heads)
    pub group_size: u32,

    pub q_len: u32,
}

#[derive(MetalStruct, Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct SdpaPrefillSplitKParams {
    pub kv_len: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub stride_k_s: u32,
    pub stride_v_s: u32,
    pub query_offset: u32,

    // Strides for manual calculation
    pub q_stride_b: u32,
    pub q_stride_h: u32,
    pub k_stride_b: u32,
    pub k_stride_h: u32,
    pub v_stride_b: u32,
    pub v_stride_h: u32,
    pub out_stride_b: u32,
    pub out_stride_h: u32,

    // Stride for Q sequence dim (e.g. D vs n_heads*D)
    pub q_stride_m: u32,

    // Stride for Output sequence dim (n_heads*D)
    pub out_stride_m: u32,

    // GQA Group Size (n_heads / n_kv_heads)
    pub group_size: u32,

    pub q_len: u32,

    // Split-K parameters (prefill only)
    pub n_heads: u32,
    pub split_k: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SdpaPrefillVariant {
    pub warps: u32,
}

impl SdpaPrefillVariant {
    pub fn validate(self) {
        // We only support 128 (4 warps) and 256 (8 warps) threads for now, matching the tile loader.
        assert!(matches!(self.warps, 4 | 8), "SdpaPrefillVariant unsupported warps={}", self.warps);
    }
}

#[derive(KernelArgs, Clone, Debug, Stage)]
#[stage(
    includes("simd.metal", "flashattention/flash_prefill.metal"),
    template_bindings(warps = "validated_prefill_warps(self.variant)"),
    struct_defs("SdpaPrefillParams", "SdpaPrefillSplitKParams"),
    emit = r#"
    SDPA_PREFILL_DECLARE_SHARED(k_shared, v_shared);
    uint simd_lane_id = lid.x & 31u;
    uint simd_group_id = lid.x >> 5;
    run_sdpa_prefill_stage<{warps}>(
        q, k, v, output, sdpa_prefill_params, gid, lid, tptg, simd_lane_id, simd_group_id, k_shared, v_shared
    );
"#,
    out_var = "output"
)]
pub struct SdpaPrefillStage {
    #[arg(buffer = 0)]
    pub q: TensorArg,
    #[arg(buffer = 1)]
    pub k: TensorArg,
    #[arg(buffer = 2)]
    pub v: TensorArg,
    #[arg(buffer = 3, output)]
    pub output: TensorArg,
    #[arg(buffer = 4, metal_type = "constant SdpaPrefillParams&")]
    pub sdpa_prefill_params: SdpaPrefillParams,
    #[arg(skip, stage_skip)]
    pub variant: SdpaPrefillVariant,
}

impl SdpaPrefillStage {
    pub fn new(params: SdpaPrefillParams) -> Self {
        Self {
            q: TensorArg::default(),
            k: TensorArg::default(),
            v: TensorArg::default(),
            output: TensorArg::default(),
            sdpa_prefill_params: params,
            variant: SdpaPrefillVariant { warps: 8 },
        }
    }
}

#[derive(KernelArgs, Clone, Debug, Stage)]
#[stage(
    includes("simd.metal", "flashattention/flash_prefill.metal"),
    template_bindings(warps = "validated_prefill_warps(self.variant)"),
    struct_defs("SdpaPrefillParams", "SdpaPrefillSplitKParams"),
    emit = r#"
    SDPA_PREFILL_DECLARE_SHARED(k_shared, v_shared);
    uint simd_lane_id = lid.x & 31u;
    uint simd_group_id = lid.x >> 5;
    run_sdpa_prefill_splitk_part_stage<{warps}>(
        q, k, v, partial_acc, partial_m, partial_l, sdpa_prefill_splitk_params, gid, lid, tptg,
        simd_lane_id, simd_group_id, k_shared, v_shared
    );
"#,
    out_var = "partial_acc"
)]
pub struct SdpaPrefillSplitKPartStage {
    #[arg(buffer = 0)]
    pub q: TensorArg,
    #[arg(buffer = 1)]
    pub k: TensorArg,
    #[arg(buffer = 2)]
    pub v: TensorArg,
    // Outputs: partial accumulators / stats for each split.
    #[arg(buffer = 3, metal_type = "device float*")]
    pub partial_acc: TensorArg,
    #[arg(buffer = 4, metal_type = "device float*")]
    pub partial_m: TensorArg,
    #[arg(buffer = 5, metal_type = "device float*")]
    pub partial_l: TensorArg,
    #[arg(buffer = 6, metal_type = "constant SdpaPrefillSplitKParams&")]
    pub sdpa_prefill_splitk_params: SdpaPrefillSplitKParams,
    #[arg(skip, stage_skip)]
    pub variant: SdpaPrefillVariant,
}

impl SdpaPrefillSplitKPartStage {
    pub fn new(params: SdpaPrefillSplitKParams) -> Self {
        Self {
            q: TensorArg::default(),
            k: TensorArg::default(),
            v: TensorArg::default(),
            partial_acc: TensorArg::default(),
            partial_m: TensorArg::default(),
            partial_l: TensorArg::default(),
            sdpa_prefill_splitk_params: params,
            variant: SdpaPrefillVariant { warps: 8 },
        }
    }
}

#[derive(KernelArgs, Clone, Debug, Stage)]
#[stage(
    includes("simd.metal", "flashattention/flash_prefill.metal"),
    template_bindings(warps = "validated_prefill_warps(self.variant)"),
    struct_defs("SdpaPrefillParams", "SdpaPrefillSplitKParams"),
    emit = r#"
    uint simd_lane_id = lid.x & 31u;
    uint simd_group_id = lid.x >> 5;
    run_sdpa_prefill_splitk_reduce_stage<{warps}>(
        partial_acc, partial_m, partial_l, output, sdpa_prefill_splitk_params, gid, lid, tptg, simd_lane_id, simd_group_id
    );
"#,
    out_var = "output"
)]
pub struct SdpaPrefillSplitKReduceStage {
    #[arg(buffer = 0, output)]
    pub output: TensorArg,
    #[arg(buffer = 1, metal_type = "const device float*")]
    pub partial_acc: TensorArg,
    #[arg(buffer = 2, metal_type = "const device float*")]
    pub partial_m: TensorArg,
    #[arg(buffer = 3, metal_type = "const device float*")]
    pub partial_l: TensorArg,
    #[arg(buffer = 4, metal_type = "constant SdpaPrefillSplitKParams&")]
    pub sdpa_prefill_splitk_params: SdpaPrefillSplitKParams,
    #[arg(skip, stage_skip)]
    pub variant: SdpaPrefillVariant,
}

impl SdpaPrefillSplitKReduceStage {
    pub fn new(params: SdpaPrefillSplitKParams) -> Self {
        Self {
            output: TensorArg::default(),
            partial_acc: TensorArg::default(),
            partial_m: TensorArg::default(),
            partial_l: TensorArg::default(),
            sdpa_prefill_splitk_params: params,
            variant: SdpaPrefillVariant { warps: 8 },
        }
    }
}

#[derive(MetalStruct, Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct SdpaParams {
    pub kv_len: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub stride_k_s: u32,
    pub stride_v_s: u32,
}

#[inline]
fn validated_prefill_warps(variant: SdpaPrefillVariant) -> u32 {
    variant.validate();
    variant.warps
}
