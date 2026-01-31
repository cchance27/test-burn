use metallic_macros::{KernelArgs, MetalStruct, Stage};

use super::variants::{FlashDecodeScalar, FlashDecodeTgOut, FlashDecodeVariant};
use crate::{
    compound::{BufferArg, Stage}, types::TensorArg
};

/// Handles layout and indexing for Multi-Head Attention (Decode).
///
/// Computes base pointers for Q, K, V, Out for a specific Batch and Head.
/// Assumes Dispatch Grid: (1, Heads, Batch) or similar where gid.y/z map to Head/Batch.
/// Intended for Single-Token Decode (Seq=1).
#[derive(Stage, KernelArgs, Clone, Debug)]
#[stage(emit = r#"
    // Grid: x=dim (implicit), y=head, z=batch
    // Metal gid usually is (x, y, z) in ELEMENTS (global threads).
    // If we dispatch (1, Heads, Batch) groups with (HeadDim, 1, 1) threads:
    // gid.x range 0..HeadDim-1.
    // gid.y range 0..Heads-1.
    // gid.z range 0..Batch-1.
    
    // We do NOT use gid.x for sequence index. 
    // It corresponds to `tid` (intra-head dimension).
    
    uint head_idx = gid.y;
    uint batch_idx = gid.z;
    uint tid = lid.x;

    ulong q_offset = batch_idx * q_stride_b + head_idx * q_stride_h;
    const device half* q_ptr = q + q_offset;
    
    ulong k_offset = batch_idx * k_stride_b + head_idx * k_stride_h;
    const device half* k_ptr = k + k_offset;
    
    ulong v_offset = batch_idx * v_stride_b + head_idx * v_stride_h;
    const device half* v_ptr = v + v_offset;
    
    ulong out_offset = batch_idx * out_stride_b + head_idx * out_stride_h;
    device half* output_ptr = output + out_offset;
"#)]
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
}

impl HeadLayoutStage {
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
/// DEBT: This stage currently hardcodes F16 loading instead of using the policy abstraction.
/// This violates the SLP architecture. It should be refactored to take a policy object
/// and use policy-driven loads instead of raw pointer arithmetic.
#[derive(KernelArgs, Clone, Debug)]
pub struct FlashDecodeFusedStage<const HEAD_DIM: usize> {
    pub sdpa_params: SdpaParams,
    #[arg(skip)]
    pub variant: FlashDecodeVariant,
}

impl<const HEAD_DIM: usize> Stage for FlashDecodeFusedStage<HEAD_DIM> {
    fn includes(&self) -> Vec<&'static str> {
        vec!["simd.metal", "flashattention/flash_decode.metal", "softmax/streaming.metal"]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![BufferArg {
            name: "sdpa_params",
            metal_type: "constant SdpaParams&",
            buffer_index: 15,
        }]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        let head_dim = HEAD_DIM as u32;
        self.variant
            .validate_for_head_dim(head_dim)
            .unwrap_or_else(|e| panic!("Invalid FlashDecodeFusedStage variant for head_dim={head_dim}: {e:?}"));

        let scalar_t = match self.variant.scalar {
            FlashDecodeScalar::Half2 => "half2",
            FlashDecodeScalar::Half4 => "half4",
        };
        let out_t = match (self.variant.scalar, self.variant.tg_out) {
            (FlashDecodeScalar::Half2, FlashDecodeTgOut::Float) => "float2",
            (FlashDecodeScalar::Half2, FlashDecodeTgOut::Half) => "half2",
            (FlashDecodeScalar::Half4, FlashDecodeTgOut::Float) => "float4",
            (FlashDecodeScalar::Half4, FlashDecodeTgOut::Half) => "half4",
        };
        let tg_out_half_define = match self.variant.tg_out {
            FlashDecodeTgOut::Float => 0,
            FlashDecodeTgOut::Half => 1,
        };

        let code = format!(
            r#"
    // FlashDecodeFusedStage (RoPE→SDPA fused; head_dim={D})
    #define FLASH_WARPS {warps}
    #define FLASH_KEYS_PER_WARP {keys_per_warp}
    #define FLASH_TG_OUT_HALF {tg_out_half}
    threadgroup float shared_warp_max[FLASH_WARPS];
    threadgroup float shared_warp_sums[FLASH_WARPS];
    threadgroup {out_t} shared_warp_out[FLASH_WARPS * 32];

        // Use Metal-provided SIMD-group builtins for warp/lane IDs (see fusion::THREAD_ARGS).
        uint warp = (uint)__simd_group_id;
        uint lane = (uint)__simd_lane_id;

        // RopeStage stores Q as half4; reinterpret as needed.
        const threadgroup {scalar_t}* q_vec = (const threadgroup {scalar_t}*)q_shared;
    flash_decode_warp_tiled_m1_{scalar_t}<FLASH_WARPS, FLASH_KEYS_PER_WARP, FLASH_TG_OUT_HALF>(
        q_vec,
        k_ptr,
        v_ptr,
        output_ptr,
        warp,
        lane,
        sdpa_params,
        shared_warp_max,
        shared_warp_sums,
        shared_warp_out
    );
"#,
            D = HEAD_DIM,
            warps = self.variant.warps,
            keys_per_warp = self.variant.keys_per_warp,
            scalar_t = scalar_t,
            out_t = out_t,
            tg_out_half = tg_out_half_define
        );

        ("output_ptr".to_string(), code)
    }

    fn struct_defs(&self) -> String {
        sdpa_params_struct_defs()
    }
}

impl<const HEAD_DIM: usize> FlashDecodeFusedStage<HEAD_DIM> {
    pub fn new(sdpa_params: SdpaParams, variant: FlashDecodeVariant) -> Self {
        Self { sdpa_params, variant }
    }
}

/// Standalone FlashDecode Stage - loads Q directly from buffer (no RoPE fusion).
///
/// // DEBT: This stage currently hardcodes F16 loading instead of using the policy abstraction.
/// // This violates the SLP architecture. It should be refactored to take a policy object
/// // and use policy-driven loads instead of raw pointer arithmetic.
#[derive(KernelArgs, Clone, Debug)]
pub struct FlashDecodeStage<const HEAD_DIM: usize> {
    pub sdpa_params: SdpaParams,
    #[arg(skip)]
    pub variant: FlashDecodeVariant,
}

impl<const HEAD_DIM: usize> Stage for FlashDecodeStage<HEAD_DIM> {
    fn includes(&self) -> Vec<&'static str> {
        vec!["simd.metal", "flashattention/flash_decode.metal", "softmax/streaming.metal"]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![BufferArg {
            name: "sdpa_params",
            metal_type: "constant SdpaParams&",
            buffer_index: 12,
        }]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        let head_dim = HEAD_DIM as u32;
        self.variant
            .validate_for_head_dim(head_dim)
            .unwrap_or_else(|e| panic!("Invalid FlashDecodeStage variant for head_dim={head_dim}: {e:?}"));

        let (scalar_t, q_load) = match self.variant.scalar {
            FlashDecodeScalar::Half2 => (
                "half2",
                "            q_shared[lane] = ((const device half2*)q_ptr)[lane];".to_string(),
            ),
            FlashDecodeScalar::Half4 => {
                let vec4 = HEAD_DIM / 4;
                (
                    "half4",
                    format!(
                        "            if (lane < {vec4}) {{\n                q_shared[lane] = ((const device half4*)q_ptr)[lane];\n            }}"
                    ),
                )
            }
        };
        let out_t = match (self.variant.scalar, self.variant.tg_out) {
            (FlashDecodeScalar::Half2, FlashDecodeTgOut::Float) => "float2",
            (FlashDecodeScalar::Half2, FlashDecodeTgOut::Half) => "half2",
            (FlashDecodeScalar::Half4, FlashDecodeTgOut::Float) => "float4",
            (FlashDecodeScalar::Half4, FlashDecodeTgOut::Half) => "half4",
        };
        let tg_out_half_define = match self.variant.tg_out {
            FlashDecodeTgOut::Float => 0,
            FlashDecodeTgOut::Half => 1,
        };

        let code = format!(
            r#"
    // FlashDecodeStage D{D}
    #define FLASH_WARPS {warps}
    threadgroup {scalar_t} q_shared[32];
    #define FLASH_KEYS_PER_WARP {keys_per_warp}
    #define FLASH_TG_OUT_HALF {tg_out_half}
    threadgroup float shared_warp_max[FLASH_WARPS];
    threadgroup float shared_warp_sums[FLASH_WARPS];
    threadgroup {out_t} shared_warp_out[FLASH_WARPS * 32];
    
        uint warp = (uint)__simd_group_id;
        uint lane = (uint)__simd_lane_id;

        if (warp == 0) {{
{q_load}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_decode_warp_tiled_m1_{scalar_t}<FLASH_WARPS, FLASH_KEYS_PER_WARP, FLASH_TG_OUT_HALF>(
        q_shared,
        k_ptr,
        v_ptr,
        output_ptr,
        warp,
        lane,
        sdpa_params,
        shared_warp_max,
        shared_warp_sums,
        shared_warp_out
    );
"#,
            D = HEAD_DIM,
            warps = self.variant.warps,
            keys_per_warp = self.variant.keys_per_warp,
            scalar_t = scalar_t,
            q_load = q_load,
            out_t = out_t,
            tg_out_half = tg_out_half_define
        );

        ("output_ptr".to_string(), code)
    }

    fn struct_defs(&self) -> String {
        sdpa_params_struct_defs()
    }
}

impl<const HEAD_DIM: usize> FlashDecodeStage<HEAD_DIM> {
    pub fn new(sdpa_params: SdpaParams, variant: FlashDecodeVariant) -> Self {
        Self { sdpa_params, variant }
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

#[derive(KernelArgs, Clone, Debug)]
pub struct SdpaPrefillStage {
    pub q: TensorArg,
    #[arg(stage_skip)]
    pub k: TensorArg,
    #[arg(stage_skip)]
    pub v: TensorArg,
    #[arg(output)]
    pub output: TensorArg,
    pub params: SdpaPrefillParams,
    #[arg(skip)]
    pub variant: SdpaPrefillVariant,
}

impl Stage for SdpaPrefillStage {
    fn includes(&self) -> Vec<&'static str> {
        vec!["simd.metal", "flashattention/flash_prefill.metal"]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "q",
                metal_type: "const device half*",
                buffer_index: 0,
            },
            BufferArg {
                name: "k",
                metal_type: "const device half*",
                buffer_index: 1,
            },
            BufferArg {
                name: "v",
                metal_type: "const device half*",
                buffer_index: 2,
            },
            BufferArg {
                name: "output",
                metal_type: "device half*",
                buffer_index: 3,
            },
            BufferArg {
                name: "sdpa_prefill_params",
                metal_type: "constant SdpaPrefillParams&",
                buffer_index: 4,
            },
        ]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        self.variant.validate();
        let warps = self.variant.warps;
        // Redefine pointer logic locally to avoid dependency issues or scoping problems.
        // We use the strides passed in params.
        (
            "output_ptr".to_string(),
            format!(
                r#"
        // SdpaPrefillStage (D64/D128)
        #define FLASH_PREFILL_WARPS {warps}
        // Define threadgroup memory (max needed: 32x128)
        threadgroup half k_shared[32 * 128];
        threadgroup half v_shared[32 * 128];
        
        uint3 tid = lid;
        
        // Use Metal-provided SIMD-group builtins for warp/lane IDs (matches fusion::THREAD_ARGS).
        uint simd_lane_id = (uint)__simd_lane_id;
        uint simd_group_id = (uint)__simd_group_id;
        
        // Manual Pointer Arithmetic
        // Grid: x=TileM, y=Head, z=Batch
        uint head_idx = gid.y;
        uint batch_idx = gid.z;
        uint kv_head_idx = head_idx / sdpa_prefill_params.group_size;
        
        ulong q_offset = batch_idx * sdpa_prefill_params.q_stride_b + head_idx * sdpa_prefill_params.q_stride_h;
        const device half* q_ptr = q + q_offset;
        
        ulong k_offset = batch_idx * sdpa_prefill_params.k_stride_b + kv_head_idx * sdpa_prefill_params.k_stride_h;
        const device half* k_ptr = k + k_offset;
        
        ulong v_offset = batch_idx * sdpa_prefill_params.v_stride_b + kv_head_idx * sdpa_prefill_params.v_stride_h;
        const device half* v_ptr = v + v_offset;
        
        ulong out_offset = batch_idx * sdpa_prefill_params.out_stride_b + head_idx * sdpa_prefill_params.out_stride_h;
        device half* output_ptr = output + out_offset;

        if (sdpa_prefill_params.head_dim == 64) {{
            flash_prefill_tiled_d64<FLASH_PREFILL_WARPS>(
                q_ptr,
                k_ptr,
                v_ptr,
                output_ptr,
                sdpa_prefill_params,
                k_shared,
                v_shared,
                gid,
                tid,
                tptg, // Use kernel argument directly
                simd_lane_id,
                simd_group_id
            );
        }} else if (sdpa_prefill_params.head_dim == 128) {{
            flash_prefill_tiled_d128<FLASH_PREFILL_WARPS>(
                q_ptr,
                k_ptr,
                v_ptr,
                output_ptr,
                sdpa_prefill_params,
                k_shared,
                v_shared,
                gid,
                tid,
                tptg, // Use kernel argument directly
                simd_lane_id,
                simd_group_id
            );
        }}
        "#,
                warps = warps
            ),
        )
    }

    fn struct_defs(&self) -> String {
        format!(
            r#"
    #ifndef METALLIC_SDPA_PREFILL_PARAMS_DEFINED
    #define METALLIC_SDPA_PREFILL_PARAMS_DEFINED
    {}
    #endif
            "#,
            SdpaPrefillParams::METAL_STRUCT_DEF
        )
    }
}

impl SdpaPrefillStage {
    pub fn new(params: SdpaPrefillParams) -> Self {
        Self {
            q: TensorArg::default(),
            k: TensorArg::default(),
            v: TensorArg::default(),
            output: TensorArg::default(),
            params,
            variant: SdpaPrefillVariant { warps: 8 },
        }
    }
}

#[derive(KernelArgs, Clone, Debug)]
pub struct SdpaPrefillSplitKPartStage {
    pub q: TensorArg,
    #[arg(stage_skip)]
    pub k: TensorArg,
    #[arg(stage_skip)]
    pub v: TensorArg,
    // Outputs: partial accumulators / stats for each split.
    pub partial_acc: TensorArg,
    pub partial_m: TensorArg,
    pub partial_l: TensorArg,
    pub params: SdpaPrefillSplitKParams,
    #[arg(skip)]
    pub variant: SdpaPrefillVariant,
}

impl Stage for SdpaPrefillSplitKPartStage {
    fn includes(&self) -> Vec<&'static str> {
        vec!["simd.metal", "flashattention/flash_prefill.metal"]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "q",
                metal_type: "const device half*",
                buffer_index: 0,
            },
            BufferArg {
                name: "k",
                metal_type: "const device half*",
                buffer_index: 1,
            },
            BufferArg {
                name: "v",
                metal_type: "const device half*",
                buffer_index: 2,
            },
            BufferArg {
                name: "partial_acc",
                metal_type: "device float*",
                buffer_index: 3,
            },
            BufferArg {
                name: "partial_m",
                metal_type: "device float*",
                buffer_index: 4,
            },
            BufferArg {
                name: "partial_l",
                metal_type: "device float*",
                buffer_index: 5,
            },
            BufferArg {
                name: "sdpa_prefill_splitk_params",
                metal_type: "constant SdpaPrefillSplitKParams&",
                buffer_index: 6,
            },
        ]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        self.variant.validate();
        let warps = self.variant.warps;
        (
            "partial_acc_ptr".to_string(),
            format!(
                r#"
        // SdpaPrefillSplitKPartStage (D64/D128)
        #define FLASH_PREFILL_WARPS {warps}
        threadgroup half k_shared[32 * 128];
        threadgroup half v_shared[32 * 128];

        uint3 tid = lid;
        uint simd_lane_id = (uint)__simd_lane_id;
        uint simd_group_id = (uint)__simd_group_id;

        // Grid: x=TileM, y=Head, z=SplitK
        uint head_idx = gid.y;
        uint split_id = gid.z;
        uint kv_head_idx = head_idx / sdpa_prefill_splitk_params.group_size;

        const uint batch_idx = 0;

        ulong q_offset = batch_idx * sdpa_prefill_splitk_params.q_stride_b + head_idx * sdpa_prefill_splitk_params.q_stride_h;
        const device half* q_ptr = q + q_offset;

        ulong k_offset = batch_idx * sdpa_prefill_splitk_params.k_stride_b + kv_head_idx * sdpa_prefill_splitk_params.k_stride_h;
        const device half* k_ptr = k + k_offset;

        ulong v_offset = batch_idx * sdpa_prefill_splitk_params.v_stride_b + kv_head_idx * sdpa_prefill_splitk_params.v_stride_h;
        const device half* v_ptr = v + v_offset;

        device float* partial_acc_ptr = partial_acc;
        device float* partial_m_ptr = partial_m;
        device float* partial_l_ptr = partial_l;

        if (sdpa_prefill_splitk_params.head_dim == 64) {{
            flash_prefill_splitk_part_d64<FLASH_PREFILL_WARPS>(
                q_ptr,
                k_ptr,
                v_ptr,
                partial_acc_ptr,
                partial_m_ptr,
                partial_l_ptr,
                sdpa_prefill_splitk_params,
                k_shared,
                v_shared,
                gid,
                tid,
                tptg,
                simd_lane_id,
                simd_group_id
            );
        }} else if (sdpa_prefill_splitk_params.head_dim == 128) {{
            flash_prefill_splitk_part_d128<FLASH_PREFILL_WARPS>(
                q_ptr,
                k_ptr,
                v_ptr,
                partial_acc_ptr,
                partial_m_ptr,
                partial_l_ptr,
                sdpa_prefill_splitk_params,
                k_shared,
                v_shared,
                gid,
                tid,
                tptg,
                simd_lane_id,
                simd_group_id
            );
        }}
        "#,
                warps = warps
            ),
        )
    }

    fn struct_defs(&self) -> String {
        format!(
            r#"
    #ifndef METALLIC_SDPA_PREFILL_SPLITK_PARAMS_DEFINED
    #define METALLIC_SDPA_PREFILL_SPLITK_PARAMS_DEFINED
    {}
    #endif
            "#,
            SdpaPrefillSplitKParams::METAL_STRUCT_DEF
        )
    }
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
            params,
            variant: SdpaPrefillVariant { warps: 8 },
        }
    }
}

#[derive(KernelArgs, Clone, Debug)]
pub struct SdpaPrefillSplitKReduceStage {
    #[arg(output)]
    pub output: TensorArg,
    pub partial_acc: TensorArg,
    pub partial_m: TensorArg,
    pub partial_l: TensorArg,
    pub params: SdpaPrefillSplitKParams,
    #[arg(skip)]
    pub variant: SdpaPrefillVariant,
}

impl Stage for SdpaPrefillSplitKReduceStage {
    fn includes(&self) -> Vec<&'static str> {
        vec!["simd.metal", "flashattention/flash_prefill.metal"]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "output",
                metal_type: "device half*",
                buffer_index: 0,
            },
            BufferArg {
                name: "partial_acc",
                metal_type: "const device float*",
                buffer_index: 1,
            },
            BufferArg {
                name: "partial_m",
                metal_type: "const device float*",
                buffer_index: 2,
            },
            BufferArg {
                name: "partial_l",
                metal_type: "const device float*",
                buffer_index: 3,
            },
            BufferArg {
                name: "sdpa_prefill_splitk_params",
                metal_type: "constant SdpaPrefillSplitKParams&",
                buffer_index: 4,
            },
        ]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        self.variant.validate();
        let warps = self.variant.warps;
        (
            "output_ptr".to_string(),
            format!(
                r#"
        // SdpaPrefillSplitKReduceStage (D64/D128)
        #define FLASH_PREFILL_WARPS {warps}

        uint3 tid = lid;
        uint simd_lane_id = (uint)__simd_lane_id;
        uint simd_group_id = (uint)__simd_group_id;

        // Grid: x=TileM, y=Head, z=1
        uint head_idx = gid.y;
        const uint batch_idx = 0;

        ulong out_offset = batch_idx * sdpa_prefill_splitk_params.out_stride_b + head_idx * sdpa_prefill_splitk_params.out_stride_h;
        device half* output_ptr = output + out_offset;

        if (sdpa_prefill_splitk_params.head_dim == 64) {{
            flash_prefill_splitk_reduce_d64<FLASH_PREFILL_WARPS>(
                partial_acc,
                partial_m,
                partial_l,
                output_ptr,
                sdpa_prefill_splitk_params,
                gid,
                tid,
                tptg,
                simd_lane_id,
                simd_group_id
            );
        }} else if (sdpa_prefill_splitk_params.head_dim == 128) {{
            flash_prefill_splitk_reduce_d128<FLASH_PREFILL_WARPS>(
                partial_acc,
                partial_m,
                partial_l,
                output_ptr,
                sdpa_prefill_splitk_params,
                gid,
                tid,
                tptg,
                simd_lane_id,
                simd_group_id
            );
        }}
        "#,
                warps = warps
            ),
        )
    }

    fn struct_defs(&self) -> String {
        format!(
            r#"
    #ifndef METALLIC_SDPA_PREFILL_SPLITK_PARAMS_DEFINED
    #define METALLIC_SDPA_PREFILL_SPLITK_PARAMS_DEFINED
    {}
    #endif
            "#,
            SdpaPrefillSplitKParams::METAL_STRUCT_DEF
        )
    }
}

impl SdpaPrefillSplitKReduceStage {
    pub fn new(params: SdpaPrefillSplitKParams) -> Self {
        Self {
            output: TensorArg::default(),
            partial_acc: TensorArg::default(),
            partial_m: TensorArg::default(),
            partial_l: TensorArg::default(),
            params,
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

/// Resolved version of SdpaParams (identical since no DynamicValues).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct SdpaParamsResolved {
    pub kv_len: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub stride_k_s: u32,
    pub stride_v_s: u32,
}

#[inline]
fn sdpa_params_struct_defs() -> String {
    // Defensive: ensure the struct is only emitted once even if multiple stages request it.
    // This also removes duplication from the Metal sources (no manual `struct SdpaParams`).
    format!(
        r#"
#ifndef METALLIC_SDPA_PARAMS_DEFINED
#define METALLIC_SDPA_PARAMS_DEFINED
{}
#endif
"#,
        SdpaParams::METAL_STRUCT_DEF
    )
}

impl From<SdpaParams> for SdpaParamsResolved {
    fn from(p: SdpaParams) -> Self {
        Self {
            kv_len: p.kv_len,
            head_dim: p.head_dim,
            scale: p.scale,
            stride_k_s: p.stride_k_s,
            stride_v_s: p.stride_v_s,
        }
    }
}
