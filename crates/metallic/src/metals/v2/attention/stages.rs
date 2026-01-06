use metallic_macros::{KernelArgs, MetalStruct, Stage};

use crate::{
    compound::{BufferArg, Stage}, types::TensorArg
};

// ... HeadLayoutStage is here ... (preserving it)

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

/// Core SDPA Attention Loop (Flash/Decode).
///
/// Iterates over Key/Value sequence, computes Q*K^T, updates Online Softmax, accumulates V.
/// Assumes Q is in `q_shared` (from RopeStage).
#[derive(KernelArgs, Clone, Debug)]
pub struct SdpaCoreStage {
    pub sdpa_params: SdpaParams,
}

impl Stage for SdpaCoreStage {
    fn includes(&self) -> Vec<&'static str> {
        vec!["v2/attention/sdpa_decode.metal"]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![BufferArg {
            name: "sdpa_params",
            metal_type: "constant SdpaParams&",
            buffer_index: 15,
        }]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        (
            "output_ptr".to_string(),
            r#"
    // SdpaCoreStage (Refactored)
    // Invokes vectorized decoder from sdpa_decode.metal

    #define MAX_REDUCE_DIM_VEC 64 // 256/4
    threadgroup float reduce_shared[MAX_REDUCE_DIM_VEC];
    
    // Load Q vector from shared memory (populated by RopeStage)
    // q_shared is threadgroup half4*
    half4 q_vec = q_shared[tid];
    
    sdpa_decode_vectorized<MAX_REDUCE_DIM_VEC>(
        q_vec,
        k_ptr,
        v_ptr,
        output_ptr,
        tid,
        sdpa_params,
        reduce_shared
    );
"#
            .to_string(),
        )
    }

    fn struct_defs(&self) -> String {
        // Struct defined in sdpa_decode.metal
        String::new()
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

impl SdpaCoreStage {
    pub fn new(sdpa_params: SdpaParams) -> Self {
        Self { sdpa_params }
    }
}
