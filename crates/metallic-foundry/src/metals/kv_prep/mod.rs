//! Fused KV preparation kernel(s).
//!
//! This fuses the common decode-time KV preparation chain:
//! - KvRearrange(Q) + Rope(Q) -> q_rot
//! - KvRearrange(K) + Rope(K) + compact KV write -> k_cache
//! - KvRearrange(V) + compact KV write -> v_cache
//!
//! The target is to reduce per-token dispatch fanout during decode.

use metallic_macros::{Kernel, KernelArgs, MetalStruct};
use serde::Deserialize;

use crate::{
    spec::{DynamicValue, TensorBindings}, types::TensorArg
};

pub mod step;

const ROPE_MODE_NEOX: u32 = 0;
const ROPE_MODE_NORMAL: u32 = 1;

#[derive(Clone, Copy, Debug, Default, serde::Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum RopeMode {
    Neox,
    #[default]
    Normal,
}

impl RopeMode {
    const fn as_u32(self) -> u32 {
        match self {
            Self::Neox => ROPE_MODE_NEOX,
            Self::Normal => ROPE_MODE_NORMAL,
        }
    }
}

fn default_rope_mode() -> DynamicValue<u32> {
    DynamicValue::Literal(RopeMode::Normal.as_u32())
}

fn default_layer_idx() -> DynamicValue<u32> {
    DynamicValue::Literal(0)
}

fn default_no_rope_layer_step() -> DynamicValue<u32> {
    DynamicValue::Literal(0)
}

fn bind_u32_if_scoped(value: &DynamicValue<u32>, bindings: &TensorBindings) -> DynamicValue<u32> {
    match value {
        DynamicValue::Literal(v) => DynamicValue::Literal(*v),
        DynamicValue::Variable(name) => bindings
            .get_var(name)
            .and_then(|v| v.parse::<u32>().ok())
            .map(DynamicValue::Literal)
            .unwrap_or_else(|| DynamicValue::Variable(name.clone())),
    }
}

fn deserialize_rope_mode<'de, D>(deserializer: D) -> Result<DynamicValue<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum RopeModeInput {
        Mode(RopeMode),
        Dynamic(DynamicValue<u32>),
    }

    match RopeModeInput::deserialize(deserializer)? {
        RopeModeInput::Mode(mode) => Ok(DynamicValue::Literal(mode.as_u32())),
        RopeModeInput::Dynamic(value) => Ok(value),
    }
}

#[derive(MetalStruct, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct KvPrepFusedParams {
    /// d_model (n_heads * head_dim).
    pub d_model: DynamicValue<u32>,
    /// kv_dim (n_kv_heads * head_dim).
    pub kv_dim: DynamicValue<u32>,
    /// head_dim (must be even for RoPE).
    pub head_dim: DynamicValue<u32>,
    pub n_heads: DynamicValue<u32>,
    pub n_kv_heads: DynamicValue<u32>,
    /// n_heads / n_kv_heads.
    pub group_size: DynamicValue<u32>,
    /// Number of tokens processed in this step.
    pub seq_len: DynamicValue<u32>,
    /// Offset into the KV cache / RoPE tables.
    pub position_offset: DynamicValue<u32>,
    /// KV cache stride dimension.
    pub max_seq_len: DynamicValue<u32>,
    /// Total Q elements (n_heads * seq_len * head_dim).
    /// Per-element dispatch expects this name.
    pub total_elements: DynamicValue<u32>,
    /// RoPE pair layout selector:
    /// - "neox": pair hd with hd+half_dim
    /// - "normal": pair adjacent dims: 0<->1, 2<->3, ...
    #[serde(default = "default_rope_mode", deserialize_with = "deserialize_rope_mode")]
    pub rope_mode: DynamicValue<u32>,
    /// Current layer index (0-based).
    #[serde(default = "default_layer_idx")]
    pub layer_idx: DynamicValue<u32>,
    /// Skip RoPE every Nth layer when >0 (SmolLM3 uses N=4).
    ///
    /// Disabled when set to 0.
    #[serde(default = "default_no_rope_layer_step")]
    pub no_rope_layer_step: DynamicValue<u32>,
}

impl KvPrepFusedParams {
    /// Bind scope-only dynamic fields to literals at compile time.
    ///
    /// Repeat variables like `{i}` are unavailable during compiled-step execution,
    /// so layer-aware controls must be materialized while the repeat scope exists.
    #[inline]
    pub fn bind_scope_literals(&self, bindings: &TensorBindings) -> Self {
        let mut out = self.clone();
        out.layer_idx = bind_u32_if_scoped(&self.layer_idx, bindings);
        out.no_rope_layer_step = bind_u32_if_scoped(&self.no_rope_layer_step, bindings);
        out
    }

    #[inline]
    pub fn resolve(&self, bindings: &TensorBindings) -> KvPrepFusedParamsResolved {
        KvPrepFusedParamsResolved {
            d_model: self.d_model.resolve(bindings),
            kv_dim: self.kv_dim.resolve(bindings),
            head_dim: self.head_dim.resolve(bindings),
            n_heads: self.n_heads.resolve(bindings),
            n_kv_heads: self.n_kv_heads.resolve(bindings),
            group_size: self.group_size.resolve(bindings),
            seq_len: self.seq_len.resolve(bindings),
            position_offset: self.position_offset.resolve(bindings),
            max_seq_len: self.max_seq_len.resolve(bindings),
            total_elements: self.total_elements.resolve(bindings),
            rope_mode: self.rope_mode.resolve(bindings),
            layer_idx: self.layer_idx.resolve(bindings),
            no_rope_layer_step: self.no_rope_layer_step.resolve(bindings),
        }
    }
}

/// Fused KV-prep kernel for F16 activations.
///
/// Constraints:
/// - head_dim must be even (RoPE pairs)
/// - n_heads must be divisible by n_kv_heads (GQA group_size)
/// - Inputs are row-major with row_stride == dim
/// - Assumes batch == 1 (matches current decode/prefill usage)
/// - position_offset + seq_len must be <= max_seq_len
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "kv_prep/kv_prep_fused.metal",
    function = "kv_prep_fused_kernel_f16",
    args = KvPrepFusedParamsResolved,
    dispatch = per_element,
    dtype = F16,
    step = false
)]
pub struct KvPrepFused {
    pub q: TensorArg,
    pub k: TensorArg,
    pub v: TensorArg,

    #[arg(output)]
    pub q_rot: TensorArg,
    #[arg(output)]
    pub k_cache: TensorArg,
    #[arg(output)]
    pub v_cache: TensorArg,

    pub cos: TensorArg,
    pub sin: TensorArg,

    pub params: KvPrepFusedParamsResolved,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct RopeOnly {
        #[serde(default = "default_rope_mode", deserialize_with = "deserialize_rope_mode")]
        rope_mode: DynamicValue<u32>,
        #[serde(default = "default_layer_idx")]
        layer_idx: DynamicValue<u32>,
        #[serde(default = "default_no_rope_layer_step")]
        no_rope_layer_step: DynamicValue<u32>,
    }

    #[test]
    fn rope_mode_defaults_to_normal() {
        let parsed: RopeOnly = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.rope_mode, DynamicValue::Literal(ROPE_MODE_NORMAL));
        assert_eq!(parsed.layer_idx, DynamicValue::Literal(0));
        assert_eq!(parsed.no_rope_layer_step, DynamicValue::Literal(0));
    }

    #[test]
    fn rope_mode_accepts_named_literals() {
        let parsed: RopeOnly = serde_json::from_str(r#"{"rope_mode":"neox"}"#).unwrap();
        assert_eq!(parsed.rope_mode, DynamicValue::Literal(ROPE_MODE_NEOX));
        let parsed: RopeOnly = serde_json::from_str(r#"{"rope_mode":"normal"}"#).unwrap();
        assert_eq!(parsed.rope_mode, DynamicValue::Literal(ROPE_MODE_NORMAL));
    }

    #[test]
    fn no_rope_layer_params_accept_literals_and_dynamic_refs() {
        let parsed: RopeOnly = serde_json::from_str(r#"{"layer_idx":"{i}","no_rope_layer_step":4}"#).unwrap();
        assert_eq!(parsed.layer_idx, DynamicValue::Variable("i".to_string()));
        assert_eq!(parsed.no_rope_layer_step, DynamicValue::Literal(4));
    }

    #[test]
    fn bind_scope_literals_materializes_repeat_index() {
        let params = KvPrepFusedParams {
            layer_idx: DynamicValue::Variable("i".to_string()),
            no_rope_layer_step: DynamicValue::Literal(4),
            ..Default::default()
        };
        let mut bindings = TensorBindings::new();
        bindings.push_scope();
        bindings.set_var("i", "7");
        let bound = params.bind_scope_literals(&bindings);
        assert_eq!(bound.layer_idx, DynamicValue::Literal(7));
        assert_eq!(bound.no_rope_layer_step, DynamicValue::Literal(4));
    }
}
