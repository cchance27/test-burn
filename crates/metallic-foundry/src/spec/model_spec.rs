//! Model specification for declarative execution plans.
//!
//! The forward pass is defined as a sequence of `Step` trait objects,
//! deserialized from JSON via typetag.

use rustc_hash::FxHashMap;
use serde::Deserialize;

use crate::{
    error::MetalError, spec::{IntExpr, Step}, tensor::Dtype
};

/// A model execution specification loaded from JSON.
///
/// Contains architecture parameters and a forward pass definition.
#[derive(Debug, Deserialize)]
pub struct ModelSpec {
    /// Model name (e.g., "qwen2.5-0.5b")
    pub name: String,
    /// Model architecture configuration
    pub architecture: Architecture,
    /// Optional chat template override (Jinja2)
    #[serde(default)]
    pub chat_template: Option<String>,
}

/// Tensor naming conventions for model weights.
/// Each key maps to an array of possible names (first match wins).
#[derive(Debug, Deserialize, Default)]
pub struct TensorNames {
    /// Embedding table names
    #[serde(default)]
    pub embedding: Vec<String>,
    /// Output/LM head weight names
    #[serde(default)]
    pub output_weight: Vec<String>,
    /// Final layer norm names
    #[serde(default)]
    pub final_norm: Vec<String>,
    /// RoPE cosine cache names
    #[serde(default)]
    pub rope_cos: Vec<String>,
    /// RoPE sine cache names
    #[serde(default)]
    pub rope_sin: Vec<String>,
    /// Per-layer tensor names (use {i} for layer index)
    #[serde(default)]
    pub layer: LayerTensorNames,
}

/// Per-layer tensor naming patterns.
/// Use `{i}` as placeholder for layer index.
#[derive(Debug, Deserialize, Default)]
pub struct LayerTensorNames {
    #[serde(default)]
    pub attn_norm: Vec<String>,
    #[serde(default)]
    pub ffn_norm: Vec<String>,
    #[serde(default)]
    pub attn_q: Vec<String>,
    #[serde(default)]
    pub attn_k: Vec<String>,
    #[serde(default)]
    pub attn_v: Vec<String>,
    #[serde(default)]
    pub attn_q_bias: Vec<String>,
    #[serde(default)]
    pub attn_k_bias: Vec<String>,
    #[serde(default)]
    pub attn_v_bias: Vec<String>,
    #[serde(default)]
    pub attn_output: Vec<String>,
    #[serde(default)]
    pub ffn_gate: Vec<String>,
    #[serde(default)]
    pub ffn_up: Vec<String>,
    #[serde(default)]
    pub ffn_down: Vec<String>,
    #[serde(default)]
    pub ffn_gate_bias: Vec<String>,
    #[serde(default)]
    pub ffn_up_bias: Vec<String>,
    #[serde(default)]
    pub ffn_down_bias: Vec<String>,
}

/// Architecture configuration and execution graph.
#[derive(Debug, Deserialize)]
pub struct Architecture {
    /// Generic parameters inferred from GGUF metadata or provided in the DSL.
    #[serde(flatten)]
    pub params: FxHashMap<String, ArchValue>,
    /// Tensor naming conventions for GGUF loading
    #[serde(default)]
    pub tensor_names: TensorNames,
    /// Optional GGUF metadata key mapping used to infer baseline architecture values.
    ///
    /// If absent, the loader falls back to a built-in key set (DEBT).
    #[serde(default)]
    pub metadata_keys: MetadataKeysSpec,
    /// Executor preparation plan (globals + intermediates + KV caches).
    #[serde(default)]
    pub prepare: PrepareSpec,
    /// Weight tensors the executor must bind from GGUF before inference.
    ///
    /// This is used to avoid hardcoding architecture-specific weight binding logic
    /// in the executor (e.g. which weights are "canonical" vs row-major).
    #[serde(default)]
    pub weight_bindings: Vec<WeightBindingSpec>,
    /// Forward pass execution graph - sequence of kernel steps
    #[serde(default)]
    pub forward: Vec<Box<dyn Step>>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct MetadataKeysSpec {
    /// Map from architecture field name -> ordered list of GGUF keys (first match wins).
    ///
    /// Expected field names (current): d_model, n_heads, n_kv_heads, n_layers, ff_dim, vocab_size, max_seq_len,
    /// rope_base, rms_eps.
    #[serde(default)]
    pub keys: FxHashMap<String, Vec<String>>,
}

/// A value inferred from GGUF metadata or provided in the DSL.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArchValue {
    USize(usize),
    F32(f32),
}

impl ArchValue {
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            ArchValue::USize(v) => Some(*v),
            ArchValue::F32(v) => Some(*v as usize),
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            ArchValue::USize(v) => Some(*v as f32),
            ArchValue::F32(v) => Some(*v),
        }
    }
}

impl<'de> serde::Deserialize<'de> for ArchValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{Error, Visitor};
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ArchValue;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a number (usize or f32)")
            }
            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ArchValue::USize(v as usize))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
            where
                E: Error,
            {
                if v < 0 {
                    return Err(E::custom("negative integer not allowed for ArchValue"));
                }
                Ok(ArchValue::USize(v as usize))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(ArchValue::F32(v as f32))
            }
        }
        deserializer.deserialize_any(V)
    }
}

/// Baseline architecture values inferred from GGUF metadata.
///
/// The executor uses the following precedence:
/// GGUF baseline < DSL overrides < runtime overrides.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ArchitectureDefaults {
    pub values: FxHashMap<String, ArchValue>,
}

impl ArchitectureDefaults {
    #[inline]
    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.values.get(key).and_then(|v| match v {
            ArchValue::USize(v) => Some(*v),
            _ => None,
        })
    }

    #[inline]
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.values.get(key).and_then(|v| match v {
            ArchValue::F32(v) => Some(*v),
            _ => None,
        })
    }

    /// Merges these defaults into the provided architecture params if not already set.
    #[inline]
    pub fn apply_to(&self, params: &mut FxHashMap<String, ArchValue>) {
        for (k, v) in &self.values {
            params.entry(k.clone()).or_insert(*v);
        }
    }
}

impl Architecture {
    /// Apply GGUF-derived baseline values to this spec.
    ///
    /// Fields that are zero/unset in the DSL are filled from `defaults`.
    /// Non-zero fields are treated as DSL overrides and preserved.
    pub fn apply_metadata_baseline(&mut self, defaults: &ArchitectureDefaults) -> Result<(), MetalError> {
        defaults.apply_to(&mut self.params);
        Ok(())
    }

    #[inline]
    pub fn d_model(&self) -> usize {
        self.params.get("d_model").and_then(|v| v.as_usize()).unwrap_or(0)
    }

    #[inline]
    pub fn n_heads(&self) -> usize {
        self.params.get("n_heads").and_then(|v| v.as_usize()).unwrap_or(0)
    }

    #[inline]
    pub fn n_kv_heads(&self) -> usize {
        self.params.get("n_kv_heads").and_then(|v| v.as_usize()).unwrap_or(0)
    }

    #[inline]
    pub fn n_layers(&self) -> usize {
        self.params.get("n_layers").and_then(|v| v.as_usize()).unwrap_or(0)
    }

    #[inline]
    pub fn ff_dim(&self) -> usize {
        self.params.get("ff_dim").and_then(|v| v.as_usize()).unwrap_or(0)
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.params.get("vocab_size").and_then(|v| v.as_usize()).unwrap_or(0)
    }

    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.params.get("max_seq_len").and_then(|v| v.as_usize()).unwrap_or(0)
    }

    #[inline]
    pub fn rope_base(&self) -> f32 {
        self.params.get("rope_base").and_then(|v| v.as_f32()).unwrap_or(0.0)
    }

    #[inline]
    pub fn rms_eps(&self) -> f32 {
        self.params.get("rms_eps").and_then(|v| v.as_f32()).unwrap_or(0.0)
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct PrepareSpec {
    /// One-time globals evaluated at session initialization.
    #[serde(default)]
    pub globals: FxHashMap<String, IntExpr>,
    /// Dynamic variables (e.g. "m", "seq_len") with default values for compilation/static analysis.
    #[serde(default)]
    pub dynamics: FxHashMap<String, usize>,
    /// Derived globals evaluated at runtime (per prefill chunk / per decode step).
    #[serde(default)]
    pub derived_globals: Vec<DerivedGlobalSpec>,
    /// Tensors the executor must allocate and bind before inference.
    #[serde(default)]
    pub tensors: Vec<TensorAllocSpec>,
    /// RoPE cache naming (executor computes/upload values).
    #[serde(default)]
    pub rope: Option<RopePrepareSpec>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DerivedGlobalSpec {
    pub name: String,
    pub expr: IntExpr,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RopePrepareSpec {
    pub cos: String,
    pub sin: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RepeatAllocSpec {
    /// Variable name or integer literal for iteration count.
    /// e.g. "n_layers" or "24"
    pub count: String,
    /// Variable name to bind the current index to.
    /// e.g. "i" -> becomes "0", "1", ...
    pub var: String,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StorageClass {
    Intermediate,
    KvCache,
    RopeCache,
    Shared,
    Private,
}

impl Default for StorageClass {
    fn default() -> Self {
        StorageClass::Intermediate
    }
}

fn default_dtype_f16() -> Dtype {
    Dtype::F16
}

#[derive(Debug, Deserialize, Clone)]
pub struct TensorAllocSpec {
    pub name: String,
    #[serde(default)]
    pub repeat: Option<RepeatAllocSpec>,
    #[serde(default = "default_dtype_f16")]
    pub dtype: Dtype,
    #[serde(default)]
    pub storage: StorageClass,
    pub dims: Vec<IntExpr>,
    #[serde(default)]
    pub strides: Option<Vec<IntExpr>>,
    /// If true, this tensor is resized when KV capacity grows.
    /// Intended for KV caches and any other buffers indexed by max_seq_len.
    #[serde(default)]
    pub grow_with_kv: bool,
    /// If true, the executor will zero-fill the buffer after allocation.
    #[serde(default)]
    pub zero_fill: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WeightBindingSpec {
    /// Key used to resolve a GGUF tensor name via `Architecture.tensor_names`.
    ///
    /// Examples:
    /// - "embedding"
    /// - "output_weight"
    /// - "layer.attn_q"
    pub key: String,
    /// Logical tensor name inserted into bindings (may contain "{i}" with repeat).
    pub logical_name: String,
    #[serde(default)]
    pub repeat: Option<RepeatAllocSpec>,
    /// Optional fallback: if the GGUF tensor is missing, bind a zero vector of this length (F16).
    ///
    /// Intended for optional biases.
    #[serde(default)]
    pub fallback_zero_len: Option<IntExpr>,
    #[serde(default)]
    pub layout: WeightLayoutSpec,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WeightLayoutSpec {
    /// Bind weights as-is (row-major / GGUF native layout).
    RowMajor,
    /// Bind weights in canonical k-block-major layout (used by some GEMV/GEMM variants).
    Canonical { expected_k: IntExpr, expected_n: IntExpr },
}

impl Default for WeightLayoutSpec {
    fn default() -> Self {
        WeightLayoutSpec::RowMajor
    }
}

impl ModelSpec {
    /// Load a model spec from a JSON file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, std::io::Error> {
        let contents = std::fs::read_to_string(path)?;
        serde_json::from_str(&contents).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Load a model spec from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl TensorNames {
    /// Find the first matching tensor name from a GGUF model's tensor list.
    pub fn resolve(&self, key: &str, layer_idx: Option<usize>, available: &FxHashMap<String, ()>) -> Option<String> {
        let candidates = match key {
            "embedding" => &self.embedding,
            "output_weight" => &self.output_weight,
            "final_norm" => &self.final_norm,
            "rope_cos" => &self.rope_cos,
            "rope_sin" => &self.rope_sin,
            _ if key.starts_with("layer.") => {
                let layer_key = &key[6..]; // Strip "layer."
                match layer_key {
                    "attn_norm" => &self.layer.attn_norm,
                    "ffn_norm" => &self.layer.ffn_norm,
                    "attn_q" => &self.layer.attn_q,
                    "attn_k" => &self.layer.attn_k,
                    "attn_v" => &self.layer.attn_v,
                    "attn_q_bias" => &self.layer.attn_q_bias,
                    "attn_k_bias" => &self.layer.attn_k_bias,
                    "attn_v_bias" => &self.layer.attn_v_bias,
                    "attn_output" => &self.layer.attn_output,
                    "ffn_gate" => &self.layer.ffn_gate,
                    "ffn_up" => &self.layer.ffn_up,
                    "ffn_down" => &self.layer.ffn_down,
                    "ffn_gate_bias" => &self.layer.ffn_gate_bias,
                    "ffn_up_bias" => &self.layer.ffn_up_bias,
                    "ffn_down_bias" => &self.layer.ffn_down_bias,
                    _ => return None,
                }
            }
            _ => return None,
        };

        for pattern in candidates {
            let name = if let Some(i) = layer_idx {
                pattern.replace("{i}", &i.to_string())
            } else {
                pattern.clone()
            };
            if available.contains_key(&name) {
                return Some(name);
            }
        }
        None
    }
}
