//! Model specification for declarative execution plans.
//!
//! The forward pass is defined as a sequence of `Step` trait objects,
//! deserialized from JSON via typetag.

use rustc_hash::FxHashMap;
use serde::Deserialize;

use crate::foundry::spec::Step;

/// A model execution specification loaded from JSON.
///
/// Contains architecture parameters and a forward pass definition.
#[derive(Debug, Deserialize)]
pub struct ModelSpec {
    /// Model name (e.g., "qwen2.5-0.5b")
    pub name: String,
    /// Model architecture configuration
    pub architecture: Architecture,
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
    /// Hidden dimension size
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of key-value heads (for GQA)
    pub n_kv_heads: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// FFN intermediate dimension
    pub ff_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RoPE base frequency
    #[serde(default = "default_rope_base")]
    pub rope_base: f32,
    /// RMSNorm epsilon
    #[serde(default = "default_rms_eps")]
    pub rms_eps: f32,
    /// Tensor naming conventions for GGUF loading
    #[serde(default)]
    pub tensor_names: TensorNames,
    /// Forward pass execution graph - sequence of kernel steps
    #[serde(default)]
    pub forward: Vec<Box<dyn Step>>,
}

fn default_rope_base() -> f32 {
    10000.0
}

fn default_rms_eps() -> f32 {
    1e-6
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
    pub fn resolve<'a>(&self, key: &str, layer_idx: Option<usize>, available: &FxHashMap<String, ()>) -> Option<String> {
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
