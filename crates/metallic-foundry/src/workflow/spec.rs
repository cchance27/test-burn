use rustc_hash::FxHashMap;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub enum Param<T> {
    Literal(T),
    Input(String),
}

impl<T: Default> Default for Param<T> {
    fn default() -> Self {
        Param::Literal(T::default())
    }
}

fn parse_ref_or_err<E: serde::de::Error>(value: &str) -> Result<String, E> {
    if value.starts_with('{') && value.ends_with('}') && value.len() > 2 {
        Ok(value[1..value.len() - 1].to_string())
    } else {
        Err(E::custom("expected \"{var_name}\" for Param reference"))
    }
}

impl<'de> Deserialize<'de> for Param<usize> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{Error, Visitor};

        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Param<usize>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a usize literal or a string like \"{var_name}\"")
            }
            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(Param::Literal(v as usize))
            }
            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(Param::Input(parse_ref_or_err::<E>(value)?))
            }
        }
        deserializer.deserialize_any(V)
    }
}

impl<'de> Deserialize<'de> for Param<u32> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{Error, Visitor};

        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Param<u32>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a u32 literal or a string like \"{var_name}\"")
            }
            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(Param::Literal(v as u32))
            }
            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(Param::Input(parse_ref_or_err::<E>(value)?))
            }
        }
        deserializer.deserialize_any(V)
    }
}

impl<'de> Deserialize<'de> for Param<f32> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{Error, Visitor};

        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Param<f32>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a float literal or a string like \"{var_name}\"")
            }
            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(Param::Literal(v as f32))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(Param::Literal(v as f32))
            }
            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(Param::Input(parse_ref_or_err::<E>(value)?))
            }
        }
        deserializer.deserialize_any(V)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkflowSpec {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Default model id used when a step omits `model_id`.
    #[serde(default)]
    pub default_model: Option<String>,
    /// Workflow variable to return as the final result.
    #[serde(default)]
    pub return_value: Option<String>,
    #[serde(default)]
    pub resources: Option<WorkflowResourcesSpec>,
    #[serde(default)]
    pub inputs: Vec<WorkflowInputSpec>,
    pub steps: Vec<WorkflowStepSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkflowResourcesSpec {
    #[serde(default)]
    pub models: Vec<WorkflowModelResourceSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkflowModelResourceSpec {
    pub id: String,
    pub gguf_path: String,
    pub spec_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkflowInputSpec {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub default: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PrefillSpec {
    #[serde(default)]
    pub model_id: Option<String>,
    pub input: String,
    /// Name of the model binding used for token ids (defaults to "input_ids").
    #[serde(default)]
    pub input_ids_binding: Option<String>,
    #[serde(default)]
    pub logits_binding: Option<String>,
    #[serde(default)]
    pub position_offset_key: Option<String>,
    #[serde(default)]
    pub m_key: Option<String>,
    #[serde(default)]
    pub seq_len_key: Option<String>,
    #[serde(default = "default_true")]
    pub apply_derived_globals: bool,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ForwardSpec {
    #[serde(default)]
    pub model_id: Option<String>,
    /// Bindings from workflow variables to model inputs.
    #[serde(default)]
    pub inputs: FxHashMap<String, String>,
    /// Extraction from model outputs to workflow variables.
    #[serde(default)]
    pub outputs: FxHashMap<String, String>,
    /// Update model globals before forward pass.
    #[serde(default)]
    pub update_globals: FxHashMap<String, Param<usize>>,
    #[serde(default = "default_true")]
    pub apply_derived_globals: bool,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SampleSpec {
    /// Input logits variable (Tensor).
    pub logits: String,
    /// Output variable for sampled token (u32).
    pub output: String,
    #[serde(default)]
    pub temperature: Param<f32>,
    #[serde(default)]
    pub top_k: Param<u32>,
    #[serde(default)]
    pub top_p: Param<f32>,
    #[serde(default)]
    pub seed: Param<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeSpec {
    #[serde(default)]
    pub model_id: Option<String>,
    /// Input text variable.
    pub input: String,
    /// Output tokens variable (TokensU32).
    pub output: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DetokenizeSpec {
    #[serde(default)]
    pub model_id: Option<String>,
    /// Input tokens variable (TokensU32 or u32).
    pub input: String,
    /// Output text variable.
    pub output: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SetGlobalsSpec {
    #[serde(default)]
    pub model_id: Option<String>,
    pub globals: FxHashMap<String, Param<usize>>,
    #[serde(default = "default_true")]
    pub apply_derived_globals: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReturnSpec {
    #[serde(default)]
    pub output: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ComputeIntSpec {
    /// Destination workflow variable.
    pub output: String,
    /// Integer expression using workflow variables (e.g. "{pos} + 1").
    pub expr: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IfSpec {
    pub condition: String,
    pub then: Vec<WorkflowStepSpec>,
    #[serde(default)]
    pub else_: Vec<WorkflowStepSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WhileSpec {
    pub condition: String,
    #[serde(default)]
    pub max_iterations: Option<Param<usize>>,
    pub body: Vec<WorkflowStepSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CheckEosSpec {
    pub input: String,
    pub output: String,
    #[serde(default)]
    pub eos_token: Param<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AppendTokenSpec {
    pub input: String,
    pub output: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraphForwardSpec {
    #[serde(default)]
    pub model_id: Option<String>,
    pub token_var: String,
    #[serde(default)]
    pub input_ids_binding: Option<String>,
    pub logits_binding: String,
    #[serde(default)]
    pub position_offset_key: Option<String>,
    #[serde(default)]
    pub position: Option<Param<usize>>,
    #[serde(default = "default_true")]
    pub apply_derived_globals: bool,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkflowStepSpec {
    pub op: String,
    #[serde(flatten)]
    pub params: serde_json::Value,
}

fn default_true() -> bool {
    true
}
