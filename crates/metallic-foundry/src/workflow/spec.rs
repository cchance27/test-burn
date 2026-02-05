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
    // DX/back-compat: accept `phases` as an alias for `steps` to avoid confusion with model DSL "steps".
    #[serde(alias = "phases")]
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
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub step: Option<f64>,
    #[serde(default)]
    pub options: Option<Vec<String>>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub hidden: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PrefillSpec {
    #[serde(default)]
    pub model_id: Option<String>,
    pub input: String,
    /// Optional workflow variable to receive the session's end position after prefill.
    ///
    /// This is preferred over `{prompt_tokens.len}` for multi-turn flows, because `prompt_tokens`
    /// may be either a delta slice (continuation tokens) or the full transcript tokens.
    #[serde(default)]
    pub output_pos: Option<String>,
    /// Prefill input interpretation mode.
    ///
    /// - `"delta"` (default): `input` is a token slice to be appended at `session.current_pos`.
    /// - `"full_append_only"`: `input` is the full prompt tokenization starting at position 0 and
    ///   monotonically growing; prefill will consume only the suffix `[session.current_pos..]`.
    #[serde(default)]
    pub mode: Option<String>,
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
    pub min_p: Param<f32>,
    #[serde(default)]
    pub repeat_penalty: Param<f32>,
    #[serde(default)]
    pub repeat_last_n: Param<usize>,
    #[serde(default)]
    pub presence_penalty: Param<f32>,
    #[serde(default)]
    pub frequency_penalty: Param<f32>,
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
    /// Optional base tokens variable (TokensU32) used when `mode="delta"` to append.
    #[serde(default)]
    pub base_tokens: Option<String>,
    /// Tokenization mode.
    ///
    /// Supported values:
    /// - `"raw"` (default): `tokenizer.encode(text)`
    /// - `"chat_single_turn"`: `tokenizer.encode_single_turn_chat_prompt(text)`
    /// - `"delta"`: `tokenizer.encode(text)` and append to `base_tokens` into `output`
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FormatChatSpec {
    #[serde(default)]
    pub model_id: Option<String>,
    /// Input messages variable (Value::Array of message maps).
    pub input: String,
    /// Output formatted prompt text variable.
    pub output: String,
    pub add_generation_prompt: bool,
    /// Optional system prompt variable name to prepend to the first turn.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Formatting mode.
    ///
    /// - `"full"` (default): render all messages
    /// - `"delta"`: render only newly appended messages since the last run of this op instance
    #[serde(default)]
    pub mode: Option<String>,
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
    // Use `else` in JSON (Rust keyword).
    #[serde(rename = "else", default)]
    pub else_: Vec<WorkflowStepSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WhileSpec {
    pub condition: String,
    #[serde(default)]
    pub max_iterations: Option<Param<usize>>,
    // Back-compat: accept `phases` as an alias for `body`.
    #[serde(alias = "phases")]
    pub body: Vec<WorkflowStepSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WhileBatchedSpec {
    pub condition: String,
    #[serde(default)]
    pub max_iterations: Option<Param<usize>>,
    /// Optional decode batch size. If omitted, defaults to `METALLIC_FOUNDRY_DECODE_BATCH_SIZE` or 1.
    #[serde(default)]
    pub batch_size: Option<Param<usize>>,
    /// Safety valve: allow `batch_size > 1` while EOS stopping is enabled.
    ///
    /// This can cause "KV overshoot" (tokens after EOS are still computed within the batch).
    /// Default is false to prevent accidental misuse in multi-turn workflows.
    #[serde(default)]
    pub unsafe_allow_overshoot: bool,
    /// Name of the per-iteration token variable produced by the loop body (u32 or u32[1] Tensor).
    pub token_var: String,
    /// Optional channel variable name (`Value::ChannelU32`) to use for token emission.
    ///
    /// When set, `while_batched` drains tokens from the channel after the captured batch completes,
    /// instead of synchronously reading `token_var` per iteration.
    #[serde(default)]
    pub stream_channel: Option<String>,
    /// When `stream_channel` is set, optionally poll/drain the channel while the GPU command buffer
    /// is executing, overlapping CPU work (e.g. detokenize/UI) with GPU decode.
    #[serde(default)]
    pub stream_async_poll: bool,
    /// Poll interval in microseconds when `stream_async_poll` is enabled.
    ///
    /// Lower values reduce latency but can increase CPU overhead.
    #[serde(default = "default_stream_poll_us")]
    pub stream_poll_interval_us: u32,
    /// Output workflow variable to append generated tokens into.
    pub output_tokens: String,
    /// EOS token id (used when `METALLIC_IGNORE_EOS_STOP` is not set).
    #[serde(default)]
    pub eos_token: Param<u32>,
    // Back-compat: accept `phases` as an alias for `body`.
    #[serde(alias = "phases")]
    pub body: Vec<WorkflowStepSpec>,
}

fn default_stream_poll_us() -> u32 {
    200
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
pub struct StreamInitSpec {
    /// Output workflow variable name to store the channel in.
    pub output: String,
    /// Ring capacity (items).
    pub capacity: Param<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamWriteU32Spec {
    /// Channel workflow variable (Value::ChannelU32).
    pub channel: String,
    /// Input token variable (u32 or Tensor u32[1]).
    pub input: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CaptureBeginSpec {}

#[derive(Debug, Clone, Deserialize)]
pub struct CaptureEndSpec {
    /// If true, waits for completion before continuing.
    #[serde(default = "default_true")]
    pub wait: bool,
    /// Optional output variable name to store the command buffer in (Value::CommandBuffer).
    #[serde(default)]
    pub output: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CaptureWaitSpec {
    /// Command buffer workflow variable (Value::CommandBuffer).
    pub input: String,
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
