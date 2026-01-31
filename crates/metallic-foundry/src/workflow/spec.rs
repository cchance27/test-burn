use serde::Deserialize;

#[derive(Debug, Clone)]
pub enum Param<T> {
    Literal(T),
    Input(String),
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
                let vv: u32 = v.try_into().map_err(|_| E::custom("u32 literal out of range"))?;
                Ok(Param::Literal(vv))
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
                f.write_str("a f32 literal or a string like \"{var_name}\"")
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
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WorkflowStepSpec {
    Prefill {
        #[serde(default)]
        model_id: Option<String>,
        input: String,
        /// Name of the model binding used for token ids (defaults to "input_ids").
        #[serde(default)]
        input_ids_binding: Option<String>,
        /// Global key for the current position (defaults to "position_offset").
        #[serde(default)]
        position_offset_key: Option<String>,
        /// Global key for batch width (defaults to "m").
        #[serde(default)]
        m_key: Option<String>,
        /// Global key for sequence length (defaults to "seq_len").
        #[serde(default)]
        seq_len_key: Option<String>,
        /// Whether to apply derived globals after updating globals (default true).
        #[serde(default = "default_true")]
        apply_derived_globals: bool,
        #[serde(default)]
        description: Option<String>,
    },
    SetGlobals {
        #[serde(default)]
        model_id: Option<String>,
        /// Integer globals to set on the target model's bindings.
        ///
        /// Values may be literals or "{var}" references to workflow inputs/values.
        globals: std::collections::BTreeMap<String, Param<usize>>,
        /// Whether to apply derived globals after updating globals (default true).
        #[serde(default = "default_true")]
        apply_derived_globals: bool,
    },
    Synchronize,
    Loop {
        #[serde(default)]
        model_id: Option<String>,
        #[serde(default)]
        condition: Option<String>,
        #[serde(default)]
        args: Vec<String>,
        stages: Vec<WorkflowStageSpec>,
    },
    Return {
        output: String,
    },
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WorkflowStageSpec {
    Sample {
        #[serde(default)]
        model_id: Option<String>,
        /// Model binding name for logits (defaults to legacy "input" field name).
        #[serde(alias = "input")]
        logits_binding: String,
        output: String,
        temperature: Param<f32>,
        top_k: Param<u32>,
        top_p: Param<f32>,
        seed: Param<u32>,
    },
    CheckEos {
        input: String,
        output: String,
        eos_token: Param<u32>,
    },
    AppendToken {
        input: String,
        output: String,
    },
    GraphForward {
        #[serde(default)]
        model_id: Option<String>,
        /// Token variable name (legacy "input").
        #[serde(alias = "input")]
        token_var: String,
        /// Model binding name for the token ids buffer (defaults to "input_ids").
        #[serde(default)]
        input_ids_binding: Option<String>,
        /// Model binding name for logits (defaults to legacy "output" value).
        #[serde(alias = "output")]
        logits_binding: String,
        /// Global key for the current position (defaults to "position_offset").
        #[serde(default)]
        position_offset_key: Option<String>,
        /// Whether to apply derived globals after updating position (default true).
        #[serde(default = "default_true")]
        apply_derived_globals: bool,
    },
}
