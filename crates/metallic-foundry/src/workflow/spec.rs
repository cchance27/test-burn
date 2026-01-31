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
    pub inputs: Vec<WorkflowInputSpec>,
    pub steps: Vec<WorkflowStepSpec>,
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
        #[serde(default)]
        description: Option<String>,
    },
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

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WorkflowStageSpec {
    Sample {
        #[serde(default)]
        model_id: Option<String>,
        input: String,
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
        input: String,
        output: String,
    },
}
