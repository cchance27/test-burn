use std::{any::Any, borrow::Cow};

use crate::tensor::Dtype;

#[derive(thiserror::Error, Debug)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
    #[error("Platform error: {0}")]
    PlatformError(String),
}

/// Metadata about a tensor in the model
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub offset: u64,
    /// The strict data type of the tensor.
    pub data_type: Dtype,
}

/// A view into tensor data
pub enum TensorData<'a> {
    Slice(&'a [u8]),
    Owned(Vec<u8>),
}

impl AsRef<[u8]> for TensorData<'_> {
    fn as_ref(&self) -> &[u8] {
        match self {
            TensorData::Slice(s) => s,
            TensorData::Owned(v) => v,
        }
    }
}

impl TensorData<'_> {
    #[must_use] 
    pub fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }
}

/// Abstract metadata wrapper.
#[derive(Debug, Clone)]
pub enum MetadataValue<'a> {
    String(Cow<'a, str>),
    Int(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<MetadataValue<'a>>),
}

impl MetadataValue<'_> {
    #[must_use] 
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_ref()),
            _ => None,
        }
    }

    #[must_use] 
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }

    #[must_use] 
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }
}

pub trait ModelMetadata: Send + Sync {
    fn get(&self, key: &str) -> Option<MetadataValue<'_>>;

    fn get_string(&self, key: &str) -> Option<Cow<'_, str>> {
        match self.get(key)? {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    fn get_u32(&self, key: &str) -> Option<u32> {
        match self.get(key)? {
            MetadataValue::Int(i) => Some(i as u32),
            _ => None,
        }
    }

    fn get_array(&self, key: &str) -> Option<Vec<MetadataValue<'_>>> {
        match self.get(key)? {
            MetadataValue::Array(a) => Some(a),
            _ => None,
        }
    }

    fn get_array_string(&self, key: &str) -> Option<Vec<String>> {
        let arr = self.get_array(key)?;
        let mut result = Vec::with_capacity(arr.len());
        for val in arr {
            if let MetadataValue::String(s) = val {
                result.push(s.to_string());
            } else {
                return None;
            }
        }
        Some(result)
    }

    fn get_i64(&self, key: &str) -> Option<i64> {
        match self.get(key)? {
            MetadataValue::Int(i) => Some(i),
            _ => None,
        }
    }

    fn get_f32(&self, key: &str) -> Option<f32> {
        match self.get(key)? {
            MetadataValue::Float(f) => Some(f as f32),
            _ => None,
        }
    }

    fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            MetadataValue::Bool(b) => Some(b),
            _ => None,
        }
    }

    /// Parse a source-specific data type string into a generic Dtype.
    fn parse_dtype(&self, s: &str) -> Option<Dtype>;

    /// Extract tokenizer vocabulary tokens.
    fn tokenizer_tokens(&self) -> Option<Vec<String>>;

    /// Extract tokenizer merges.
    fn tokenizer_merges(&self) -> Option<Vec<String>>;
}

pub trait LoadedModel: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    // --- Metadata Access ---
    fn architecture(&self) -> Option<&str>;
    fn metadata(&self) -> &dyn ModelMetadata;

    // --- Tensor Access ---
    fn tensor_info(&self, name: &str) -> Option<TensorInfo>;
    fn tensor_data(&self, name: &str) -> Result<TensorData<'_>, LoaderError>;
    fn tensor_names(&self) -> Vec<String>;

    // --- Memory Management ---
    fn estimated_memory_usage(&self) -> usize;
    fn offload_tensor(&self, name: &str) -> Result<(), LoaderError>;
    fn load_tensor(&self, name: &str) -> Result<(), LoaderError>;

    // --- Fallback & Defaults (Legacy/Compatibility) ---
    /// Returns a list of supported fallback keys (e.g. "`rope_cos`", "`final_norm`") that the model can provide
    /// default tensors for if missing in the source file.
    fn available_fallbacks(&self) -> &[String];

    /// Retrieve a fallback tensor value by key.
    /// IMPLEMENTORS: This should return warnings if used, as models should prefer explicit metadata.
    fn get_fallback(&self, key: &str) -> Result<Option<TensorData<'_>>, LoaderError>;

    /// Returns a list of inferred architecture parameters (e.g. "`d_model`", "`n_heads`")
    /// that the loader can derive from its internal metadata or knowledge of the architecture.
    fn inferred_architecture_params(&self) -> Vec<(String, MetadataValue<'_>)>;
}
