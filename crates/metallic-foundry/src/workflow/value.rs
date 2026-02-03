use std::sync::Arc;

use rustc_hash::FxHashMap;

/// Runtime value passed between workflow steps.
#[derive(Debug, Clone)]
pub enum Value {
    U32(u32),
    Usize(usize),
    F32(f32),
    Bool(bool),
    Text(Arc<str>),
    Array(Vec<Value>),
    Map(FxHashMap<String, Value>),
    TokensU32(Vec<u32>),
    Tensor(crate::types::TensorArg),
}

impl Value {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::U32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Value::Usize(v) => Some(*v),
            Value::U32(v) => Some(*v as usize),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_tokens_u32(&self) -> Option<&[u32]> {
        match self {
            Value::TokensU32(v) => Some(v.as_ref()),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s.as_ref()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&FxHashMap<String, Value>> {
        match self {
            Value::Map(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_tensor(&self) -> Option<&crate::types::TensorArg> {
        match self {
            Value::Tensor(t) => Some(t),
            _ => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::U32(_) => "u32",
            Value::Usize(_) => "usize",
            Value::F32(_) => "f32",
            Value::Bool(_) => "bool",
            Value::Text(_) => "text",
            Value::Array(_) => "array",
            Value::Map(_) => "map",
            Value::TokensU32(_) => "tokens_u32",
            Value::Tensor(_) => "tensor",
        }
    }
}
