use std::sync::Arc;

/// Runtime value passed between workflow steps.
#[derive(Debug, Clone)]
pub enum Value {
    U32(u32),
    Usize(usize),
    F32(f32),
    Bool(bool),
    Text(Arc<str>),
    TokensU32(Arc<[u32]>),
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
}
