use std::{
    hash::{Hash, Hasher}, sync::Arc
};

use rustc_hash::FxHashMap;

use super::channel::ChannelU32;
use crate::types::MetalCommandBuffer;

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
    ChannelU32(ChannelU32),
    CommandBuffer(MetalCommandBuffer),
}

impl Value {
    pub fn from_json(v: serde_json::Value) -> Self {
        match v {
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_u64() {
                    Value::Usize(i as usize)
                } else if let Some(f) = n.as_f64() {
                    Value::F32(f as f32)
                } else {
                    Value::F32(0.0)
                }
            }
            serde_json::Value::Bool(b) => Value::Bool(b),
            serde_json::Value::String(s) => Value::Text(Arc::from(s.as_str())),
            serde_json::Value::Array(arr) => Value::Array(arr.into_iter().map(Value::from_json).collect()),
            serde_json::Value::Object(obj) => {
                let mut map = FxHashMap::default();
                for (k, v) in obj {
                    map.insert(k, Value::from_json(v));
                }
                Value::Map(map)
            }
            _ => Value::Bool(false), // Fallback
        }
    }

    pub fn fingerprint64(&self) -> u64 {
        let mut h = rustc_hash::FxHasher::default();
        self.hash_into(&mut h);
        h.finish()
    }

    fn hash_into(&self, h: &mut rustc_hash::FxHasher) {
        match self {
            Value::U32(v) => {
                0u8.hash(h);
                v.hash(h);
            }
            Value::Usize(v) => {
                1u8.hash(h);
                (*v as u64).hash(h);
            }
            Value::F32(v) => {
                2u8.hash(h);
                v.to_bits().hash(h);
            }
            Value::Bool(v) => {
                3u8.hash(h);
                v.hash(h);
            }
            Value::Text(s) => {
                4u8.hash(h);
                s.as_ref().hash(h);
            }
            Value::Array(arr) => {
                5u8.hash(h);
                arr.len().hash(h);
                for v in arr {
                    v.hash_into(h);
                }
            }
            Value::Map(map) => {
                6u8.hash(h);
                map.len().hash(h);
                // Deterministic ordering.
                let mut keys: Vec<&str> = map.keys().map(|s| s.as_str()).collect();
                keys.sort_unstable();
                for k in keys {
                    k.hash(h);
                    if let Some(v) = map.get(k) {
                        v.hash_into(h);
                    }
                }
            }
            Value::TokensU32(tokens) => {
                7u8.hash(h);
                tokens.len().hash(h);
                for t in tokens {
                    t.hash(h);
                }
            }
            // Tensors are not fingerprinted for memoization (expensive/unclear semantics).
            Value::Tensor(_) => {
                8u8.hash(h);
            }
            // Channels are not fingerprinted for memoization (live state / expensive semantics).
            Value::ChannelU32(_) => {
                9u8.hash(h);
            }
            // Command buffers are not fingerprinted for memoization (live state).
            Value::CommandBuffer(_) => {
                10u8.hash(h);
            }
        }
    }

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
            Value::ChannelU32(_) => "channel_u32",
            Value::CommandBuffer(_) => "command_buffer",
        }
    }

    pub fn as_channel_u32(&self) -> Option<&ChannelU32> {
        match self {
            Value::ChannelU32(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_command_buffer(&self) -> Option<&MetalCommandBuffer> {
        match self {
            Value::CommandBuffer(c) => Some(c),
            _ => None,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Value::U32(v) => serde_json::Value::from(*v),
            Value::Usize(v) => serde_json::Value::from(*v),
            Value::F32(v) => serde_json::Value::from(*v),
            Value::Bool(v) => serde_json::Value::from(*v),
            Value::Text(s) => serde_json::Value::from(s.to_string()),
            Value::Array(arr) => serde_json::Value::Array(arr.iter().map(|v| v.to_json()).collect()),
            Value::Map(map) => {
                let mut obj = serde_json::Map::new();
                for (k, v) in map {
                    obj.insert(k.clone(), v.to_json());
                }
                serde_json::Value::Object(obj)
            }
            Value::TokensU32(tokens) => serde_json::Value::Array(tokens.iter().map(|&t| serde_json::Value::from(t)).collect()),
            _ => serde_json::Value::Null,
        }
    }
}
