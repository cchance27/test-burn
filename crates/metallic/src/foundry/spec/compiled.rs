use rustc_hash::FxHashMap;

use super::TensorBindings;
use crate::{error::MetalError, foundry::Foundry, types::TensorArg};

/// A compiled execution step that uses integer indices for fast lookup.
pub trait CompiledStep: Send + Sync + std::fmt::Debug {
    /// Execute this step using fast bindings (Vec lookup) for tensors,
    /// and slow bindings (HashMap) for global variables if needed.
    fn execute(&self, foundry: &mut Foundry, bindings: &FastBindings, globals: &TensorBindings) -> Result<(), MetalError>;
}

/// Runtime bindings storage optimized for vector access.
#[derive(Default)]
pub struct FastBindings {
    /// The actual tensor arguments, indexed by the compiler-assigned slot ID.
    storage: Vec<Option<TensorArg>>,
}

impl FastBindings {
    pub fn new(capacity: usize) -> Self {
        Self {
            storage: vec![None; capacity],
        }
    }

    #[inline(always)]
    pub fn set(&mut self, index: usize, arg: TensorArg) {
        if index >= self.storage.len() {
            self.storage.resize(index + 1, None);
        }
        self.storage[index] = Some(arg);
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&TensorArg> {
        self.storage.get(index).and_then(|opt| opt.as_ref())
    }
}

/// A symbol table that maps logical names (with interpolation resolved) to indices.
/// Used during the compilation phase.
pub struct SymbolTable {
    map: FxHashMap<String, usize>,
    next_id: usize,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self {
            map: FxHashMap::default(),
            next_id: 0,
        }
    }
}

impl SymbolTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, name: &str) -> Option<usize> {
        self.map.get(name).copied()
    }

    pub fn get_or_create(&mut self, name: String) -> usize {
        if let Some(&id) = self.map.get(&name) {
            id
        } else {
            let id = self.next_id;
            self.map.insert(name, id);
            self.next_id += 1;
            id
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}
