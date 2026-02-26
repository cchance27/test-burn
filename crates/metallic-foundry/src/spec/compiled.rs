use rustc_hash::FxHashMap;

use super::TensorBindings;
use crate::{Foundry, error::MetalError, types::TensorArg};

/// A compiled execution step that uses integer indices for fast lookup.
pub trait CompiledStep: Send + Sync + std::fmt::Debug {
    /// Execute this step using fast bindings (Vec lookup) for tensors,
    /// and slow bindings (HashMap) for global variables if needed.
    fn execute(
        &self,
        foundry: &mut Foundry,
        bindings: &FastBindings,
        globals: &TensorBindings,
        symbols: &SymbolTable,
    ) -> Result<(), MetalError>;

    /// Human-readable name for this step (for debugging).
    fn name(&self) -> &'static str {
        "UnnamedStep"
    }

    /// Optional perf metadata label used for decode hot-step diagnostics.
    /// Keep this lightweight: it may be called on hot paths when diagnostics are enabled.
    fn perf_metadata(&self, _globals: &TensorBindings) -> Option<String> {
        None
    }
}

/// Pre-resolved indices for tensor arguments (weights, scales, etc.).
/// This allows LoaderStage to bind arguments without hash lookups.
#[derive(Debug, Clone, Default)]
pub struct ResolvedSymbols {
    pub weights: usize,
    pub scales: Option<usize>,
    pub bias: Option<usize>,
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

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }
}

/// A symbol table that maps logical names (with interpolation resolved) to indices.
/// Used during the compilation phase.
#[derive(Default)]
pub struct SymbolTable {
    map: FxHashMap<String, usize>,
    next_id: usize,
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

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &usize)> {
        self.map.iter()
    }
}
