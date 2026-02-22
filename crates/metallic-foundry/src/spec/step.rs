//! DSL Step trait and TensorBindings for model execution plans.
//!
//! Kernels with `#[derive(Kernel)]` + `step = true` auto-generate Step impls.
//! The macro generates a `{Kernel}Step` struct with string refs that resolves
//! to actual TensorArgs via TensorBindings at execute time.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{Foundry, error::MetalError, types::TensorArg};

/// Reference to a named tensor in the execution graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub struct Ref(pub String);

impl From<&str> for Ref {
    fn from(s: &str) -> Self {
        Ref(s.to_string())
    }
}

impl From<String> for Ref {
    fn from(s: String) -> Self {
        Ref(s)
    }
}

/// Runtime tensor bindings - maps string refs to actual TensorArgs.
///
/// The Executor populates this with:
/// - Weight tensors materialized from GGUF
/// - Intermediate tensors allocated for the forward pass
/// - KV cache tensors
#[derive(Default)]
pub struct TensorBindings {
    bindings: FxHashMap<String, TensorArg>,
    /// Stack of variable scopes for interpolation (e.g. "i" -> "0")
    scopes: Vec<FxHashMap<String, String>>,
    /// Global variables (e.g. config values like "n_layers")
    globals: FxHashMap<String, String>,
    /// Integer globals for fast lookup without String allocation/parsing
    int_globals: FxHashMap<std::sync::Arc<str>, usize>,
}

impl std::fmt::Debug for TensorBindings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorBindings")
            .field("bindings", &format!("{} entries", self.bindings.len()))
            .field("scopes", &self.scopes)
            .field("globals", &self.globals)
            .finish()
    }
}

impl TensorBindings {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a tensor binding.
    pub fn insert(&mut self, name: impl Into<String>, tensor: TensorArg) {
        self.bindings.insert(name.into(), tensor);
    }

    /// Remove a tensor binding.
    pub fn remove(&mut self, name: &str) -> Option<TensorArg> {
        let resolved_name = self.interpolate(name.to_string());
        self.bindings.remove(&resolved_name)
    }

    /// Set (insert or replace) a tensor binding by name.
    ///
    /// Fast-path avoids allocations when `name` does not require interpolation.
    pub fn set_binding(&mut self, name: &str, tensor: TensorArg) {
        if name.contains('{') {
            let resolved_name = self.interpolate(name.to_string());
            if let Some(existing) = self.bindings.get_mut(&resolved_name) {
                *existing = tensor;
            } else {
                self.bindings.insert(resolved_name, tensor);
            }
            return;
        }

        if let Some(existing) = self.bindings.get_mut(name) {
            *existing = tensor;
        } else {
            self.bindings.insert(name.to_string(), tensor);
        }
    }

    /// Set a global variable.
    pub fn set_global(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.globals.insert(key.into(), value.into());
    }

    /// Push a new variable scope.
    pub fn push_scope(&mut self) {
        self.scopes.push(FxHashMap::default());
    }

    /// Pop the top variable scope.
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Set a variable in the current scope.
    pub fn set_var(&mut self, key: impl Into<String>, value: impl Into<String>) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(key.into(), value.into());
        } else {
            // Fallback to globals if no scope pushed? Or just error/warn?
            // Proactively pushing a default scope might be safer, but for now set global.
            self.globals.insert(key.into(), value.into());
        }
    }

    /// Resolve a variable value (check scopes top-down, then globals).
    pub fn get_var(&self, key: &str) -> Option<&String> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(key) {
                return Some(val);
            }
        }
        self.globals.get(key)
    }

    /// Set an integer global for fast lookup (no String allocation).
    pub fn set_int_global(&mut self, key: &str, value: usize) {
        self.int_globals.insert(std::sync::Arc::from(key), value);
    }

    /// Get an integer global directly (no String parsing).
    pub fn get_int_global(&self, key: &str) -> Option<usize> {
        self.int_globals.get(key).copied()
    }

    /// Interpolate a string using current variables (e.g. "blk.{i}.weight" -> "blk.0.weight").
    pub fn interpolate(&self, mut s: String) -> String {
        // Simple scan for {key}
        while let Some(start) = s.find('{') {
            if let Some(end) = s[start..].find('}') {
                let end = start + end; // absolute index
                let key = &s[start + 1..end];
                if let Some(val) = self.get_var(key) {
                    s.replace_range(start..=end, val);
                } else {
                    // Unknown variable, leave it? Or error?
                    // For now, break to avoid infinite loop if user actually wanted braces
                    break;
                }
            } else {
                break;
            }
        }
        s
    }

    /// Get a tensor by ref name (with interpolation).
    pub fn get(&self, name: &str) -> Result<TensorArg, MetalError> {
        // 1. Interpolate name
        let resolved_name = self.interpolate(name.to_string());

        // 2. Lookup
        self.bindings
            .get(&resolved_name)
            .cloned()
            .ok_or_else(|| MetalError::InvalidShape(format!("Tensor binding '{}' (resolved from '{}') not found", resolved_name, name)))
    }

    /// Get a tensor by Ref.
    pub fn resolve(&self, r: &Ref) -> Result<TensorArg, MetalError> {
        self.get(&r.0)
    }

    /// Number of bindings.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Check if bindings is empty.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Check if a binding exists.
    pub fn contains(&self, name: &str) -> bool {
        let resolved_name = self.interpolate(name.to_string());
        self.bindings.contains_key(&resolved_name)
    }

    /// Iterate over all bindings.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorArg)> {
        self.bindings.iter()
    }
}

/// A step in a model execution plan.
///
/// This trait is auto-implemented by `#[derive(Kernel)]` when `step = true`.
/// The macro generates a `{Kernel}Step` struct with string refs that deserializes
/// from JSON and resolves to actual TensorArgs at execute time.
///
/// # Auto-Generation
///
/// For a kernel like:
/// ```text
/// #[derive(Kernel, KernelArgs)]
/// #[kernel(step = true, ...)]
/// pub struct RmsNorm {
///     pub input: TensorArg,
///     pub gamma: TensorArg,
///     pub output: TensorArg,
/// }
/// ```
///
/// The macro generates:
/// ```text
/// #[derive(Serialize, Deserialize)]
/// pub struct RmsNormStep {
///     pub input: Ref,
///     pub gamma: Ref,
///     pub output: Ref,
/// }
///
/// #[typetag::serde(name = "RmsNorm")]
/// impl Step for RmsNormStep {
///     fn execute(&self, f: &mut Foundry, bindings: &TensorBindings) -> Result<(), MetalError> {
///         let kernel = RmsNorm {
///             input: bindings.resolve(&self.input)?,
///             gamma: bindings.resolve(&self.gamma)?,
///             output: bindings.resolve(&self.output)?,
///             ..Default::default()
///         };
///         f.run(&kernel)
///     }
/// }
/// ```
#[typetag::serde(tag = "op")]
pub trait Step: Send + Sync + std::fmt::Debug {
    /// Execute this step, resolving refs from bindings.
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError>;

    /// Compile this step into an optimized executable form.
    fn compile(&self, _resolver: &mut TensorBindings, _symbols: &mut super::SymbolTable) -> Vec<Box<dyn super::CompiledStep>> {
        unimplemented!("compile not implemented for {}", self.name())
    }

    /// Returns a human-readable name for this step.
    fn name(&self) -> &'static str;
}
