use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::{
    error::MetalError, foundry::{
        Foundry, spec::step::{Step, TensorBindings}
    }, types::KernelArg
};

const METALLIC_FOUNDRY_TRACE_ENV: &str = "METALLIC_FOUNDRY_TRACE";

fn foundry_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(METALLIC_FOUNDRY_TRACE_ENV)
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(false)
    })
}

/// A step that repeats a sequence of sub-steps.
///
/// Useful for repeating layers in transformer blocks.
#[derive(Debug, Serialize, Deserialize)]
pub struct Repeat {
    /// Variable name or integer literal for iteration count.
    /// e.g. "n_layers" or "24"
    pub count: String,

    /// Variable name to bind the current index to.
    /// e.g. "i" -> becomes "0", "1", ...
    pub var: String,

    /// Sequence of steps to repeat.
    pub steps: Vec<Box<dyn Step>>,
}

#[typetag::serde(name = "Repeat")]
impl Step for Repeat {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let count_val = if let Ok(val) = self.count.parse::<usize>() {
            val
        } else {
            // Lookup variable
            bindings
                .get_var(&self.count)
                .ok_or_else(|| MetalError::InvalidOperation(format!("Repeat count variable '{}' not found", self.count)))?
                .parse::<usize>()
                .map_err(|e| {
                    MetalError::InvalidOperation(format!("Repeat count variable '{}' is not a valid integer: {}", self.count, e))
                })?
        };

        bindings.push_scope();

        for i in 0..count_val {
            bindings.set_var(&self.var, i.to_string());

            for (step_idx, step) in self.steps.iter().enumerate() {
                // Debug logging for first layer only
                if i == 0 && foundry_trace_enabled() {
                    eprintln!("    [Layer 0, SubStep {}] {}", step_idx, step.name());
                }
                step.execute(foundry, bindings)?;

                // Dump key tensors after specific sub-steps in layer 0
                if i == 0 && foundry_trace_enabled() {
                    let tensor_name = match step_idx {
                        0 => Some("norm_out"),
                        1 => Some("q"),
                        4 => Some("q_heads"),
                        6 => Some("v_heads"),
                        7 => Some("q_rot"),
                        12 => Some("v_expanded"),
                        13 => Some("attn_out"),
                        14 => Some("hidden"),
                        _ => None,
                    };

                    if let Some(name) = tensor_name {
                        if let Ok(arg) = bindings.get(name) {
                            let len = arg.dims().iter().product::<usize>().min(5);
                            if len > 0 {
                                let data: Vec<half::f16> = unsafe {
                                    use objc2_metal::MTLBuffer;
                                    let ptr = arg.buffer().contents().as_ptr() as *const half::f16;
                                    std::slice::from_raw_parts(ptr, len).to_vec()
                                };
                                eprintln!("      → {} first {}: {:?}", name, len, data);
                            }
                        } else {
                            eprintln!("      → {} not found in bindings", name);
                        }
                    }
                }
            }
        }

        bindings.pop_scope();

        Ok(())
    }

    fn compile(
        &self,
        resolver: &mut TensorBindings,
        symbols: &mut crate::foundry::spec::SymbolTable,
    ) -> Vec<Box<dyn crate::foundry::spec::CompiledStep>> {
        let count_val = if let Ok(val) = self.count.parse::<usize>() {
            val
        } else {
            // Lookup variable
            resolver
                .get_var(&self.count)
                .expect(&format!("Repeat count variable '{}' not found during compilation", self.count))
                .parse::<usize>()
                .expect(&format!("Repeat count variable '{}' is not a valid integer", self.count))
        };

        let mut compiled = Vec::new();

        // Scope management for compilation (simulated scope)
        resolver.push_scope();

        for i in 0..count_val {
            resolver.set_var(&self.var, i.to_string());
            for step in &self.steps {
                let substeps = step.compile(resolver, symbols);
                compiled.extend(substeps);
            }
        }

        resolver.pop_scope();
        compiled
    }

    fn name(&self) -> &'static str {
        "Repeat"
    }
}
