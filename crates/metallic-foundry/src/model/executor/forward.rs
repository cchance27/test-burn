use metallic_env::{FoundryEnvVar, is_set};

use super::*;

impl CompiledModel {
    /// Run a single forward step by executing all DSL steps.
    ///
    /// Each step in `spec.architecture.forward` is executed via `Step::execute()`.
    /// Run a single forward step by executing all compiled steps.
    pub fn forward(&self, foundry: &mut Foundry, bindings: &mut TensorBindings, fast_bindings: &FastBindings) -> Result<(), MetalError> {
        //eprintln!(
        //    "Forward: m={}, seq_len={}, pos={}, kv_seq_len={} | StrPos={}",
        //    bindings.get_int_global("m").unwrap_or(0),
        //    bindings.get_int_global("seq_len").unwrap_or(0),
        //    bindings.get_int_global("position_offset").unwrap_or(0),
        //    bindings.get_int_global("kv_seq_len").unwrap_or(0),
        //    bindings.get_var("position_offset").map(|s| s.as_str()).unwrap_or("MISSING")
        //);
        // If we are already capturing (e.g. batched prompt processing), don't start a new capture.
        let nested_capture = foundry.is_capturing();
        let profiling_per_kernel = crate::instrument::foundry_per_kernel_profiling_enabled();
        let debug_step_log = is_set(FoundryEnvVar::DebugStepLog);
        let debug_step_sync = is_set(FoundryEnvVar::DebugCompiledStepSync);

        // Always start capture, even in profiling mode (dispatch_pipeline handles per-kernel sync)
        if !nested_capture {
            foundry.start_capture()?;
        }

        for (idx, step) in self.compiled_steps.iter().enumerate() {
            if debug_step_log {
                tracing::info!("Forward compiled step {:03}: {}", idx, step.name());
            }
            step.execute(foundry, fast_bindings, bindings, &self.symbol_table)?;
            if debug_step_sync {
                // Flush after each compiled step to isolate GPU hangs.
                // This is intentionally expensive and intended only for debugging.
                foundry.restart_capture_sync()?;
            }
        }

        // End capture only if we're not profiling per-kernel (profiling mode syncs per dispatch)
        if !nested_capture && !profiling_per_kernel {
            foundry.end_capture()?;
        }

        Ok(())
    }

    /// Run a single forward step by executing DSL steps (uncompiled/interpreted).
    ///
    /// Unlike `forward()` which uses pre-compiled steps, this method executes the
    /// original `Step::execute()` method on each step in `spec.architecture.forward`.
    /// This allows runtime modification of variables like `n_layers` via `bindings.set_global()`.
    pub fn forward_uncompiled(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        // Start a new command buffer for this forward pass (token)
        foundry.start_capture()?;

        for step in &self.spec.architecture.forward {
            if step.name() == "Sample" {
                continue;
            }
            step.execute(foundry, bindings)?;
        }

        // Commit and wait for the token to complete
        let cmd_buffer = foundry.end_capture()?;
        cmd_buffer.wait_until_completed();

        Ok(())
    }
}
