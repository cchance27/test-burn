use crate::{
    compound::{BufferArg, Stage}, foundry::{Includes, Kernel}
};

/// Adapts any `Kernel` to be used as a `Stage` in a `CompoundKernel`.
///
/// This allows "bare" kernels (like `Gemv`) to be composed with policies
/// without needing internal modification.
pub struct GenericKernelStage<K> {
    kernel: K,
}

impl<K: Kernel> GenericKernelStage<K> {
    pub fn new(kernel: K) -> Self {
        Self { kernel }
    }
}

impl<K: Kernel + Send + Sync> Stage for GenericKernelStage<K> {
    fn includes(&self) -> Vec<&'static str> {
        let Includes(incs) = self.kernel.includes();
        // Kernel::includes returns ownership, so we leak strings if they aren't static?
        // Wait, Kernel::includes returns Vec<&'static str>. We are safe.
        incs
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![] // Placeholder - this won't work for generic kernels yet
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        let name = self.kernel.function_name();
        // A generic kernel call inside a fused kernel is tricky.
        // We typically "inline" the body.
        // If `source()` returns a file, we rely on helper function call.
        //
        // Return: (output_var, code)
        ("output".to_string(), format!("{}(); // Generic call not fully implemented", name))
    }

    fn struct_defs(&self) -> String {
        self.kernel.struct_defs().to_string()
    }
}
