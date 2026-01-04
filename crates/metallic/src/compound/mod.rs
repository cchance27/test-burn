//! Compound Kernel System
//! Enables composing Metal helper functions into fused kernels.
//!
//! # Example
//! ```ignore
//! use metallic::compound::{CompoundKernel, GemvCoreStage, PolicyStage};
//! use metallic::policies::PolicyQ8;
//!
//! let kernel = CompoundKernel::new("fused_q8_gemv")
//!     .prologue(PolicyStage::<PolicyQ8>::new())
//!     .main(GemvCoreStage::new())
//!     .build();
//!
//! foundry.run(&kernel)?;
//! ```

mod code_builder;
mod stages;

pub use code_builder::CodeBuilder;
pub use stages::*;

use crate::{
    foundry::{Includes, Kernel, KernelSource}, fusion::THREAD_ARGS, types::{ComputeCommandEncoder, DispatchConfig, GridSize, ThreadgroupSize}
};

/// A buffer argument contributed by a stage.
#[derive(Clone, Debug)]
pub struct BufferArg {
    /// Argument name in Metal signature.
    pub name: &'static str,
    /// Metal type string (e.g., "const device half*").
    pub metal_type: &'static str,
    /// Buffer index in kernel signature.
    pub buffer_index: u32,
}

/// Trait for types that can bind themselves to a Metal encoder.
/// This is implemented by `#[derive(KernelArgs)]` types.
pub trait BindArgs {
    fn bind_args(&self, encoder: &ComputeCommandEncoder);
}

/// A stage in a compound kernel.
///
/// Each stage wraps a hand-written Metal helper function and provides:
/// - Headers to include
/// - Buffer arguments for the kernel signature
/// - Metal code to emit (calling the helper function)
pub trait Stage: Send + Sync {
    /// Metal headers required by this stage.
    fn includes(&self) -> Vec<&'static str>;

    /// Buffer arguments this stage contributes to the kernel signature.
    fn buffer_args(&self) -> Vec<BufferArg>;

    /// Generate Metal code for this stage.
    ///
    /// # Arguments
    /// * `input_var` - Variable name from the previous stage (or "input" for first stage)
    ///
    /// # Returns
    /// Tuple of (output_variable_name, metal_code_string)
    fn emit(&self, input_var: &str) -> (String, String);

    /// Generate Metal code for SIMD GEMV prologue (threadgroup setup).
    /// Used when this Stage is fused into a GEMV kernel.
    fn emit_simd_prologue(&self) -> Option<String> {
        None
    }

    /// Generate Metal code for SIMD GEMV reduction loop (shuffle/xor).
    /// Used when this Stage is fused into a GEMV kernel.
    fn emit_simd_reduce(&self) -> Option<String> {
        None
    }

    /// Generate C-struct definitions required by this stage.
    /// Default implementation returns empty string.
    fn struct_defs(&self) -> String {
        String::new()
    }
}

/// Compile-time metadata for stages (not dyn-compatible).
/// Used by #[derive(CompoundKernel)] macro.
pub trait StageMeta {
    /// Number of buffer arguments this stage contributes.
    const BUFFER_ARG_COUNT: usize;

    /// Buffer argument metadata for macro code generation.
    /// Each entry: (name, metal_type, rust_type, is_output)
    const BUFFER_ARG_META: &'static [(&'static str, &'static str, &'static str, bool)];
}

impl<S: Stage + ?Sized> Stage for Box<S> {
    fn includes(&self) -> Vec<&'static str> {
        (**self).includes()
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        (**self).buffer_args()
    }

    fn emit(&self, input_var: &str) -> (String, String) {
        (**self).emit(input_var)
    }

    fn struct_defs(&self) -> String {
        (**self).struct_defs()
    }
}

/// Typestate: Kernel is being built, not yet compiled.
pub struct Unfused;

/// Typestate: Kernel is compiled and ready to dispatch.
pub struct Fused;

/// Typestate: Kernel is bound with runtime arguments.
pub struct Bound<A: BindArgs> {
    args: A,
    dispatch: DispatchConfig,
}

/// A compound kernel composed of multiple stages.
///
/// Uses typestate pattern to ensure kernels are built before dispatch.
pub struct CompoundKernel<State = Unfused> {
    /// Kernel function name.
    name: String,
    /// Prologue stages (input loading, dequant). Can have multiple for multi-input ops.
    prologues: Vec<Box<dyn Stage>>,
    /// Main computation stage (e.g., GEMV core).
    main: Option<Box<dyn Stage>>,
    /// Epilogue stages (activations, norms).
    epilogues: Vec<Box<dyn Stage>>,

    // --- Fused state only ---
    /// Generated Metal source code.
    source: Option<String>,

    /// Typestate with embedded data (e.g., Bound<A> holds args and dispatch config).
    state: State,

    /// Disable automatic output assignment (output[idx] = var).
    /// Used when a stage handles writing to output manually (e.g. Gemv).
    pub manual_output: bool,
}

/// A compiled compound kernel template that no longer retains stage objects.
///
/// This is the recommended representation for hot-path usage (e.g. per-token decode),
/// since it avoids rebuilding source strings and avoids owning non-cloneable stage graphs.
pub struct CompiledCompoundKernel {
    fn_name: &'static str,
    includes: Vec<&'static str>,
    struct_defs: String,
    source: String,
}

/// A bound (dispatchable) compiled compound kernel.
pub struct BoundCompiledCompoundKernel<A: BindArgs> {
    template: &'static CompiledCompoundKernel,
    args: A,
    dispatch: DispatchConfig,
}

impl CompoundKernel<Unfused> {
    /// Create a new compound kernel builder.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            prologues: Vec::new(),
            main: None,
            epilogues: Vec::new(),
            source: None,
            state: Unfused,
            manual_output: false,
        }
    }

    /// Enable manual output mode (disables default assignment).
    pub fn with_manual_output(mut self, manual: bool) -> Self {
        self.manual_output = manual;
        self
    }

    /// Add a prologue stage (input loading, dequant).
    ///
    /// Can be called multiple times for multi-input operations (QKV, attention).
    pub fn prologue<S: Stage + 'static>(mut self, stage: S) -> Self {
        self.prologues.push(Box::new(stage));
        self
    }

    /// Set the main computation stage.
    pub fn main<S: Stage + 'static>(mut self, stage: S) -> Self {
        self.main = Some(Box::new(stage));
        self
    }

    /// Add an epilogue stage (activation, normalization).
    ///
    /// Can be called multiple times to chain post-processing.
    pub fn epilogue<S: Stage + 'static>(mut self, stage: S) -> Self {
        self.epilogues.push(Box::new(stage));
        self
    }

    // --- Dynamic variants for derive macro use ---

    /// Add a boxed prologue stage (for macro-generated code).
    pub fn prologue_dyn(mut self, stage: Box<dyn Stage>) -> Self {
        self.prologues.push(stage);
        self
    }

    /// Set a boxed main stage (for macro-generated code).
    pub fn main_dyn(mut self, stage: Box<dyn Stage>) -> Self {
        self.main = Some(stage);
        self
    }

    /// Add a boxed epilogue stage (for macro-generated code).
    pub fn epilogue_dyn(mut self, stage: Box<dyn Stage>) -> Self {
        self.epilogues.push(stage);
        self
    }

    /// Compile the compound kernel into a fused, dispatchable kernel.
    pub fn build(self) -> CompoundKernel<Fused> {
        let source = self.generate_source();

        CompoundKernel {
            name: self.name,
            prologues: self.prologues,
            main: self.main,
            epilogues: self.epilogues,
            source: Some(source),
            state: Fused,
            manual_output: self.manual_output,
        }
    }

    /// Compile into a reusable template that can be bound many times without rebuilding.
    pub fn compile(self) -> CompiledCompoundKernel {
        let source = self.generate_source();
        let includes = self.collect_includes();
        let struct_defs = format!("#define FUSED_KERNEL 1\n\n{}", self.collect_struct_defs());
        let fn_name = Box::leak(self.name.into_boxed_str());
        CompiledCompoundKernel {
            fn_name,
            includes,
            struct_defs,
            source,
        }
    }

    fn generate_source(&self) -> String {
        let mut code = String::new();

        // 1. Collect all includes
        let all_includes = self.collect_includes();

        for inc in &all_includes {
            code.push_str(&format!("#include \"{}\"\n", inc));
        }
        code.push('\n');

        // 2. Generate signature
        let all_args = self.collect_buffer_args();
        code.push_str(&format!("kernel void {}(\n", self.name));

        for (i, arg) in all_args.iter().enumerate() {
            let is_last = i == all_args.len() - 1;
            // Add comma unless this is the last buffer arg AND there are no thread args
            let comma = if is_last && THREAD_ARGS.is_empty() { "" } else { "," };
            code.push_str(&format!(
                "    {} {} [[buffer({})]]{}\n",
                arg.metal_type, arg.name, arg.buffer_index, comma
            ));
        }
        // Use THREAD_ARGS from fusion module
        for (i, arg) in THREAD_ARGS.iter().enumerate() {
            let comma = if i < THREAD_ARGS.len() - 1 { "," } else { "" };
            code.push_str(&format!("    {}{}\n", arg, comma));
        }
        code.push_str(") {\n");

        code.push_str("    uint idx = gid.x;\n\n");

        // 3. Generate body by chaining stages
        let mut current_var = "input".to_string();

        // Prologues
        for (i, stage) in self.prologues.iter().enumerate() {
            let (out_var, stage_code) = stage.emit(&current_var);
            code.push_str(&format!("    // Prologue {}\n", i));
            code.push_str(&stage_code);
            code.push('\n');
            current_var = out_var;
        }

        // Main
        if let Some(main) = &self.main {
            let (out_var, stage_code) = main.emit(&current_var);
            code.push_str("    // Main computation\n");
            code.push_str(&stage_code);
            code.push('\n');
            current_var = out_var;
        }

        // Epilogues
        for (i, stage) in self.epilogues.iter().enumerate() {
            let (out_var, stage_code) = stage.emit(&current_var);
            code.push_str(&format!("    // Epilogue {}\n", i));
            code.push_str(&stage_code);
            code.push('\n');
            current_var = out_var;
        }

        // Write final output
        if !self.manual_output {
            code.push_str(&format!("    output[idx] = {};\n", current_var));
        }
        code.push_str("}\n");

        code
    }
}

impl<S> CompoundKernel<S> {
    /// Get the kernel function name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Collect includes from all stages.
    fn collect_includes(&self) -> Vec<&'static str> {
        let mut all_includes = Vec::new();
        for stage in &self.prologues {
            all_includes.extend(stage.includes());
        }
        if let Some(main) = &self.main {
            all_includes.extend(main.includes());
        }
        for stage in &self.epilogues {
            all_includes.extend(stage.includes());
        }

        // Dedup preserving order
        let mut seen = std::collections::HashSet::new();
        all_includes.retain(|item| seen.insert(*item));
        all_includes
    }

    /// Collect struct definitions from all stages.
    pub fn collect_struct_defs(&self) -> String {
        let mut all_struct_defs = Vec::new();

        for stage in &self.prologues {
            let def = stage.struct_defs();
            if !def.is_empty() {
                all_struct_defs.push(def);
            }
        }
        if let Some(main) = &self.main {
            let def = main.struct_defs();
            if !def.is_empty() {
                all_struct_defs.push(def);
            }
        }
        for stage in &self.epilogues {
            let def = stage.struct_defs();
            if !def.is_empty() {
                all_struct_defs.push(def);
            }
        }

        // Dedup preserving order
        let mut seen = std::collections::HashSet::new();
        all_struct_defs.retain(|item| seen.insert(item.clone()));
        all_struct_defs.join("\n\n")
    }

    /// Collect all buffer arguments from all stages.
    pub fn collect_buffer_args(&self) -> Vec<BufferArg> {
        let mut args = Vec::new();

        for stage in &self.prologues {
            args.extend(stage.buffer_args());
        }
        if let Some(main) = &self.main {
            args.extend(main.buffer_args());
        }
        for stage in &self.epilogues {
            args.extend(stage.buffer_args());
        }

        // Sort by buffer index
        args.sort_by_key(|a| a.buffer_index);
        args
    }
}

impl CompiledCompoundKernel {
    /// Bind this kernel with runtime arguments for dispatch.
    pub fn bind<A: BindArgs>(self: &'static Self, args: A, dispatch: DispatchConfig) -> BoundCompiledCompoundKernel<A> {
        BoundCompiledCompoundKernel {
            template: self,
            args,
            dispatch,
        }
    }
}

impl Kernel for CompiledCompoundKernel {
    type Id = String;
    type Args = ();

    fn function_name(&self) -> &'static str {
        self.fn_name
    }

    fn source(&self) -> KernelSource {
        KernelSource::String(self.source.clone())
    }

    fn includes(&self) -> Includes {
        Includes(self.includes.clone())
    }

    fn struct_defs(&self) -> String {
        self.struct_defs.clone()
    }

    fn bind(&self, _encoder: &ComputeCommandEncoder) {}
}

impl<A: BindArgs> Kernel for BoundCompiledCompoundKernel<A> {
    type Id = String;
    type Args = A;

    fn function_name(&self) -> &'static str {
        self.template.fn_name
    }

    fn source(&self) -> KernelSource {
        KernelSource::String(self.template.source.clone())
    }

    fn includes(&self) -> Includes {
        Includes(self.template.includes.clone())
    }

    fn struct_defs(&self) -> String {
        self.template.struct_defs.clone()
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.args.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        self.dispatch.clone()
    }
}

impl CompoundKernel<Fused> {
    /// Get the generated Metal source.
    pub fn source_code(&self) -> &str {
        self.source.as_ref().expect("Fused kernel should have source")
    }
}

// --- Kernel Trait Implementation ---

impl Kernel for CompoundKernel<Fused> {
    type Id = String;
    type Args = ();

    fn function_name(&self) -> &'static str {
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn source(&self) -> KernelSource {
        KernelSource::String(self.source.clone().unwrap())
    }

    fn includes(&self) -> Includes {
        Includes(self.collect_includes())
    }

    fn struct_defs(&self) -> String {
        format!("#define FUSED_KERNEL 1\n\n{}", self.collect_struct_defs())
    }

    fn bind(&self, _encoder: &ComputeCommandEncoder) {
        // Note: For unbound compound kernels, binding is a no-op.
        // Use BoundCompoundKernel for actual dispatch with buffer binding.
    }

    fn dispatch_config(&self) -> DispatchConfig {
        // Default dispatch - should be overridden by BoundCompoundKernel
        DispatchConfig {
            grid: GridSize::d1(1),
            group: ThreadgroupSize::d1(64),
        }
    }
}

impl<A: BindArgs + 'static> Kernel for CompoundKernel<Bound<A>> {
    type Id = String;
    type Args = A;

    fn function_name(&self) -> &'static str {
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn source(&self) -> KernelSource {
        KernelSource::String(self.source.clone().unwrap())
    }

    fn includes(&self) -> Includes {
        let mut all_includes = Vec::new();
        for stage in &self.prologues {
            all_includes.extend(stage.includes());
        }
        if let Some(main) = &self.main {
            all_includes.extend(main.includes());
        }
        for stage in &self.epilogues {
            all_includes.extend(stage.includes());
        }
        all_includes.sort();
        all_includes.dedup();
        Includes(all_includes)
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.state.args.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        self.state.dispatch.clone()
    }
}

impl CompoundKernel<Fused> {
    /// Bind this kernel with runtime arguments for dispatch.
    pub fn bind<A: BindArgs>(self, args: A, dispatch: DispatchConfig) -> CompoundKernel<Bound<A>> {
        CompoundKernel {
            name: self.name,
            prologues: self.prologues,
            main: self.main,
            epilogues: self.epilogues,
            source: self.source,
            state: Bound { args, dispatch },
            manual_output: self.manual_output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::{EpilogueNone, PolicyQ8};

    #[test]
    fn test_policy_stage_includes() {
        let stage = PolicyStage::<PolicyQ8>::new();
        let includes = stage.includes();
        assert!(includes.contains(&"policies/policy_q8.metal"));
    }

    #[test]
    fn test_policy_stage_emit() {
        let stage = PolicyStage::<PolicyQ8>::new();
        let (out_var, code) = stage.emit("input");
        // PolicyQ8::load_and_dequant returns "w_vec"
        assert_eq!(out_var, "w_vec");
        assert!(code.contains("data loaded via macro") || code.contains("float4"));
    }

    #[test]
    fn test_epilogue_stage_emit() {
        let stage = EpilogueStage::<EpilogueNone>::new();
        let (out_var, code) = stage.emit("prev_result");
        assert_eq!(out_var, "epilogue_out");
        assert!(code.contains("prev_result"));
    }
}
