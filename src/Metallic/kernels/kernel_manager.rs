use super::*;

/// A trait for kernel operations that can be invoked via `Context::call`.
pub trait KernelInvocable {
    type Args<'a>;

    fn function_id() -> Option<KernelFunction>;

    #[allow(clippy::new_ret_no_self)]
    fn new<'a>(
        ctx: &mut Context,
        args: Self::Args<'a>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError>;
}

/// Manages the compilation and caching of Metal kernel libraries and functions.
#[derive(Default)]
pub struct KernelManager {
    libraries: FxHashMap<KernelLibrary, Retained<ProtocolObject<dyn MTLLibrary>>>,
    pipelines: FxHashMap<KernelFunction, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl KernelManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_pipeline(
        &mut self,
        func: KernelFunction,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        if let Some(pipeline) = self.pipelines.get(&func) {
            return Ok(pipeline.clone());
        }

        let lib_id = func.library();
        let library = if let Some(lib) = self.libraries.get(&lib_id) {
            lib.clone()
        } else {
            let source = lib_id.source();
            let source_ns = NSString::from_str(source);
            let lib = device
                .newLibraryWithSource_options_error(&source_ns, None)
                .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;
            self.libraries.insert(lib_id, lib.clone());
            lib
        };

        let fn_name = NSString::from_str(func.name());
        let metal_fn = library
            .newFunctionWithName(&fn_name)
            .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&metal_fn)
            .map_err(|_err| MetalError::PipelineCreationFailed)?;

        self.pipelines.insert(func, pipeline.clone());
        Ok(pipeline)
    }
}
