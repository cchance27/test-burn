use objc2_metal::{MTLCompileOptions, MTLLanguageVersion};

use super::*;
use crate::{Dtype, TensorElement};

/// A trait for kernel operations that can be invoked via `Context::call`.
pub trait CustomKernelInvocable {
    type Args<'a, T: TensorElement>;
    type OutputTensor<U: TensorElement>: TensorElement + 'static;

    fn function_id() -> Option<KernelFunction>;

    #[allow(clippy::new_ret_no_self)]
    fn new<'a, T: TensorElement, U: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<Self::OutputTensor<U>>), MetalError>;
}

/// Helper trait for the common case where output tensor has same type as context
pub trait DefaultKernelInvocable {
    type Args<'a, T: TensorElement>;

    fn function_id() -> Option<KernelFunction>;

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError>;
}

/// Manages the compilation and caching of Metal kernel libraries and functions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct KernelPipelineKey {
    function: KernelFunction,
    dtype: Dtype,
}

#[derive(Default)]
pub struct KernelManager {
    libraries: FxHashMap<KernelLibrary, Retained<ProtocolObject<dyn MTLLibrary>>>,
    pipelines: FxHashMap<KernelPipelineKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl KernelManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_pipeline(
        &mut self,
        func: KernelFunction,
        dtype: Dtype,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        let key = KernelPipelineKey { function: func, dtype };

        if let Some(pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline.clone());
        }

        let lib_id = func.library();
        let library = if let Some(lib) = self.libraries.get(&lib_id) {
            lib.clone()
        } else {
            let source = lib_id.source();
            let source_ns = NSString::from_str(source);
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version4_0);
            let lib = device
                .newLibraryWithSource_options_error(&source_ns, Some(&options))
                .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;
            self.libraries.insert(lib_id, lib.clone());
            lib
        };

        let fn_name = NSString::from_str(func.name_for_dtype(dtype)?);
        let metal_fn = library
            .newFunctionWithName(&fn_name)
            .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
        let pipeline = device.newComputePipelineStateWithFunction_error(&metal_fn).unwrap();

        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}
