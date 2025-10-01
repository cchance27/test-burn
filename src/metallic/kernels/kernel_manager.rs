use crate::metallic::{Dtype, MetalError, TensorElement};
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLDataType, MTLFunctionConstantValues};

use super::*;

/// A trait for kernel operations that can be invoked via `Context::call`.
pub trait KernelInvocable {
    type Args<'a, T: TensorElement>;

    fn function_id() -> Option<KernelFunction>;

    #[allow(clippy::new_ret_no_self)]
    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError>;
}

/// Manages the compilation and caching of Metal kernel libraries and functions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct KernelFunctionConstantsKey(Vec<(u16, bool)>);

impl KernelFunctionConstantsKey {
    fn from_bools(constants: &[(u16, bool)]) -> Self {
        let mut values = constants.to_vec();
        values.sort_by_key(|(index, _)| *index);
        Self(values)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct KernelPipelineKey {
    function: KernelFunction,
    dtype: Dtype,
    constants: Option<KernelFunctionConstantsKey>,
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
        self.get_pipeline_with_constants(func, dtype, device, None)
    }

    pub fn get_pipeline_with_constants(
        &mut self,
        func: KernelFunction,
        dtype: Dtype,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        constants: Option<&[(u16, bool)]>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        let constants_key = constants.map(KernelFunctionConstantsKey::from_bools);
        let key = KernelPipelineKey {
            function: func,
            dtype,
            constants: constants_key.clone(),
        };

        if let Some(pipeline) = self.pipelines.get(&key) {
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

        let fn_name = NSString::from_str(func.name_for_dtype(dtype)?);
        let metal_fn = if let Some(constants) = constants {
            let values = MTLFunctionConstantValues::new();
            for (index, value) in constants {
                let mut raw_value = *value;
                let ptr = std::ptr::NonNull::from(&mut raw_value).cast();
                unsafe {
                    values.setConstantValue_type_atIndex(ptr, MTLDataType::Bool, *index as NSUInteger);
                }
            }
            library
                .newFunctionWithName_constantValues_error(&fn_name, &values)
                .map_err(|err| MetalError::FunctionCreationFailed(err.to_string()))?
        } else {
            library
                .newFunctionWithName(&fn_name)
                .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?
        };
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&metal_fn)
            .map_err(|_err| MetalError::PipelineCreationFailed)?;

        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}
