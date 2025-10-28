use objc2_metal::{MTLCompileOptions, MTLLanguageVersion};

use super::*;
use crate::{Dtype, TensorElement};

/// A trait for kernel operations that can be invoked via `Context::call`.
/// Supports both single and multiple tensor outputs of different types.
pub trait CustomKernelInvocable {
    type Args<'a, T: TensorElement>;

    /// A type that represents the output tensor element types as a tuple - e.g., (F32,), (F32, U32), (F32, U32, F16)
    type OutputTuple<T: TensorElement>: MultiTensorOutput<T>;

    fn function_id() -> Option<KernelFunction>;

    #[allow(clippy::new_ret_no_self)]
    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, <Self::OutputTuple<T> as MultiTensorOutput<T>>::Tensors), MetalError>;
}

/// A trait that represents the output of a multi-tensor operation
/// This trait uses associated types to avoid dynamic dispatch
pub trait MultiTensorOutput<T: TensorElement> {
    type Tensors;

    /// Mark the tensors as pending in the context
    fn mark_pending(ctx: &mut Context<T>, tensors: &Self::Tensors);
}

/// Implementation for single tensor output as a tuple with one element
impl<T: TensorElement, T1: TensorElement> MultiTensorOutput<T> for (T1,) {
    type Tensors = (Tensor<T1>,);

    fn mark_pending(ctx: &mut Context<T>, tensors: &Self::Tensors) {
        ctx.mark_tensor_pending(&tensors.0);
    }
}

/// Implementation for 2 tensor outputs
impl<T: TensorElement, T1: TensorElement, T2: TensorElement> MultiTensorOutput<T> for (T1, T2) {
    type Tensors = (Tensor<T1>, Tensor<T2>);

    fn mark_pending(ctx: &mut Context<T>, tensors: &Self::Tensors) {
        ctx.mark_tensor_pending(&tensors.0);
        ctx.mark_tensor_pending(&tensors.1);
    }
}

/// Implementation for 3 tensor outputs
impl<T: TensorElement, T1: TensorElement, T2: TensorElement, T3: TensorElement> MultiTensorOutput<T> for (T1, T2, T3) {
    type Tensors = (Tensor<T1>, Tensor<T2>, Tensor<T3>);

    fn mark_pending(ctx: &mut Context<T>, tensors: &Self::Tensors) {
        ctx.mark_tensor_pending(&tensors.0);
        ctx.mark_tensor_pending(&tensors.1);
        ctx.mark_tensor_pending(&tensors.2);
    }
}

/// Implementation for 4 tensor outputs
impl<T: TensorElement, T1: TensorElement, T2: TensorElement, T3: TensorElement, T4: TensorElement> MultiTensorOutput<T>
    for (T1, T2, T3, T4)
{
    type Tensors = (Tensor<T1>, Tensor<T2>, Tensor<T3>, Tensor<T4>);

    fn mark_pending(ctx: &mut Context<T>, tensors: &Self::Tensors) {
        ctx.mark_tensor_pending(&tensors.0);
        ctx.mark_tensor_pending(&tensors.1);
        ctx.mark_tensor_pending(&tensors.2);
        ctx.mark_tensor_pending(&tensors.3);
    }
}

/// Implementation for 5 tensor outputs
impl<T: TensorElement, T1: TensorElement, T2: TensorElement, T3: TensorElement, T4: TensorElement, T5: TensorElement> MultiTensorOutput<T>
    for (T1, T2, T3, T4, T5)
{
    type Tensors = (Tensor<T1>, Tensor<T2>, Tensor<T3>, Tensor<T4>, Tensor<T5>);

    fn mark_pending(ctx: &mut Context<T>, tensors: &Self::Tensors) {
        ctx.mark_tensor_pending(&tensors.0);
        ctx.mark_tensor_pending(&tensors.1);
        ctx.mark_tensor_pending(&tensors.2);
        ctx.mark_tensor_pending(&tensors.3);
        ctx.mark_tensor_pending(&tensors.4);
    }
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

            #[cfg(debug_assertions)]
            options.setEnableLogging(true);

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
