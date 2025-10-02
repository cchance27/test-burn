use crate::metallic::{Dtype, TensorElement};

use super::*;
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLDataType, MTLFunctionConstantValues};
use std::ffi::c_void;
use std::ptr::NonNull;

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
struct KernelPipelineKey {
    function: KernelFunction,
    dtype: Dtype,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MlxTileShape {
    Tile32x32,
    Tile16x32,
    Tile8x32,
}

impl MlxTileShape {
    pub const fn bm(self) -> usize {
        match self {
            Self::Tile32x32 => 32,
            Self::Tile16x32 => 16,
            Self::Tile8x32 => 8,
        }
    }

    pub const fn bn(self) -> usize {
        32
    }

    pub const fn bk(self) -> usize {
        16
    }

    pub const fn wm(self) -> usize {
        match self {
            Self::Tile32x32 | Self::Tile16x32 => 2,
            Self::Tile8x32 => 1,
        }
    }

    pub const fn wn(self) -> usize {
        2
    }

    pub const fn threadgroup_size(self) -> (usize, usize, usize) {
        (32, self.wm(), self.wn())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MlxPipelineKey {
    pub dtype: Dtype,
    pub transpose_left: bool,
    pub transpose_right: bool,
    pub has_batch: bool,
    pub align_m: bool,
    pub align_n: bool,
    pub align_k: bool,
    pub use_out_source: bool,
    pub do_axpby: bool,
    pub scale_only: bool,
    pub tile_shape: MlxTileShape,
}

#[derive(Default)]
pub struct KernelManager {
    libraries: FxHashMap<KernelLibrary, Retained<ProtocolObject<dyn MTLLibrary>>>,
    pipelines: FxHashMap<KernelPipelineKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    mlx_pipelines: FxHashMap<MlxPipelineKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
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

        let library = self.get_library(func.library(), device)?;

        let fn_name = NSString::from_str(func.name_for_dtype(dtype)?);
        let metal_fn = library
            .newFunctionWithName(&fn_name)
            .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&metal_fn)
            .map_err(|_err| MetalError::PipelineCreationFailed)?;

        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    pub fn get_library(
        &mut self,
        library: KernelLibrary,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
        if let Some(lib) = self.libraries.get(&library) {
            return Ok(lib.clone());
        }

        let source = library.source();
        let source_ns = NSString::from_str(source);
        let lib = device
            .newLibraryWithSource_options_error(&source_ns, None)
            .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;
        self.libraries.insert(library, lib.clone());
        Ok(lib)
    }

    pub fn get_mlx_pipeline(
        &mut self,
        key: MlxPipelineKey,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        if let Some(pipeline) = self.mlx_pipelines.get(&key) {
            return Ok(pipeline.clone());
        }

        let library = self.get_library(KernelLibrary::MatMulMlx, device)?;
        let function_name = mlx_function_name(key.dtype, key.transpose_left, key.transpose_right, key.tile_shape)?;
        let fn_name = NSString::from_str(&function_name);

        let constants = build_mlx_function_constants(key);
        let metal_fn = library
            .newFunctionWithName_constantValues_error(&fn_name, &constants)
            .map_err(|_err| MetalError::FunctionCreationFailed(fn_name.to_string()))?;

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&metal_fn)
            .map_err(|_err| MetalError::PipelineCreationFailed)?;

        self.mlx_pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}

#[cfg(test)]
impl KernelManager {
    pub(crate) fn mlx_pipeline_keys(&self) -> Vec<MlxPipelineKey> {
        self.mlx_pipelines.keys().copied().collect()
    }
}

fn mlx_function_name(dtype: Dtype, transpose_left: bool, transpose_right: bool, tile_shape: MlxTileShape) -> Result<String, MetalError> {
    use Dtype::*;

    let suffix = match tile_shape {
        MlxTileShape::Tile32x32 => "32_32_16_2_2",
        MlxTileShape::Tile16x32 => "16_32_16_2_2",
        MlxTileShape::Tile8x32 => "8_32_16_1_2",
    };

    let prefix = match (dtype, transpose_left, transpose_right) {
        (F32, false, false) => "gemm_nn_f32_f32",
        (F32, true, false) => "gemm_tn_f32_f32",
        (F32, false, true) => "gemm_nt_f32_f32",
        (F32, true, true) => "gemm_tt_f32_f32",
        (F16, false, false) => "gemm_nn_f16_f16",
        (F16, true, false) => "gemm_tn_f16_f16",
        (F16, false, true) => "gemm_nt_f16_f16",
        (F16, true, true) => "gemm_tt_f16_f16",
    };

    Ok(format!("{}_{}", prefix, suffix))
}

fn build_mlx_function_constants(key: MlxPipelineKey) -> Retained<MTLFunctionConstantValues> {
    let constants = MTLFunctionConstantValues::new();

    set_bool_constant(&constants, 10, key.has_batch);
    set_bool_constant(&constants, 100, key.use_out_source);
    set_bool_constant(&constants, 110, key.do_axpby);
    set_bool_constant(&constants, 200, key.align_m);
    set_bool_constant(&constants, 201, key.align_n);
    set_bool_constant(&constants, 202, key.align_k);
    set_bool_constant(&constants, 300, false);
    set_bool_constant(&constants, 310, key.scale_only);

    constants
}

fn set_bool_constant(constants: &MTLFunctionConstantValues, index: NSUInteger, value: bool) {
    let mut raw = if value { 1u8 } else { 0u8 };
    let ptr = NonNull::new((&mut raw as *mut u8).cast::<c_void>()).expect("bool constant pointer must be non-null");
    unsafe {
        constants.setConstantValue_type_atIndex(ptr, MTLDataType::Bool, index);
    }
}
