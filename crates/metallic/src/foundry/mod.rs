use std::{
    any::{Any, TypeId}, path::PathBuf
};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice as _, MTLLibrary};
use rustc_hash::FxHashMap;

use crate::{
    error::MetalError, tensor::Dtype, types::{DispatchConfig, MetalDevice, MetalQueue}
};

pub mod pool;
pub mod storage;
pub mod tensor;

/// The central hub for Metal operations, managing the device, queue, and resources.
pub struct Foundry {
    pub device: MetalDevice,
    pub queue: MetalQueue,
    /// Type-safe registry for resources (caches, pools, etc.)
    resources: FxHashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Cache for compiled pipelines
    pipelines: FxHashMap<TypeId, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

/// Metal kernel source (file path or raw string).
pub enum KernelSource {
    /// A relative path to a .metal file in `src/metals/`.
    File(&'static str),
    /// Raw Metal source code string.
    String(String),
}

impl Foundry {
    /// Create a new Foundry with the system default device.
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceNotFound)?;
        let queue = device.newCommandQueue().ok_or(MetalError::CommandQueueCreationFailed)?;

        let device = crate::types::MetalDevice(device);
        let queue = crate::types::MetalQueue(queue);

        let mut resources = FxHashMap::default();
        let pool = pool::MemoryPool::new(device.clone(), queue.clone())?;
        resources.insert(TypeId::of::<pool::MemoryPool>(), Box::new(pool) as Box<dyn Any + Send + Sync>);

        Ok(Self {
            device,
            queue,
            resources,
            pipelines: FxHashMap::default(),
        })
    }

    /// Create a new Foundry with an existing device and queue.
    pub fn new_with(device: MetalDevice, queue: MetalQueue) -> Self {
        let mut resources = FxHashMap::default();
        let pool = pool::MemoryPool::new(device.clone(), queue.clone()).expect("Failed to create MemoryPool");
        resources.insert(TypeId::of::<pool::MemoryPool>(), Box::new(pool) as Box<dyn Any + Send + Sync>);

        Self {
            device,
            queue,
            resources,
            pipelines: FxHashMap::default(),
        }
    }

    /// Retrieve a mutable reference to a registered resource.
    pub fn get_resource<R: 'static>(&mut self) -> Option<&mut R> {
        self.resources.get_mut(&TypeId::of::<R>()).and_then(|any| any.downcast_mut::<R>())
    }

    /// Register a new resource.
    pub fn register_resource<R: 'static + Send + Sync>(&mut self, resource: R) {
        self.resources.insert(TypeId::of::<R>(), Box::new(resource));
    }

    /// Loads or retrieves a compute pipeline for the given Kernel type.
    pub fn load_kernel<K: Kernel>(&mut self, kernel: &K) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        use objc2_foundation::NSString;

        if let Some(pipeline) = self.pipelines.get(&std::any::TypeId::of::<K::Id>()) {
            return Ok(pipeline.clone());
        }

        let function_name = kernel.function_name();

        // 1. Try Default Library (loading from Bundle/built output)
        if let Some(lib) = self.device.newDefaultLibrary() {
            let ns_name = NSString::from_str(function_name);
            if let Some(func) = lib.newFunctionWithName(&ns_name) {
                let pipeline = self
                    .device
                    .newComputePipelineStateWithFunction_error(&func)
                    .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;
                self.pipelines.insert(TypeId::of::<K::Id>(), pipeline.clone());
                return Ok(pipeline);
            }
        }

        // 2. Fallback to Source
        // Helper to find include file
        let find_include = |name: &str, relative_to: Option<&PathBuf>| -> Option<PathBuf> {
            let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/metals");
            let p = base_path.join(name);
            if p.exists() {
                return Some(p);
            }
            if let Some(parent) = relative_to {
                let p = parent.join(name);
                if p.exists() {
                    return Some(p);
                }
            }
            None
        };

        let mut full_source = String::new();
        let Includes(mut includes) = kernel.includes();

        // Always include policy base if not explicitly present
        if !includes.iter().any(|&i| i == "policies/base.metal") {
            includes.insert(0, "policies/base.metal");
        }

        // Automatic policy inclusion based on Dtype
        if let Some(dtype) = kernel.dtype() {
            let policy_h = match dtype {
                Dtype::F16 => Some("policies/policy_f16.metal"),
                Dtype::U8 => Some("policies/policy_q8.metal"),
                _ => None,
            };
            if let Some(h) = policy_h {
                if !includes.iter().any(|&i| i == h) {
                    includes.push(h);
                }
            }
        }

        // Prepend struct definitions
        let struct_defs = kernel.struct_defs();
        if !struct_defs.is_empty() {
            full_source.push_str("// Auto-generated struct definitions\n");
            full_source.push_str(&struct_defs);
            full_source.push_str("\n\n");
        }

        // Define source_path only for File variant context
        let mut source_path_ctx: Option<PathBuf> = None;

        // Identify main source content
        let main_content = match kernel.source() {
            KernelSource::File(filename) => {
                let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/metals");
                let p = base_path.join(filename);
                if !p.exists() {
                    return Err(MetalError::LoadLibraryFailed(format!("Kernel source {} not found", p.display())));
                }
                source_path_ctx = p.parent().map(|p| p.to_path_buf());
                std::fs::read_to_string(&p).map_err(|e| MetalError::LoadLibraryFailed(format!("Failed to read {}: {}", p.display(), e)))?
            }
            KernelSource::String(s) => s,
        };

        // Load includes
        for include in includes {
            let p = find_include(include, source_path_ctx.as_ref())
                .ok_or_else(|| MetalError::LoadLibraryFailed(format!("Include file {} not found", include)))?;
            let content = std::fs::read_to_string(&p)
                .map_err(|e| MetalError::LoadLibraryFailed(format!("Failed to read include {}: {}", p.display(), e)))?;

            // Strip local includes (#include "...") to prevent recursive resolution errors
            // since we are manually concatenating the dependent files.
            for line in content.lines() {
                if line.trim().starts_with("#include \"") {
                    full_source.push_str(&format!("// Skipped: {}\n", line));
                } else {
                    full_source.push_str(line);
                    full_source.push('\n');
                }
            }
            full_source.push('\n');
        }

        // Also strip includes from main content (if it's from a file, it might have them).
        for line in main_content.lines() {
            if line.trim().starts_with("#include \"") {
                full_source.push_str(&format!("// Skipped: {}\n", line));
            } else {
                full_source.push_str(line);
                full_source.push('\n');
            }
        }

        let _library_options = None::<&objc2_metal::MTLCompileOptions>;

        let options = objc2_metal::MTLCompileOptions::new();
        options.setLanguageVersion(objc2_metal::MTLLanguageVersion::Version2_4); // Or 3.0?
        options.setEnableLogging(true);
        // If we had library options, we'd apply them here

        let ns_source = NSString::from_str(&full_source);
        let library = self
            .device
            .newLibraryWithSource_options_error(&ns_source, Some(&options))
            .map_err(|e| MetalError::LoadLibraryFailed(format!("{:?}", e)))?;

        let ns_name = NSString::from_str(function_name);
        let function = library
            .newFunctionWithName(&ns_name)
            .ok_or_else(|| MetalError::LoadLibraryFailed(format!("Function {} not found in source", function_name)))?;

        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;

        self.pipelines.insert(std::any::TypeId::of::<K::Id>(), pipeline.clone());
        Ok(pipeline)
    }

    /// Dispatches a kernel.
    ///
    /// This handles:
    /// 1. Getting the pipeline (compiling if needed).
    /// 2. Creating an encoder.
    /// 3. Binding arguments via `K::bind`.
    /// 4. Calculating grid size.
    /// 5. Committing the command.
    pub fn dispatch<K: Kernel>(&mut self, kernel: &K, config: DispatchConfig) -> Result<(), MetalError> {
        use objc2_metal::{MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _, MTLComputeCommandEncoder as _, MTLSize};

        let pipeline = self.load_kernel(kernel)?;

        let command_buffer = self.queue.commandBuffer().ok_or(MetalError::CommandBufferCreationFailed)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::CommandQueueCreationFailed)?;

        encoder.setComputePipelineState(&pipeline);

        let encoder_wrapper = crate::types::ComputeCommandEncoder(encoder.clone());
        kernel.bind(&encoder_wrapper);

        let (grid_size, group_size): (MTLSize, MTLSize) = config.into();
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, group_size);
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(())
    }

    /// Simplified dispatch using the kernel's built-in dispatch configuration.
    /// Requires the kernel to implement `dispatch_config()`.
    ///
    /// Automatically applies policy wrapping if the kernel implements `WithPolicy`.
    pub fn run<K: Kernel>(&mut self, kernel: &K) -> Result<(), MetalError> {
        let config = kernel.dispatch_config();
        self.dispatch(kernel, config)
    }

    /// Dispatch a kernel with a specific policy applied via PolicyStage.
    pub fn run_with_policy<P, K>(&mut self, kernel: &K) -> Result<(), MetalError>
    where
        P: crate::fusion::MetalPolicy + Default + 'static,
        K: Kernel + Clone + Send + Sync + 'static, // Needs Clone+Send+Sync for Stage wrapping
    {
        use crate::compound::{CompoundKernel, PolicyStage};

        let fused_kernel = CompoundKernel::new(kernel.function_name())
            .prologue(PolicyStage::<P>::new())
            .main_dyn(kernel.as_stage())
            .with_manual_output(true) // Helper manages output manually
            .build();

        // 2. Bind inputs
        let pipeline = self.load_kernel(&fused_kernel)?;

        // Dispatch with manual binding
        use objc2_metal::{MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _, MTLComputeCommandEncoder as _};
        let command_buffer = self.queue.commandBuffer().ok_or(MetalError::CommandBufferCreationFailed)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::CommandQueueCreationFailed)?;

        encoder.setComputePipelineState(&pipeline);

        // Bind arguments using the ORIGINAL kernel
        let encoder_wrapper = crate::types::ComputeCommandEncoder(encoder.clone());
        kernel.bind(&encoder_wrapper);

        let config = kernel.dispatch_config();
        let (grid_size, group_size): (objc2_metal::MTLSize, objc2_metal::MTLSize) = config.into();

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, group_size);
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(())
    }
}

pub struct Includes(pub Vec<&'static str>);

pub trait Kernel {
    type Args;
    /// A static marker type used for pipeline caching.
    /// This allows kernels to have lifetimes while the cache key remains static.
    type Id: 'static;

    fn function_name(&self) -> &'static str;
    fn source(&self) -> KernelSource;
    fn includes(&self) -> Includes {
        Includes(vec![])
    }
    /// Returns the primary data type for this kernel, used for automatic policy inclusion.
    fn dtype(&self) -> Option<Dtype> {
        None
    }
    /// Binds the kernel arguments to the encoder.
    /// This is auto-generated by the `KernelArgs` derive macro.
    fn bind(&self, encoder: &crate::types::ComputeCommandEncoder);

    /// Returns the dispatch configuration (grid_size, group_size) for this kernel.
    /// Override this to enable the simplified `Foundry::run()` API.
    /// Default implementation panics - kernels must opt-in by implementing this.
    fn dispatch_config(&self) -> DispatchConfig {
        panic!(
            "dispatch_config() not implemented for this kernel. Use dispatch() with explicit grid/group sizes, or implement dispatch_config()."
        )
    }

    /// Returns the compiled metallib bytes if available (via `built_kernels` feature).
    fn metallib_bytes(&self) -> Option<&'static [u8]> {
        None
    }

    /// Returns Metal struct definitions to prepend to the kernel source.
    /// Used to inline structs like `GemvParams` that implement `MetalStruct`.
    fn struct_defs(&self) -> String {
        String::new()
    }

    /// Convert this kernel into a Stage for use in a CompoundKernel.
    /// This enables `foundry.run_with_policy(kernel)`.
    ///
    /// Default implementation wraps the kernel in a GenericKernelStage.
    /// Specialized kernels (like Gemv) should override this to return a proper Stage (e.g. GemvCoreStage).
    fn as_stage(&self) -> Box<dyn crate::compound::Stage>
    where
        Self: Sized + Clone + Send + Sync + 'static,
    {
        Box::new(crate::compound::GenericKernelStage::new(self.clone()))
    }
}
