use std::{
    any::{Any, TypeId}, path::PathBuf
};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice as _, MTLLibrary};
use rustc_hash::FxHashMap;

use crate::{
    error::MetalError, tensor::Dtype, types::{DispatchConfig, MetalDevice, MetalQueue}
};

pub mod constants;
pub mod model;
pub mod pool;
pub mod spec;
pub mod storage;
pub mod tensor;

/// The central hub for Metal operations, managing the device, queue, and resources.
pub struct Foundry {
    pub device: MetalDevice,
    pub queue: MetalQueue,
    /// Type-safe registry for resources (caches, pools, etc.)
    resources: FxHashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Cache for compiled pipelines
    pipelines: FxHashMap<(TypeId, u64), Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// Active command buffer for batched dispatch
    active_command_buffer: Option<Retained<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>>,
    /// Helper: Active compute encoder to reuse across dispatches
    active_compute_encoder: Option<Retained<ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>>>,
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
            active_command_buffer: None,
            active_compute_encoder: None,
        })
    }

    /// Create a new Foundry with an existing device and queue.
    pub fn new_with(device: MetalDevice, queue: MetalQueue) -> Self {
        let mut resources = FxHashMap::default();
        let pool = pool::MemoryPool::new(device.clone(), queue.clone()).expect("Failed to create memory pool");
        resources.insert(TypeId::of::<pool::MemoryPool>(), Box::new(pool) as Box<dyn Any + Send + Sync>);

        Self {
            device,
            queue,
            resources,
            pipelines: FxHashMap::default(),
            active_command_buffer: None,
            active_compute_encoder: None,
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

    /// Start capturing commands into a single command buffer (batched dispatch).
    pub fn start_capture(&mut self) -> Result<(), MetalError> {
        use objc2_metal::MTLCommandQueue as _;
        if self.active_command_buffer.is_some() {
            return Err(MetalError::OperationFailed("Capture already active".to_string()));
        }
        let buffer = self.queue.0.commandBuffer().ok_or(MetalError::CommandBufferCreationFailed)?;
        self.active_command_buffer = Some(buffer);
        self.active_compute_encoder = None;
        Ok(())
    }

    /// End capture and return the command buffer (committed but not waited).
    pub fn end_capture(&mut self) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>, MetalError> {
        use objc2_metal::{MTLCommandBuffer as _, MTLCommandEncoder as _};
        // End the persistent encoder before committing
        if let Some(encoder) = self.active_compute_encoder.take() {
            encoder.endEncoding();
        }

        let buffer = self
            .active_command_buffer
            .take()
            .ok_or(MetalError::OperationFailed("No active capture".to_string()))?;

        buffer.commit();
        Ok(buffer)
    }

    /// Ensure all pending commands have completed.
    /// If capture is active, completes the current batch and starts a new one.
    pub fn synchronize(&mut self) -> Result<(), MetalError> {
        use objc2_metal::MTLCommandBuffer as _;
        if self.is_capturing() {
            let buf = self.end_capture()?;
            buf.waitUntilCompleted();
            self.start_capture()?;
        } else {
            // Not capturing - commands are already committed and waited in dispatch() but
            // we might have a committed buffer that hasn't finished yet if they were manually committed.
            // For safety, we'll just ensure the queue is empty.
            // Actually, currently dispatch() is synchronous for non-capturing, so this is a no-op except
            // for ensuring any previously committed buffers are done.
        }
        Ok(())
    }

    /// Loads or retrieves a compute pipeline for the given Kernel type.
    pub fn load_kernel<K: Kernel>(&mut self, kernel: &K) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        use std::hash::{Hash as _, Hasher as _};

        use objc2_foundation::NSString;

        let mut hasher = rustc_hash::FxHasher::default();
        kernel.function_name().hash(&mut hasher);
        let key = (TypeId::of::<K::Id>(), hasher.finish());

        if let Some(pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline.clone());
        }

        // 1. Try Default Library (loading from Bundle/built output)
        let function_name = kernel.function_name();
        if let Some(lib) = self.device.newDefaultLibrary() {
            let ns_name = NSString::from_str(function_name);
            if let Some(func) = lib.newFunctionWithName(&ns_name) {
                let pipeline = self
                    .device
                    .newComputePipelineStateWithFunction_error(&func)
                    .map_err(|e| MetalError::PipelineCreationFailed(format!("{:?}", e)))?;
                self.pipelines.insert(key, pipeline.clone());
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

        // Helper to check if an include is a policy/system header
        let is_policy = |inc: &str| inc.starts_with("policies/");

        // 1. Load policy includes (base, f16, q8, etc.)
        // These provide types (uint, half) needed by struct_defs
        for include in includes.iter().filter(|&&i| is_policy(i)) {
            let p = find_include(include, source_path_ctx.as_ref())
                .ok_or_else(|| MetalError::LoadLibraryFailed(format!("Include file {} not found", include)))?;
            let content = std::fs::read_to_string(&p)
                .map_err(|e| MetalError::LoadLibraryFailed(format!("Failed to read include {}: {}", p.display(), e)))?;

            // Strip local includes
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

        // 2. Inject struct definitions
        // These rely on types from policies, and are relied upon by user helper/stage includes
        let struct_defs = kernel.struct_defs();
        if !struct_defs.is_empty() {
            full_source.push_str("// Auto-generated struct definitions\n");
            full_source.push_str(&struct_defs);
            full_source.push_str("\n\n");
        }

        // 3. Load remaining user includes (stages, helpers)
        for include in includes.iter().filter(|&&i| !is_policy(i)) {
            let p = find_include(include, source_path_ctx.as_ref())
                .ok_or_else(|| MetalError::LoadLibraryFailed(format!("Include file {} not found", include)))?;
            let content = std::fs::read_to_string(&p)
                .map_err(|e| MetalError::LoadLibraryFailed(format!("Failed to read include {}: {}", p.display(), e)))?;

            // Strip local includes
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
        // Match legacy kernel compilation settings to avoid source-level feature mismatches
        // (the matmul_gemv fused kernels rely on newer MSL features).
        options.setLanguageVersion(objc2_metal::MTLLanguageVersion::Version4_0);
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

        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// Dispatches a kernel using a pre-loaded pipeline.
    ///
    /// This bypasses the pipeline lookup/compilation step, which is useful for
    /// repeatedly dispatching the same kernel (e.g. in autoregressive loops).
    pub fn dispatch_pipeline<K: Kernel>(
        &mut self,
        pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        kernel: &K,
        config: DispatchConfig,
    ) -> Result<(), MetalError> {
        use objc2_metal::{MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _, MTLComputeCommandEncoder as _, MTLSize};

        if self.active_command_buffer.is_some() {
            // Batched dispatch path
            let encoder = if let Some(enc) = self.active_compute_encoder.clone() {
                enc
            } else {
                let active = self.active_command_buffer.as_ref().unwrap();
                let enc = active.computeCommandEncoder().ok_or(MetalError::CommandQueueCreationFailed)?;
                self.active_compute_encoder = Some(enc.clone());
                enc
            };

            encoder.setComputePipelineState(pipeline);
            let encoder_wrapper = crate::types::ComputeCommandEncoder(encoder.clone());
            kernel.bind(&encoder_wrapper);
            let (grid_size, group_size): (MTLSize, MTLSize) = config.into();
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, group_size);
        } else {
            // Non-batched path
            let command_buffer = self.queue.0.commandBuffer().ok_or(MetalError::CommandBufferCreationFailed)?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or(MetalError::CommandQueueCreationFailed)?;

            encoder.setComputePipelineState(pipeline);
            let encoder_wrapper = crate::types::ComputeCommandEncoder(encoder.clone());
            kernel.bind(&encoder_wrapper);
            let (grid_size, group_size): (MTLSize, MTLSize) = config.into();
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, group_size);
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
        }

        Ok(())
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
        let pipeline = self.load_kernel(kernel)?;
        self.dispatch_pipeline(&pipeline, kernel, config)
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

        if let Some(ref buffer) = self.active_command_buffer {
            // Batched path: reuse or create encoder on active buffer
            let encoder = if let Some(ref active_enc) = self.active_compute_encoder {
                active_enc.clone()
            } else {
                let enc = buffer.computeCommandEncoder().ok_or(MetalError::CommandQueueCreationFailed)?;
                self.active_compute_encoder = Some(enc.clone());
                enc
            };

            encoder.setComputePipelineState(&pipeline);
            let encoder_wrapper = crate::types::ComputeCommandEncoder(encoder.clone());
            kernel.bind(&encoder_wrapper);

            let config = kernel.dispatch_config();
            let (grid_size, group_size): (objc2_metal::MTLSize, objc2_metal::MTLSize) = config.into();
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, group_size);

            // Do NOT end encoding or commit - wait for sync() or end_capture()
        } else {
            // Standalone path: create new buffer, encode, and block wait
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
        }

        Ok(())
    }

    /// Check if command buffer capture is currently active.
    pub fn is_capturing(&self) -> bool {
        self.active_command_buffer.is_some()
    }

    /// Copy data between buffers using a blit encoder.
    ///
    /// When batched dispatch is active, this encodes the blit into the active command buffer
    /// (ending the compute encoder temporarily). When not batched, creates a standalone
    /// command buffer that commits and waits.
    pub fn blit_copy(
        &mut self,
        src_buffer: &crate::types::MetalBuffer,
        src_offset: usize,
        dst_buffer: &crate::types::MetalBuffer,
        dst_offset: usize,
        size_bytes: usize,
    ) -> Result<(), MetalError> {
        use objc2_metal::{MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _};

        if let Some(ref active_buffer) = self.active_command_buffer {
            // Batched path: encode blit into the active command buffer

            // End the compute encoder if one is active (Metal requires only one encoder at a time)
            if let Some(encoder) = self.active_compute_encoder.take() {
                encoder.endEncoding();
            }

            // Create a blit encoder on the same command buffer
            let blit = active_buffer.blitCommandEncoder().ok_or(MetalError::CommandQueueCreationFailed)?;

            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src_buffer, src_offset, dst_buffer, dst_offset, size_bytes,
                );
            }
            blit.endEncoding();

            // The next dispatch() call will recreate the compute encoder as needed
        } else {
            // Non-batched path: create a standalone command buffer
            let cmd = self.queue.0.commandBuffer().ok_or(MetalError::CommandBufferCreationFailed)?;
            let blit = cmd.blitCommandEncoder().ok_or(MetalError::CommandQueueCreationFailed)?;

            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src_buffer, src_offset, dst_buffer, dst_offset, size_bytes,
                );
            }
            blit.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        Ok(())
    }
}

impl Drop for Foundry {
    fn drop(&mut self) {
        use objc2_metal::MTLCommandEncoder as _;
        if let Some(encoder) = self.active_compute_encoder.take() {
            encoder.endEncoding();
        }
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
