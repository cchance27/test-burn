#![allow(clippy::manual_div_ceil)]
#![allow(clippy::needless_update)]
use std::any::{Any, TypeId};

pub use error::MetalError;
use instrument::CaptureMetrics;
use rustc_hash::FxHashMap;
pub use spec::*;
pub use tensor::*;
pub use tokenizer::BPETokenizer;
pub use types::*;

pub mod compound;
pub mod constants;
mod error;
pub mod fusion;
pub mod generation;
pub mod instrument;
pub mod kernel_registry;
pub mod metals;
pub mod model;
pub mod policy;
pub mod pool;
pub mod spec;
pub mod storage;
pub mod template;
pub mod tensor;
pub mod tokenizer;
pub mod types;
pub mod workflow;

pub use kernel_registry::kernel_registry;

/// Metal kernel source (file path or raw string).
pub enum KernelSource {
    /// A relative path to a .metal file in `src/metals/`.
    File(&'static str),
    /// Raw Metal source code string.
    String(String),
}

/// The central hub for Metal operations, managing the device, queue, and resources.
pub struct Foundry {
    pub device: MetalDevice,
    pub queue: MetalQueue,
    /// Type-safe registry for resources (caches, pools, etc.)
    resources: FxHashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Current stream for batched commands
    active_stream: Option<CommandStream>,
    /// Optional capture-level metrics accumulator (enabled only when profiling is on).
    capture_metrics: Option<CaptureMetrics>,
}

impl Foundry {
    /// Create a new Foundry with the system default device.
    pub fn new() -> Result<Self, MetalError> {
        let device = crate::types::MetalDevice::create_system_default_device()?;
        let queue = device.new_command_queue()?;
        Foundry::new_with_result(device, queue)
    }

    /// Create a new Foundry with an existing device and queue.
    pub fn new_with(device: MetalDevice, queue: MetalQueue) -> Self {
        Self::new_with_result(device, queue).expect("Failed to create Foundry with existing device and queue")
    }

    /// Internal constructor that returns a Result.
    fn new_with_result(device: MetalDevice, queue: MetalQueue) -> Result<Self, MetalError> {
        let mut resources: FxHashMap<TypeId, Box<dyn Any + Send + Sync>> = FxHashMap::default();
        let pool = pool::MemoryPool::new(device.clone(), queue.clone())?;
        resources.insert(TypeId::of::<pool::MemoryPool>(), Box::new(pool) as Box<dyn Any + Send + Sync>);

        Ok(Self {
            device,
            queue,
            resources,
            active_stream: None,
            capture_metrics: None,
        })
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
        if self.active_stream.is_some() {
            return Err(MetalError::OperationFailed("Capture already active".to_string()));
        }
        let buffer = self.queue.command_buffer()?;
        self.active_stream = Some(CommandStream::new(buffer));
        if instrument::emit_cb_timing_metrics() {
            self.capture_metrics = Some(CaptureMetrics::new(instrument::next_capture_id()));
        }
        Ok(())
    }

    /// End capture and return the command buffer (committed but not waited).
    pub fn end_capture(&mut self) -> Result<MetalCommandBuffer, MetalError> {
        let mut stream = self
            .active_stream
            .take()
            .ok_or(MetalError::OperationFailed("No active capture".to_string()))?;

        stream.end_encoding();
        let buffer = stream.command_buffer().clone();

        // Attach capture-level completion metrics before commit so the handler is retained by Metal.
        if instrument::emit_cb_timing_metrics() {
            use std::{sync::Mutex, time::Instant};

            let commit_instant = Instant::now();
            let captured = self
                .capture_metrics
                .take()
                .unwrap_or_else(|| CaptureMetrics::new(instrument::next_capture_id()));
            let op_name = format!("foundry_capture#{}", captured.id);
            let data = instrument::summarize_kernel_counts(&captured, 12);

            let handler = Mutex::new(Some((op_name, data, commit_instant)));

            // Use wrapper method
            buffer.add_completed_handler(move |cmd| {
                let Some((op_name, data, commit_instant)) = handler.lock().ok().and_then(|mut slot| slot.take()) else {
                    return;
                };
                let cpu_elapsed_us = commit_instant.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;

                let start = cmd.gpu_start_time();
                let end = cmd.gpu_end_time();

                let gpu_elapsed_us = if start.is_finite() && end.is_finite() && end > start && start > 0.0 {
                    ((end - start) * 1_000_000.0).clamp(0.0, u64::MAX as f64) as u64
                } else {
                    cpu_elapsed_us
                };

                let mut data = data;
                data.insert("cpu_elapsed_us".to_string(), cpu_elapsed_us.to_string());
                data.insert("gpu_elapsed_us".to_string(), gpu_elapsed_us.to_string());
                metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                    op_name,
                    backend: "Foundry".to_string(),
                    duration_us: gpu_elapsed_us,
                    data: Some(data),
                });
            });
        } else {
            self.capture_metrics = None;
        }

        buffer.commit();
        Ok(buffer)
    }

    /// End the current capture, commit, wait for completion, and start a new capture.
    /// Useful for per-kernel profiling or checkpoints.
    pub fn restart_capture_sync(&mut self) -> Result<MetalCommandBuffer, MetalError> {
        let buf = self.end_capture()?;
        buf.wait_until_completed();
        self.start_capture()?;
        Ok(buf)
    }

    /// Ensure all pending commands have completed.
    /// If capture is active, completes the current batch and starts a new one.
    pub fn synchronize(&mut self) -> Result<(), MetalError> {
        if self.is_capturing() {
            let buf = self.end_capture()?;
            buf.wait_until_completed();
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
    pub fn load_kernel<K: Kernel>(&mut self, kernel: &K) -> Result<MetalPipeline, MetalError> {
        let registry = kernel_registry();
        let pipeline = registry.get_or_load_pipeline(&self.device, kernel)?;
        Ok((*pipeline).clone())
    }
    /// Dispatches a kernel using a pre-loaded pipeline.
    ///
    /// This bypasses the pipeline lookup/compilation step, which is useful for
    /// repeatedly dispatching the same kernel (e.g. in autoregressive loops).
    pub fn dispatch_pipeline<K: Kernel>(&mut self, pipeline: &MetalPipeline, kernel: &K, config: DispatchConfig) -> Result<(), MetalError> {
        if let Some(stream) = self.active_stream.as_mut() {
            // Batched dispatch path
            if let Some(metrics) = self.capture_metrics.as_mut() {
                metrics.record_kernel(kernel.function_name());
            }
            let encoder = stream.compute_encoder()?;

            // Unified dispatch logic
            let dispatch_start = std::time::Instant::now();
            encoder.execute(pipeline, config.grid, config.group, |e| kernel.bind(e));

            // Profiling mode: sync after each dispatch to get actual GPU timing
            // Non-profiling mode: just use dispatch overhead (batched CB wait handles GPU time)
            let duration_us = if instrument::foundry_per_kernel_profiling_enabled() {
                let kernel_start = std::time::Instant::now();
                self.restart_capture_sync()?;
                let kernel_duration = kernel_start.elapsed();

                kernel_duration.as_micros().min(u128::from(u64::MAX)) as u64
            } else {
                dispatch_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64
            };

            // Emit metric unconditionally (single call point)
            if instrument::foundry_metrics_enabled() {
                let op_name = kernel.function_name();
                metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                    op_name: format!("Generation Loop/Forward Step/{}", op_name),
                    backend: "Foundry".to_string(),
                    duration_us,
                    data: None,
                });
            }
        } else {
            // Non-batched path: use a temporary stream for consistent encoder management
            let command_buffer = self.queue.command_buffer()?;
            {
                let mut stream = CommandStream::new(command_buffer.clone());
                let encoder = stream.compute_encoder()?;
                encoder.execute(pipeline, config.grid, config.group, |e| kernel.bind(e));

                if instrument::emit_cb_timing_metrics() {
                    let op_name = kernel.function_name().to_string();
                    let metric_data = kernel.metric_data();
                    let commit_instant = std::time::Instant::now();
                    let handler = std::sync::Mutex::new(Some((op_name, metric_data, commit_instant)));

                    command_buffer.add_completed_handler(move |cmd: &crate::types::MetalCommandBuffer| {
                        let Some((op_name, metric_data, commit_instant)) = handler.lock().ok().and_then(|mut slot| slot.take()) else {
                            return;
                        };
                        let cpu_elapsed_us = commit_instant.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;

                        let start = cmd.gpu_start_time();
                        let end = cmd.gpu_end_time();

                        let gpu_elapsed_us = if start.is_finite() && end.is_finite() && end > start && start > 0.0 {
                            ((end - start) * 1_000_000.0).clamp(0.0, u64::MAX as f64) as u64
                        } else {
                            cpu_elapsed_us
                        };
                        let metric_data = metric_data.map(|mut data: FxHashMap<String, String>| {
                            data.insert("cpu_elapsed_us".to_string(), cpu_elapsed_us.to_string());
                            data.insert("gpu_elapsed_us".to_string(), gpu_elapsed_us.to_string());
                            data
                        });
                        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                            op_name: op_name.to_string(),
                            backend: "Foundry".to_string(),
                            duration_us: gpu_elapsed_us,
                            data: metric_data,
                        });
                    });
                }
                // End encoding happens implicitly via stream drop
            }
            command_buffer.commit();
            command_buffer.wait_until_completed();
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
    /// Note: this does not implicitly wrap kernels with a policy. Callers should pass the final
    /// kernel value they intend to dispatch (including any policy wrappers).
    pub fn run<K: Kernel>(&mut self, kernel: &K) -> Result<(), MetalError> {
        let config = kernel.dispatch_config();
        self.dispatch(kernel, config)
    }

    /// Check if command buffer capture is currently active.
    pub fn is_capturing(&self) -> bool {
        self.active_stream.is_some()
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
        if let Some(stream) = self.active_stream.as_mut() {
            // Batched path: encode blit into the active stream
            let blit = stream.blit_encoder()?;
            blit.copy_from_buffer(src_buffer, src_offset, dst_buffer, dst_offset, size_bytes);
            // The next dispatch() call will switch back to compute as needed
        } else {
            self.blit_copy_sync(src_buffer, src_offset, dst_buffer, dst_offset, size_bytes)?;
        }

        Ok(())
    }

    pub fn blit_copy_sync(
        &self,
        src_buffer: &crate::types::MetalBuffer,
        src_offset: usize,
        dst_buffer: &crate::types::MetalBuffer,
        dst_offset: usize,
        size_bytes: usize,
    ) -> Result<(), MetalError> {
        let cmd = self.queue.command_buffer()?;
        let blit = cmd.blit_command_encoder()?;

        blit.copy_from_buffer(src_buffer, src_offset, dst_buffer, dst_offset, size_bytes);
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    pub fn upload_bytes(&mut self, dst_buffer: &crate::types::MetalBuffer, dst_offset: usize, data: &[u8]) -> Result<(), MetalError> {
        if data.is_empty() {
            return Ok(());
        }

        let staging = self
            .device
            .new_buffer_from_slice(data, MetalResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferFromBytesCreationFailed)?;

        self.blit_copy(&staging, 0, dst_buffer, dst_offset, data.len())
    }
}

impl Drop for Foundry {
    fn drop(&mut self) {
        if let Some(mut stream) = self.active_stream.take() {
            stream.end_encoding();
        }
    }
}

pub struct Includes(pub Vec<&'static str>);

pub trait Kernel {
    type Args;

    fn function_name(&self) -> &str;
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

    /// Optional metric metadata to attach to `GpuOpCompleted` events when profiling is enabled.
    ///
    /// This is only consulted when instrumentation is active (e.g. `METALLIC_ENABLE_PROFILING=1`
    /// with a configured metrics sink).
    fn metric_data(&self) -> Option<FxHashMap<String, String>> {
        None
    }

    /// Hash of the source code (optional, used for collision-proof caching).
    fn source_hash(&self) -> u64 {
        0
    }

    /// Convert this kernel into a Stage for use in a CompoundKernel.
    ///
    /// Only kernels designed for fusion should implement this. The default implementation
    /// panics to fail fast instead of silently producing an invalid fused kernel.
    fn to_stage(&self) -> Box<dyn crate::compound::Stage>
    where
        Self: Sized + Clone + Send + Sync + 'static,
    {
        panic!(
            "Kernel::to_stage is not implemented for {}. \
This kernel cannot be used as a Stage in a CompoundKernel.",
            self.function_name()
        )
    }
}
/// Internal helper for pipeline compilation from kernel metadata.
pub fn compile_pipeline<K: Kernel>(device: &MetalDevice, kernel: &K) -> Result<MetalPipeline, MetalError> {
    use std::path::PathBuf;

    // 1. Try Default Library (loading from Bundle/built output)
    let function_name = kernel.function_name();
    if let Some(lib) = device.new_default_library()
        && let Some(func) = lib.new_function(function_name)
    {
        let pipeline = device.new_compute_pipeline_state_with_function(&func)?;
        return Ok(pipeline);
    }

    // 2. Fallback to Source
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

    if !includes.contains(&"policies/base.metal") {
        includes.insert(0, "policies/base.metal");
    }

    if let Some(dtype) = kernel.dtype() {
        let policy = crate::policy::resolve_policy(dtype);
        let policy_h = policy.header();
        if !includes.contains(&policy_h) {
            includes.push(policy_h);
        }
    }

    let mut source_path_ctx: Option<PathBuf> = None;
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

    let process_lines = |target: &mut String, content: &str, source_name: &str| {
        for line in content.lines() {
            if line.trim().starts_with("#include \"") {
                tracing::warn!(
                    "Ignored local #include in {}: '{}'. Use Kernel::includes() trait method to specify dependencies.",
                    source_name,
                    line.trim()
                );
                target.push_str(&format!("// Skipped: {}\n", line));
            } else {
                target.push_str(line);
                target.push('\n');
            }
        }
        target.push('\n');
    };

    let is_policy = |inc: &str| inc.starts_with("policies/");

    for include in includes.iter().filter(|&&i| is_policy(i)) {
        let p = find_include(include, source_path_ctx.as_ref())
            .ok_or_else(|| MetalError::LoadLibraryFailed(format!("Include file {} not found", include)))?;
        let content = std::fs::read_to_string(&p)
            .map_err(|e| MetalError::LoadLibraryFailed(format!("Failed to read include {}: {}", p.display(), e)))?;
        process_lines(&mut full_source, &content, include);
    }

    let struct_defs = kernel.struct_defs();
    if !struct_defs.is_empty() {
        full_source.push_str("// Auto-generated struct definitions\n");
        full_source.push_str(&struct_defs);
        full_source.push_str("\n\n");
    }

    for include in includes.iter().filter(|&&i| !is_policy(i)) {
        let p = find_include(include, source_path_ctx.as_ref())
            .ok_or_else(|| MetalError::LoadLibraryFailed(format!("Include file {} not found", include)))?;
        let content = std::fs::read_to_string(&p)
            .map_err(|e| MetalError::LoadLibraryFailed(format!("Failed to read include {}: {}", p.display(), e)))?;
        process_lines(&mut full_source, &content, include);
    }

    process_lines(&mut full_source, &main_content, kernel.function_name());

    if let Ok(dump_dir) = std::env::var("METALLIC_DUMP_METAL_SOURCE_DIR") {
        let mut safe_name = function_name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();
        if safe_name.is_empty() {
            safe_name = "kernel".to_string();
        }

        let dump_dir = std::path::Path::new(&dump_dir);
        std::fs::create_dir_all(dump_dir).map_err(|e| {
            MetalError::LoadLibraryFailed(format!(
                "Failed to create METALLIC_DUMP_METAL_SOURCE_DIR {}: {}",
                dump_dir.display(),
                e
            ))
        })?;

        let dump_path = dump_dir.join(format!("{}.metal", safe_name));
        std::fs::write(&dump_path, &full_source)
            .map_err(|e| MetalError::LoadLibraryFailed(format!("Failed to write Metal source dump {}: {}", dump_path.display(), e)))?;
    }

    let options = crate::types::MetalCompileOptions::new();

    let library = device.new_library_with_source(&full_source, Some(&options))?;

    let function = library
        .new_function(function_name)
        .ok_or_else(|| MetalError::LoadLibraryFailed(format!("Function {} not found in source", function_name)))?;

    let pipeline = device.new_compute_pipeline_state_with_function(&function)?;

    Ok(pipeline)
}
