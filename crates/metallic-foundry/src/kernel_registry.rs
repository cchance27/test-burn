//! Unified Kernel Registry for Foundry.
//!
//! Provides centralized caching for compiled kernels and Metal pipelines using
//! `moka` for concurrent access with bounded capacity and time-based eviction.
//!
//! # Features
//! - **Bounded memory**: Capacity-bounded cache (eviction policy is internal to `moka`)
//! - **Time-based eviction**: Expire idle kernels/pipelines after configurable timeout
//! - **Multi-GPU safe**: Pipeline keys include device ID
//!
//! # Usage
//! ```text
//! use metallic_foundry::kernel_registry::{kernel_registry, KernelCacheKey};
//!
//! let key = KernelCacheKey::new("gemm", "f16_f16_nn_default_false_false_none");
//! let kernel = kernel_registry().get_or_build(key, || build_gemm_kernel(...));
//! ```

use std::{
    borrow::Cow, hash::{Hash, Hasher}, sync::{Arc, OnceLock}, time::Duration
};

use moka::sync::Cache;

use crate::{MetalError, compound::CompiledCompoundKernel, types::MetalDevice};

/// Open kernel-family identifier.
///
/// This intentionally accepts arbitrary names so first-party and third-party (WASM/WASI/ABI)
/// backends can register families without touching Foundry core enums.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelFamily(Cow<'static, str>);

impl KernelFamily {
    #[inline]
    pub const fn static_name(name: &'static str) -> Self {
        Self(Cow::Borrowed(name))
    }

    #[inline]
    pub fn owned_name(name: impl Into<String>) -> Self {
        Self(Cow::Owned(name.into()))
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for KernelFamily {
    fn from(value: String) -> Self {
        Self::owned_name(value)
    }
}

impl From<&str> for KernelFamily {
    fn from(value: &str) -> Self {
        Self::owned_name(value.to_string())
    }
}

/// Optional trait for typed key specs.
pub trait TypedKernelCacheKey {
    fn family(&self) -> KernelFamily;
    fn variant(&self) -> String;
}

#[inline]
pub fn key_from_typed(spec: &impl TypedKernelCacheKey) -> KernelCacheKey {
    KernelCacheKey::from_family(spec.family(), spec.variant())
}

/// Cache key for compiled kernel templates.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelCacheKey {
    /// Kernel family (e.g., "gemm", "gemv", "embedding")
    pub family: String,
    /// Unique variant identifier (policy names, configs, etc.)
    pub variant: String,
}

impl KernelCacheKey {
    /// Create a new cache key with the given family and variant.
    pub fn new(family: impl Into<String>, variant: impl Into<String>) -> Self {
        Self {
            family: family.into(),
            variant: variant.into(),
        }
    }

    /// Create a key from a canonical [`KernelFamily`].
    #[inline]
    pub fn from_family(family: KernelFamily, variant: impl Into<String>) -> Self {
        Self {
            family: family.as_str().to_string(),
            variant: variant.into(),
        }
    }
}

/// Shorthand constructor for kernel cache keys.
///
/// Usage:
/// - `kernel_cache_key!("gemv", "row_q8_auto")`
/// - `kernel_cache_key!("gemm", "m{}_n{}_k{}", m, n, k)`
#[macro_export]
macro_rules! kernel_cache_key {
    ($family:expr, $variant:expr) => {
        $crate::kernel_registry::KernelCacheKey::new($family, $variant)
    };
    ($family:expr, $fmt:literal, $($arg:tt)*) => {
        $crate::kernel_registry::KernelCacheKey::new($family, format!($fmt, $($arg)*))
    };
}

/// Pipeline cache key - includes device ID for multi-GPU safety.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineCacheKey {
    pub kernel_id_hash: u64,
    pub source_hash: u64,
    pub runtime_dtype_hash: u64,
    pub device_id: u64,
}

impl PipelineCacheKey {
    /// Create a pipeline cache key for the given kernel and device.
    pub fn new<K: crate::Kernel>(kernel: &K, device: &MetalDevice) -> Self {
        let mut hasher = rustc_hash::FxHasher::default();
        kernel.function_name().hash(&mut hasher);
        let kernel_id_hash = hasher.finish();

        Self {
            kernel_id_hash,
            source_hash: kernel.source_hash(),
            runtime_dtype_hash: kernel.runtime_dtype_hash(),
            device_id: device.registry_id(),
        }
    }
}

/// Configuration for the kernel registry.
#[derive(Clone, Debug)]
pub struct RegistryConfig {
    /// Max kernel templates to cache (default: 256).
    pub max_kernels: u64,
    /// Max pipelines to cache (default: 512).
    pub max_pipelines: u64,
    /// Time-to-idle for kernels not accessed (default: 10 min).
    pub kernel_ttl: Duration,
    /// Time-to-idle for pipelines not accessed (default: 30 min).
    pub pipeline_ttl: Duration,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_kernels: 256,
            max_pipelines: 512,
            kernel_ttl: Duration::from_secs(600),    // 10 minutes
            pipeline_ttl: Duration::from_secs(1800), // 30 minutes
        }
    }
}

/// Cache statistics for monitoring.
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    /// Number of cached kernel templates.
    pub kernel_count: u64,
    /// Number of cached pipelines.
    pub pipeline_count: u64,
}

/// Type alias for the Metal pipeline state object.
pub type Pipeline = crate::types::MetalPipeline;

/// Centralized kernel registry with moka concurrent caching.
///
/// This is the single source of truth for all compiled kernels and pipelines
/// in Foundry, replacing the scattered `OnceLock<Mutex<FxHashMap<...>>>` caches.
pub struct KernelRegistry {
    /// Compiled kernel templates (source + metadata).
    kernels: Cache<KernelCacheKey, Arc<CompiledCompoundKernel>>,
    /// Compiled pipelines (Metal PSO objects).
    pipelines: Cache<PipelineCacheKey, Arc<Pipeline>>,
}

impl KernelRegistry {
    /// Create a new registry with the given configuration.
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            kernels: Cache::builder()
                .max_capacity(config.max_kernels)
                .time_to_idle(config.kernel_ttl)
                .build(),
            pipelines: Cache::builder()
                .max_capacity(config.max_pipelines)
                .time_to_idle(config.pipeline_ttl)
                .build(),
        }
    }

    /// Get or build a kernel template.
    ///
    /// Uses moka's `get_with` for atomic get-or-insert semantics.
    /// Only one thread builds if multiple request the same key simultaneously.
    ///
    /// # Arguments
    /// * `key` - Cache key (family + variant).
    /// * `builder` - Closure that builds the kernel if not cached.
    ///
    /// # Returns
    /// Arc to the cached or newly-built kernel.
    pub fn get_or_build<F>(&self, key: KernelCacheKey, builder: F) -> Arc<CompiledCompoundKernel>
    where
        F: FnOnce() -> CompiledCompoundKernel,
    {
        self.kernels.get_with(key, || Arc::new(builder()))
    }

    /// Get or load a pipeline for the given kernel.
    ///
    /// Uses moka's `try_get_with` for fallible pipeline compilation.
    ///
    /// # Arguments
    /// * `device` - Metal device to compile on.
    /// * `kernel` - The kernel template to compile.
    /// * `key` - The kernel cache key (used to derive pipeline key).
    ///
    /// # Returns
    /// Arc to the cached or newly-compiled pipeline.
    /// Get or load a pipeline for the given kernel.
    ///
    /// Uses moka's `try_get_with` for fallible pipeline compilation.
    ///
    /// # Arguments
    /// * `device` - Metal device to compile on.
    /// * `kernel` - The kernel template to compile.
    ///
    /// # Returns
    /// Arc to the cached or newly-compiled pipeline.
    pub fn get_or_load_pipeline<K: crate::Kernel>(&self, device: &MetalDevice, kernel: &K) -> Result<Arc<Pipeline>, MetalError> {
        let pipeline_key = PipelineCacheKey::new(kernel, device);

        self.pipelines
            .try_get_with(pipeline_key, || crate::compile_pipeline(device, kernel).map(Arc::new))
            .map_err(|e| MetalError::PipelineCreationFailed(format!("Pipeline compilation failed: {}", e)))
    }

    /// Clear all caches.
    pub fn clear(&self) {
        self.kernels.invalidate_all();
        self.pipelines.invalidate_all();
    }

    /// Run pending maintenance tasks (eviction, etc.).
    ///
    /// Call this periodically in long-running sessions to ensure
    /// TTL eviction happens even if there are no cache accesses.
    pub fn run_pending_tasks(&self) {
        self.kernels.run_pending_tasks();
        self.pipelines.run_pending_tasks();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            kernel_count: self.kernels.entry_count(),
            pipeline_count: self.pipelines.entry_count(),
        }
    }
}

/// Global kernel registry singleton.
///
/// This is the recommended way to access the registry in most cases.
/// The singleton is lazily initialized with default configuration.
pub fn kernel_registry() -> &'static KernelRegistry {
    static REGISTRY: OnceLock<KernelRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| KernelRegistry::new(RegistryConfig::default()))
}

#[path = "kernel_registry.test.rs"]
mod tests;
