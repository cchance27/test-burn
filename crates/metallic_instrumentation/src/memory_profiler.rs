//! Efficient memory profiling utilities that cache system information to reduce performance impact.
//!
//! The main benefit of this profiler is that it caches the `sysinfo::System` instance and process ID,
//! avoiding expensive recreation of these objects on each call, which was happening in the generation loop.

use std::{
    sync::{Arc, Mutex}, time::{Duration, Instant}
};

use sysinfo::{Pid, ProcessRefreshKind, System, get_current_pid};

const MEMORY_CACHE_DURATION: Duration = Duration::from_millis(50); // Cache system memory value for 50ms

/// Cached memory profiler that reuses the System instance to avoid expensive recreation.
pub struct CachedMemoryProfiler {
    /// Cached process ID to avoid repeated calls to `process::id()`
    cached_pid: Pid,
    /// Cached System instance wrapped in a mutex for thread-safe access
    cached_system: Arc<Mutex<System>>,
    /// Cached memory value to avoid frequent system calls
    cached_memory_bytes: Arc<Mutex<u64>>,
    /// Timestamp of when the memory was last cached
    last_cache_time: Arc<Mutex<Instant>>,
}

impl CachedMemoryProfiler {
    /// Create a new cached memory profiler.
    #[must_use]
    pub fn new() -> Self {
        // Get the current process ID once and cache it
        let pid = get_current_pid().expect("Failed to get current process ID");

        // Create a system instance once and cache it
        let mut sys = System::new_all();
        sys.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::everything().with_memory().with_cpu().with_disk_usage(),
        );

        // Get initial memory usage
        let initial_memory = match sys.process(pid) {
            Some(process) => process.memory(),
            None => 0,
        };

        Self {
            cached_pid: pid,
            cached_system: Arc::new(Mutex::new(sys)),
            cached_memory_bytes: Arc::new(Mutex::new(initial_memory)),
            last_cache_time: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Get the current process memory usage in bytes.
    /// This is the main method that should be called from the generation loop.
    /// Uses caching to avoid expensive system calls on every invocation.
    #[must_use]
    pub fn get_process_memory_usage(&self) -> u64 {
        let now = Instant::now();

        // Check if cached value is still valid
        let last_cache_time = self.last_cache_time.lock().expect("Failed to lock cache time");
        let is_cache_valid = now.duration_since(*last_cache_time) < MEMORY_CACHE_DURATION;
        drop(last_cache_time); // Release the lock early

        if is_cache_valid {
            return *self.cached_memory_bytes.lock().expect("Failed to lock cached memory");
        }

        let mut sys = self.cached_system.lock().expect("Failed to lock system mutex");

        // Refresh only the specific process instead of the entire system
        sys.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::Some(&[self.cached_pid]),
            true,
            ProcessRefreshKind::everything().with_memory().with_cpu().with_disk_usage(),
        );

        // Get the memory usage for the current process
        let memory = match sys.process(self.cached_pid) {
            Some(process) => process.memory(),
            None => 0,
        };

        // Update the cached values
        *self.cached_memory_bytes.lock().expect("Failed to lock cached memory") = memory;
        *self.last_cache_time.lock().expect("Failed to lock cache time") = now;

        memory
    }

    /// Get the cached process ID
    #[must_use]
    pub fn get_pid(&self) -> Pid {
        self.cached_pid
    }

    /// Refresh all system information (only when needed, not in generation loop)
    pub fn refresh_all_system_info(&mut self) {
        let mut sys = self.cached_system.lock().expect("Failed to lock system mutex");
        sys.refresh_all();
    }
}

impl Default for CachedMemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Global cached memory profiler instance for convenient access.
pub fn global_cached_memory_profiler() -> &'static CachedMemoryProfiler {
    static INSTANCE: std::sync::OnceLock<CachedMemoryProfiler> = std::sync::OnceLock::new();
    INSTANCE.get_or_init(CachedMemoryProfiler::new)
}

#[path = "memory_profiler.test.rs"]
mod tests;
