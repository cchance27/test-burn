#![cfg(test)]

use super::*;

#[test]
fn test_cached_memory_profiler_creation() {
    let profiler = CachedMemoryProfiler::new();
    assert!(profiler.get_pid().as_u32() != 0); // PID should be valid
}

#[test]
fn test_get_process_memory_usage() {
    let profiler = CachedMemoryProfiler::new();
    let memory = profiler.get_process_memory_usage();
    assert!(memory > 0); // Process should have some memory usage
}
