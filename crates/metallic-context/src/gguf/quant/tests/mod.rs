#![cfg(test)]
//! Benchmark for Q8 quantization operations
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use crate::gguf::quant::q8_simd;
use crate::gguf::{GGUFDataType, quant::q8};

mod benchmark;
mod large_benchmark;

#[cfg(test)]
/// Get current memory usage in bytes
pub fn get_memory_usage() -> usize {
    use std::process::Command;

    // Use vm_stat to get memory information on macOS
    let output = Command::new("vm_stat").output().expect("Failed to execute vm_stat");

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Parse the output to get memory usage
        // This is a simplified approach - in a real implementation, you'd want to parse more carefully
        for line in stdout.lines() {
            if line.contains("Pages active") {
                // Extract the number of pages and convert to bytes
                // This is a very rough approximation
                if let Some(pages) = line.split(':').nth(1)
                    && let Ok(page_count) = pages.trim().parse::<usize>()
                {
                    // Assuming 4KB pages
                    return page_count * 4096;
                }
            }
        }
    }

    0 // Return 0 if we can't get memory usage
}
