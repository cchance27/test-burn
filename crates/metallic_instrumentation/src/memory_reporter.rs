//! Memory reporting utilities that recreate the historical memory sidebar format.

use std::collections::BTreeMap;

use rustc_hash::FxHashMap;

use crate::memory_collector::MemoryCollector;

/// Memory reporting utilities for displaying memory usage in the historical sidebar format.
pub struct MemoryReporter {
    collector: MemoryCollector,
}

impl MemoryReporter {
    /// Create a new memory reporter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            collector: MemoryCollector::new(),
        }
    }

    /// Report GGUF file memory-mapped usage.
    pub fn report_gguf_mmap(&self, size_bytes: u64) {
        self.collector.collect_gguf_mmap(size_bytes);
        println!("GGUF File MMAP: {:.2} GB", size_bytes as f64 / 1_073_741_824.0);
    }

    /// Report model weights with detailed breakdown.
    pub fn report_model_weights(&self, total_bytes: u64, breakdown: FxHashMap<String, u64>) {
        self.collector.collect_model_weights(total_bytes, breakdown.clone());
        println!("Model Weights: {:.2} GB", total_bytes as f64 / 1_073_741_824.0);

        for (category, bytes) in &breakdown {
            println!("  {}: {:.2} MB", category, (*bytes) as f64 / 1_048_576.0);
        }
    }

    /// Report host memory usage including pools.
    pub fn report_host_memory(
        &self,
        total_bytes: u64,
        tensor_pool_reserved_bytes: u64,
        tensor_pool_used_bytes: u64,
        kv_pool_reserved_bytes: u64,
        kv_pool_used_bytes: u64,
        forward_pass_breakdown: BTreeMap<usize, (String, BTreeMap<String, u64>)>,
    ) {
        self.collector.collect_host_memory(
            total_bytes,
            tensor_pool_reserved_bytes,
            tensor_pool_used_bytes,
            kv_pool_reserved_bytes,
            kv_pool_used_bytes,
            forward_pass_breakdown,
        );

        println!("Host Memory (MB): {:.2} GB", total_bytes as f64 / 1_073_741_824.0);
        println!(
            "  Tensor Pool (Reserved): {:.2} MB",
            tensor_pool_reserved_bytes as f64 / 1_048_576.0
        );
        println!("  Tensor Pool (Used): {:.2} MB", tensor_pool_used_bytes as f64 / 1_048_576.0);
        println!("  KV Pool (Reserved): {:.2} MB", kv_pool_reserved_bytes as f64 / 1_048_576.0);
        println!("  KV Pool (Used): {:.2} MB", kv_pool_used_bytes as f64 / 1_048_576.0);
    }

    /// Report forward step memory usage with detailed breakdown.
    pub fn report_forward_step(&self, total_bytes: u64, breakdown: FxHashMap<String, u64>) {
        self.collector.collect_forward_step(total_bytes, breakdown.clone());
        println!("Forward Step: {:.2} MB", total_bytes as f64 / 1_048_576.0);

        for (component, bytes) in &breakdown {
            println!("  {}: {:.2} KB", component, (*bytes) as f64 / 1_024.0);
        }
    }

    /// Report tensor memory usage statistics.
    pub fn report_tensor_memory(&self, total_bytes: u64, tensor_count: u64, breakdown: FxHashMap<String, u64>) {
        self.collector.collect_tensor_memory(total_bytes, tensor_count, breakdown.clone());
        println!("Tensors: {:.2} MB ({} tensors)", total_bytes as f64 / 1_048_576.0, tensor_count);

        for (category, bytes) in &breakdown {
            println!("  {}: {:.2} MB", category, (*bytes) as f64 / 1_048_576.0);
        }
    }

    /// Generate a complete memory report in the historical format.
    pub fn generate_complete_report(&self) {
        // This would be called from your model loading and inference code
        // to collect and display all memory statistics

        // Example usage (these would be replaced with actual memory collection):
        /*
        self.report_gguf_mmap(1_258_291_200); // 1.18 GB

        let mut model_weights = FxHashMap::default();
        model_weights.insert("Token Embeddings".to_string(), 278_921_216); // 259.66 MB
        model_weights.insert("Output Projection".to_string(), 278_921_216); // 259.66 MB
        model_weights.insert("Final Layer Norm".to_string(), 1_884_160); // 1.75 KB
        model_weights.insert("RoPE Cache".to_string(), 4_194_304); // 4.00 MB
        model_weights.insert("Transformer Blocks".to_string(), 734_003_200); // 683.12 MB
        self.report_model_weights(1_258_291_200, model_weights);

        self.report_host_memory(
            2_828_854_272, // 2.63 GB total
            268_435_456,   // 256.00 MB tensor pool
            671_088_640,   // 640.00 MB KV pool reserved
            251_658_240,   // 240.00 MB KV pool used
        );

        let mut forward_breakdown = FxHashMap::default();
        forward_breakdown.insert("Embedding".to_string(), 1_884_160); // 1.75 KB
        self.report_forward_step(1_228_800, forward_breakdown); // 1.14 MB total

        // Report individual blocks...
        let mut block1_breakdown = FxHashMap::default();
        block1_breakdown.insert("attn_norm".to_string(), 1_884_160); // 1.75 KB
        block1_breakdown.insert("attn_qkv_proj".to_string(), 4_194_304); // 4.00 KB
        block1_breakdown.insert("attn_rearrange".to_string(), 6_553_600); // 6.25 KB
        block1_breakdown.insert("rope".to_string(), 8_912_896); // 8.25 KB
        block1_breakdown.insert("kv_cache".to_string(), 8_912_896); // 8.25 KB
        block1_breakdown.insert("kv_repeat".to_string(), 8_912_896); // 8.25 KB
        block1_breakdown.insert("sdpa".to_string(), 19_922_944); // 19 KB
        block1_breakdown.insert("attn_output".to_string(), 25_169_536); // 24 KB
        block1_breakdown.insert("mlp_swiglu".to_string(), 49_331_200); // 47 KB
        block1_breakdown.insert("mlp_output".to_string(), 52_428_800); // 49 KB
        self.report_forward_block(1, "Block 1".to_string(), block1_breakdown);
        */
    }
}

impl Default for MemoryReporter {
    fn default() -> Self {
        Self::new()
    }
}
