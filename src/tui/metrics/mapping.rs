//! Metrics mapping utilities for converting metric events to latency rows for TUI display

use metallic_cli_helpers::app_event::{LatencyRow, MemoryRow};
use metallic_instrumentation::MetricEvent;

/// Label for the generation loop in the metrics hierarchy
pub const GENERATION_LOOP_LABEL: &str = "Generation Loop";
/// Label for prompt processing in the metrics hierarchy
pub const PROMPT_PROCESSING_LABEL: &str = "Prompt Processing";

const BLOCK_STAGE_PREFIXES: &[(&str, &str)] = &[
    ("attn_residual_clone_block_", "attn_residual_clone"),
    ("attn_norm_block_", "attn_norm"),
    ("qkv_proj_block_", "attn_qkv_proj"),
    ("attn_rearrange_block_", "attn_rearrange"),
    ("rope_block_", "Rope"),
    ("kv_cache_block_", "kv_cache"),
    ("kv_repeat_block_", "kv_repeat"),
    ("sdpa_block_", "Sdpa"),
    ("attn_reassembly_block_", "attn_reassembly"),
    ("attn_output_block_", "attn_output"),
    ("attn_residual_block_", "attn_residual"),
    ("mlp_residual_clone_block_", "mlp_residual_clone"),
    ("mlp_norm_block_", "mlp_norm"),
    ("mlp_swiglu_block_", "mlp_swiglu"),
    ("mlp_reshape_block_", "mlp_reshape"),
    ("mlp_residual_block_", "mlp_residual"),
    ("mlp_output_block_", "mlp_output"),
];

/// Convert a metric event to latency rows for display in the TUI
pub fn metric_event_to_latency_rows(event: &MetricEvent) -> Vec<LatencyRow> {
    match event {
        MetricEvent::GpuOpCompleted {
            op_name,
            duration_us,
            data,
            ..
        } => {
            // Convert FxHashMap to std::collections::HashMap for LatencyRow
            let metadata = data.as_ref().map(|d| d.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
            map_gpu_op_completed(op_name)
                .into_iter()
                .map(|segments| build_latency_row(segments, *duration_us, metadata.clone()))
                .collect()
        }
        MetricEvent::InternalKernelCompleted {
            parent_op_name,
            internal_kernel_name,
            duration_us,
        } => map_internal_kernel(parent_op_name, internal_kernel_name)
            .into_iter()
            .map(|segments| build_latency_row(segments, *duration_us, None))
            .collect(),
        _ => Vec::new(),
    }
}

/// Map internal kernel names to hierarchical display segments
pub fn map_internal_kernel(parent: &str, kernel: &str) -> Option<Vec<String>> {
    let base_parent = parent.to_ascii_lowercase();
    match base_parent.as_str() {
        "sampling" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Sampling".to_string()]),
        "decoding" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Decode".to_string()]),
        "generation_loop" => match kernel {
            k if k.starts_with("block_") && k.ends_with("_total") => {
                let idx_str = k.strip_prefix("block_").unwrap().strip_suffix("_total").unwrap();
                let idx = idx_str.parse::<usize>().ok()?;
                Some(vec![
                    GENERATION_LOOP_LABEL.to_string(),
                    "Forward Step".to_string(),
                    format!("Block {}", idx),
                ])
            }
            "pool_reset" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Pool Reset".to_string()]),
            "embedding" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Embedding".to_string()]),
            "forward_step_total" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Forward Step".to_string()]),
            "token_push" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Token Push".to_string()]),
            "cache_logging" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Cache Logging".to_string()]),
            "token_callback" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Token Callback".to_string()]),
            "eos_check" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "EOS Check".to_string()]),
            "metric_recording_overhead" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Metric Recording Overhead".to_string()]),
            "logits_sync" => Some(vec![
                GENERATION_LOOP_LABEL.to_string(),
                "Forward Step".to_string(),
                "Logits Sync".to_string(),
            ]),
            "iteration_total" => Some(vec![GENERATION_LOOP_LABEL.to_string()]),
            _ => None,
        },
        "prompt_processing" => match kernel {
            "logits_sync" => Some(vec![PROMPT_PROCESSING_LABEL.to_string(), "Logits Sync".to_string()]),
            _ => None,
        },
        _ => None,
    }
}

/// Map GPU operation names to hierarchical display segments
pub fn map_gpu_op_completed(op_name: &str) -> Option<Vec<String>> {
    let base_name = op_name.split('#').next().unwrap_or(op_name);

    // New: handle hierarchical op names produced by Context::with_gpu_scope which uses '/' separators
    if base_name.contains('/')
        && let Some(segments) = map_hierarchical_gpu_op(base_name)
    {
        return Some(segments);
    }

    if let Some(segments) = map_generation_stage(base_name) {
        return Some(segments);
    }
    if let Some(segments) = map_block_stage(base_name) {
        return Some(segments);
    }
    map_prompt_stage(base_name)
}

/// Map hierarchical GPU operation paths to display segments
pub fn map_hierarchical_gpu_op(path: &str) -> Option<Vec<String>> {
    // Parse the '/'-separated path directly to build a hierarchical display path
    let segments: Vec<&str> = path.split('/').collect();

    if segments.is_empty() {
        return None;
    }

    // Check for special cases first
    if segments.iter().any(|&s| s.eq_ignore_ascii_case(PROMPT_PROCESSING_LABEL)) {
        // Collapse prompt processing into a single line summary
        return Some(vec![PROMPT_PROCESSING_LABEL.to_string()]);
    }

    // Build the result path step by step
    let mut result_path: Vec<String> = Vec::new();
    result_path.push(GENERATION_LOOP_LABEL.to_string());

    let mut has_forward_step = false;
    let mut has_block = false;

    for &seg in &segments {
        // Skip generation loop if it appears in the path (we already added it)
        if seg.eq_ignore_ascii_case("generation loop") {
            continue;
        }

        // Check if this segment is a block identifier
        if let Some(rest) = seg.strip_prefix("block_")
            && let Ok(idx) = rest.parse::<usize>()
        {
            // Only add the block if we don't already have one in the path
            if !has_block {
                if !has_forward_step {
                    result_path.push("Forward Step".to_string());
                    has_forward_step = true;
                }
                result_path.push(format!("Block {}", idx));
                has_block = true;
            }
            continue;
        }

        // Special handling for some operations
        match seg {
            "generation_step_output" => {
                if !has_forward_step && !has_block {
                    result_path.push("Forward Step".to_string());
                    has_forward_step = true;
                }
                result_path.push("Output".to_string());
            }
            "cb_wait" | "CB Wait" => {
                if !has_forward_step && !has_block {
                    result_path.push("Forward Step".to_string());
                    has_forward_step = true;
                }
                result_path.push("CB Wait".to_string());
            }
            "dep_wait" | "Dep Wait" => {
                if !has_forward_step && !has_block {
                    result_path.push("Forward Step".to_string());
                    has_forward_step = true;
                }
                result_path.push("Dep Wait".to_string());
            }
            _ => {
                // Check for matmul operations (format: matmul/backend or matmul_*/backend)
                if seg.starts_with("matmul") {
                    // Look ahead to see if next segment is a backend
                    let seg_idx = segments.iter().position(|&s| s == seg);
                    if let Some(idx) = seg_idx
                        && idx + 1 < segments.len()
                    {
                        let backend = segments[idx + 1];
                        // Format as "MatMul [OpType (BACKEND)]"
                        let op_display = if seg == "matmul" {
                            format!("MatMul [{}]", backend.to_uppercase())
                        } else if seg == "matmul_bias_add" {
                            format!("MatMul [BiasAdd ({})]", backend.to_uppercase())
                        } else if seg == "matmul_alpha_beta" {
                            format!("MatMul [AlphaBeta ({})]", backend.to_uppercase())
                        } else if seg == "matmul_alpha_beta_cache" {
                            format!("MatMul [AlphaBetaCache ({})]", backend.to_uppercase())
                        } else {
                            format!("MatMul [{}]", backend.to_uppercase())
                        };

                        if !has_forward_step && !has_block {
                            result_path.push("Forward Step".to_string());
                            has_forward_step = true;
                        }
                        if !result_path.contains(&op_display) {
                            result_path.push(op_display);
                        }
                        continue; // Skip the backend segment (will be handled as part of matmul)
                    }
                }

                // Skip standalone backend names that follow matmul
                if matches!(seg, "mlx" | "mps" | "gemv") {
                    // Check if previous segment was matmul-related
                    let seg_idx = segments.iter().position(|&s| s == seg);
                    if let Some(idx) = seg_idx
                        && idx > 0
                        && segments[idx - 1].starts_with("matmul")
                    {
                        continue; // Skip this backend segment
                    }
                }

                // Check if this segment matches a known block stage
                if let Some(stage_segments) = map_block_stage(seg) {
                    if !has_forward_step {
                        result_path.push("Forward Step".to_string());
                        has_forward_step = true;
                    }
                    // Add the stage parts that aren't already in the path
                    for stage_part in &stage_segments {
                        if !result_path.contains(stage_part) && stage_part != GENERATION_LOOP_LABEL {
                            result_path.push(stage_part.clone());
                        }
                    }
                } else {
                    // For other segments, clean them and add to the path
                    let clean_segment = if seg.contains("::") {
                        // Extract the last part after the final :: (e.g., "...::softmax_dispatcher::dispatch_op::SoftmaxDispatchOp" -> "SoftmaxDispatchOp")
                        let seg = seg.split("::").last().unwrap_or(seg);
                        format!("{} Kernel", seg)
                    } else {
                        seg.to_string()
                    };

                    // Map to friendly names for known operation types
                    let display_name = match clean_segment.trim_end_matches("Op") {
                        s if s.starts_with("sdpa_matmul_qk") => "QK MatMul".to_string(),
                        s if s.starts_with("sdpa_softmax") => "Softmax".to_string(),
                        s if s.starts_with("sdpa_matmul_av") => "AV MatMul".to_string(),
                        s if s.starts_with("attn_qkv_proj") => "QKV Proj".to_string(),
                        s if s.starts_with("attn_rearrange") => "Rearrange".to_string(),
                        s if s.starts_with("attn_output") => "Attn Output".to_string(),
                        s if s.starts_with("mlp_swiglu") => "SwiGLU".to_string(),
                        s if s.starts_with("mlp_norm") => "MLP Norm".to_string(),
                        s if s.starts_with("mlp_output") => "MLP Output".to_string(),
                        s if s.starts_with("rope_block") || s == "rope" => "Rope".to_string(),
                        s if s.starts_with("MatmulDispatch") => "MatMul".to_string(),
                        s if s.starts_with("SoftmaxDispatch") => "Softmax".to_string(),
                        s if s.starts_with("RMSNorm") => "RMS Norm".to_string(),
                        s if s.starts_with("SwiGLU") => "SwiGLU".to_string(),
                        s if s.starts_with("ScaledDotProductAttention") => "SDPA".to_string(),
                        "generation_step_output" => "Output".to_string(),
                        other => other.to_string(),
                    };

                    if !has_forward_step && !has_block && !display_name.contains("Wait") {
                        result_path.push("Forward Step".to_string());
                        has_forward_step = true;
                    }

                    // Only add if it's not already in the path
                    if !result_path.contains(&display_name) {
                        result_path.push(display_name);
                    }
                }
            }
        }
    }

    if result_path.len() <= 1 { None } else { Some(result_path) }
}

/// Map generation stage names to display segments
pub fn map_generation_stage(name: &str) -> Option<Vec<String>> {
    if let Some(rest) = name.strip_prefix("generation_step_")
        && rest.ends_with("_output")
    {
        return Some(vec![GENERATION_LOOP_LABEL.to_string(), "Output".to_string()]);
    }

    if let Some(_idx) = name.strip_prefix("iteration_") {
        return Some(vec![GENERATION_LOOP_LABEL.to_string()]);
    }

    if name == "forward_step" {
        return Some(vec![GENERATION_LOOP_LABEL.to_string(), "Forward Step".to_string()]);
    }

    if let Some(idx_str) = name.strip_prefix("block_")
        && let Ok(idx) = idx_str.parse::<usize>()
    {
        return Some(vec![
            GENERATION_LOOP_LABEL.to_string(),
            "Forward Step".to_string(),
            format!("Block {}", idx),
        ]);
    }

    None
}

/// Map block stage names to display segments
pub fn map_block_stage(name: &str) -> Option<Vec<String>> {
    let base = name.strip_suffix("_op").unwrap_or(name);
    for (prefix, display) in BLOCK_STAGE_PREFIXES {
        if let Some(rest) = base.strip_prefix(prefix)
            && let Ok(idx) = rest.parse::<usize>()
        {
            return Some(vec![
                GENERATION_LOOP_LABEL.to_string(),
                "Forward Step".to_string(),
                format!("Block {}", idx),
                (*display).to_string(),
            ]);
        }
    }
    None
}

/// Build a latency row from segments, duration, and optional metadata
pub fn build_latency_row(
    segments: Vec<String>,
    duration_us: u64,
    metadata: Option<std::collections::HashMap<String, String>>,
) -> LatencyRow {
    let level = segments.len().saturating_sub(1) as u8;
    let label = segments.join("::");
    let duration_ms = duration_us as f64 / 1000.0;
    LatencyRow {
        label,
        last_ms: duration_ms,
        average_ms: duration_ms,
        level,
        metadata,
    }
}

/// Map prompt stage names to display segments
pub fn map_prompt_stage(name: &str) -> Option<Vec<String>> {
    if let Some(rest) = name.strip_prefix("prompt_step_")
        && rest.parse::<usize>().is_ok()
    {
        return Some(vec![PROMPT_PROCESSING_LABEL.to_string()]);
    }
    None
}

/// Convert a metric event to memory rows for display in the TUI
pub fn metric_event_to_memory_rows(event: &MetricEvent) -> Vec<MemoryRow> {
    match event {
        MetricEvent::GgufFileMmap { size_bytes } => {
            vec![MemoryRow {
                label: "GGUF File MMAP".to_string(),
                level: 0,
                current_total_mb: (*size_bytes as f64) / 1_048_576.0,
                peak_total_mb: (*size_bytes as f64) / 1_048_576.0,
                current_pool_mb: 0.0,
                peak_pool_mb: 0.0,
                current_kv_mb: 0.0,
                peak_kv_mb: 0.0,
                current_kv_cache_mb: 0.0,
                peak_kv_cache_mb: 0.0,
                absolute_pool_mb: 0.0,
                absolute_kv_mb: 0.0,
                absolute_kv_cache_mb: 0.0,
                show_absolute: false,
            }]
        }
        MetricEvent::ModelWeights { total_bytes, breakdown } => {
            let mut rows = vec![MemoryRow {
                label: "Model Weights".to_string(),
                level: 0,
                current_total_mb: (*total_bytes as f64) / 1_048_576.0,
                peak_total_mb: (*total_bytes as f64) / 1_048_576.0,
                current_pool_mb: 0.0,
                peak_pool_mb: 0.0,
                current_kv_mb: 0.0,
                peak_kv_mb: 0.0,
                current_kv_cache_mb: 0.0,
                peak_kv_cache_mb: 0.0,
                absolute_pool_mb: 0.0,
                absolute_kv_mb: 0.0,
                absolute_kv_cache_mb: 0.0,
                show_absolute: false,
            }];

            #[derive(Debug)]
            struct Node {
                name: String,
                value: u64,
                children: std::collections::BTreeMap<String, Node>, // BTreeMap to keep order
            }

            impl Node {
                fn new(name: &str, value: u64) -> Self {
                    Self {
                        name: name.to_string(),
                        value,
                        children: std::collections::BTreeMap::new(),
                    }
                }
            }

            let mut root = Node::new("", 0);

            for (key, &value) in breakdown {
                let mut current_node = &mut root;
                for part in key.split('.') {
                    current_node = current_node.children.entry(part.to_string()).or_insert_with(|| Node::new(part, 0));
                }
                current_node.value = value;
            }

            fn aggregate_values(node: &mut Node) -> u64 {
                let mut sum = node.value;
                for child in node.children.values_mut() {
                    sum += aggregate_values(child);
                }
                // If node has no value but has children, its value is the sum of children
                // If it has a value (leaf), sum includes it.
                // However, the display logic expects 'value' to be the total size of the node (inclusive of children)
                // So we update node.value to sum.
                // BUT: if we update node.value, we double count if we run this multiple times (we won't).
                // Issue: If leaf has value 10, sum is 10. node.value becomes 10.
                // If parent has value 0, sum is 10. parent.value becomes 10.
                node.value = sum;
                sum
            }

            aggregate_values(&mut root);

            fn build_rows(node: &Node, level: u8, rows: &mut Vec<MemoryRow>) {
                let mut children_to_process: Vec<_> = node.children.iter().collect();

                if node.name == "Transformer Blocks" {
                    children_to_process.sort_by_key(|(name, _)| {
                        name.strip_prefix("Weight Block ")
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(0)
                    });
                }

                for (name, child) in children_to_process {
                    rows.push(MemoryRow {
                        label: format!("{}{}", "  ".repeat(level as usize), name),
                        level,
                        current_total_mb: (child.value as f64) / 1_048_576.0,
                        peak_total_mb: (child.value as f64) / 1_048_576.0,
                        current_pool_mb: 0.0,
                        peak_pool_mb: 0.0,
                        current_kv_mb: 0.0,
                        peak_kv_mb: 0.0,
                        current_kv_cache_mb: 0.0,
                        peak_kv_cache_mb: 0.0,
                        absolute_pool_mb: 0.0,
                        absolute_kv_mb: 0.0,
                        absolute_kv_cache_mb: 0.0,
                        show_absolute: false,
                    });
                    build_rows(child, level + 1, rows);
                }
            }

            build_rows(&root, 1, &mut rows);

            rows
        }
        MetricEvent::HostMemory {
            total_bytes,
            tensor_pool_reserved_bytes,
            tensor_pool_used_bytes,
            kv_pool_reserved_bytes,
            kv_pool_used_bytes,
            forward_pass_breakdown,
        } => {
            let mut rows = vec![MemoryRow {
                label: "Host Memory (MB)".to_string(),
                level: 0,
                current_total_mb: (*total_bytes as f64) / 1_048_576.0,
                peak_total_mb: (*total_bytes as f64) / 1_048_576.0,
                current_pool_mb: 0.0,
                peak_pool_mb: 0.0,
                current_kv_mb: 0.0,
                peak_kv_mb: 0.0,
                current_kv_cache_mb: 0.0,
                peak_kv_cache_mb: 0.0,
                absolute_pool_mb: 0.0,
                absolute_kv_mb: 0.0,
                absolute_kv_cache_mb: 0.0,
                show_absolute: false,
            }];

            // Consolidated one-liners for pool information
            rows.push(MemoryRow {
                label: "  Tensor Pool".to_string(),
                level: 1,
                // current_total_mb holds the USED amount for the consolidated display
                current_total_mb: (*tensor_pool_used_bytes as f64) / 1_048_576.0,
                peak_total_mb: (*tensor_pool_used_bytes as f64) / 1_048_576.0,
                current_pool_mb: 0.0,
                peak_pool_mb: 0.0,
                current_kv_mb: 0.0,
                peak_kv_mb: 0.0,
                current_kv_cache_mb: 0.0,
                peak_kv_cache_mb: 0.0,
                // absolute_pool_mb holds the RESERVED amount for consolidated formatting
                absolute_pool_mb: (*tensor_pool_reserved_bytes as f64) / 1_048_576.0,
                absolute_kv_mb: 0.0,
                absolute_kv_cache_mb: 0.0,
                show_absolute: false,
            });

            rows.push(MemoryRow {
                label: "  KV Pool".to_string(),
                level: 1,
                // current_total_mb holds the USED amount for the consolidated display
                current_total_mb: (*kv_pool_used_bytes as f64) / 1_048_576.0,
                peak_total_mb: (*kv_pool_used_bytes as f64) / 1_048_576.0,
                current_pool_mb: 0.0,
                peak_pool_mb: 0.0,
                current_kv_mb: 0.0,
                peak_kv_mb: 0.0,
                current_kv_cache_mb: 0.0,
                peak_kv_cache_mb: 0.0,
                absolute_pool_mb: 0.0,
                // absolute_kv_mb holds the RESERVED amount for consolidated formatting
                absolute_kv_mb: (*kv_pool_reserved_bytes as f64) / 1_048_576.0,
                absolute_kv_cache_mb: 0.0,
                show_absolute: false,
            });

            if !forward_pass_breakdown.is_empty() {
                let total_blocks_bytes: u64 = forward_pass_breakdown
                    .values()
                    .map(|(_, breakdown)| breakdown.values().sum::<u64>())
                    .sum();

                rows.push(MemoryRow {
                    label: "  Activations".to_string(),
                    level: 1,
                    current_total_mb: (total_blocks_bytes as f64) / 1_048_576.0,
                    peak_total_mb: (total_blocks_bytes as f64) / 1_048_576.0,
                    current_pool_mb: 0.0,
                    peak_pool_mb: 0.0,
                    current_kv_mb: 0.0,
                    peak_kv_mb: 0.0,
                    current_kv_cache_mb: 0.0,
                    peak_kv_cache_mb: 0.0,
                    absolute_pool_mb: 0.0,
                    absolute_kv_mb: 0.0,
                    absolute_kv_cache_mb: 0.0,
                    show_absolute: false,
                });

                for (_, (block_name, breakdown)) in forward_pass_breakdown.iter() {
                    let total_block_bytes: u64 = breakdown.values().sum();
                    rows.push(MemoryRow {
                        label: format!("    {}", block_name),
                        level: 2,
                        current_total_mb: (total_block_bytes as f64) / 1_048_576.0,
                        peak_total_mb: (total_block_bytes as f64) / 1_048_576.0,
                        current_pool_mb: 0.0,
                        peak_pool_mb: 0.0,
                        current_kv_mb: 0.0,
                        peak_kv_mb: 0.0,
                        current_kv_cache_mb: 0.0,
                        peak_kv_cache_mb: 0.0,
                        absolute_pool_mb: 0.0,
                        absolute_kv_mb: 0.0,
                        absolute_kv_cache_mb: 0.0,
                        show_absolute: false,
                    });

                    let mut sorted_breakdown: Vec<_> = breakdown.iter().collect();
                    sorted_breakdown.sort_by_key(|(k, _)| *k);

                    for (component, bytes) in sorted_breakdown {
                        rows.push(MemoryRow {
                            label: format!("       {}", component),
                            level: 3,
                            current_total_mb: (*bytes as f64) / 1_048_576.0,
                            peak_total_mb: (*bytes as f64) / 1_048_576.0,
                            current_pool_mb: 0.0,
                            peak_pool_mb: 0.0,
                            current_kv_mb: 0.0,
                            peak_kv_mb: 0.0,
                            current_kv_cache_mb: 0.0,
                            peak_kv_cache_mb: 0.0,
                            absolute_pool_mb: 0.0,
                            absolute_kv_mb: 0.0,
                            absolute_kv_cache_mb: 0.0,
                            show_absolute: false,
                        });
                    }
                }
            }

            rows
        }
        MetricEvent::ForwardStep { total_bytes, breakdown } => {
            let mut rows = vec![MemoryRow {
                label: "Forward Step".to_string(),
                level: 0,
                current_total_mb: (*total_bytes as f64) / 1_048_576.0,
                peak_total_mb: (*total_bytes as f64) / 1_048_576.0,
                current_pool_mb: 0.0,
                peak_pool_mb: 0.0,
                current_kv_mb: 0.0,
                peak_kv_mb: 0.0,
                current_kv_cache_mb: 0.0,
                peak_kv_cache_mb: 0.0,
                absolute_pool_mb: 0.0,
                absolute_kv_mb: 0.0,
                absolute_kv_cache_mb: 0.0,
                show_absolute: false,
            }];

            for (component, bytes) in breakdown {
                rows.push(MemoryRow {
                    label: format!("  {}", component),
                    level: 1,
                    current_total_mb: (*bytes as f64) / 1_048_576.0,
                    peak_total_mb: (*bytes as f64) / 1_048_576.0,
                    current_pool_mb: 0.0,
                    peak_pool_mb: 0.0,
                    current_kv_mb: 0.0,
                    peak_kv_mb: 0.0,
                    current_kv_cache_mb: 0.0,
                    peak_kv_cache_mb: 0.0,
                    absolute_pool_mb: 0.0,
                    absolute_kv_mb: 0.0,
                    absolute_kv_cache_mb: 0.0,
                    show_absolute: false,
                });
            }

            rows
        }
        MetricEvent::TensorMemory {
            total_bytes,
            tensor_count: _,
            breakdown,
        } => {
            let mut rows = vec![MemoryRow {
                label: "Tensors".to_string(),
                level: 0,
                current_total_mb: (*total_bytes as f64) / 1_048_576.0,
                peak_total_mb: (*total_bytes as f64) / 1_048_576.0,
                current_pool_mb: 0.0,
                peak_pool_mb: 0.0,
                current_kv_mb: 0.0,
                peak_kv_mb: 0.0,
                current_kv_cache_mb: 0.0,
                peak_kv_cache_mb: 0.0,
                absolute_pool_mb: 0.0,
                absolute_kv_mb: 0.0,
                absolute_kv_cache_mb: 0.0,
                show_absolute: false,
            }];

            for (category, bytes) in breakdown {
                rows.push(MemoryRow {
                    label: format!("  {}", category),
                    level: 1,
                    current_total_mb: (*bytes as f64) / 1_048_576.0,
                    peak_total_mb: (*bytes as f64) / 1_048_576.0,
                    current_pool_mb: 0.0,
                    peak_pool_mb: 0.0,
                    current_kv_mb: 0.0,
                    peak_kv_mb: 0.0,
                    current_kv_cache_mb: 0.0,
                    peak_kv_cache_mb: 0.0,
                    absolute_pool_mb: 0.0,
                    absolute_kv_mb: 0.0,
                    absolute_kv_cache_mb: 0.0,
                    show_absolute: false,
                });
            }

            rows
        }
        _ => Vec::new(),
    }
}

/// Convert a metric event to stats rows for display in the TUI
pub fn metric_event_to_stats_rows(event: &MetricEvent) -> Vec<metallic_cli_helpers::app_event::StatsRow> {
    match event {
        MetricEvent::TensorPreparationStats {
            cache_hits,
            cache_misses,
            total_preparation_time_us,
            estimated_time_saved_us,
            hit_rate,
        } => {
            use metallic_cli_helpers::app_event::StatsRow;
            vec![
                StatsRow {
                    label: "Tensor Preparation Cache".to_string(),
                    value: String::new(), // Empty for section header
                    level: 0,
                    description: String::new(),
                },
                StatsRow {
                    label: "  Cache Hits".to_string(),
                    value: format!("{}", cache_hits),
                    level: 1,
                    description: String::new(),
                },
                StatsRow {
                    label: "  Cache Misses".to_string(),
                    value: format!("{}", cache_misses),
                    level: 1,
                    description: String::new(),
                },
                StatsRow {
                    label: "  Total Prep Time".to_string(),
                    value: format!("{:.2} ms", *total_preparation_time_us as f64 / 1000.0),
                    level: 1,
                    description: String::new(),
                },
                StatsRow {
                    label: "  Estimated Time Saved".to_string(),
                    value: format!("{:.2} ms", *estimated_time_saved_us as f64 / 1000.0),
                    level: 1,
                    description: String::new(),
                },
                StatsRow {
                    label: "  Hit Rate".to_string(),
                    value: format!("{:.2}%", hit_rate),
                    level: 1,
                    description: String::new(),
                },
            ]
        }
        MetricEvent::ResourceCacheSummary {
            cache,
            hits,
            misses,
            hit_rate,
            size,
        } => {
            use metallic_cli_helpers::app_event::StatsRow;
            // Create a cache-specific entry without the Resource Cache header since it will be grouped
            vec![
                StatsRow {
                    label: format!("{} Cache", capitalize_first(cache)).to_string(),
                    value: String::new(), // Empty for section header
                    level: 0,             // This is now the top level for this cache type
                    description: String::new(),
                },
                StatsRow {
                    label: "  Size".to_string(),
                    value: format!("{}", size),
                    level: 1,
                    description: String::new(),
                },
                StatsRow {
                    label: "  Hits".to_string(),
                    value: format!("{}", hits),
                    level: 1,
                    description: String::new(),
                },
                StatsRow {
                    label: "  Misses".to_string(),
                    value: format!("{}", misses),
                    level: 1,
                    description: String::new(),
                },
                StatsRow {
                    label: "  Hit Rate".to_string(),
                    value: format!("{:.2}%", hit_rate),
                    level: 1,
                    description: String::new(),
                },
            ]
        }
        _ => Vec::new(),
    }
}

/// Helper function to capitalize the first letter of a string
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}
