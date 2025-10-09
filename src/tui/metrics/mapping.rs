//! Metrics mapping utilities for converting metric events to latency rows for TUI display

use metallic_cli_helpers::app_event::LatencyRow;
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
        MetricEvent::GpuOpCompleted { op_name, duration_us, .. } => map_gpu_op_completed(op_name)
            .into_iter()
            .map(|segments| build_latency_row(segments, *duration_us))
            .collect(),
        MetricEvent::InternalKernelCompleted {
            parent_op_name,
            internal_kernel_name,
            duration_us,
        } => map_internal_kernel(parent_op_name, internal_kernel_name)
            .into_iter()
            .map(|segments| build_latency_row(segments, *duration_us))
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
    // Parse the '/'-separated path and reconstruct a canonical hierarchical label path
    // Default to Generation Loop context. Insert Forward Step when we have block/stage context
    // or an inner label that is part of the forward pass. Plain waits (cb_wait/dep_wait) without
    // a block/stage should live directly under Generation Loop.
    let mut has_generation = false;
    let mut block_label: Option<String> = None;
    let mut stage_label: Option<String> = None;
    let mut inner_label: Option<String> = None;

    for seg in path.split('/') {
        if seg.eq_ignore_ascii_case("generation loop") {
            has_generation = true;
            continue;
        }
        if seg.eq_ignore_ascii_case(PROMPT_PROCESSING_LABEL) {
            // Collapse prompt processing into a single line summary
            return Some(vec![PROMPT_PROCESSING_LABEL.to_string()]);
        }
        if let Some(rest) = seg.strip_prefix("block_")
            && let Ok(idx) = rest.parse::<usize>()
        {
            block_label = Some(format!("Block {}", idx));
            continue;
        }
        if seg.eq_ignore_ascii_case("generation_step_output") {
            inner_label = Some("Output".to_string());
            continue;
        }
        // Explicitly surface wait segments as visible leaves
        if seg.eq_ignore_ascii_case("cb_wait") {
            inner_label = Some("CB Wait".to_string());
            continue;
        }
        if seg.eq_ignore_ascii_case("dep_wait") {
            inner_label = Some("Dep Wait".to_string());
            continue;
        }
        if let Some(stage) = map_block_stage(seg) {
            // Expect [Generation Loop, Forward Step, Block N, Stage]; capture Block and Stage
            if stage.len() >= 3 {
                block_label = Some(stage[2].clone());
            }
            if let Some(last) = stage.last() {
                stage_label = Some(last.clone());
            }
            continue;
        }
        // Map inner op friendly names
        let inner = seg.trim_end_matches("_op");
        let friendly = match inner {
            s if s.starts_with("sdpa_matmul_qk") => Some("QK MatMul"),
            s if s.starts_with("sdpa_softmax") => Some("Softmax"),
            s if s.starts_with("sdpa_matmul_av") => Some("AV MatMul"),
            s if s.starts_with("attn_qkv_proj") => Some("QKV Proj"),
            s if s.starts_with("attn_rearrange") => Some("Rearrange"),
            s if s.starts_with("attn_output") => Some("Attn Output"),
            s if s.starts_with("mlp_swiglu") => Some("SwiGLU"),
            s if s.starts_with("mlp_norm") => Some("MLP Norm"),
            s if s.starts_with("mlp_output") => Some("MLP Output"),
            s if s.starts_with("rope_block") || s == "rope" => Some("Rope"),
            "generation_step_output" => Some("Output"),
            _ => None,
        };
        if let Some(name) = friendly {
            inner_label = Some(name.to_string());
            continue;
        }
    }

    // Build final path
    let mut out: Vec<String> = Vec::new();
    if has_generation {
        out.push(GENERATION_LOOP_LABEL.to_string());
    } else {
        // If not explicitly present, inject it for generation-related scopes
        out.push(GENERATION_LOOP_LABEL.to_string());
    }

    // Decide whether to insert "Forward Step" as an intermediate node.
    let is_plain_wait = matches!(inner_label.as_deref(), Some("CB Wait") | Some("Dep Wait"));
    let has_block_or_stage = block_label.is_some() || stage_label.is_some();
    if has_block_or_stage || !is_plain_wait {
        out.push("Forward Step".to_string());
    }

    if let Some(block) = block_label {
        out.push(block);
    }
    if let Some(stage) = stage_label {
        out.push(stage);
    }
    if let Some(inner) = inner_label {
        out.push(inner);
    }

    if out.is_empty() { None } else { Some(out) }
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

/// Build a latency row from segments and duration
pub fn build_latency_row(segments: Vec<String>, duration_us: u64) -> LatencyRow {
    let level = segments.len().saturating_sub(1) as u8;
    let label = segments.join("::");
    let duration_ms = duration_us as f64 / 1000.0;
    LatencyRow {
        label,
        last_ms: duration_ms,
        average_ms: duration_ms,
        level,
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
