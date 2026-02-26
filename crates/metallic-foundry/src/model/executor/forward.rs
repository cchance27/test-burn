use std::time::{Duration, Instant};

use metallic_env::{DEBUG_DECODE_STAGE_TIMING, FoundryEnvVar, is_set};
use rustc_hash::FxHashMap;

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeStage {
    QkvProj,
    Rope,
    FlashAttention,
    OProj,
    Other,
}

#[derive(Default, Clone, Copy)]
struct DecodeStageStats {
    qkv: Duration,
    rope: Duration,
    fa: Duration,
    oproj: Duration,
    other: Duration,
}

#[derive(Clone)]
struct DecodeStepStats {
    name: String,
    total: Duration,
    count: u32,
    stage: DecodeStage,
}

impl Default for DecodeStepStats {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            total: Duration::ZERO,
            count: 0,
            stage: DecodeStage::Other,
        }
    }
}

impl DecodeStageStats {
    #[inline]
    fn add(&mut self, stage: DecodeStage, dur: Duration) {
        match stage {
            DecodeStage::QkvProj => self.qkv += dur,
            DecodeStage::Rope => self.rope += dur,
            DecodeStage::FlashAttention => self.fa += dur,
            DecodeStage::OProj => self.oproj += dur,
            DecodeStage::Other => self.other += dur,
        }
    }

    #[inline]
    fn total(self) -> Duration {
        self.qkv + self.rope + self.fa + self.oproj + self.other
    }
}

#[inline]
fn classify_decode_stage(step_name: &str) -> DecodeStage {
    if step_name.contains("Qkv") {
        DecodeStage::QkvProj
    } else if step_name.contains("Rope") {
        DecodeStage::Rope
    } else if step_name.contains("Flash") || step_name.contains("Sdpa") {
        DecodeStage::FlashAttention
    } else if step_name.contains("Ffn")
        || step_name.contains("SwiGlu")
        || step_name.contains("Gemm")
        || step_name.contains("Gemv")
        || step_name.contains("Matmul")
    {
        DecodeStage::OProj
    } else {
        DecodeStage::Other
    }
}

impl CompiledModel {
    /// Run a single forward step by executing all DSL steps.
    ///
    /// Each step in `spec.architecture.forward` is executed via `Step::execute()`.
    /// Run a single forward step by executing all compiled steps.
    pub fn forward(&self, foundry: &mut Foundry, bindings: &mut TensorBindings, fast_bindings: &FastBindings) -> Result<(), MetalError> {
        //eprintln!(
        //    "Forward: m={}, seq_len={}, pos={}, kv_seq_len={} | StrPos={}",
        //    bindings.get_int_global("m").unwrap_or(0),
        //    bindings.get_int_global("seq_len").unwrap_or(0),
        //    bindings.get_int_global("position_offset").unwrap_or(0),
        //    bindings.get_int_global("kv_seq_len").unwrap_or(0),
        //    bindings.get_var("position_offset").map(|s| s.as_str()).unwrap_or("MISSING")
        //);
        // If we are already capturing (e.g. batched prompt processing), don't start a new capture.
        let nested_capture = foundry.is_capturing();
        let profiling_per_kernel = crate::instrument::foundry_per_kernel_profiling_enabled();
        let debug_step_log = is_set(FoundryEnvVar::DebugStepLog);
        let debug_step_sync = is_set(FoundryEnvVar::DebugCompiledStepSync);
        let debug_decode_stage_timing = DEBUG_DECODE_STAGE_TIMING.get_valid().unwrap_or(false);
        let is_decode = bindings.get_int_global("m").unwrap_or(0) == 1;
        let collect_decode_stage_timing = debug_decode_stage_timing && is_decode;
        let mut decode_stage_stats = DecodeStageStats::default();
        let mut decode_step_stats: FxHashMap<usize, DecodeStepStats> = FxHashMap::default();

        // Always start capture, even in profiling mode (dispatch_pipeline handles per-kernel sync)
        if !nested_capture {
            foundry.start_capture()?;
        }

        for (idx, step) in self.compiled_steps.iter().enumerate() {
            let step_name = step.name();
            let step_perf_name = step.perf_metadata(bindings).unwrap_or_else(|| step_name.to_string());
            if debug_step_log {
                tracing::info!("Forward compiled step {:03}: {}", idx, step_name);
            }
            if collect_decode_stage_timing {
                let stage = classify_decode_stage(step_name);
                let start = Instant::now();
                step.execute(foundry, fast_bindings, bindings, &self.symbol_table)?;
                // For actionable per-step timings, force completion after each step in diagnostics mode.
                if !profiling_per_kernel {
                    foundry.restart_capture_sync()?;
                }
                let elapsed = start.elapsed();
                decode_stage_stats.add(stage, elapsed);
                let entry = decode_step_stats.entry(idx).or_insert(DecodeStepStats {
                    name: step_perf_name,
                    stage,
                    ..DecodeStepStats::default()
                });
                entry.total += elapsed;
                entry.count = entry.count.saturating_add(1);
            } else {
                step.execute(foundry, fast_bindings, bindings, &self.symbol_table)?;
            }
            if debug_step_sync && !collect_decode_stage_timing {
                // Flush after each compiled step to isolate GPU hangs.
                // This is intentionally expensive and intended only for debugging.
                foundry.restart_capture_sync()?;
            }
        }

        if collect_decode_stage_timing {
            let total = decode_stage_stats.total().as_secs_f64().max(1e-12);
            let pct = |d: Duration| (d.as_secs_f64() / total) * 100.0;
            tracing::info!(
                target: "metallic_foundry::model::executor::forward",
                "Decode stage timing (m=1): total={:.2} us | qkv={:.2} us ({:.1}%) | rope={:.2} us ({:.1}%) | fa={:.2} us ({:.1}%) | oproj={:.2} us ({:.1}%) | other={:.2} us ({:.1}%)",
                decode_stage_stats.total().as_secs_f64() * 1e6,
                decode_stage_stats.qkv.as_secs_f64() * 1e6,
                pct(decode_stage_stats.qkv),
                decode_stage_stats.rope.as_secs_f64() * 1e6,
                pct(decode_stage_stats.rope),
                decode_stage_stats.fa.as_secs_f64() * 1e6,
                pct(decode_stage_stats.fa),
                decode_stage_stats.oproj.as_secs_f64() * 1e6,
                pct(decode_stage_stats.oproj),
                decode_stage_stats.other.as_secs_f64() * 1e6,
                pct(decode_stage_stats.other),
            );

            let mut ranked: Vec<(usize, DecodeStepStats)> = decode_step_stats.into_iter().collect();
            ranked.sort_by(|a, b| b.1.total.cmp(&a.1.total));
            const TOP_N: usize = 8;
            for (rank, (step_idx, stats)) in ranked.into_iter().take(TOP_N).enumerate() {
                let total_us = stats.total.as_secs_f64() * 1e6;
                let avg_us = if stats.count > 0 { total_us / (stats.count as f64) } else { 0.0 };
                let pct_total = pct(stats.total);
                tracing::info!(
                    target: "metallic_foundry::model::executor::forward",
                    "Decode step hot[{rank}] idx={step_idx} {}: total={total_us:.2} us ({pct_total:.1}%) avg={avg_us:.2} us count={} stage={:?}",
                    stats.name,
                    stats.count,
                    stats.stage,
                );
            }
        }

        // End capture only if we're not profiling per-kernel (profiling mode syncs per dispatch)
        if !nested_capture && !profiling_per_kernel {
            foundry.end_capture()?;
        }

        Ok(())
    }

    /// Run a single forward step by executing DSL steps (uncompiled/interpreted).
    ///
    /// Unlike `forward()` which uses pre-compiled steps, this method executes the
    /// original `Step::execute()` method on each step in `spec.architecture.forward`.
    /// This allows runtime modification of variables like `n_layers` via `bindings.set_global()`.
    pub fn forward_uncompiled(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        // Start a new command buffer for this forward pass (token)
        foundry.start_capture()?;

        for step in &self.spec.architecture.forward {
            if step.name() == "Sample" {
                continue;
            }
            step.execute(foundry, bindings)?;
        }

        // Commit and wait for the token to complete
        let cmd_buffer = foundry.end_capture()?;
        cmd_buffer.wait_until_completed();

        Ok(())
    }
}
