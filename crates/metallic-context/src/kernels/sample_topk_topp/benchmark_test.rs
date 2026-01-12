use std::{
    fs, sync::mpsc::{self, RecvTimeoutError}, time::{Duration, Instant}
};

use metallic_instrumentation::{event::MetricEvent, prelude::*};
use rustc_hash::FxHashSet;
use serde::Serialize;
use serial_test::serial;

use crate::{
    Context, F16Element, MetalError, SamplerBuffers, Tensor, TensorInit, TensorStorage, generation::{gpu_sample_top_k_top_p, sample_top_k_top_p}, kernels::elemwise_add::ElemwiseAddOp
};

fn create_test_logits<T: crate::TensorElement>(ctx: &mut Context<T>, size: usize) -> Result<Tensor<T>, MetalError> {
    let mut logits_data = Vec::with_capacity(size);
    for i in 0..size {
        let val = (size - i) as f32 * 0.1;
        logits_data.push(T::from_f32(val));
    }

    Tensor::new(vec![size], TensorStorage::Pooled(ctx), TensorInit::CopyFrom(&logits_data))
}

#[derive(Debug, Clone, Serialize, Default)]
struct DurationStats {
    avg_us: f64,
    min_us: f64,
    max_us: f64,
    p50_us: f64,
    p95_us: f64,
    stddev_us: f64,
}

impl DurationStats {
    fn from_slice(samples: &[Duration]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let mut micros: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        micros.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = micros.len();
        let sum: f64 = micros.iter().sum();
        let mean = sum / len as f64;
        let variance = if len > 1 {
            let ss: f64 = micros
                .iter()
                .map(|v| {
                    let diff = *v - mean;
                    diff * diff
                })
                .sum();
            ss / (len as f64 - 1.0)
        } else {
            0.0
        };

        let percentile = |p: f64| -> f64 {
            if len == 1 {
                return micros[0];
            }
            let rank = p * (len as f64 - 1.0);
            let lower = rank.floor() as usize;
            let upper = rank.ceil() as usize;
            if lower == upper {
                micros[lower]
            } else {
                let weight = rank - lower as f64;
                micros[lower] * (1.0 - weight) + micros[upper] * weight
            }
        };

        Self {
            avg_us: mean,
            min_us: *micros.first().unwrap(),
            max_us: *micros.last().unwrap(),
            p50_us: percentile(0.5),
            p95_us: percentile(0.95),
            stddev_us: variance.max(0.0).sqrt(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct KernelDispatchRecord {
    kernel: String,
    op_name: String,
    thread_groups: (u32, u32, u32),
}

#[derive(Debug, Clone, Serialize)]
struct FailedGpuConfig {
    threads_per_tg: usize,
    per_thread_m: u32,
    error: String,
}

#[derive(Debug, Serialize)]
struct BenchmarkSnapshot {
    vocab_size: usize,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    cpu_only: DurationStats,
    sync_only: DurationStats,
    cpu_with_sync: DurationStats,
    gpu: DurationStats,
    sync_overhead_us: f64,
    cpu_vs_gpu_speedup: f64,
    cpu_only_vs_gpu_speedup: f64,
    cpu_only_iterations: usize,
    sync_iterations: usize,
    cpu_iterations: usize,
    gpu_iterations: usize,
    metrics_jsonl_path: String,
    dispatch_records: Vec<KernelDispatchRecord>,
    failed_gpu_configs: Vec<FailedGpuConfig>,
}

#[test]
#[serial]
#[ignore]
fn benchmark_cpu_vs_gpu_sampling() -> Result<(), MetalError> {
    let _profiling_env = ENABLE_PROFILING_VAR.set_guard(true).unwrap();
    reset_app_config_for_tests();
    let _profiling_guard = AppConfig::force_enable_profiling_guard();

    let metrics_path = {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        std::env::temp_dir().join(format!("sample_topk_topp_benchmark_{}_{}.jsonl", std::process::id(), timestamp,))
    };
    let metrics_path_guard = METRICS_JSONL_PATH_VAR.set_guard(metrics_path.clone()).unwrap();

    let (sender, receiver) = mpsc::channel();
    let mut exporters: Vec<Box<dyn MetricExporter>> =
        vec![Box::new(JsonlExporter::new(&metrics_path).expect("jsonl exporter for benchmark"))];
    exporters.push(Box::new(ChannelExporter::new(sender)));
    let layer = MetricsLayer::new(exporters);
    let subscriber = tracing_subscriber::registry().with(layer);

    let metrics_path_display = metrics_path.display().to_string();

    let bench_result = subscriber::with_default(subscriber, move || -> Result<(), MetalError> {
        let _metric_bypass = MetricQueueBypassGuard::new();
        let mut ctx = Context::new()?;
        ctx.force_enable_profiling_for_tests();

        struct EnvVarGuard {
            key: &'static str,
            previous: Option<String>,
        }

        impl EnvVarGuard {
            fn set(key: &'static str, value: String) -> Self {
                let previous = std::env::var(key).ok();
                unsafe {
                    std::env::set_var(key, value);
                }
                Self { key, previous }
            }
        }

        impl Drop for EnvVarGuard {
            fn drop(&mut self) {
                match &self.previous {
                    Some(value) => unsafe { std::env::set_var(self.key, value) },
                    None => unsafe { std::env::remove_var(self.key) },
                }
            }
        }

        let vocab_size = 100_000usize;
        let k = 50usize;
        let top_p = 0.9f32;
        let temperature = 0.8f32;

        println!("Starting CPU-only sampling benchmark (no GPU operations)...");
        let cpu_only_iterations = 1000;
        let mut cpu_only_times = Vec::with_capacity(cpu_only_iterations);
        let mut logits_data = Vec::<half::f16>::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let val = (vocab_size - i) as f32 * 0.1;
            logits_data.push(half::f16::from_f32(val));
        }

        for _ in 0..cpu_only_iterations {
            let start = Instant::now();
            let mut sampler_buffers = SamplerBuffers::default();
            let _cpu_token = sample_top_k_top_p::<F16Element>(&logits_data, k, top_p, temperature, &mut sampler_buffers) as u32;
            cpu_only_times.push(start.elapsed());
        }
        let cpu_only_stats = DurationStats::from_slice(&cpu_only_times);

        let base_logits = create_test_logits::<F16Element>(&mut ctx, vocab_size)?;
        let ones = Tensor::ones(vec![vocab_size], &mut ctx)?;
        let mut sync_tensor = ctx.call::<ElemwiseAddOp>((base_logits.clone(), ones.clone()), None)?;
        ctx.synchronize();

        println!("Measuring GPU->CPU sync times...");
        let sync_iterations = 1000;
        let mut sync_times = Vec::with_capacity(sync_iterations);
        for _ in 0..sync_iterations {
            sync_tensor = ctx.call::<ElemwiseAddOp>((sync_tensor.clone(), ones.clone()), None)?;
            ctx.synchronize();

            let sync_start = Instant::now();
            let _ = sync_tensor.as_slice();
            sync_times.push(sync_start.elapsed());
        }
        let sync_stats = DurationStats::from_slice(&sync_times);

        println!("Starting CPU sampling benchmark (including GPU->CPU sync)...");
        let cpu_iterations = 1000;
        let mut cpu_total_times = Vec::with_capacity(cpu_iterations);
        let cpu_logits = create_test_logits::<F16Element>(&mut ctx, vocab_size)?;
        for i in 0..cpu_iterations {
            let staged = ctx.call::<ElemwiseAddOp>((cpu_logits.clone(), ones.clone()), None)?;
            ctx.synchronize();
            let start = Instant::now();
            let logits_slice = staged.as_slice();
            let vocab_logits = &logits_slice[..vocab_size];
            let mut sampler_buffers = SamplerBuffers::default();
            let _cpu_token = sample_top_k_top_p::<F16Element>(vocab_logits, k, top_p, temperature, &mut sampler_buffers) as u32;
            cpu_total_times.push(start.elapsed());

            if i % 25 == 0 {
                ctx.reset_pool();
            }
        }
        let cpu_total_stats = DurationStats::from_slice(&cpu_total_times);

        println!("Starting GPU sampling benchmark (no GPU->CPU sync for logits)...");
        let max_threads_per_tg = ctx.max_threads_per_threadgroup().max(128);
        let candidate_threadgroups = [128usize, 192, 256, 320, 352, 384, 448, 512, 640, 768, 896, 1024];

        let mut sweep_configs = Vec::new();
        for &threads_per_tg in &candidate_threadgroups {
            if threads_per_tg > max_threads_per_tg {
                continue;
            }

            let (min_m, max_m) = match threads_per_tg {
                0..=128 => (2u32, 8u32),
                129..=192 => (4, 8),
                193..=256 => (2, 8),
                257..=320 => (4, 8),
                321..=352 => (4, 8),
                353..=384 => (4, 7),
                _ => (4, 6),
            };

            for per_thread_m in min_m..=max_m {
                sweep_configs.push((threads_per_tg, per_thread_m));
            }
        }

        if sweep_configs.is_empty() {
            sweep_configs.extend([(128usize, 2u32), (128, 3), (128, 4)]);
        }

        let gpu_iterations = 100;
        let gpu_logits = create_test_logits::<F16Element>(&mut ctx, vocab_size)?;
        let mut best_gpu_stats = None;
        let mut best_config = None;
        let mut failed_configs = Vec::new();

        for (cfg_tptg, cfg_m) in sweep_configs.into_iter() {
            let config_result = (|| -> Result<DurationStats, MetalError> {
                let _tptg_guard = EnvVarGuard::set("METALLIC_SAMPLE_TPTG", cfg_tptg.to_string());
                let _m_guard = EnvVarGuard::set("METALLIC_SAMPLE_PER_THREAD_M", cfg_m.to_string());

                let mut gpu_times = Vec::with_capacity(gpu_iterations);
                for i in 0..gpu_iterations {
                    let staged = ctx.call::<ElemwiseAddOp>((gpu_logits.clone(), ones.clone()), None)?;
                    ctx.synchronize();

                    let start = Instant::now();
                    let _gpu_token = ctx.with_gpu_scope("sample_topk_topp_benchmark", |ctx| {
                        gpu_sample_top_k_top_p::<F16Element>(&staged, vocab_size, k, top_p, temperature, None, ctx)
                    })?;
                    gpu_times.push(start.elapsed());

                    if i % 25 == 0 {
                        ctx.reset_pool();
                    }
                }

                Ok(DurationStats::from_slice(&gpu_times))
            })();

            match config_result {
                Ok(stats) => {
                    println!(
                        "GPU config tptg={} per_thread_m={} avg={:.3}µs p95={:.3}µs stddev={:.3}µs",
                        cfg_tptg, cfg_m, stats.avg_us, stats.p95_us, stats.stddev_us
                    );

                    println!(
                        "GPU config JSON: {}",
                        serde_json::to_string(&serde_json::json!({
                            "threads_per_tg": cfg_tptg,
                            "per_thread_m": cfg_m,
                            "stats": {
                                "avg_us": stats.avg_us,
                                "min_us": stats.min_us,
                                "max_us": stats.max_us,
                                "p50_us": stats.p50_us,
                                "p95_us": stats.p95_us,
                                "stddev_us": stats.stddev_us,
                            }
                        }))
                        .unwrap()
                    );

                    if best_gpu_stats
                        .as_ref()
                        .map(|best: &DurationStats| stats.avg_us < best.avg_us)
                        .unwrap_or(true)
                    {
                        best_gpu_stats = Some(stats.clone());
                        best_config = Some((cfg_tptg, cfg_m));
                    }
                }
                Err(err) => {
                    println!("GPU config tptg={} per_thread_m={} failed: {err}", cfg_tptg, cfg_m);
                    failed_configs.push(FailedGpuConfig {
                        threads_per_tg: cfg_tptg,
                        per_thread_m: cfg_m,
                        error: err.to_string(),
                    });
                }
            }

            ctx.reset_pool();
        }

        let best_gpu_stats =
            best_gpu_stats.ok_or_else(|| MetalError::OperationFailed("GPU sweep returned no successful configurations".to_string()))?;
        let (best_tptg, best_m) = best_config.expect("best config missing");
        println!(
            "Best GPU config tptg={} per_thread_m={} avg={:.3}µs",
            best_tptg, best_m, best_gpu_stats.avg_us
        );

        let gpu_stats = best_gpu_stats;

        ctx.synchronize();
        ctx.reset_pool();

        println!("CPU Sampling (CPU-only baseline) - Average time: {:.3}µs", cpu_only_stats.avg_us);
        println!("GPU->CPU sync only - Average time: {:.3}µs", sync_stats.avg_us);
        println!("CPU Sampling (with GPU->CPU sync) - Average time: {:.3}µs", cpu_total_stats.avg_us);
        println!(
            "GPU Sampling (no GPU->CPU sync for logits) - Average time: {:.3}µs",
            gpu_stats.avg_us
        );

        let sync_overhead_us = sync_stats.avg_us;
        let cpu_vs_gpu_speedup = if gpu_stats.avg_us > 0.0 {
            cpu_total_stats.avg_us / gpu_stats.avg_us
        } else {
            0.0
        };
        let cpu_only_vs_gpu_speedup = if gpu_stats.avg_us > 0.0 {
            cpu_only_stats.avg_us / gpu_stats.avg_us
        } else {
            0.0
        };

        println!("Estimated GPU->CPU sync overhead: {:.2}µs", sync_overhead_us);
        println!(
            "GPU sampling is {:.2}x faster than CPU sampling (including sync overhead)",
            cpu_vs_gpu_speedup
        );
        println!(
            "GPU sampling is {:.2}x faster than CPU sampling (CPU-only baseline)",
            cpu_only_vs_gpu_speedup
        );

        let mut metric_events = Vec::new();
        let drain_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match receiver.recv_timeout(Duration::from_millis(50)) {
                Ok(event) => metric_events.push(event),
                Err(RecvTimeoutError::Timeout) => {
                    if Instant::now() >= drain_deadline {
                        break;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        let mut dispatch_records = Vec::new();
        let mut seen_dispatch = FxHashSet::default();
        let mut partial_count = 0usize;
        let mut merge_count = 0usize;
        let mut fused_count = 0usize;

        for event in &metric_events {
            match &event.event {
                MetricEvent::GpuKernelDispatched {
                    kernel_name,
                    op_name,
                    thread_groups,
                } => {
                    if kernel_name.starts_with("sample_topk") && seen_dispatch.insert((kernel_name.clone(), op_name.clone())) {
                        dispatch_records.push(KernelDispatchRecord {
                            kernel: kernel_name.clone(),
                            op_name: op_name.clone(),
                            thread_groups: *thread_groups,
                        });
                    }
                }
                MetricEvent::GpuOpCompleted { op_name, .. } => {
                    if op_name.contains("sample_topk_partials") {
                        partial_count += 1;
                    } else if op_name.contains("sample_topk_merge_and_sample") {
                        merge_count += 1;
                    } else if op_name.contains("sample_topk_fused") {
                        fused_count += 1;
                    }
                }
                _ => {}
            }
        }

        assert!(
            (partial_count > 0 && merge_count > 0) || fused_count > 0,
            "expected sample_topk kernels to emit GPU completion metrics"
        );

        let snapshot = BenchmarkSnapshot {
            vocab_size,
            top_k: k,
            top_p,
            temperature,
            cpu_only: cpu_only_stats.clone(),
            sync_only: sync_stats.clone(),
            cpu_with_sync: cpu_total_stats.clone(),
            gpu: gpu_stats.clone(),
            sync_overhead_us,
            cpu_vs_gpu_speedup,
            cpu_only_vs_gpu_speedup,
            cpu_only_iterations,
            sync_iterations,
            cpu_iterations,
            gpu_iterations,
            metrics_jsonl_path: metrics_path_display,
            dispatch_records,
            failed_gpu_configs: failed_configs,
        };

        println!("Benchmark snapshot (JSON):\n{}", serde_json::to_string_pretty(&snapshot).unwrap());

        Ok(())
    });

    drop(metrics_path_guard);

    bench_result?;

    if let Err(err) = fs::remove_file(&metrics_path) {
        eprintln!(
            "Warning: failed to remove benchmark metrics file {}: {err:?}",
            metrics_path.display()
        );
    }

    Ok(())
}

#[test]
fn test_logits_download_overhead_simulation() {
    use std::time::Duration;

    let vocab_size = 150000;
    let element_size = std::mem::size_of::<f32>();
    let tensor_size_bytes = vocab_size * element_size;
    let gbps_bandwidth = 50.0;

    let sync_time_estimate = (tensor_size_bytes as f64) / (gbps_bandwidth * 1_000_000_000.0);
    let sync_time = Duration::from_secs_f64(sync_time_estimate);

    println!(
        "Logits tensor size: {} bytes (~{:.1} KB)",
        tensor_size_bytes,
        tensor_size_bytes as f64 / 1024.0
    );
    println!("Estimated GPU->CPU sync time: {:?}", sync_time);
    println!("This overhead is eliminated with GPU-based sampling!");
}
