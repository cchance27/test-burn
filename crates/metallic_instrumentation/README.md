# Metallic Instrumentation

This crate provides a unified, extensible, and ergonomic instrumentation system for the Metallic project, built upon the `tracing` ecosystem. It is designed to capture structured, type-safe metrics for application behavior, resource usage, and GPU performance on Apple Metal.

## Core Concepts

The system is founded on a few key principles:

1.  **`tracing` as the Foundation**: We use `tracing` for structured logging and asynchronous context propagation (spans). Spans are used to define logical hierarchies, allowing metrics to be automatically nested and correlated.

2.  **Structured, Type-Safe Metrics**: All metrics are defined as variants of the `MetricEvent` enum. This ensures a canonical, machine-readable, and extensible schema for all instrumentation data.

3.  **Pluggable Exporters**: A `MetricExporter` trait and a custom `tracing::Layer` (`MetricsLayer`) allow metric events to be dispatched to multiple backends simultaneously. You can log to a file, the console, and an in-process channel all at once.

4.  **Centralized Configuration**: A global `AppConfig` struct, configured via environment variables, acts as the single source of truth for logging levels and exporter settings.

5.  **High-Performance GPU Profiling**: A dedicated `GpuProfiler` provides accurate, per-operation timing for Metal kernels within a single `MTLCommandBuffer`, encapsulating all `unsafe` Objective-C interop.

## Features

-   **Hierarchical Metrics**: Automatically associate metrics with their parent operations using `tracing` spans.
-   **GPU Kernel Timing**: Measure the precise execution time of individual Metal kernels.
-   **Multiple Exporters**:
    -   `JsonlExporter`: Persists metrics as a stream of JSON objects to a file.
    -   `ConsoleExporter`: Prints metrics to `stdout` for easy debugging.
    -   `ChannelExporter`: Sends metrics over a standard `mpsc::channel` for real-time, in-process consumption (e.g., by a UI).
-   **Environment-Based Configuration**: Easily configure logging and metrics without changing code.
-   **Ergonomic API**: Use simple macros (`record_metric!`) and RAII guards for clean and safe instrumentation.
-   **Latency-Oriented GPU Command Buffers**: When latency emission is enabled (the default), every kernel executes in its own
    Metal command buffer so we can capture accurate scheduling and execution timings even without GPU counter support.

## Getting Started

### 1. Initialization

To enable instrumentation, you must initialize a `tracing` subscriber with the `MetricsLayer`. This is typically done once at application startup.

```rust
use metallic_instrumentation::prelude::*;
use std::sync::mpsc;

fn main() {
    // 1. (Optional) Create exporters based on configuration or need.
    let (sender, receiver) = mpsc::channel();
    let channel_exporter = Box::new(ChannelExporter::new(sender));
    let console_exporter = Box::new(ConsoleExporter::new());

    let exporters: Vec<Box<dyn MetricExporter>> = vec![
        channel_exporter,
        console_exporter,
    ];

    // 2. Create the MetricsLayer.
    let metrics_layer = MetricsLayer::new(exporters);

    // 3. Create a tracing subscriber and set it as the global default.
    let subscriber = tracing_subscriber::registry().with(metrics_layer);
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting global default subscriber failed");

    // Your application logic here...
}
```

### 2. Emitting Metrics

Use the `record_metric!` macro to emit `MetricEvent` instances.

```rust
use metallic_instrumentation::prelude::*;

// This code assumes a subscriber has been initialized.
fn do_work() {
    record_metric!(MetricEvent::ResourceCacheAccess {
        cache_key: "model_weights".to_string(),
        hit: true,
        bytes: 1024 * 1024,
    });
}
```

### 3. Hierarchical Spans

Use `tracing::info_span!` to create nested scopes. Any metrics recorded within a span will be automatically associated with it and its parents.

```rust
use metallic_instrumentation::prelude::*;
use tracing::info_span;

fn run_transformer_block() {
    let _block_span = info_span!("transformer_block", block_id = 5).entered();

    // This metric will be associated with the "transformer_block" span.
    record_metric!(MetricEvent::GpuKernelDispatched {
        kernel_name: "attention_matmul".to_string(),
        op_name: "attention_matmul_op".to_string(),
        thread_groups: (16, 1, 1),
    });
}
```

## GPU Profiling

The `GpuProfiler` is designed to measure the execution time of kernels within a `MTLCommandBuffer`.

### Usage

1.  **Attach**: Create a `GpuProfiler` by attaching it to a command buffer. This automatically registers a completion handler.
2.  **Profile**: Use `GpuProfiler::profile_compute` or `GpuProfiler::profile_blit` to get a RAII scope guard. The guard automatically samples the GPU timestamp counters when it's created and when it's dropped.
3.  **Commit**: When the command buffer is committed and completes execution, the profiler calculates the duration of each scope and emits a `GpuOpCompleted` metric.

```rust
use metallic_instrumentation::{gpu_profiler::GpuProfiler, prelude::*};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLCommandQueue};
// Assuming you have a wrapper for CommandBuffer that implements ProfiledCommandBuffer
// and a way to get encoders.

fn profile_gpu_work() {
    let device = MTLCreateSystemDefaultDevice().unwrap();
    let queue = device.newCommandQueue().unwrap();

    // Create your command buffer wrapper
    let mut command_buffer = MyCommandBuffer::new(&queue).unwrap();

    // 1. Attach the profiler
    let profiler = GpuProfiler::attach(&command_buffer).expect("Profiler should attach");

    // 2. Get an encoder and profile a scope of work
    let encoder = command_buffer.new_compute_command_encoder();
    if let Some(scope) = GpuProfiler::profile_compute(
        command_buffer.raw(),
        encoder.raw(),
        "MyKernel".to_string(),
        "Metal".to_string(),
    ) {
        // ... dispatch your kernel ...
        
        // The scope is automatically finished when `scope` is dropped.
    }
    encoder.end_encoding();

    // 3. Commit the work
    command_buffer.commit();
    command_buffer.wait(); // Wait for completion to receive metrics

    // A `GpuOpCompleted` metric for "MyKernel" will now be emitted.
}
```

## Configuration

The instrumentation system is configured via environment variables:

-   **`METALLIC_LOG_LEVEL`**: Sets the minimum level for logs.
    -   Values: `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`
    -   Default: `INFO`
-   **`METALLIC_METRICS_CONSOLE`**: Enables or disables the `ConsoleExporter`.
    -   Values: `true`, `false`
    -   Default: `false`
-   **`METALLIC_METRICS_JSONL_PATH`**: If set, enables the `JsonlExporter` and writes metrics to the specified file path.
    -   Example: `/tmp/metrics.jsonl`
-   **`METALLIC_EMIT_LATENCY`**: Controls whether GPU kernels execute in dedicated command buffers to
    surface precise `kernelStartTime`/`GPUEndTime` measurements. Defaults to `true`. Set to `false` to
    reuse command buffers when prioritising throughput over latency observability.

`AppConfig::get_or_init_from_env()` can be called at startup to populate the global configuration from
these variables. Consumers such as the Metal `Context` automatically use this helper, so setting the
environment is typically sufficient; explicit initialisation is only required when customisation beyond
environment variables is desired.

## Extending the System

### Adding a New Metric Type

1.  Add a new variant to the `MetricEvent` enum in `src/event.rs`.
2.  Use the `record_metric!` macro with your new event type where needed.

### Creating a Custom Exporter

1.  Create a struct that implements the `MetricExporter` trait.
2.  Implement the `export` method, which defines how to handle an incoming `EnrichedMetricEvent`.
3.  Add an instance of your new exporter to the list passed to `MetricsLayer::new()` during initialization.

```rust
use metallic_instrumentation::recorder::{EnrichedMetricEvent, MetricExporter};

// A simple custom exporter that just prints the operation name.
struct OpNameExporter;

impl MetricExporter for OpNameExporter {
    fn export(&self, event: &EnrichedMetricEvent) {
        match &event.event {
            MetricEvent::GpuOpCompleted { op_name, .. } => {
                println!("Completed GPU Op: {}", op_name);
            }
            _ => {}
        }
    }
}
```
