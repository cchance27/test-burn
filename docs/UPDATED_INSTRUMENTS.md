# Unified Instrumentation and Logging Plan (Corrected)

## 1. Overview and Core Principles

This document outlines a comprehensive plan to refactor and unify the project's instrumentation, metrics, and logging systems. The new system is founded on these core principles:

1.  **`tracing` as the Foundation**: We will adopt the `tracing` crate for structured logging and asynchronous context (spans).

2.  **Structured, Type-Safe Metrics**: All metrics will be defined as variants of a single `MetricEvent` enum, ensuring a canonical, type-safe, and extensible schema.

3.  **Integration with Existing `objc2` Patterns**: We will upgrade the project's existing `MatMulDispatchTiming` mechanism. Instead of creating a new timer, we will adapt the current `msg_send!`-based counter sampling to feed into the new, unified `MetricEvent` system. This correctly handles per-operation timing within a single command buffer.

4.  **Pluggable Exporter System**: A `MetricExporter` trait and a custom `tracing` layer will allow metrics to be routed to multiple backends (JSONL, SQLite, console) based on configuration.

5.  **Centralized Configuration**: A single, static `AppConfig` struct will be the source of truth for all logging, metric, and feature-gating settings.

6.  **Ergonomic Developer Experience (DX)**: Simple, intention-revealing macros will provide a clean API for developers.

---

## 2. Proposed File Topology

The new instrumentation system will reside entirely within a new module in the `metallic` crate.

```
/src/metallic/
├── instrument/
│   ├── mod.rs          # Public API exports and module definitions.
│   ├── prelude.rs      # Convenience re-exports for easy `use`.
│   ├── config.rs       # `AppConfig` struct and `std::env` parsing.
│   ├── event.rs        # The canonical `MetricEvent` enum.
│   ├── recorder.rs     # The `MetricsLayer` and `MetricExporter` trait.
│   ├── exporters.rs    # Concrete exporter implementations (e.g., Jsonl).
│   └── macros.rs       # All developer-facing macros (`record_metric!`, etc.).
└── ... (other metallic modules)
```

---

## 3. Detailed Implementation

This plan **replaces** the previous `GpuTimer` proposal with a system that integrates with your existing patterns.

### `instrument/event.rs`

The `MetricEvent` enum will be the heart of the system.

```rust
//! The canonical, structured `MetricEvent` enum.

use serde::Serialize;

#[derive(Debug, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum MetricEvent {
    GpuKernelDispatched {
        kernel_name: String,
        /// A unique name for this specific operation instance.
        op_name: String,
        thread_groups: (u32, u32, u32),
    },
    /// This event is generated upon command buffer completion.
    GpuOpCompleted {
        op_name: String,
        /// The backend that executed the operation (e.g., "Mlx", "Mps")
        backend: String,
        duration_us: u64,
    },
    /// For timing internal kernels from frameworks like MPS.
    InternalKernelCompleted {
        parent_op_name: String,
        internal_kernel_name: String,
        duration_us: u64,
    },
    ResourceCacheAccess {
        cache_key: String,
        hit: bool,
        bytes: u64,
    },
}
```

### `instrument/macros.rs`

```rust
//! Developer-facing macros for clean and easy instrumentation.

#[macro_export]
macro_rules! record_metric {
    ($event:expr) => {
        // This check prevents serialization from impacting performance if
        // no metric recorders are active for this log level.
        if tracing::enabled!(target: "metrics", tracing::Level::INFO) {
            if let Ok(metric_json) = serde_json::to_string(&$event) {
                tracing::event!(
                    target: "metrics",
                    tracing::Level::INFO,
                    metric = %metric_json
                );
            }
        }
    };
}
```

### Other Files

The files `config.rs`, `recorder.rs`, `exporters.rs`, `mod.rs`, and `prelude.rs` should be implemented as detailed in the previous (now-obsolete) `UPDATED_INSTRUMENTS.md` file. The core change is the removal of `gpu_timer.rs` and the adaptation of the code that uses it.

--- 

## 4. GPU Performance Timing: The `GpuProfiler`

To provide accurate, per-operation GPU timing while completely encapsulating the underlying `objc2` and `msg_send!` logic, we will introduce a dedicated `GpuProfiler`.

This profiler will be the sole entry point for timing GPU work. It will manage the `MTLCounterSampleBuffer`, handle the `msg_send!` calls for sampling, and integrate with the completion handler to report metrics, thus replacing the old `MatMulDispatchTiming` mechanism entirely.

### A. Design and Responsibilities

1.  **Lifecycle**: One `GpuProfiler` instance will be created per `MTLCommandBuffer`.
2.  **Encapsulation**: It will contain the `MTLCounterSampleBuffer` and all state needed for timing. All `unsafe` code and `msg_send!` calls are hidden within its methods.
3.  **RAII for Custom Kernels**: It will provide a `timed_scope` method that returns a RAII guard. The guard's creation and destruction will trigger the `msg_send!` calls to sample the GPU counters, making timing your own kernels trivial.
4.  **MPS Internal Timing**: It will have a dedicated method for timing MPS operations, which internally uses your established `MpsRecorder` logic (via `objc` calls) to capture and report internal kernel timings.
5.  **Automated Reporting**: The `GpuProfiler` will attach a completion handler to the command buffer at creation time. This handler will automatically process the sample buffer and report all recorded timings via the `record_metric!` macro when the GPU work is finished.

### B. Implementation Sketch

**New File: `src/metallic/instrument/gpu_profiler.rs`**

```rust
//! A profiler for capturing per-operation GPU timings within a single command buffer.

use super::event::MetricEvent;
use crate::record_metric;
use metal::{CommandBufferRef, ComputeCommandEncoderRef, CounterSampleBuffer, Device};
use objc2::{msg_send, rc::Retained, runtime::Bool};
use std::cell::RefCell;
use std::sync::Arc;

// A record of a single timed operation.
struct GpuOpRecord {
    op_name: String,
    backend: String,
    start_index: u32,
    end_index: u32,
}

// The internal state of the profiler, shared with the completion handler.
struct ProfilerState {
    records: Vec<GpuOpRecord>,
    sample_buffer: Retained<CounterSampleBuffer>,
    sample_index: u32,
}

/// The public-facing profiler object.
pub struct GpuProfiler {
    // We use a RefCell because the methods that add records need mutable access,
    // but the profiler itself is captured by an Fn closure in the completion handler.
    state: Arc<RefCell<ProfilerState>>,
}

impl GpuProfiler {
    /// Creates a new profiler for the given command buffer and attaches the completion handler.
    pub fn new(device: &Device, cmdbuf: &CommandBufferRef) -> Self {
        let sample_buffer = device.new_counter_sample_buffer_with_descriptor(&...);
        let state = Arc::new(RefCell::new(ProfilerState {
            records: Vec::new(),
            sample_buffer: Retained::from_owned(sample_buffer),
            sample_index: 0,
        }));

        // Attach the completion handler that will report the metrics.
        let state_clone = state.clone();
        cmdbuf.add_completed_handler(move |cbuf| {
            let mut state = state_clone.borrow_mut();
            let data = state.sample_buffer.resolve_counters_with_range(0..state.sample_index).unwrap();
            // ... logic to resolve timestamps and calculate conversion factor ...

            for record in state.records.drain(..) {
                let start_ts = ...; // get from data
                let end_ts = ...;   // get from data
                let duration_us = ...; // calculate

                record_metric!(MetricEvent::GpuOpCompleted {
                    op_name: record.op_name,
                    backend: record.backend,
                    duration_us,
                });
            }
        });

        Self { state }
    }

    /// Creates a RAII guard that times a scope within a compute encoder.
    pub fn timed_scope<'a>(
        &'a self,
        encoder: &ComputeCommandEncoderRef,
        op_name: String,
        backend: String,
    ) -> TimedGpuScope<'a> {
        let mut state = self.state.borrow_mut();
        let start_index = state.sample_index;
        state.sample_index += 1;
        let end_index = state.sample_index;
        state.sample_index += 1;

        state.records.push(GpuOpRecord {
            op_name,
            backend,
            start_index,
            end_index,
        });

        // ENCAPSULATED: The msg_send! call is an implementation detail.
        unsafe {
            let _: () = msg_send![
                encoder,
                sampleCountersInBuffer: &*state.sample_buffer,
                atSampleIndex: start_index,
                withBarrier: Bool::YES
            ];
        }

        TimedGpuScope {
            encoder,
            state: &self.state,
            end_index,
        }
    }
    
    // Placeholder for the encapsulated MPS timing logic
    pub fn timed_mps_matmul(&self, /* ... mps params ... */) {
        // 1. Call your Objective-C MpsRecorder via msg_send!.
        // 2. The completion handler will need to be augmented to also
        //    retrieve results from the MpsRecorder and fire
        //    `MetricEvent::InternalKernelCompleted` events.
    }
}

/// RAII guard for a timed GPU scope. Its Drop implementation records the end sample.
#[must_use]
pub struct TimedGpuScope<'a> {
    encoder: &'a ComputeCommandEncoderRef,
    state: &'a Arc<RefCell<ProfilerState>>,
    end_index: u32,
}

impl<'a> Drop for TimedGpuScope<'a> {
    fn drop(&mut self) {
        let state = self.state.borrow();
        // ENCAPSULATED: The user of the guard doesn't see this.
        unsafe {
            let _: () = msg_send![
                self.encoder,
                sampleCountersInBuffer: &*state.sample_buffer,
                atSampleIndex: self.end_index,
                withBarrier: Bool::NO
            ];
        }
    }
}
```

### C. Clean, Encapsulated Usage

With the `GpuProfiler`, the application code becomes clean, safe, and completely free of `objc` details.

```rust
use metallic::instrument::gpu_profiler::GpuProfiler;
use metal::{Device, CommandQueue};

fn run_model_pass(device: &Device, queue: &CommandQueue) {
    let cmdbuf = queue.new_command_buffer();
    let encoder = cmdbuf.new_compute_command_encoder();

    // 1. Create the profiler. The completion handler is attached automatically.
    let profiler = GpuProfiler::new(device, cmdbuf);

    // 2. Time a custom kernel using the RAII guard.
    {
        let _timed_scope = profiler.timed_scope(encoder, "matmul_q_k".to_string(), "Mlx".to_string());
        // ... dispatch matmul_q_k kernel ...
    } // <-- Guard is dropped, end sample is recorded automatically.

    // 3. Time another kernel.
    {
        let _timed_scope = profiler.timed_scope(encoder, "softmax".to_string(), "Mlx".to_string());
        // ... dispatch softmax kernel ...
    } 

    // 4. Time a high-level MPS operation.
    // profiler.timed_mps_matmul(...);

    encoder.end_encoding();
    cmdbuf.commit();
    // cmdbuf completes, the handler runs, and all metrics are reported automatically.
}
```

This design achieves the desired encapsulation, providing a safe and ergonomic Rust API while reusing your proven, low-level timing mechanisms as an internal implementation detail.

---

## 5. Hierarchical Metrics: Capturing Nesting

A flat list of timings is of limited use. To understand performance, we need to see which operations are children of other operations (e.g., multiple `GpuOpCompleted` events nested within a single `TransformerBlock` scope). The `tracing` crate's span system is designed for exactly this.

### A. Strategy: Enriching Events with Span Context

We will leverage `tracing` to automatically associate metrics with their parent operations. The implementation requires no changes to the developer-facing macros, only to the backend `MetricsLayer`.

1.  **Spans Define Hierarchy**: Developers will use `tracing::span!` to define logical scopes. These spans can be nested to any depth.
2.  **`MetricsLayer` Extracts Context**: Our `MetricsLayer` will be updated to inspect the `tracing` context whenever it sees a metric event.
3.  **Events are Enriched**: The layer will extract the current span's ID and its parent's ID, adding them to the metric data before exporting. This provides the necessary information for a client-side tool to reconstruct the metric tree.

### B. Updated `recorder.rs` (`MetricsLayer`)

The `on_event` function will be modified to access the span registry.

```rust
// In src/metallic/instrument/recorder.rs

use tracing::span::{Id, Record};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;
use serde::Serialize;

// ... MetricVisitor struct ...

// New struct for the final, enriched output.
#[derive(Serialize)]
struct EnrichedMetricEvent<'a> {
    span_id: u64,
    parent_id: Option<u64>,
    #[serde(flatten)]
    event: &'a MetricEvent,
}

impl<S> Layer<S> for MetricsLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        if *event.metadata().target() != "metrics" {
            return;
        }

        // ... logic to extract the metric_json string ...
        let metric_json = ...;
        if let Ok(metric_event) = serde_json::from_str::<MetricEvent>(&metric_json) {
            // Find the current span and its parent in the registry.
            let current_span = ctx.current_span();
            let span_id = current_span.id().map(|id| id.into_u64()).unwrap_or(0);
            let parent_id = current_span.parent().map(|span| span.id().into_u64());

            let enriched_event = EnrichedMetricEvent {
                span_id,
                parent_id,
                event: &metric_event,
            };

            for exporter in &self.exporters {
                // The exporter now receives the enriched event.
                exporter.export(&enriched_event);
            }
        }
    }
}
```

### C. Example of Nested Usage and Output

Developers can now create nested scopes, and the metrics will be linked automatically.

**Usage with Nested Spans:**

```rust
use metallic::instrument::gpu_profiler::GpuProfiler;
use tracing::info_span;

fn run_full_generation(device: &Device, queue: &CommandQueue) {
    // 1. Create a top-level span for the entire generation task.
    let generation_span = info_span!("text_generation", model = "qwen2.5");
    let _gen_guard = generation_span.enter();

    for block_id in 0..24 {
        // 2. Create a nested span for each transformer block.
        let block_span = info_span!("transformer_block", block_id);
        let _block_guard = block_span.enter();

        let cmdbuf = queue.new_command_buffer();
        let encoder = cmdbuf.new_compute_command_encoder();
        let profiler = GpuProfiler::new(device, cmdbuf);

        // 3. Record a metric inside the nested span.
        {
            let _timed_scope = profiler.timed_scope(encoder, "attention_matmul".to_string(), "Mlx".to_string());
            // ... dispatch attention kernel ...
        }

        encoder.end_encoding();
        cmdbuf.commit();
        // ... wait for completion ...
    }
}
```

**Resulting JSONL Output:**

The `JsonlExporter` will now output structured data containing the hierarchy, which a UI can easily parse into a tree.

```json
{"span_id":2,"parent_id":1,"event":{"type":"GpuOpCompleted","data":{"op_name":"attention_matmul","backend":"Mlx","duration_us":152}}}
{"span_id":3,"parent_id":1,"event":{"type":"GpuOpCompleted","data":{"op_name":"attention_matmul","backend":"Mlx","duration_us":149}}}
{"span_id":4,"parent_id":1,"event":{"type":"GpuOpCompleted","data":{"op_name":"attention_matmul","backend":"Mlx","duration_us":155}}}
```

Here, span `1` would correspond to `text_generation`, and spans `2`, `3`, and `4` correspond to the `attention_matmul` operations inside three different `transformer_block` spans, all correctly parented under the main generation task.

This approach fully solves the nesting requirement in a clean, idiomatic way using the power of the `tracing` ecosystem.

---

## 6. In-Process Subscriptions: The `ChannelExporter`

For a consuming application (like a GUI or TUI) to display metrics in real-time, it needs a way to subscribe to the stream of events from within the same process. The ideal, idiomatic Rust solution for this is a channel.

We will implement a `ChannelExporter` that sends every metric event into a sender-half of a channel provided by the consuming application.

### A. Strategy: In-Process Messaging via Channels

1.  **`ChannelExporter`**: A new exporter will be created that holds the `Sender` end of a standard `std::sync::mpsc` channel.
2.  **Application Creates Channel**: The consuming application will create the channel and pass the `Sender` to the instrumentation system during initialization.
3.  **Real-time Receiving**: The application holds onto the `Receiver` end, which it can poll in its main loop (e.g., using `try_recv()`) to get a non-blocking stream of metric events to update its UI.
4.  **Decoupled & Thread-Safe**: This pattern cleanly decouples the `metallic` library from the application's UI logic and is inherently thread-safe, as events are safely passed from the Metal completion handler thread to the application's main/UI thread.

### B. Implementation Sketch

**Updated `exporters.rs`:**

```rust
// In src/metallic/instrument/exporters.rs

// ... existing code ...
use std::sync::mpsc::Sender;

// The event struct needs to be cloneable and sendable across threads.
#[derive(Serialize, Clone)]
struct EnrichedMetricEvent<'a> { ... }

/// An exporter that sends metrics to a channel for in-process consumption.
pub struct ChannelExporter<T> {
    sender: Sender<T>,
}

impl<T> ChannelExporter<T> {
    pub fn new(sender: Sender<T>) -> Self {
        Self { sender }
    }
}

// We will have the exporter handle the enriched event directly.
impl<T: Send + Clone + 'static> MetricExporter for ChannelExporter<T> where T: From<EnrichedMetricEvent> {
    fn export(&self, event: &EnrichedMetricEvent) {
        // Attempt to send a clone of the event. If the receiver has been
        // dropped, this will fail gracefully.
        self.sender.send(event.clone().into()).ok();
    }
}
```

### C. Consuming Application Usage

This is how the application using the `metallic` library would subscribe to and display live metrics.

**Example Application (`main.rs` of the consuming app):**

```rust
use metallic::instrument::exporters::ChannelExporter;
use metallic::instrument::recorder::MetricsLayer;
use std::sync::mpsc;

fn main() {
    // 1. Application creates the channel.
    let (metric_sender, metric_receiver) = mpsc::channel();

    // 2. Application creates the ChannelExporter with its sender.
    let channel_exporter = Box::new(ChannelExporter::new(metric_sender));

    // 3. Initialize the instrumentation system with the channel exporter.
    let mut exporters: Vec<Box<dyn MetricExporter>> = vec![channel_exporter];
    // ... add other exporters like JsonlExporter if needed ...
    let metrics_layer = MetricsLayer::new(exporters);
    // ... initialize tracing with the metrics_layer ...

    // 4. Start the work that uses the metallic library.
    std::thread::spawn(move || {
        // ... run_full_generation() or other metallic functions ...
    });

    // 5. The application's main loop polls the receiver for live updates.
    loop {
        // Use `try_recv` for non-blocking polling in a UI loop.
        if let Ok(metric_event) = metric_receiver.try_recv() {
            // Update the UI with the live metric.
            println!("LIVE METRIC: {:?}", metric_event);
            // e.g., update_latency_widget(metric_event.duration_us);
        }
        
        // ... other UI work ...
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}
```

This channel-based approach provides a clean, safe, and highly performant way for your library to provide real-time data to its consumers without requiring any networking or complex dependencies.

---

## 7. Conclusion

By channelling both your hand-written kernel timings and your internal MPS kernel timings into `record_metric!`, all performance data ends up in the same structured, flexible, and extensible system. This achieves the goal of unification while respecting and reusing the critical, low-level work you have already successfully implemented.