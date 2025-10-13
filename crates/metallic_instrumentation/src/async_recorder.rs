//! Asynchronous metric recording system using lock-free queues for zero-overhead metric collection.

use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam::queue::SegQueue;
use tracing_subscriber::layer::SubscriberExt as _;

use crate::event::MetricEvent;
use crate::exporters::ChannelExporter;
use crate::recorder::{EnrichedMetricEvent, MetricExporter, MetricsLayer};

/// Lock-free queue for sending metrics from the recording thread to the processing thread.
pub type MetricQueue = Arc<SegQueue<MetricEvent>>;

/// Asynchronous metric recorder that processes metrics in a background thread.
pub struct AsyncMetricRecorder {
    /// Handle to the background processing thread
    _handle: JoinHandle<()>,
    /// Queue for sending metrics to the background thread
    pub queue: MetricQueue,
    /// Channel for receiving processed metrics in the main thread
    pub receiver: mpsc::Receiver<EnrichedMetricEvent>,
}

impl AsyncMetricRecorder {
    /// Create a new async metric recorder with the given exporters.
    pub fn new(exporters: Vec<Box<dyn MetricExporter>>) -> Self {
        let (sender, receiver) = mpsc::channel();
        let channel_exporter = Box::new(ChannelExporter::new(sender));
        let mut all_exporters = exporters;
        all_exporters.push(channel_exporter);

        let metrics_layer = MetricsLayer::new(all_exporters);
        let queue = Arc::new(SegQueue::new());

        // Create the background processing thread
        let queue_clone = queue.clone();
        let handle = thread::spawn(move || {
            Self::background_processor(&queue_clone, &metrics_layer);
        });

        Self {
            _handle: handle,
            queue,
            receiver,
        }
    }

    /// Background thread function that processes metrics from the queue.
    fn background_processor(queue: &MetricQueue, metrics_layer: &MetricsLayer) {
        // Create a dummy subscriber for the tracing layer to work with
        let _subscriber = tracing_subscriber::registry().with(metrics_layer.clone());

        while !shutdown_requested() {
            // Try to get metrics from the queue with a small timeout to avoid busy waiting
            if let Some(metric_event) = queue.pop() {
                // Process the metric event through the tracing system
                // We need to create a fake span context since we don't have access to the real one
                let enriched_event = Self::create_enriched_event(metric_event);

                // Dispatch to all exporters through the metrics layer
                // Note: This is a simplified approach - in a real implementation,
                // we'd want to properly integrate with the tracing system
                metrics_layer.dispatch(&enriched_event);
            } else {
                // Small sleep to avoid busy waiting
                thread::sleep(Duration::from_millis(1));
            }
        }
    }

    /// Create an enriched metric event from a basic metric event.
    /// This is a simplified version - in practice, we'd want to capture
    /// the actual span context from where the metric was recorded.
    fn create_enriched_event(event: MetricEvent) -> EnrichedMetricEvent {
        EnrichedMetricEvent {
            timestamp: chrono::Utc::now(),
            span_id: None, // Would be populated from actual span context
            parent_span_id: None,
            span_name: None, // Would be populated from actual span context
            event,
        }
    }
}

/// Global flag to signal shutdown to background threads.
static mut SHUTDOWN_REQUESTED: bool = false;

/// Check if shutdown has been requested.
fn shutdown_requested() -> bool {
    unsafe { SHUTDOWN_REQUESTED }
}

/// Request shutdown of all background threads.
pub fn request_shutdown() {
    unsafe {
        SHUTDOWN_REQUESTED = true;
    }
}
