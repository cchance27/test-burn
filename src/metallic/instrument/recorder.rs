//! Tracing layer and exporter abstractions for metric collection.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::Serialize;
use tracing::{Event, Subscriber};
use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

use crate::metallic::instrument::event::MetricEvent;

/// Metric events enriched with tracing context metadata.
#[derive(Debug, Clone, Serialize)]
pub struct EnrichedMetricEvent {
    /// Timestamp captured at emission time.
    pub timestamp: DateTime<Utc>,
    /// Identifier of the active span, if any.
    pub span_id: Option<u64>,
    /// Identifier of the parent span.
    pub parent_span_id: Option<u64>,
    /// Name of the active span for ergonomic debugging.
    pub span_name: Option<String>,
    /// The structured metric event payload.
    pub event: MetricEvent,
}

/// Exporter trait implemented by concrete sinks for metric events.
pub trait MetricExporter: Send + Sync {
    /// Export a single enriched metric event.
    fn export(&self, event: &EnrichedMetricEvent);
}

/// Custom tracing layer responsible for routing metric events to exporters.
#[derive(Clone)]
pub struct MetricsLayer {
    exporters: Arc<Vec<Box<dyn MetricExporter>>>,
}

impl MetricsLayer {
    /// Construct a new metrics layer using the provided exporters.
    pub fn new(exporters: Vec<Box<dyn MetricExporter>>) -> Self {
        Self {
            exporters: Arc::new(exporters),
        }
    }

    fn dispatch(&self, event: &EnrichedMetricEvent) {
        for exporter in self.exporters.iter() {
            exporter.export(event);
        }
    }
}

impl<S> Layer<S> for MetricsLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        if event.metadata().target() != "metrics" {
            return;
        }

        let metric_json = {
            let mut visitor = MetricVisitor::default();
            event.record(&mut visitor);
            match visitor.metric_json {
                Some(value) => value,
                None => return,
            }
        };

        let metric_event: MetricEvent = match serde_json::from_str(&metric_json) {
            Ok(event) => event,
            Err(error) => {
                tracing::error!(
                    target: "instrument",
                    ?error,
                    "failed to deserialize metric event"
                );
                return;
            }
        };

        let span = ctx.lookup_current();
        let enriched = EnrichedMetricEvent {
            timestamp: Utc::now(),
            span_id: span.map(|s| s.id().into_u64()),
            parent_span_id: span.and_then(|s| s.parent()).map(|parent| parent.id().into_u64()),
            span_name: span.and_then(|s| s.metadata().map(|meta| meta.name().to_string())),
            event: metric_event,
        };

        self.dispatch(&enriched);
    }
}

#[derive(Default)]
struct MetricVisitor {
    metric_json: Option<String>,
}

impl tracing::field::Visit for MetricVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "metric" {
            self.metric_json = Some(format!("{:?}", value));
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "metric" {
            self.metric_json = Some(value.to_string());
        }
    }
}
