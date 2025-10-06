use crate::metallic::instrument::prelude::*;

use std::sync::mpsc;
use std::time::Duration;

#[test]
fn metrics_layer_enriches_span_context() {
    let (sender, receiver) = mpsc::channel();
    let exporters: Vec<Box<dyn MetricExporter>> = vec![Box::new(ChannelExporter::new(sender))];
    let layer = MetricsLayer::new(exporters);
    let subscriber = tracing_subscriber::registry().with(layer);

    let metric_event = MetricEvent::GpuKernelDispatched {
        kernel_name: "matmul_kernel".to_string(),
        op_name: "matmul_op".to_string(),
        thread_groups: (8, 8, 1),
    };

    let (parent_id, child_id) = subscriber::with_default(subscriber, || {
        let parent_span = info_span!("parent_span");
        let parent_id = parent_span.id().map(|id| id.into_u64()).expect("parent span should be active");
        let _parent_guard = parent_span.enter();

        let child_span = info_span!("kernel_span");
        let child_id = child_span.id().map(|id| id.into_u64()).expect("child span should be active");
        let _child_guard = child_span.enter();

        record_metric!(metric_event.clone());
        (Some(parent_id), Some(child_id))
    });

    let enriched = receiver.recv_timeout(Duration::from_secs(1)).expect("metric should be dispatched");

    assert_eq!(enriched.span_id, child_id);
    assert_eq!(enriched.parent_span_id, parent_id);
    assert_eq!(enriched.span_name.as_deref(), Some("kernel_span"));

    match enriched.event {
        MetricEvent::GpuKernelDispatched {
            ref kernel_name,
            ref op_name,
            thread_groups,
        } => {
            assert_eq!(kernel_name, "matmul_kernel");
            assert_eq!(op_name, "matmul_op");
            assert_eq!(thread_groups, (8, 8, 1));
        }
        other => panic!("expected gpu dispatch metric, got {other:?}"),
    }
}

#[test]
fn metrics_layer_ignores_non_metric_events() {
    let (sender, receiver) = mpsc::channel();
    let exporters: Vec<Box<dyn MetricExporter>> = vec![Box::new(ChannelExporter::new(sender))];
    let layer = MetricsLayer::new(exporters);
    let subscriber = tracing_subscriber::registry().with(layer);

    subscriber::with_default(subscriber, || {
        let span = info_span!("non_metric_span");
        let _guard = span.enter();
        info!("non-metric event should be ignored");
    });

    assert!(receiver.try_recv().is_err(), "channel should remain empty");
}
