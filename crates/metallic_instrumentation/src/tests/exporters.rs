use std::{
    sync::mpsc, time::{Duration, SystemTime, UNIX_EPOCH}
};

use crate::prelude::*;

#[test]
fn jsonl_exporter_writes_serialised_metrics() {
    let mut path = std::env::temp_dir();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_nanos();
    path.push(format!("metallic_metrics_test_{}.jsonl", unique));

    let event = EnrichedMetricEvent {
        timestamp: Utc::now(),
        span_id: Some(42),
        parent_span_id: Some(24),
        span_name: Some("jsonl_span".to_string()),
        event: MetricEvent::GpuKernelDispatched {
            kernel_name: "jsonl_kernel".to_string(),
            op_name: "jsonl_op".to_string(),
            thread_groups: (4, 2, 1),
        },
    };

    {
        let exporter = JsonlExporter::new(&path).expect("jsonl exporter should open file");
        exporter.export(&event);
    }

    let contents = std::fs::read_to_string(&path).expect("jsonl exporter should write file");
    let expected = serde_json::to_string(&event).expect("event should serialise");
    assert_eq!(contents.trim_end_matches('\n'), expected);

    std::fs::remove_file(&path).expect("temporary jsonl file should be removable");
}

#[test]
fn channel_exporter_clones_events() {
    let (sender, receiver) = mpsc::channel();
    let exporter = ChannelExporter::new(sender);

    let mut event = EnrichedMetricEvent {
        timestamp: Utc::now(),
        span_id: Some(7),
        parent_span_id: Some(3),
        span_name: Some("channel".to_string()),
        event: MetricEvent::GpuKernelDispatched {
            kernel_name: "channel_kernel".to_string(),
            op_name: "channel_op".to_string(),
            thread_groups: (1, 1, 1),
        },
    };

    exporter.export(&event);

    event.span_name = Some("mutated".to_string());
    event.event = MetricEvent::GpuOpCompleted {
        op_name: "channel_op".to_string(),
        backend: "TestBackend".to_string(),
        duration_us: 123,
        data: None,
    };

    let received = receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("channel should receive cloned event");

    assert_eq!(received.span_name.as_deref(), Some("channel"));
    assert_ne!(received.span_name, event.span_name);

    match received.event {
        MetricEvent::GpuKernelDispatched {
            kernel_name,
            op_name,
            thread_groups,
        } => {
            assert_eq!(kernel_name, "channel_kernel");
            assert_eq!(op_name, "channel_op");
            assert_eq!(thread_groups, (1, 1, 1));
        }
        other => panic!("expected cloned gpu dispatch event, got {other:?}"),
    }
}
