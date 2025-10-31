use std::{sync::mpsc, time::Duration};

use metallic_env::ENABLE_PROFILING_VAR;
use metallic_instrumentation::{event::MetricEvent, gpu_profiler::GpuProfiler, prelude::*};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice as _, MTLResourceOptions};
use tracing::subscriber;

use crate::{
    caching::ResourceCache, operation::{CommandBuffer, FillConstant}, tensor::{Dtype, F32Element, Tensor}
};

// Maintainers: run this test on Apple Silicon hardware before releasing.
#[test]
#[ignore]
fn batch_profiler_emits_individual_kernel_events() {
    let _guard = ENABLE_PROFILING_VAR.set_guard(true).unwrap();
    let _profiling_guard = AppConfig::force_enable_profiling_guard();
    let (sender, receiver) = mpsc::channel();
    let exporters: Vec<Box<dyn MetricExporter>> = vec![Box::new(ChannelExporter::new(sender))];
    let layer = MetricsLayer::new(exporters);
    let subscriber = tracing_subscriber::registry().with(layer);

    subscriber::with_default(subscriber, || {
        let _metric_bypass = MetricQueueBypassGuard::new();
        let device = MTLCreateSystemDefaultDevice().expect("default metal device");
        let queue = device.newCommandQueue().expect("command queue");

        let mut cache = ResourceCache::with_device(device.clone());
        let command_buffer = CommandBuffer::new(&queue).expect("command buffer");
        let profiler = GpuProfiler::attach(&command_buffer, true).expect("gpu profiler support");

        let element_count = 16usize;
        let byte_len = element_count * std::mem::size_of::<f32>();
        let buffer = device
            .newBufferWithLength_options(byte_len as _, MTLResourceOptions::StorageModeShared)
            .expect("shared buffer");

        let tensor = Tensor::<F32Element>::from_existing_buffer(buffer, vec![element_count], Dtype::F32, &device, &queue, 0, true)
            .expect("tensor from buffer");

        let op_a = FillConstant {
            dst: tensor.clone(),
            value: 0.0,
            ones_pipeline: None,
        };
        let op_b = FillConstant {
            dst: tensor.clone(),
            value: 0.0,
            ones_pipeline: None,
        };

        let ops: [&dyn crate::operation::Operation; 2] = [&op_a, &op_b];
        command_buffer.record_batch(&ops, &mut cache).expect("record batch of ops");
        command_buffer.commit();
        command_buffer.wait();
        drop(profiler);

        // Expect two distinct GPU events for the two FillConstantZero kernels
        let mut gpu_events = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while gpu_events.len() < 2 {
            let now = std::time::Instant::now();
            if now >= deadline {
                panic!("metric event: Timeout");
            }
            let wait = std::cmp::min(Duration::from_millis(500), deadline.saturating_duration_since(now));
            if let Ok(enriched) = receiver.recv_timeout(wait)
                && let MetricEvent::GpuOpCompleted {
                    op_name,
                    backend,
                    duration_us,
                } = enriched.event
                && backend == "Metal"
                && op_name.starts_with("FillConstantZero")
            {
                gpu_events.push((op_name, duration_us));
            }
        }
        assert_eq!(gpu_events.len(), 2, "expected two kernel events");
        let mut names: Vec<_> = gpu_events.iter().map(|(name, _)| name.clone()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 2, "kernel event names must remain distinct");
        for name in &names {
            assert!(
                name.starts_with("FillConstantZero@"),
                "unexpected kernel name for FillConstantZero: {name}"
            );
            assert!(!name.contains('#'), "kernel names should be de-duplicated before emission: {name}");
        }
        for (_, duration) in &gpu_events {
            assert!(*duration > 0, "duration must be non-zero");
        }
    });
}

#[test]
fn record_batch_edge_cases() {
    let device = MTLCreateSystemDefaultDevice().expect("default metal device");
    let queue = device.newCommandQueue().expect("command queue");

    let mut cache = ResourceCache::with_device(device.clone());
    let command_buffer = CommandBuffer::new(&queue).expect("command buffer");

    // Empty batch is a no-op
    let empty_ops: [&dyn crate::operation::Operation; 0] = [];
    command_buffer
        .record_batch(&empty_ops, &mut cache)
        .expect("empty batch should be ok");

    // Recording after an empty batch still works
    let element_count = 4usize;
    let byte_len = element_count * std::mem::size_of::<f32>();
    let buffer = device
        .newBufferWithLength_options(byte_len as _, MTLResourceOptions::StorageModeShared)
        .expect("shared buffer");
    let tensor = Tensor::<F32Element>::from_existing_buffer(buffer, vec![element_count], Dtype::F32, &device, &queue, 0, true)
        .expect("tensor from buffer");

    let op = FillConstant {
        dst: tensor,
        value: 0.0,
        ones_pipeline: None,
    };
    command_buffer.record(&op, &mut cache).expect("record after empty batch");

    // Commit and wait, then ensure recording on a committed buffer errors
    command_buffer.commit();
    command_buffer.wait();

    let err = command_buffer
        .record_batch(&empty_ops, &mut cache)
        .expect_err("recording on committed buffer should fail");
    match err {
        crate::error::MetalError::InvalidOperation(msg) => {
            assert!(msg.contains("committed"))
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
