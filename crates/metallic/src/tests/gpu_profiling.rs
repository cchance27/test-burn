use metallic_instrumentation::{
    event::MetricEvent,
    gpu_profiler::GpuProfiler,
    prelude::*,
};

use crate::{
    operation::{CommandBuffer, FillConstant},
    resource_cache::ResourceCache,
    tensor::{Dtype, F32Element, Tensor},
};

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice as _, MTLResourceOptions};
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};
use tracing::subscriber;

// Maintainers: run this test on Apple Silicon hardware before releasing.
#[test]
fn gpu_profiler_emits_individual_kernel_events() {
    let (sender, receiver) = mpsc::channel();
    let exporters: Vec<Box<dyn MetricExporter>> = vec![Box::new(ChannelExporter::new(sender))];
    let layer = MetricsLayer::new(exporters);
    let subscriber = tracing_subscriber::registry().with(layer);

    subscriber::with_default(subscriber, || {
        let device = MTLCreateSystemDefaultDevice().expect("default metal device");
        let queue = device.newCommandQueue().expect("command queue");

        let mut cache = ResourceCache::with_device(device.clone());
        let mut command_buffer = CommandBuffer::new(&queue).expect("command buffer");
        let profiler = GpuProfiler::attach(&command_buffer).expect("gpu profiler support");

        let element_count = 16usize;
        let byte_len = element_count * std::mem::size_of::<f32>();
        let buffer = device
            .newBufferWithLength_options(byte_len as _, MTLResourceOptions::StorageModeShared)
            .expect("shared buffer");

        let tensor = Tensor::<F32Element>::from_existing_buffer(
            buffer,
            vec![element_count],
            Dtype::F32,
            &device,
            &queue,
            0,
            true,
        )
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

        command_buffer
            .record(&op_a, &mut cache)
            .expect("record first fill");
        command_buffer
            .record(&op_b, &mut cache)
            .expect("record second fill");

        let cpu_start = Instant::now();
        command_buffer.commit();
        command_buffer.wait();
        let total_us = cpu_start.elapsed().as_micros().max(1).try_into().unwrap_or(u64::MAX);
        drop(profiler);

        let mut gpu_events = Vec::new();
        while gpu_events.len() < 2 {
            let enriched = receiver.recv_timeout(Duration::from_secs(2)).expect("metric event");
            if let MetricEvent::GpuOpCompleted {
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
        assert!(gpu_events[0].0.ends_with("#0"));
        assert!(gpu_events[1].0.ends_with("#1"));
        for (_, duration) in &gpu_events {
            assert!(duration > &0, "duration must be non-zero");
        }

        let sum_us: u64 = gpu_events.iter().map(|(_, dur)| *dur).sum();
        assert_ne!(
            sum_us,
            total_us,
            "per-kernel durations must not collapse into a single total"
        );
    });
}
