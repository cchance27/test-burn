use metallic_instrumentation::{event::MetricEvent, gpu_profiler::GpuProfiler, prelude::*};
use metallic_env::ENABLE_PROFILING_VAR;

use crate::{
    context::Context,
    kernels::elemwise_add::ElemwiseAddOp,
    operation::{CommandBuffer, FillConstant},
    resource_cache::ResourceCache,
    tensor::{Dtype, F32Element, Tensor, TensorInit, TensorStorage},
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
        let mut command_buffer = CommandBuffer::new(&queue).expect("command buffer");
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

        command_buffer.record(&op_a, &mut cache).expect("record first fill");
        command_buffer.record(&op_b, &mut cache).expect("record second fill");

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

        let sum_us: u64 = gpu_events.iter().map(|(_, dur)| *dur).sum();
        assert_ne!(sum_us, total_us, "per-kernel durations must not collapse into a single total");
    });
}

// Maintainers: run this test on Apple Silicon hardware before releasing.
#[test]
fn context_call_attaches_gpu_profiler() {
    let _guard = ENABLE_PROFILING_VAR.set_guard(true).unwrap();
    reset_app_config_for_tests();
    let _profiling_guard = AppConfig::force_enable_profiling_guard();
    let (sender, receiver) = mpsc::channel();
    let exporters: Vec<Box<dyn MetricExporter>> = vec![Box::new(ChannelExporter::new(sender))];
    let layer = MetricsLayer::new(exporters);
    let subscriber = tracing_subscriber::registry().with(layer);

    subscriber::with_default(subscriber, || {
        let _metric_bypass = MetricQueueBypassGuard::new();
        let mut ctx = Context::<F32Element>::new().expect("metal context");
        ctx.force_enable_profiling_for_tests();

        let dims = vec![8usize];
        let mut a = Tensor::new(dims.clone(), TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("tensor a");
        let mut b = Tensor::new(dims.clone(), TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("tensor b");

        for (idx, value) in a.as_mut_slice().iter_mut().enumerate() {
            *value = idx as f32;
        }
        for (idx, value) in b.as_mut_slice().iter_mut().enumerate() {
            *value = (idx as f32) * 2.0;
        }

        ctx.with_gpu_scope("elemwise_add_op", |ctx| {
            let out = ctx.call::<ElemwiseAddOp>((a.clone(), b.clone())).expect("elemwise add call");
            ctx.synchronize();
            out
        });

        let mut observed = None;
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            println!("Waiting for event...");
            if let Ok(enriched) = receiver.recv_timeout(Duration::from_millis(1000)) {
                println!("Received event: {:#?}", enriched);
                if let MetricEvent::GpuOpCompleted {
                    op_name,
                    backend,
                    duration_us,
                } = enriched.event
                    && backend == "Metal"
                    && op_name.starts_with("elemwise_add_op")
                {
                    assert!(duration_us > 0, "duration must be positive");
                    observed = Some((op_name, duration_us));
                    break;
                }
            }
        }

        assert!(observed.is_some(), "expected elemwise_add_op GPU event");
    });
}

// Maintainers: run this test on Apple Silicon hardware before releasing.
#[test]
fn matmul_mps_emits_gpu_event() {
    let _guard = ENABLE_PROFILING_VAR.set_guard(true).unwrap();
    reset_app_config_for_tests();
    let _profiling_guard = AppConfig::force_enable_profiling_guard();
    let (sender, receiver) = mpsc::channel();
    let exporters: Vec<Box<dyn MetricExporter>> = vec![Box::new(ChannelExporter::new(sender))];
    let layer = MetricsLayer::new(exporters);
    let subscriber = tracing_subscriber::registry().with(layer);

    subscriber::with_default(subscriber, || {
        let _metric_bypass = MetricQueueBypassGuard::new();
        let mut ctx = Context::<F32Element>::new().expect("metal context");

        let dims = vec![2usize, 2];
        let mut a = Tensor::new(dims.clone(), TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("tensor a");
        let mut b = Tensor::new(dims.clone(), TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("tensor b");
        let out = Tensor::new(dims.clone(), TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized).expect("tensor out");

        for (idx, value) in a.as_mut_slice().iter_mut().enumerate() {
            *value = idx as f32;
        }
        for (idx, value) in b.as_mut_slice().iter_mut().enumerate() {
            *value = (idx as f32) * 0.5;
        }

        ctx.with_gpu_scope("matmul_test_scope", |ctx| {
            let result = ctx.matmul_alpha_beta(&a, &b, &out, false, false, 1.0, 0.0).expect("matmul call");
            ctx.synchronize();
            result
        });

        let mut observed = None;
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if let Ok(enriched) = receiver.recv_timeout(Duration::from_millis(100))
                && let MetricEvent::GpuOpCompleted {
                    op_name,
                    backend,
                    duration_us,
                } = enriched.event
                && backend == "Metal"
                && op_name.starts_with("matmul_test_scope")
            {
                assert!(duration_us > 0, "duration must be positive");
                observed = Some((op_name, duration_us));
                break;
            }
        }

        assert!(observed.is_some(), "expected matmul_test_scope GPU event");
    });
}
