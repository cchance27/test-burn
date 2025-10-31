use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use metallic::{
    caching::ResourceCache, kernels::{KernelFunction, KernelManager}, operation::{CommandBuffer, FillConstant, Operation, RandomUniform}, tensor::{Dtype, F32Element, Tensor}
};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice as _, MTLResourceOptions};

fn bench_individual_vs_batched(c: &mut Criterion) {
    let device = MTLCreateSystemDefaultDevice().expect("default metal device");
    let queue = device.newCommandQueue().expect("command queue");
    let mut km = KernelManager::new();

    let mut group = c.benchmark_group("record_vs_batch_record_mixed");

    for &n_ops in &[1usize, 10, 100] {
        group.bench_with_input(BenchmarkId::new("individual", n_ops), &n_ops, |b, &n| {
            b.iter(|| {
                let mut cache = ResourceCache::with_device(device.clone());
                let command_buffer = CommandBuffer::new(&queue).expect("command buffer");
                let element_count = 1024usize;
                let byte_len = element_count * std::mem::size_of::<f32>();

                let buffer = device
                    .newBufferWithLength_options(byte_len as _, MTLResourceOptions::StorageModeShared)
                    .expect("shared buffer");

                let tensor = Tensor::<F32Element>::from_existing_buffer(buffer, vec![element_count], Dtype::F32, &device, &queue, 0, true)
                    .expect("tensor from buffer");

                for i in 0..n {
                    if i % 2 == 0 {
                        // Blit op (zero-fill)
                        let op = FillConstant {
                            dst: tensor.clone(),
                            value: 0.0,
                            ones_pipeline: None,
                        };
                        command_buffer.record(&op, &mut cache).unwrap();
                    } else {
                        // Compute op (random uniform)
                        let op = RandomUniform {
                            dst: tensor.clone(),
                            seed: 42,
                            pipeline: km
                                .get_pipeline(KernelFunction::RandomUniform, Dtype::F32, &device)
                                .expect("random_uniform pipeline"),
                        };
                        command_buffer.record(&op, &mut cache).unwrap();
                    }
                }
                command_buffer.commit();
                command_buffer.wait();
            });
        });

        group.bench_with_input(BenchmarkId::new("batched", n_ops), &n_ops, |b, &n| {
            b.iter(|| {
                let mut cache = ResourceCache::with_device(device.clone());
                let command_buffer = CommandBuffer::new(&queue).expect("command buffer");
                let element_count = 1024usize;
                let byte_len = element_count * std::mem::size_of::<f32>();

                let buffer = device
                    .newBufferWithLength_options(byte_len as _, MTLResourceOptions::StorageModeShared)
                    .expect("shared buffer");

                let tensor = Tensor::<F32Element>::from_existing_buffer(buffer, vec![element_count], Dtype::F32, &device, &queue, 0, true)
                    .expect("tensor from buffer");

                let mut ops: Vec<Box<dyn Operation>> = Vec::with_capacity(n);
                for i in 0..n {
                    if i % 2 == 0 {
                        let op = FillConstant {
                            dst: tensor.clone(),
                            value: 0.0,
                            ones_pipeline: None,
                        };
                        ops.push(Box::new(op));
                    } else {
                        let op = RandomUniform {
                            dst: tensor.clone(),
                            seed: 42,
                            pipeline: km
                                .get_pipeline(KernelFunction::RandomUniform, Dtype::F32, &device)
                                .expect("random_uniform pipeline"),
                        };
                        ops.push(Box::new(op));
                    }
                }
                let borrowed: Vec<&dyn Operation> = ops.iter().map(|b| b.as_ref()).collect();
                command_buffer.record_batch(&borrowed, &mut cache).unwrap();
                command_buffer.commit();
                command_buffer.wait();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_individual_vs_batched);
criterion_main!(benches);
