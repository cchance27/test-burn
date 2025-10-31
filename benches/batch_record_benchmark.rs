use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use metallic::{
    caching::ResourceCache, operation::{CommandBuffer, Operation}, tensor::{Dtype, F32Element, Tensor}
};
use objc2_metal::{MTLBlitCommandEncoder as _, MTLCreateSystemDefaultDevice, MTLDevice as _, MTLResourceOptions};

// Simple benchmark-only operation for zero-fill using blit encoder
struct BenchBlitZeroFill {
    dst: Tensor<F32Element>,
}

impl Operation for BenchBlitZeroFill {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut metallic::caching::ResourceCache) -> Result<(), metallic::MetalError> {
        let encoder = command_buffer.get_blit_encoder()?;
        encoder.fillBuffer_range_value(&self.dst.buf, (self.dst.offset..self.dst.offset + self.dst.size_bytes()).into(), 0);
        Ok(())
    }

    fn bind_kernel_args(&self, _encoder: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>>) {
        // No-op for blit operation
    }
}

fn bench_individual_vs_batched(c: &mut Criterion) {
    let device = MTLCreateSystemDefaultDevice().expect("default metal device");
    let queue = device.newCommandQueue().expect("command queue");

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

                for _i in 0..n {
                    // Blit op (zero-fill) - simplified for benchmark
                    let op = BenchBlitZeroFill { dst: tensor.clone() };
                    command_buffer.record(&op, &mut cache).unwrap();
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
                for _i in 0..n {
                    // Blit op (zero-fill) - simplified for benchmark
                    let op = BenchBlitZeroFill { dst: tensor.clone() };
                    ops.push(Box::new(op));
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
