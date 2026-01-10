use half::f16;
use objc2_metal::{MTLBuffer as _, MTLDevice as _, MTLResourceOptions};
use rustc_hash::FxHashMap;

use crate::{
    error::MetalError,
    tensor::Dtype,
    types::{MetalBuffer, MetalDevice, TensorArg},
};

use super::Foundry;

/// Cached scalar buffers for Foundry.
///
/// This avoids per-step allocations and, critically, avoids any blit uploads + waits
/// for tiny constant buffers (which are catastrophic for decode throughput).
#[derive(Default)]
pub struct ScalarBufferCache {
    f16: FxHashMap<u16, MetalBuffer>,
}

impl ScalarBufferCache {
    fn get_or_create_f16(&mut self, device: &MetalDevice, value: f16) -> Result<MetalBuffer, MetalError> {
        let bits = value.to_bits();
        if let Some(buf) = self.f16.get(&bits) {
            return Ok(buf.clone());
        }

        // Use shared storage so we can write CPU-side without a staging blit.
        // Allocate 256 bytes to avoid any alignment quirks for constant reads.
        let buffer = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(256))?;
        let buffer = MetalBuffer::from_retained(buffer);

        // Initialize first 2 bytes to the f16 value, rest zero.
        unsafe {
            let ptr = buffer.0.contents().as_ptr() as *mut u8;
            if ptr.is_null() {
                return Err(MetalError::NullPointer);
            }
            // Zero-fill 256 bytes.
            std::ptr::write_bytes(ptr, 0u8, 256);
            // Write f16 bits.
            let bytes = bits.to_ne_bytes();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, 2);
        }

        self.f16.insert(bits, buffer.clone());
        Ok(buffer)
    }
}

/// Returns a `TensorArg` that points at a cached shared buffer containing the given `f16` scalar.
pub fn f16_scalar(foundry: &mut Foundry, value: f16) -> Result<TensorArg, MetalError> {
    let device = foundry.device.clone();

    // Lazily create or fetch the cache resource.
    if foundry.get_resource::<ScalarBufferCache>().is_none() {
        foundry.register_resource(ScalarBufferCache::default());
    }
    let cache = foundry
        .get_resource::<ScalarBufferCache>()
        .ok_or_else(|| MetalError::ResourceNotCached("ScalarBufferCache".into()))?;

    let buffer = cache.get_or_create_f16(&device, value)?;

    // Represent as a 1-element tensor.
    Ok(TensorArg::from_buffer(buffer, Dtype::F16, vec![1], vec![1]))
}
