use objc2_metal::{MTLDevice as _, MTLResourceOptions};
use rustc_hash::FxHashMap;

use crate::{
    error::MetalError, foundry::Foundry, tensor::Dtype, types::{MetalBuffer, MetalDevice, TensorArg}
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SdpaScratchKey {
    dtype: Dtype,
    n_heads: u32,
}

#[derive(Debug, Clone)]
struct SdpaScratchEntry {
    scores: MetalBuffer,
    probs: MetalBuffer,
    capacity_elems: usize,
}

/// Persistent scratch buffers for Foundry SDPA.
///
/// Context's SDPA path effectively reuses pooled tensors/workspaces; Foundry SDPA previously
/// allocated scratch (scores/probs) per call, which is extremely costly for decode.
///
/// These buffers are GPU-only (private) and safe to reuse sequentially within a single command buffer.
#[derive(Default)]
pub struct SdpaScratchCache {
    entries: FxHashMap<SdpaScratchKey, SdpaScratchEntry>,
}

impl SdpaScratchCache {
    fn ensure_capacity(
        &mut self,
        device: &MetalDevice,
        dtype: Dtype,
        n_heads: u32,
        needed_elems: usize,
    ) -> Result<&SdpaScratchEntry, MetalError> {
        let key = SdpaScratchKey { dtype, n_heads };

        let (needs_new, target_capacity_elems) = match self.entries.get(&key) {
            Some(entry) if entry.capacity_elems >= needed_elems => (false, entry.capacity_elems),
            Some(entry) => {
                // Avoid reallocating every time kv_seq_len grows by 1 by growing capacity
                // geometrically. This is decode-critical.
                let grown = entry.capacity_elems.saturating_add(entry.capacity_elems / 2); // 1.5x
                (true, needed_elems.max(grown))
            }
            None => (true, needed_elems),
        };

        if needs_new {
            let elem_size = match dtype {
                Dtype::F16 => 2usize,
                _ => {
                    return Err(MetalError::UnsupportedDtype {
                        operation: "FoundrySdpaScratch",
                        dtype,
                    });
                }
            };

            let bytes = align_256(target_capacity_elems.saturating_mul(elem_size));
            let opts = MTLResourceOptions::StorageModePrivate;

            let scores = device
                .newBufferWithLength_options(bytes, opts)
                .ok_or(MetalError::BufferCreationFailed(bytes))?;
            let probs = device
                .newBufferWithLength_options(bytes, opts)
                .ok_or(MetalError::BufferCreationFailed(bytes))?;

            self.entries.insert(
                key,
                SdpaScratchEntry {
                    scores: MetalBuffer::from_retained(scores),
                    probs: MetalBuffer::from_retained(probs),
                    capacity_elems: target_capacity_elems,
                },
            );
        }

        self.entries
            .get(&key)
            .ok_or_else(|| MetalError::ResourceNotCached("SdpaScratchCache".into()))
    }
}

#[inline]
fn align_256(bytes: usize) -> usize {
    (bytes + 255) & !255
}

pub fn get_sdpa_scratch_f16(foundry: &mut Foundry, n_heads: u32, m: u32, kv_seq_len: u32) -> Result<(TensorArg, TensorArg), MetalError> {
    let needed_elems = (n_heads as usize).saturating_mul(m as usize).saturating_mul(kv_seq_len as usize);

    let device = foundry.device.clone();

    if foundry.get_resource::<SdpaScratchCache>().is_none() {
        foundry.register_resource(SdpaScratchCache::default());
    }

    // Borrow resource mutably just for sizing decision and potential allocation, then drop.
    let entry = {
        let cache = foundry
            .get_resource::<SdpaScratchCache>()
            .ok_or_else(|| MetalError::ResourceNotCached("SdpaScratchCache".into()))?;
        cache.ensure_capacity(&device, Dtype::F16, n_heads, needed_elems)?.clone()
    };

    // TensorArg views with the exact logical lengths needed for this call.
    // Offsets are applied by callers (per-head slicing) as needed.
    let scores = TensorArg::from_buffer(entry.scores, Dtype::F16, vec![needed_elems], vec![1]);
    let probs = TensorArg::from_buffer(entry.probs, Dtype::F16, vec![needed_elems], vec![1]);
    Ok((scores, probs))
}
