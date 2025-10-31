use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandQueue, MTLDevice};
use rustc_hash::FxHashMap;

use super::registry::CacheRegistrySlot;
use crate::{
    MetalError, pool::MemoryPool, tensor::{Tensor, TensorElement}
};

pub const KV_CACHE_POOL_MAX_BYTES: usize = 8 * 1024 * 1024 * 1024; // 8GB

/// Per-layer KV cache tensors retained on device memory.
#[derive(Clone)]
pub struct KvCacheEntry<T: TensorElement> {
    pub k: Tensor<T>,
    pub v: Tensor<T>,
    #[allow(dead_code)]
    pub dtype: crate::tensor::Dtype,
    pub element_size: usize,
    pub zeroing_complete: bool,
    pub capacity: usize,
}

impl<T: TensorElement> KvCacheEntry<T> {
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.k.size_bytes() + self.v.size_bytes()
    }
}

/// Execution plan describing a single KV cache write.
pub struct KvWritePlan<T: TensorElement> {
    pub k_src: Tensor<T>,
    pub v_src: Tensor<T>,
    pub k_cache: Tensor<T>,
    pub v_cache: Tensor<T>,
    pub canonical_heads: usize,
    pub repeated_heads: usize,
    pub group_size: usize,
    pub group_size_u32: u32,
    pub seq_in_src: usize,
    pub head_dim: usize,
    pub capacity_seq_val: usize,
    pub element_size: usize,
    pub src_head_stride: u32,
    pub src_seq_stride: u32,
    pub dst_head_stride: u32,
    pub dst_seq_stride: u32,
    pub total_threads: u32,
    pub heads_u32: u32,
    pub head_dim_u32: u32,
    pub seq_len_u32: u32,
    pub step_u32: u32,
    pub step: usize,
}

/// Workspace for KV cache allocations and bookkeeping.
pub struct KvCacheState<T: TensorElement> {
    pool: MemoryPool,
    caches: FxHashMap<usize, KvCacheEntry<T>>,
    total_bytes: usize,
}

impl<T: TensorElement> KvCacheState<T> {
    pub fn new(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        command_queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            pool: MemoryPool::with_limit(device, command_queue, KV_CACHE_POOL_MAX_BYTES)?,
            caches: FxHashMap::default(),
            total_bytes: 0,
        })
    }

    #[inline]
    pub fn pool(&self) -> &MemoryPool {
        &self.pool
    }

    #[inline]
    pub fn pool_mut(&mut self) -> &mut MemoryPool {
        &mut self.pool
    }

    #[inline]
    pub fn caches(&self) -> &FxHashMap<usize, KvCacheEntry<T>> {
        &self.caches
    }

    #[inline]
    pub fn caches_mut(&mut self) -> &mut FxHashMap<usize, KvCacheEntry<T>> {
        &mut self.caches
    }

    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    #[inline]
    pub fn get(&self, layer_idx: usize) -> Option<&KvCacheEntry<T>> {
        self.caches.get(&layer_idx)
    }

    #[inline]
    pub fn get_mut(&mut self, layer_idx: usize) -> Option<&mut KvCacheEntry<T>> {
        self.caches.get_mut(&layer_idx)
    }

    pub fn insert(&mut self, layer_idx: usize, entry: KvCacheEntry<T>) {
        let entry_bytes = entry.total_bytes();
        if let Some(prev) = self.caches.insert(layer_idx, entry) {
            self.total_bytes = self.total_bytes.saturating_sub(prev.total_bytes()).saturating_add(entry_bytes);
        } else {
            self.total_bytes = self.total_bytes.saturating_add(entry_bytes);
        }
    }

    pub fn clear_entries(&mut self) {
        self.caches.clear();
        self.total_bytes = 0;
    }
}

impl<T: TensorElement> CacheRegistrySlot for KvCacheState<T> {
    fn clear_slot(&mut self) {
        self.clear_entries();
    }
}
