use objc2_metal::{MTLBlitCommandEncoder as _, MTLDevice, MTLResource, MTLStorageMode};

use super::{main::Context, utils::tier_up_capacity};
use crate::{
    MetalError, caching::{KvCacheEntry, KvWritePlan}, tensor::{Tensor, TensorElement}
};

impl<T: TensorElement> Context<T> {
    #[inline]
    pub(crate) fn clear_kv_caches(&mut self) {
        self.kv_cache_state_mut().clear_entries();
        self.sdpa_workspaces.clear();
    }

    /// Allocate an on-device per-layer KV cache and register it in the centralized kv_caches map.
    /// Layout: [batch * n_heads, seq_len, head_dim].
    #[allow(clippy::too_many_arguments)]
    pub fn alloc_kv_cache(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        repeated_batch_heads: usize,
        head_dim: usize,
    ) -> Result<(), MetalError> {
        let repeated_dims = vec![repeated_batch_heads, seq_len, head_dim];

        // Allocate K and V tensors directly from the dedicated KV cache pool
        let (k_allocation, v_allocation) = {
            let state = self.kv_cache_state_mut();
            let pool = state.pool_mut();
            let k_allocation = pool.alloc_tensor::<T>(repeated_dims.clone())?;
            let v_allocation = pool.alloc_tensor::<T>(repeated_dims)?;
            (k_allocation, v_allocation)
        };

        let dtype = k_allocation.dtype();
        let element_size = k_allocation.element_size();
        debug_assert_eq!(dtype, v_allocation.dtype());
        let k = k_allocation.into_tensor();
        let v = v_allocation.into_tensor();

        // Manually zero the tensors using a blit command
        let k_size = k.size_bytes();
        let v_size = v.size_bytes();

        self.ensure_active_cmd_buffer()?;
        let cmd_buf = self.active_command_buffer_mut()?;
        {
            let encoder = cmd_buf.get_blit_encoder()?;
            encoder.fillBuffer_range_value(&k.buf, (k.offset..k.offset + k_size).into(), 0);
            encoder.fillBuffer_range_value(&v.buf, (v.offset..v.offset + v_size).into(), 0);
        }

        self.mark_tensor_pending(&k);
        self.mark_tensor_pending(&v);
        self.finalize_active_command_buffer_if_latency();

        let entry = KvCacheEntry {
            k,
            v,
            dtype,
            element_size,
            zeroing_complete: true,
            capacity: seq_len,
        };
        self.kv_cache_state_mut().insert(layer_idx, entry);
        Ok(())
    }

    /// Write a single timestep of K and V (per-head flattened) into the per-layer cache at index `step`.
    /// - `k_step` and `v_step` must be contiguous tensors with shape [batch_heads, head_dim] or [batch_heads, 1, head_dim].
    ///   This performs a device blit copy from the source buffer into the cache at the correct offset.
    #[allow(clippy::too_many_arguments)]
    pub fn write_kv_step(
        &mut self,
        layer_idx: usize,
        step: usize,
        group_size: usize,
        k_step: &Tensor<T>,
        v_step: &Tensor<T>,
    ) -> Result<(), MetalError> {
        let plan = self.build_kv_write_plan(layer_idx, step, group_size, k_step, v_step)?;
        self.dispatch_kv_write(layer_idx, plan)
    }

    fn build_kv_write_plan(
        &self,
        layer_idx: usize,
        step: usize,
        group_size: usize,
        k_step: &Tensor<T>,
        v_step: &Tensor<T>,
    ) -> Result<KvWritePlan<T>, MetalError> {
        let k_src = k_step.clone();
        let v_src = v_step.clone();

        let dims = k_src.dims().to_vec();
        let (bh, seq_in_src, hd) = match dims.len() {
            2 => (dims[0], 1, dims[1]),
            3 => (dims[0], dims[1], dims[2]),
            _ => {
                return Err(MetalError::InvalidShape("write_kv_step expects source tensor rank 2 or 3".into()));
            }
        };

        let v_dims = v_src.dims().to_vec();
        let (v_bh, v_seq_in_src, v_hd) = match v_dims.len() {
            2 => (v_dims[0], 1, v_dims[1]),
            3 => (v_dims[0], v_dims[1], v_dims[2]),
            _ => {
                return Err(MetalError::InvalidShape("write_kv_step expects V tensor rank 2 or 3".into()));
            }
        };

        if seq_in_src != 1 {
            return Err(MetalError::OperationNotSupported(
                "write_kv_step currently expects a single timestep in the source tensor".into(),
            ));
        }

        if v_seq_in_src != seq_in_src {
            return Err(MetalError::OperationNotSupported(
                "write_kv_step expects matching sequence dims for K and V".into(),
            ));
        }

        let entry = self
            .kv_cache_state()
            .get(layer_idx)
            .ok_or_else(|| MetalError::InvalidOperation(format!("KV cache for layer {} not allocated", layer_idx)))?;
        let k_cache = entry.k.clone();
        let v_cache = entry.v.clone();
        let capacity_seq_val = entry.capacity;
        let element_size = entry.element_size;

        if group_size == 0 {
            return Err(MetalError::InvalidOperation("write_kv_step requires a non-zero group size".into()));
        }

        if step >= capacity_seq_val {
            return Err(MetalError::InvalidOperation(format!(
                "Step {} exceeds KV cache capacity {} for layer {}",
                step, capacity_seq_val, layer_idx
            )));
        }

        let cache_dims = k_cache.dims();
        if cache_dims.len() != 3 {
            return Err(MetalError::InvalidShape(
                "KV cache tensor must have shape [batch_heads, seq_len, head_dim]".into(),
            ));
        }

        let expected_repeated_heads = cache_dims[0];
        let expected_hd = cache_dims[2];

        if hd != expected_hd {
            return Err(MetalError::DimensionMismatch {
                expected: expected_hd,
                actual: hd,
            });
        }

        if v_hd != expected_hd {
            return Err(MetalError::DimensionMismatch {
                expected: expected_hd,
                actual: v_hd,
            });
        }

        if bh != v_bh {
            return Err(MetalError::DimensionMismatch {
                expected: bh,
                actual: v_bh,
            });
        }

        let canonical_heads = bh;
        let repeated_heads_expected = canonical_heads
            .checked_mul(group_size)
            .ok_or_else(|| MetalError::InvalidOperation("group_size overflow while expanding KV heads".into()))?;

        if expected_repeated_heads != repeated_heads_expected {
            return Err(MetalError::DimensionMismatch {
                expected: repeated_heads_expected,
                actual: expected_repeated_heads,
            });
        }

        if step + seq_in_src > capacity_seq_val {
            return Err(MetalError::InvalidOperation(format!(
                "Writing KV step {} ({} timesteps) exceeds cache capacity {} for layer {}",
                step, seq_in_src, capacity_seq_val, layer_idx
            )));
        }

        if k_src.strides.len() != v_src.strides.len()
            || k_src.strides.first() != v_src.strides.first()
            || (k_src.strides.len() > 1 && k_src.strides[1] != v_src.strides[1])
        {
            return Err(MetalError::InvalidShape(
                "write_kv_step requires K and V to share the same layout".into(),
            ));
        }

        let head_dim_u32 = u32::try_from(expected_hd).map_err(|_| MetalError::InvalidShape("head dimension exceeds u32::MAX".into()))?;
        let heads_u32 = u32::try_from(canonical_heads).map_err(|_| MetalError::InvalidShape("batch_heads exceeds u32::MAX".into()))?;
        let seq_len_u32 = u32::try_from(seq_in_src).map_err(|_| MetalError::InvalidShape("sequence length exceeds u32::MAX".into()))?;
        let step_u32 = u32::try_from(step).map_err(|_| MetalError::InvalidShape("step index exceeds u32::MAX".into()))?;
        let src_head_stride =
            u32::try_from(k_src.strides[0]).map_err(|_| MetalError::InvalidShape("source head stride exceeds u32::MAX".into()))?;
        let src_seq_stride = if dims.len() == 3 {
            u32::try_from(k_src.strides[1]).map_err(|_| MetalError::InvalidShape("source sequence stride exceeds u32::MAX".into()))?
        } else {
            0
        };
        let dst_head_stride =
            u32::try_from(k_cache.strides[0]).map_err(|_| MetalError::InvalidShape("cache head stride exceeds u32::MAX".into()))?;
        let dst_seq_stride =
            u32::try_from(k_cache.strides[1]).map_err(|_| MetalError::InvalidShape("cache sequence stride exceeds u32::MAX".into()))?;

        let repeated_heads = expected_repeated_heads;
        let group_size_u32 = u32::try_from(group_size).map_err(|_| MetalError::InvalidShape("group size exceeds u32::MAX".into()))?;

        if !k_src.offset.is_multiple_of(element_size)
            || !v_src.offset.is_multiple_of(element_size)
            || k_cache.offset % element_size != 0
            || v_cache.offset % element_size != 0
        {
            return Err(MetalError::InvalidOperation("KV tensors must be element-aligned".into()));
        }

        let total_threads = heads_u32
            .checked_mul(head_dim_u32)
            .ok_or_else(|| MetalError::InvalidShape("thread count exceeds u32::MAX".into()))?;

        Ok(KvWritePlan {
            k_src,
            v_src,
            k_cache,
            v_cache,
            canonical_heads,
            repeated_heads,
            group_size,
            group_size_u32,
            seq_in_src,
            head_dim: expected_hd,
            capacity_seq_val,
            element_size,
            src_head_stride,
            src_seq_stride,
            dst_head_stride,
            dst_seq_stride,
            total_threads,
            heads_u32,
            head_dim_u32,
            seq_len_u32,
            step_u32,
            step,
        })
    }

    fn dispatch_kv_write(&mut self, layer_idx: usize, plan: KvWritePlan<T>) -> Result<(), MetalError> {
        let KvWritePlan {
            k_src,
            v_src,
            k_cache,
            v_cache,
            canonical_heads,
            repeated_heads,
            group_size,
            group_size_u32,
            seq_in_src,
            head_dim,
            capacity_seq_val,
            element_size,
            src_head_stride,
            src_seq_stride,
            dst_head_stride,
            dst_seq_stride,
            total_threads,
            heads_u32,
            head_dim_u32,
            seq_len_u32,
            step_u32,
            step,
        } = plan;

        let tensors: Vec<&Tensor<T>> = vec![&k_cache, &v_cache, &k_src, &v_src];
        self.prepare_tensors_for_active_cmd(&tensors)?;

        let config = crate::kernels::kv_cache_write::KvCacheWriteConfig {
            canonical_heads: heads_u32,
            head_dim: head_dim_u32,
            seq_len: seq_len_u32,
            step: step_u32,
            group_size: group_size_u32,
            src_head_stride,
            src_seq_stride,
            dst_head_stride,
            dst_seq_stride,
            total_threads,
            repeated_heads: u32::try_from(repeated_heads)
                .map_err(|_| MetalError::InvalidShape("repeated head count exceeds u32::MAX".into()))?,
        };

        match self.call::<crate::kernels::kv_cache_write::KvCacheWriteOp>(
            (k_src.clone(), v_src.clone(), k_cache.clone(), v_cache.clone(), config.clone()),
            None,
        ) {
            Ok(_) => {
                metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuKernelDispatched {
                    kernel_name: "kv_cache_write".to_string(),
                    op_name: format!("kv_cache_write_step_{}_layer_{}", step, layer_idx),
                    thread_groups: (config.total_threads, 1, 1),
                });
            }
            Err(err) if Self::kv_cache_kernel_unavailable(&err) => {
                metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuKernelDispatched {
                    kernel_name: "kv_cache_fallback_blit".to_string(),
                    op_name: format!("kv_cache_blit_step_{}_layer_{}", step, layer_idx),
                    thread_groups: (1, 1, 1),
                });
                return self.blit_write_kv_step(
                    layer_idx,
                    step,
                    &k_src,
                    &v_src,
                    &k_cache,
                    &v_cache,
                    seq_in_src,
                    canonical_heads,
                    head_dim,
                    capacity_seq_val,
                    element_size,
                    group_size,
                    repeated_heads,
                );
            }
            Err(err) => return Err(err),
        }

        {
            let state = self.kv_cache_state_mut();
            if let Some(entry) = state.get_mut(layer_idx) {
                entry.zeroing_complete = false;
            }
        }

        self.mark_tensor_pending(&k_cache);
        self.mark_tensor_pending(&v_cache);
        self.finalize_active_command_buffer_if_latency();

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn blit_write_kv_step(
        &mut self,
        layer_idx: usize,
        step: usize,
        k_src: &Tensor<T>,
        v_src: &Tensor<T>,
        k_cache: &Tensor<T>,
        v_cache: &Tensor<T>,
        seq_in_src: usize,
        canonical_heads: usize,
        head_dim: usize,
        capacity: usize,
        element_size: usize,
        group_size: usize,
        repeated_heads: usize,
    ) -> Result<(), MetalError> {
        self.ensure_active_cmd_buffer()?;
        let profiler_label = self
            .take_gpu_scope()
            .unwrap_or_else(|| super::utils::GpuProfilerLabel::fallback("kv_cache_blit_op"));
        let cmd_buf = self.active_command_buffer_mut()?;
        let raw_cmd = cmd_buf.raw();
        let encoder = cmd_buf.get_blit_encoder()?;
        let _scope = metallic_instrumentation::gpu_profiler::GpuProfiler::profile_blit(
            raw_cmd,
            &encoder,
            profiler_label.op_name,
            profiler_label.backend,
        );

        let cache_stride_elems = capacity * head_dim;
        let copy_bytes = head_dim * element_size;
        let k_dims_len = k_src.dims().len();
        let v_dims_len = v_src.dims().len();

        if repeated_heads != canonical_heads * group_size {
            return Err(MetalError::DimensionMismatch {
                expected: canonical_heads * group_size,
                actual: repeated_heads,
            });
        }

        unsafe {
            for head_idx in 0..canonical_heads {
                let src_elem_index = if k_dims_len == 2 {
                    head_idx * head_dim
                } else {
                    (head_idx * seq_in_src) * head_dim
                };
                let v_src_elem_index = if v_dims_len == 2 {
                    head_idx * head_dim
                } else {
                    (head_idx * seq_in_src) * head_dim
                };
                let src_offset_k = k_src.offset + src_elem_index * element_size;
                let src_offset_v = v_src.offset + v_src_elem_index * element_size;

                for group in 0..group_size {
                    let repeated_head = head_idx * group_size + group;
                    let dst_elem_index = repeated_head * cache_stride_elems + step * head_dim;

                    let dst_offset_k = k_cache.offset + dst_elem_index * element_size;
                    let dst_offset_v = v_cache.offset + dst_elem_index * element_size;

                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &k_src.buf,
                        src_offset_k,
                        &k_cache.buf,
                        dst_offset_k,
                        copy_bytes,
                    );
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &v_src.buf,
                        src_offset_v,
                        &v_cache.buf,
                        dst_offset_v,
                        copy_bytes,
                    );
                }
            }
        }

        self.mark_tensor_pending(k_cache);
        self.mark_tensor_pending(v_cache);
        self.finalize_active_command_buffer_if_latency();

        {
            let state = self.kv_cache_state_mut();
            if let Some(entry) = state.get_mut(layer_idx) {
                entry.zeroing_complete = false;
            }
        }

        Ok(())
    }

    #[inline]
    fn kv_cache_kernel_unavailable(err: &MetalError) -> bool {
        matches!(
            err,
            MetalError::LibraryCompilationFailed(_)
                | MetalError::FunctionCreationFailed(_)
                | MetalError::PipelineCreationFailed(_)
                | MetalError::ComputeEncoderCreationFailed
                | MetalError::UnsupportedDtype { .. }
        )
    }

    /// Create a strided view of the KV cache exposing the first `active_steps` positions in
    /// [batch_heads, steps, head_dim] order while preserving the underlying cache stride.
    pub fn kv_cache_history_view(&mut self, cache: &Tensor<T>, active_steps: usize) -> Result<(Tensor<T>, usize), MetalError> {
        let dims = cache.dims();
        if dims.len() != 3 {
            return Err(MetalError::InvalidShape(
                "KV cache tensor must have shape [batch_heads, seq_len, head_dim]".to_string(),
            ));
        }

        if active_steps == 0 || active_steps > dims[1] {
            return Err(MetalError::InvalidShape(format!(
                "Requested {} KV steps exceeds cache capacity {}",
                active_steps, dims[1]
            )));
        }

        let mut view = cache.clone();
        view.dims = vec![dims[0], active_steps, dims[2]];
        view.strides = vec![cache.strides[0], cache.strides[1], cache.strides[2]];

        self.prepare_tensors_for_active_cmd(&[&view])?;

        Ok((view, dims[1]))
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn acquire_repeat_kv_workspace(
        &mut self,
        layer_idx: usize,
        kind: super::main::RepeatKvWorkspaceKind,
        repeated_heads: usize,
        seq: usize,
        cache_capacity: usize,
        head_dim: usize,
        prefer_shared: bool,
    ) -> Result<Tensor<T>, MetalError> {
        if seq == 0 || cache_capacity == 0 {
            return Err(MetalError::InvalidShape(
                "repeat_kv_workspace requires non-zero sequence capacity".to_string(),
            ));
        }
        if seq > cache_capacity {
            return Err(MetalError::InvalidShape(format!(
                "repeat_kv_workspace sequence {seq} exceeds cache capacity {cache_capacity}"
            )));
        }

        let storage_mode = if prefer_shared {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Private
        };
        let key = super::main::RepeatKvWorkspaceKey {
            layer_idx,
            kind,
            repeated_heads,
            cache_capacity,
            head_dim,
            dtype: T::DTYPE,
            shared: prefer_shared,
        };

        // When prefer_shared is true (for MPSGraph), allocate a buffer with tiered capacity
        // to reduce allocations during incremental decode while ensuring zero-offset tensors.
        // When prefer_shared is false (for legacy), use a shared buffer with cache_capacity
        // and return views into it to minimize allocations during incremental decode.
        let tensor = if prefer_shared {
            // For MPSGraph: allocate buffer with tiered capacity to reduce number of allocations
            let tiered_capacity = tier_up_capacity(seq);
            let seq_key = super::main::RepeatKvWorkspaceKey {
                layer_idx,
                kind,
                repeated_heads,
                cache_capacity: tiered_capacity, // Use tiered capacity instead of exact seq
                head_dim,
                dtype: T::DTYPE,
                shared: prefer_shared,
            };

            if !self.kv_repeat_workspaces.contains_key(&seq_key) {
                let new_tensor = self.create_repeat_kv_tensor(repeated_heads, tiered_capacity, head_dim, storage_mode)?;
                self.kv_repeat_workspaces.insert(
                    seq_key,
                    super::main::RepeatKvWorkspaceEntry {
                        tensor: new_tensor.clone(),
                    },
                );
                new_tensor
            } else {
                self.kv_repeat_workspaces
                    .get(&seq_key)
                    .expect("workspace must exist")
                    .tensor
                    .clone()
            }
        } else {
            // For legacy: allocate shared buffer with cache_capacity
            if !self.kv_repeat_workspaces.contains_key(&key) {
                let new_tensor = self.create_repeat_kv_tensor(repeated_heads, cache_capacity, head_dim, storage_mode)?;
                self.kv_repeat_workspaces.insert(
                    key,
                    super::main::RepeatKvWorkspaceEntry {
                        tensor: new_tensor.clone(),
                    },
                );
                new_tensor
            } else {
                self.kv_repeat_workspaces.get(&key).expect("workspace must exist").tensor.clone()
            }
        };

        // Create a view with the active sequence length.
        // For prefer_shared=true (MPSGraph), the buffer has tiered capacity but we create a view for exact seq.
        // For prefer_shared=false (legacy), we slice into a larger buffer with cache_capacity.
        let mut view = tensor.clone();
        view.dims = vec![repeated_heads, seq, head_dim];
        if prefer_shared {
            // For MPSGraph: strides match dims for the view's actual sequence length
            view.strides = Tensor::<T>::compute_strides(&view.dims);
        } else {
            // For legacy: preserve cache_capacity strides for view into larger buffer
            view.strides = tensor.strides.clone();
        }
        view.offset = 0;

        // Emit metric for workspace usage
        let prep_start = std::time::Instant::now();
        self.prepare_tensors_for_active_cmd(&[&view])?;
        let prep_elapsed = prep_start.elapsed();
        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::InternalKernelCompleted {
            parent_op_name: "tensor_prepare".to_string(),
            internal_kernel_name: format!(
                "repeat_kv_workspace/{:?}/storage={:?}/bytes={}",
                kind,
                view.buf.storageMode(),
                view.size_bytes()
            ),
            duration_us: (prep_elapsed.as_micros().max(1)) as u64,
        });

        Ok(view)
    }

    fn create_repeat_kv_tensor(
        &self,
        repeated_heads: usize,
        capacity_seq: usize,
        head_dim: usize,
        storage_mode: MTLStorageMode,
    ) -> Result<Tensor<T>, MetalError> {
        let dims = vec![repeated_heads, capacity_seq, head_dim];
        let elem_size = T::DTYPE.size_bytes();
        let elements = dims
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| MetalError::InvalidShape("repeat_kv_workspace element count overflow".into()))?;
        let byte_len = elements
            .checked_mul(elem_size)
            .ok_or_else(|| MetalError::InvalidShape("repeat_kv_workspace buffer size overflow".into()))?;

        let resource_options = match storage_mode {
            MTLStorageMode::Shared => objc2_metal::MTLResourceOptions::StorageModeShared,
            MTLStorageMode::Private => objc2_metal::MTLResourceOptions::StorageModePrivate,
            other => {
                return Err(MetalError::OperationFailed(format!(
                    "Unsupported storage mode for repeat KV workspace: {other:?}"
                )));
            }
        };

        let buffer = self
            .device
            .newBufferWithLength_options(byte_len, resource_options)
            .ok_or(MetalError::BufferCreationFailed(byte_len))?;

        let host_accessible = matches!(storage_mode, MTLStorageMode::Shared | MTLStorageMode::Managed);
        Tensor::from_existing_buffer(buffer, dims, T::DTYPE, &self.device, &self.command_queue, 0, host_accessible)
    }
}
