use super::*;

impl CompiledModel {
    fn collect_kv_tensor_names(&self, bindings: &mut TensorBindings) -> Result<Vec<String>, MetalError> {
        let mut names = Vec::new();
        for tensor in &self.spec.architecture.prepare.tensors {
            if tensor.storage != StorageClass::KvCache {
                continue;
            }

            if let Some(repeat) = &tensor.repeat {
                let count_val = if let Ok(v) = repeat.count.parse::<usize>() {
                    v
                } else {
                    bindings
                        .get_var(&repeat.count)
                        .ok_or_else(|| {
                            MetalError::InvalidOperation(format!(
                                "prepare.tensors repeat count variable '{}' not found while collecting KV names",
                                repeat.count
                            ))
                        })?
                        .parse::<usize>()
                        .map_err(|e| {
                            MetalError::InvalidOperation(format!(
                                "prepare.tensors repeat count variable '{}' is not a valid integer while collecting KV names: {}",
                                repeat.count, e
                            ))
                        })?
                };

                bindings.push_scope();
                for i in 0..count_val {
                    bindings.set_var(&repeat.var, i.to_string());
                    names.push(bindings.interpolate(tensor.name.clone()));
                }
                bindings.pop_scope();
            } else {
                names.push(bindings.interpolate(tensor.name.clone()));
            }
        }

        Ok(names)
    }

    fn kv_tensor_copy_shape(arg: &TensorArg, name: &str, prefix_len: usize) -> Result<(usize, usize, usize, usize), MetalError> {
        if arg.dims.len() != 3 {
            return Err(MetalError::InvalidShape(format!(
                "KV tensor '{name}' must be rank-3 for prefix caching, got dims={:?}",
                arg.dims
            )));
        }
        let heads = arg.dims[0];
        let capacity = arg.dims[1];
        let head_dim = arg.dims[2];
        if prefix_len > capacity {
            return Err(MetalError::InvalidShape(format!(
                "KV tensor '{name}' prefix_len {} exceeds capacity {}",
                prefix_len, capacity
            )));
        }

        // Prefix snapshot/restore assumes contiguous [heads, seq, dim] layout.
        if arg.strides.len() == 3 {
            let expected = [capacity.saturating_mul(head_dim), head_dim, 1];
            if arg.strides[0] != expected[0] || arg.strides[1] != expected[1] || arg.strides[2] != expected[2] {
                return Err(MetalError::InvalidShape(format!(
                    "KV tensor '{name}' has unsupported strides {:?}, expected {:?}",
                    arg.strides, expected
                )));
            }
        }

        Ok((heads, capacity, head_dim, arg.dtype.size_bytes()))
    }

    fn capture_kv_prefix_snapshot(
        &self,
        foundry: &mut Foundry,
        session: &mut ModelSession,
        prompt_tokens: &[u32],
        prefix_len: usize,
        key: Option<&str>,
    ) -> Result<Option<KvPrefixSnapshot>, MetalError> {
        if prefix_len == 0 {
            return Ok(None);
        }

        let kv_names = self.collect_kv_tensor_names(&mut session.bindings)?;
        if kv_names.is_empty() {
            tracing::debug!(
                target: "metallic_foundry::model::executor",
                model = self.name(),
                prefix_len,
                "KV prefix snapshot skipped: no kv_cache tensors configured"
            );
            return Ok(None);
        }

        let mut total_bytes = 0usize;
        let mut kv_tensors = Vec::with_capacity(kv_names.len());
        let started_capture = !foundry.is_capturing();
        if started_capture {
            foundry.start_capture()?;
        }

        let copy_result = (|| -> Result<(), MetalError> {
            for name in kv_names {
                let live = session.bindings.get(&name)?;
                let (heads, capacity, head_dim, elem_bytes) = Self::kv_tensor_copy_shape(&live, &name, prefix_len)?;
                let bytes_per_head = prefix_len
                    .checked_mul(head_dim)
                    .and_then(|v| v.checked_mul(elem_bytes))
                    .ok_or_else(|| MetalError::InvalidShape(format!("KV tensor '{name}' snapshot size overflow")))?;
                let snapshot_bytes = heads
                    .checked_mul(bytes_per_head)
                    .ok_or_else(|| MetalError::InvalidShape(format!("KV tensor '{name}' snapshot bytes overflow")))?;
                if snapshot_bytes == 0 {
                    continue;
                }

                let snapshot_buffer = foundry
                    .device
                    .new_buffer(snapshot_bytes, MetalResourceOptions::StorageModePrivate)
                    .ok_or_else(|| {
                        MetalError::OperationFailed(format!(
                            "Failed to allocate {} bytes for KV prefix snapshot '{}'",
                            snapshot_bytes, name
                        ))
                    })?;

                let live_buffer = live
                    .buffer
                    .as_ref()
                    .ok_or_else(|| MetalError::InvalidOperation(format!("KV tensor '{}' has no backing buffer", name)))?;

                for h in 0..heads {
                    let src_head_offset = h
                        .checked_mul(capacity)
                        .and_then(|v| v.checked_mul(head_dim))
                        .and_then(|v| v.checked_mul(elem_bytes))
                        .ok_or_else(|| MetalError::InvalidShape(format!("KV tensor '{name}' source offset overflow")))?;
                    let dst_head_offset = h
                        .checked_mul(bytes_per_head)
                        .ok_or_else(|| MetalError::InvalidShape(format!("KV tensor '{name}' destination offset overflow")))?;
                    foundry.blit_copy(
                        live_buffer,
                        live.offset.saturating_add(src_head_offset),
                        &snapshot_buffer,
                        dst_head_offset,
                        bytes_per_head,
                    )?;
                }

                total_bytes = total_bytes.saturating_add(snapshot_bytes);
                kv_tensors.push(KvPrefixTensorSnapshot {
                    name,
                    buffer: snapshot_buffer,
                    dtype: live.dtype,
                    heads,
                    head_dim,
                });
            }
            Ok(())
        })();

        if started_capture {
            match foundry.end_capture() {
                Ok(cmd) => cmd.wait_until_completed(),
                Err(end_err) => {
                    if copy_result.is_ok() {
                        return Err(end_err);
                    }
                    tracing::warn!(
                        target: "metallic_foundry::model::executor",
                        model = self.name(),
                        error = %end_err,
                        "KV prefix snapshot cleanup failed after copy error"
                    );
                }
            }
        }

        copy_result?;

        if kv_tensors.is_empty() {
            tracing::debug!(
                target: "metallic_foundry::model::executor",
                model = self.name(),
                prefix_len,
                "KV prefix snapshot skipped: resolved KV tensor set was empty"
            );
            return Ok(None);
        }

        Ok(Some(KvPrefixSnapshot {
            key: key.map(std::sync::Arc::<str>::from),
            tokens: prompt_tokens.to_vec().into(),
            prefix_len,
            kv_tensors,
            bytes: total_bytes,
        }))
    }

    fn restore_kv_prefix_snapshot(
        &self,
        foundry: &mut Foundry,
        session: &mut ModelSession,
        snapshot: &KvPrefixSnapshot,
    ) -> Result<(), MetalError> {
        let started_capture = !foundry.is_capturing();
        if started_capture {
            foundry.start_capture()?;
        }

        let copy_result = (|| -> Result<(), MetalError> {
            for saved in &snapshot.kv_tensors {
                let live = session.bindings.get(&saved.name)?;
                let (heads, capacity, head_dim, elem_bytes) = Self::kv_tensor_copy_shape(&live, &saved.name, snapshot.prefix_len)?;

                if live.dtype != saved.dtype {
                    return Err(MetalError::InvalidShape(format!(
                        "KV tensor '{}' dtype mismatch during snapshot restore: live={:?} snapshot={:?}",
                        saved.name, live.dtype, saved.dtype
                    )));
                }
                if heads != saved.heads || head_dim != saved.head_dim {
                    return Err(MetalError::InvalidShape(format!(
                        "KV tensor '{}' shape mismatch during snapshot restore: live=[{}, {}, {}], snapshot=[{}, {}, {}]",
                        saved.name, heads, capacity, head_dim, saved.heads, snapshot.prefix_len, saved.head_dim
                    )));
                }

                let bytes_per_head = snapshot
                    .prefix_len
                    .checked_mul(head_dim)
                    .and_then(|v| v.checked_mul(elem_bytes))
                    .ok_or_else(|| MetalError::InvalidShape(format!("KV tensor '{}' restore size overflow", saved.name)))?;
                if bytes_per_head == 0 {
                    continue;
                }

                let live_buffer = live
                    .buffer
                    .as_ref()
                    .ok_or_else(|| MetalError::InvalidOperation(format!("KV tensor '{}' has no backing buffer", saved.name)))?;

                for h in 0..heads {
                    let src_head_offset = h
                        .checked_mul(bytes_per_head)
                        .ok_or_else(|| MetalError::InvalidShape(format!("KV tensor '{}' source offset overflow", saved.name)))?;
                    let dst_head_offset = h
                        .checked_mul(capacity)
                        .and_then(|v| v.checked_mul(head_dim))
                        .and_then(|v| v.checked_mul(elem_bytes))
                        .ok_or_else(|| MetalError::InvalidShape(format!("KV tensor '{}' destination offset overflow", saved.name)))?;

                    foundry.blit_copy(
                        &saved.buffer,
                        src_head_offset,
                        live_buffer,
                        live.offset.saturating_add(dst_head_offset),
                        bytes_per_head,
                    )?;
                }
            }
            Ok(())
        })();

        if started_capture {
            match foundry.end_capture() {
                Ok(cmd) => cmd.wait_until_completed(),
                Err(end_err) => {
                    if copy_result.is_ok() {
                        return Err(end_err);
                    }
                    tracing::warn!(
                        target: "metallic_foundry::model::executor",
                        model = self.name(),
                        error = %end_err,
                        "KV prefix restore cleanup failed after copy error"
                    );
                }
            }
        }

        copy_result?;

        session.current_pos = snapshot.prefix_len;
        Ok(())
    }

    pub(crate) fn try_restore_kv_prefix_from_cache(
        &self,
        foundry: &mut Foundry,
        session: &mut ModelSession,
        prompt_tokens: &[u32],
    ) -> Result<Option<usize>, MetalError> {
        if !Self::kv_prefix_cache_enabled() || prompt_tokens.is_empty() {
            return Ok(None);
        }

        let mut cache = self.kv_prefix_cache.lock();
        let mut best_idx: Option<usize> = None;
        let mut best_len: usize = 0;
        for (idx, entry) in cache.entries.iter().enumerate() {
            if entry.prefix_len == 0 || entry.prefix_len > prompt_tokens.len() {
                continue;
            }
            let prefix = &prompt_tokens[..entry.prefix_len];
            if entry.tokens.as_ref() == prefix && entry.prefix_len >= best_len {
                best_idx = Some(idx);
                best_len = entry.prefix_len;
            }
        }

        let Some(idx) = best_idx else {
            tracing::debug!(
                target: "metallic_foundry::model::executor",
                model = self.name(),
                cache_fingerprint = self.cache_namespace_fingerprint(),
                prompt_tokens = prompt_tokens.len(),
                cache_entries = cache.entries.len(),
                "KV prefix cache miss (token prefix fallback)"
            );
            return Ok(None);
        };

        let snapshot = cache
            .entries
            .remove(idx)
            .ok_or_else(|| MetalError::OperationFailed("KV prefix cache internal remove failed".into()))?;
        drop(cache);

        let matched_prefix = snapshot.prefix_len;
        match self.restore_kv_prefix_snapshot(foundry, session, &snapshot) {
            Ok(()) => {
                let mut cache = self.kv_prefix_cache.lock();
                cache.entries.push_back(snapshot);
                tracing::debug!(
                    target: "metallic_foundry::model::executor",
                    model = self.name(),
                    cache_fingerprint = self.cache_namespace_fingerprint(),
                    matched_prefix_tokens = matched_prefix,
                    prompt_tokens = prompt_tokens.len(),
                    cache_entries = cache.entries.len(),
                    "KV prefix cache hit (token prefix fallback)"
                );
                Ok(Some(matched_prefix))
            }
            Err(err) => {
                tracing::warn!(
                    target: "metallic_foundry::model::executor",
                    model = self.name(),
                    cache_fingerprint = self.cache_namespace_fingerprint(),
                    matched_prefix_tokens = matched_prefix,
                    prompt_tokens = prompt_tokens.len(),
                    error = %err,
                    "KV prefix cache restore failed (token prefix fallback); entry evicted"
                );
                Ok(None)
            }
        }
    }

    pub(crate) fn try_restore_kv_prefix_from_cache_key(
        &self,
        foundry: &mut Foundry,
        session: &mut ModelSession,
        key: &str,
        prompt_tokens: &[u32],
    ) -> Result<Option<usize>, MetalError> {
        if !Self::kv_prefix_cache_enabled() || key.is_empty() {
            return Ok(None);
        }
        let prompt_tokens_len = prompt_tokens.len();

        let mut cache = self.kv_prefix_cache.lock();
        let mut best_idx: Option<usize> = None;
        let mut best_match_len: usize = 0;
        let mut best_snapshot_len: usize = 0;
        let mut best_mismatch_index: Option<usize> = None;
        let mut best_mismatch_snapshot_token: Option<u32> = None;
        let mut best_mismatch_prompt_token: Option<u32> = None;
        let mut key_matches = 0usize;
        let mut len_rejects = 0usize;
        let mut zero_prefix_rejects = 0usize;
        let mut token_mismatch_rejects = 0usize;
        let mut closest_match_len = 0usize;
        let mut closest_snapshot_len = 0usize;
        let mut closest_mismatch_index: Option<usize> = None;
        let mut closest_mismatch_snapshot_token: Option<u32> = None;
        let mut closest_mismatch_prompt_token: Option<u32> = None;
        for (idx, entry) in cache.entries.iter().enumerate() {
            if entry.key.as_deref() != Some(key) {
                continue;
            }
            key_matches = key_matches.saturating_add(1);
            if entry.prefix_len == 0 || entry.prefix_len > prompt_tokens_len {
                if entry.prefix_len == 0 {
                    zero_prefix_rejects = zero_prefix_rejects.saturating_add(1);
                } else {
                    len_rejects = len_rejects.saturating_add(1);
                }
                continue;
            }

            let candidate_len = entry.prefix_len.min(entry.tokens.len());
            let matched_prefix = entry
                .tokens
                .iter()
                .zip(prompt_tokens.iter())
                .take(candidate_len)
                .take_while(|(a, b)| a == b)
                .count();
            let mismatch = if matched_prefix < candidate_len && matched_prefix < prompt_tokens_len {
                Some((matched_prefix, entry.tokens[matched_prefix], prompt_tokens[matched_prefix]))
            } else {
                None
            };
            if matched_prefix >= closest_match_len {
                closest_match_len = matched_prefix;
                closest_snapshot_len = entry.prefix_len;
                if let Some((idx, snapshot_token, prompt_token)) = mismatch {
                    closest_mismatch_index = Some(idx);
                    closest_mismatch_snapshot_token = Some(snapshot_token);
                    closest_mismatch_prompt_token = Some(prompt_token);
                } else {
                    closest_mismatch_index = None;
                    closest_mismatch_snapshot_token = None;
                    closest_mismatch_prompt_token = None;
                }
            }
            if matched_prefix == 0 {
                token_mismatch_rejects = token_mismatch_rejects.saturating_add(1);
                continue;
            }
            if matched_prefix >= best_match_len {
                best_idx = Some(idx);
                best_match_len = matched_prefix;
                best_snapshot_len = entry.prefix_len;
                if let Some((idx, snapshot_token, prompt_token)) = mismatch {
                    best_mismatch_index = Some(idx);
                    best_mismatch_snapshot_token = Some(snapshot_token);
                    best_mismatch_prompt_token = Some(prompt_token);
                } else {
                    best_mismatch_index = None;
                    best_mismatch_snapshot_token = None;
                    best_mismatch_prompt_token = None;
                }
            }
        }

        let Some(idx) = best_idx else {
            tracing::debug!(
                target: "metallic_foundry::model::executor",
                model = self.name(),
                cache_fingerprint = self.cache_namespace_fingerprint(),
                key,
                prompt_tokens = prompt_tokens_len,
                key_matches,
                len_rejects,
                zero_prefix_rejects,
                token_mismatch_rejects,
                closest_match_len,
                closest_snapshot_len,
                closest_mismatch_index = ?closest_mismatch_index,
                closest_mismatch_snapshot_token = ?closest_mismatch_snapshot_token,
                closest_mismatch_prompt_token = ?closest_mismatch_prompt_token,
                cache_entries = cache.entries.len(),
                "KV prefix cache miss (keyed)"
            );
            return Ok(None);
        };

        let snapshot = cache
            .entries
            .remove(idx)
            .ok_or_else(|| MetalError::OperationFailed("KV prefix cache keyed remove failed".into()))?;
        drop(cache);

        let matched_prefix = best_match_len;
        match self.restore_kv_prefix_snapshot(foundry, session, &snapshot) {
            Ok(()) => {
                session.current_pos = matched_prefix;
                let mut cache = self.kv_prefix_cache.lock();
                cache.entries.push_back(snapshot);
                tracing::debug!(
                    target: "metallic_foundry::model::executor",
                    model = self.name(),
                    cache_fingerprint = self.cache_namespace_fingerprint(),
                    key,
                    matched_prefix_tokens = matched_prefix,
                    snapshot_prefix_tokens = best_snapshot_len,
                    prompt_tokens = prompt_tokens_len,
                    partial = matched_prefix < best_snapshot_len,
                    first_mismatch_index = ?best_mismatch_index,
                    first_mismatch_snapshot_token = ?best_mismatch_snapshot_token,
                    first_mismatch_prompt_token = ?best_mismatch_prompt_token,
                    cache_entries = cache.entries.len(),
                    "KV prefix cache hit (keyed)"
                );
                Ok(Some(matched_prefix))
            }
            Err(err) => {
                tracing::warn!(
                    target: "metallic_foundry::model::executor",
                    model = self.name(),
                    cache_fingerprint = self.cache_namespace_fingerprint(),
                    key,
                    matched_prefix_tokens = matched_prefix,
                    snapshot_prefix_tokens = best_snapshot_len,
                    prompt_tokens = prompt_tokens_len,
                    first_mismatch_index = ?best_mismatch_index,
                    first_mismatch_snapshot_token = ?best_mismatch_snapshot_token,
                    first_mismatch_prompt_token = ?best_mismatch_prompt_token,
                    error = %err,
                    "KV prefix cache restore failed (keyed); entry evicted"
                );
                Ok(None)
            }
        }
    }

    pub(crate) fn store_kv_prefix_in_cache(
        &self,
        foundry: &mut Foundry,
        session: &mut ModelSession,
        prompt_tokens: &[u32],
        key: Option<&str>,
    ) -> Result<(), MetalError> {
        if !Self::kv_prefix_cache_enabled() || prompt_tokens.is_empty() {
            return Ok(());
        }

        let prefix_len = prompt_tokens.len();
        if session.current_pos < prefix_len {
            tracing::debug!(
                target: "metallic_foundry::model::executor",
                model = self.name(),
                cache_fingerprint = self.cache_namespace_fingerprint(),
                session_pos = session.current_pos,
                prefix_len,
                "KV prefix cache store skipped: session shorter than prefix"
            );
            return Ok(());
        }

        let Some(snapshot) = self.capture_kv_prefix_snapshot(foundry, session, prompt_tokens, prefix_len, key)? else {
            return Ok(());
        };

        let max_entries = Self::kv_prefix_cache_max_entries();
        let mut cache = self.kv_prefix_cache.lock();

        let replaced = if let Some(key) = key {
            if let Some(existing_idx) = cache.entries.iter().position(|e| e.key.as_deref() == Some(key)) {
                cache.entries.remove(existing_idx);
                true
            } else {
                false
            }
        } else if let Some(existing_idx) = cache.entries.iter().position(|e| e.tokens.as_ref() == snapshot.tokens.as_ref()) {
            cache.entries.remove(existing_idx);
            true
        } else {
            false
        };

        let evicted = if cache.entries.len() >= max_entries {
            cache.entries.pop_front()
        } else {
            None
        };

        let stored_prefix = snapshot.prefix_len;
        let stored_bytes = snapshot.bytes;
        cache.entries.push_back(snapshot);

        if let Some(evicted) = evicted {
            tracing::debug!(
                target: "metallic_foundry::model::executor",
                model = self.name(),
                cache_fingerprint = self.cache_namespace_fingerprint(),
                evicted_prefix_tokens = evicted.prefix_len,
                evicted_bytes = evicted.bytes,
                max_entries,
                "KV prefix cache eviction"
            );
        }

        tracing::debug!(
            target: "metallic_foundry::model::executor",
            model = self.name(),
            cache_fingerprint = self.cache_namespace_fingerprint(),
            stored_prefix_tokens = stored_prefix,
            stored_bytes,
            replaced,
            key = key.unwrap_or("<none>"),
            cache_entries = cache.entries.len(),
            max_entries,
            "KV prefix cache store"
        );

        Ok(())
    }

    /// Ensure that KV caches and related buffers have enough capacity for the requested length.
    /// Grows buffers on-demand if needed, respecting alignment and max context limits.
    pub(crate) fn ensure_kv_capacity(
        &self,
        foundry: &mut Foundry,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        context_config: &mut crate::model::context::ContextConfig,
        current_pos: usize,
        required_len: usize,
    ) -> Result<(), MetalError> {
        let kv_geometry = KvGeometry::from_architecture(self.architecture());
        tracing::trace!(
            layout = ?kv_geometry.layout,
            n_heads = kv_geometry.n_heads,
            n_kv_heads = kv_geometry.n_kv_heads,
            group_size = kv_geometry.group_size,
            cache_heads = kv_geometry.cache_heads(),
            current_pos,
            required_len,
            allocated_capacity = context_config.allocated_capacity,
            max_context_len = context_config.max_context_len,
            "ensure_kv_capacity: evaluating capacity request"
        );
        if required_len <= context_config.allocated_capacity {
            tracing::trace!(
                required_len,
                allocated_capacity = context_config.allocated_capacity,
                "ensure_kv_capacity: capacity already sufficient"
            );
            return Ok(());
        }

        if required_len > context_config.max_context_len {
            return Err(MetalError::InvalidOperation(format!(
                "Requested context length {} exceeds maximum allowed {}",
                required_len, context_config.max_context_len
            )));
        }

        // Calculate new capacity using growth strategy
        let mut new_capacity = match context_config.growth_strategy {
            crate::model::context::GrowthStrategy::FullReserve => context_config.max_context_len,
            crate::model::context::GrowthStrategy::GrowOnDemand { growth_factor, .. } => {
                let grown = (context_config.allocated_capacity as f32 * growth_factor) as usize;
                grown.max(required_len).min(context_config.max_context_len)
            }
        };

        // Enforce alignment to 128 elements to keep kernels happy
        new_capacity = crate::model::context::ContextConfig::align_capacity(new_capacity);

        tracing::info!(
            "Growing KV cache capacity: {} -> {} (requested {}, layout={:?}, cache_heads={})",
            context_config.allocated_capacity,
            new_capacity,
            required_len,
            kv_geometry.layout,
            kv_geometry.cache_heads()
        );

        let old_capacity = context_config.allocated_capacity;
        self.reallocate_kv_buffers(foundry, bindings, fast_bindings, current_pos, old_capacity, new_capacity)?;
        context_config.allocated_capacity = new_capacity;

        Ok(())
    }

    /// Reallocate all context-dependent buffers (KV caches, slice buffers, etc.).
    fn reallocate_kv_buffers(
        &self,
        foundry: &mut Foundry,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        current_pos: usize,
        old_capacity: usize,
        new_capacity: usize,
    ) -> Result<(), MetalError> {
        let arch = self.architecture();
        let head_dim = arch.d_model() / arch.n_heads();
        let kv_geometry = KvGeometry::from_architecture(arch);
        tracing::debug!(
            old_capacity,
            new_capacity,
            current_pos,
            layout = ?kv_geometry.layout,
            n_heads = kv_geometry.n_heads,
            n_kv_heads = kv_geometry.n_kv_heads,
            group_size = kv_geometry.group_size,
            cache_heads = kv_geometry.cache_heads(),
            head_dim = kv_geometry.head_dim,
            "Reallocating KV-aware buffers"
        );

        // Batch all buffer copies into a single command buffer when possible.
        let nested_capture = foundry.is_capturing();
        if !nested_capture {
            foundry.start_capture()?;
        }

        // 1. Update physical max context capacity for kernels and expression evaluation.
        self.set_global_usize(bindings, "max_seq_len", new_capacity);
        // Derived globals may depend on max_seq_len (e.g. kv_seq_len).
        self.apply_derived_globals(bindings);

        // 2. Reallocate any grow_with_kv tensors declared by the DSL.
        // Rope caches are handled by the dedicated RoPE grow path below.
        let preserve_kv_cache = |foundry: &mut Foundry, name: &str, old: &TensorArg, new: &TensorArg| -> Result<(), MetalError> {
            if current_pos == 0 {
                return Ok(());
            }
            if old.dtype != crate::tensor::Dtype::F16 || new.dtype != crate::tensor::Dtype::F16 {
                return Err(MetalError::InvalidShape(format!(
                    "KV cache '{name}' must be F16 for growth preservation"
                )));
            }

            let old_geom = KvGeometry::from_cache_tensor(arch, old, old_capacity)?;
            let new_geom = KvGeometry::from_cache_tensor(arch, new, new_capacity)?;
            tracing::trace!(
                tensor = %name,
                old_dims = ?old.dims,
                new_dims = ?new.dims,
                old_layout = ?old_geom.layout,
                new_layout = ?new_geom.layout,
                old_cache_heads = old_geom.cache_heads(),
                new_cache_heads = new_geom.cache_heads(),
                old_capacity,
                new_capacity,
                "Preserving KV cache tensor across growth"
            );
            if old_geom.layout != new_geom.layout {
                return Err(MetalError::InvalidShape(format!(
                    "KV cache '{name}' layout changed during growth: old={:?}, new={:?}",
                    old_geom.layout, new_geom.layout
                )));
            }
            if old_geom.cache_heads() != old.dims[0] || old_geom.head_dim != old.dims[2] {
                return Err(MetalError::InvalidShape(format!(
                    "KV cache '{name}' old dims mismatch: got {:?}, inferred heads={} head_dim={}",
                    old.dims,
                    old_geom.cache_heads(),
                    old_geom.head_dim
                )));
            }
            if new_geom.cache_heads() != new.dims[0] || new_geom.head_dim != new.dims[2] {
                return Err(MetalError::InvalidShape(format!(
                    "KV cache '{name}' new dims mismatch: got {:?}, inferred heads={} head_dim={}",
                    new.dims,
                    new_geom.cache_heads(),
                    new_geom.head_dim
                )));
            }

            let old_buf = old
                .buffer
                .as_ref()
                .ok_or_else(|| MetalError::InvalidOperation(format!("{name} buffer missing during growth")))?;
            let new_buf = new
                .buffer
                .as_ref()
                .ok_or_else(|| MetalError::InvalidOperation(format!("{name} buffer missing after growth")))?;

            let copy_size = current_pos * old_geom.head_dim * 2;
            for h in 0..old_geom.cache_heads() {
                let old_head_offset = h * old_capacity * old_geom.head_dim * 2;
                let new_head_offset = h * new_capacity * old_geom.head_dim * 2;
                foundry.blit_copy(
                    old_buf,
                    old.offset + old_head_offset,
                    new_buf,
                    new.offset + new_head_offset,
                    copy_size,
                )?;
            }
            tracing::trace!(
                tensor = %name,
                copied_heads = old_geom.cache_heads(),
                copied_tokens = current_pos,
                copied_bytes_per_head = copy_size,
                "Preserved KV cache contents for grown tensor"
            );
            Ok(())
        };

        for tensor in &arch.prepare.tensors {
            if !tensor.grow_with_kv {
                continue;
            }
            if tensor.storage == StorageClass::RopeCache {
                continue;
            }

            let mut alloc_one = |bindings: &mut TensorBindings, name: String| -> Result<(), MetalError> {
                let old = bindings.get(&name).ok();
                bindings.remove(&name);

                let dims: Vec<usize> = tensor.dims.iter().map(|e| e.eval(bindings)).collect::<Vec<usize>>();
                let strides: Vec<usize> = if let Some(strides) = &tensor.strides {
                    strides.iter().map(|e| e.eval(bindings)).collect::<Vec<usize>>()
                } else {
                    crate::tensor::compute_strides(&dims)
                };

                self.allocate_tensor_from_spec(
                    foundry,
                    bindings,
                    fast_bindings,
                    &name,
                    tensor.dtype,
                    dims,
                    strides,
                    tensor.storage,
                    tensor.zero_fill,
                )?;

                if tensor.storage == StorageClass::KvCache {
                    let old = old.ok_or_else(|| MetalError::InvalidOperation(format!("Missing KV cache '{name}' during growth")))?;
                    let new = bindings.get(&name)?;
                    tracing::trace!(
                        tensor = %name,
                        old_dims = ?old.dims,
                        new_dims = ?new.dims,
                        "KV cache reallocated; starting preservation copy"
                    );
                    preserve_kv_cache(foundry, &name, &old, &new)?;
                }

                Ok(())
            };

            if let Some(repeat) = &tensor.repeat {
                let count_val = if let Ok(v) = repeat.count.parse::<usize>() {
                    v
                } else {
                    bindings
                        .get_var(&repeat.count)
                        .ok_or_else(|| {
                            MetalError::InvalidOperation(format!(
                                "prepare.tensors repeat count variable '{}' not found during growth",
                                repeat.count
                            ))
                        })?
                        .parse::<usize>()
                        .map_err(|e| {
                            MetalError::InvalidOperation(format!(
                                "prepare.tensors repeat count variable '{}' is not a valid integer during growth: {}",
                                repeat.count, e
                            ))
                        })?
                };
                bindings.push_scope();
                for i in 0..count_val {
                    bindings.set_var(&repeat.var, i.to_string());
                    let resolved = bindings.interpolate(tensor.name.clone());
                    alloc_one(bindings, resolved)?;
                }
                bindings.pop_scope();
            } else {
                let resolved = bindings.interpolate(tensor.name.clone());
                alloc_one(bindings, resolved)?;
            }
        }

        // 3. Grow RoPE tables to match the new physical capacity.
        let dim_half = head_dim / 2;
        if let Some(rope) = arch.prepare.rope.as_ref() {
            let rope_freq_factors = self.load_rope_freq_factors(dim_half)?;
            self.ensure_rope_capacity_named(
                bindings,
                fast_bindings,
                foundry,
                arch.rope_base(),
                head_dim,
                dim_half,
                rope_freq_factors.as_deref(),
                &rope.cos,
                &rope.sin,
                old_capacity,
                new_capacity,
            )?;
        }

        if !nested_capture {
            let cmd = foundry.end_capture()?;
            cmd.wait_until_completed();
        }

        Ok(())
    }
}
