use metallic_env::{FOUNDRY_DECODE_BATCH_SIZE, KV_PREFIX_CACHE_DISABLE, KV_PREFIX_CACHE_ENTRIES, MAX_PREFILL_CHUNK, PREFILL_CHUNK_SIZE};

use super::*;

impl CompiledModel {
    pub(super) fn compute_cache_namespace_fingerprint(spec: &ModelSpec, weights: &WeightBundle) -> std::sync::Arc<str> {
        let model = weights.model();
        let metadata = model.metadata();
        let mut hasher = DefaultHasher::new();

        spec.name.hash(&mut hasher);
        spec.architecture.max_seq_len().hash(&mut hasher);
        spec.architecture.forward.len().hash(&mut hasher);
        spec.architecture.prepare.tensors.len().hash(&mut hasher);

        for (k, v) in &spec.architecture.params {
            k.hash(&mut hasher);
            match v {
                ArchValue::USize(n) => n.hash(&mut hasher),
                ArchValue::F32(f) => f.to_bits().hash(&mut hasher),
            }
        }

        for key in [
            "general.architecture",
            "general.name",
            "general.basename",
            "general.type",
            "tokenizer.ggml.model",
        ] {
            if let Some(v) = metadata.get_string(key) {
                key.hash(&mut hasher);
                v.as_ref().hash(&mut hasher);
            }
        }

        for key in ["general.file_type", "general.quantization_version", "general.parameter_count"] {
            if let Some(v) = metadata.get_i64(key) {
                key.hash(&mut hasher);
                v.hash(&mut hasher);
            }
        }

        model.estimated_memory_usage().hash(&mut hasher);
        model.tensor_names().len().hash(&mut hasher);
        if let Some(info) = model.tensor_info("output.weight") {
            info.dimensions.hash(&mut hasher);
            info.data_type.hash(&mut hasher);
        }

        std::sync::Arc::<str>::from(format!("{:016x}", hasher.finish()))
    }

    pub(crate) fn prefill_config() -> (usize, usize) {
        // Defaults chosen to be "big enough to matter" but not explode memory (logits = max*V).
        const DEFAULT_MAX_PREFILL_CHUNK: usize = 32;
        const DEFAULT_PREFILL_CHUNK_SIZE: usize = 32;
        const MAX_ALLOWED: usize = 512;

        let mut max_prefill_chunk = MAX_PREFILL_CHUNK.get().ok().flatten().unwrap_or(DEFAULT_MAX_PREFILL_CHUNK);
        let mut prefill_chunk_size = PREFILL_CHUNK_SIZE.get().ok().flatten().unwrap_or(DEFAULT_PREFILL_CHUNK_SIZE);

        max_prefill_chunk = max_prefill_chunk.clamp(1, MAX_ALLOWED);
        prefill_chunk_size = prefill_chunk_size.clamp(1, MAX_ALLOWED);

        // Allocation must cover the largest runtime M.
        if prefill_chunk_size > max_prefill_chunk {
            max_prefill_chunk = prefill_chunk_size;
        }

        (max_prefill_chunk, prefill_chunk_size)
    }

    #[inline]
    pub(crate) fn rebalance_prefill_chunk_size(prompt_len: usize, requested: usize, max_allowed: usize) -> usize {
        let requested = requested.max(1).min(max_allowed.max(1));
        if prompt_len <= 1 {
            return 1;
        }
        let chunks = prompt_len.div_ceil(requested);
        let balanced = prompt_len.div_ceil(chunks);
        balanced.max(1).min(max_allowed)
    }

    pub(super) fn kv_prefix_cache_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| !KV_PREFIX_CACHE_DISABLE.get().ok().flatten().unwrap_or(false))
    }

    pub(super) fn kv_prefix_cache_max_entries() -> usize {
        const DEFAULT_MAX_ENTRIES: usize = 8;
        const MAX_ALLOWED: usize = 64;
        static MAX_ENTRIES: OnceLock<usize> = OnceLock::new();
        *MAX_ENTRIES.get_or_init(|| {
            KV_PREFIX_CACHE_ENTRIES
                .get()
                .ok()
                .flatten()
                .unwrap_or(DEFAULT_MAX_ENTRIES)
                .clamp(1, MAX_ALLOWED)
        })
    }

    #[inline]
    pub(super) fn decode_batch_size_or(default_decode_batch_size: usize, max: usize) -> usize {
        FOUNDRY_DECODE_BATCH_SIZE
            .get()
            .ok()
            .flatten()
            .unwrap_or(default_decode_batch_size)
            .clamp(1, max)
    }
}
