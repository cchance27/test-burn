use crate::{
    error::MetalError, spec::{Architecture, StorageClass, TensorBindings}, types::TensorArg
};

/// KV cache layout used by attention kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheLayout {
    /// Cache stores one slot per query head: [n_heads, seq, head_dim].
    ExpandedHeads,
    /// Cache stores one slot per KV head: [n_kv_heads, seq, head_dim].
    CompactKvHeads,
}

/// Resolved KV geometry shared across model/runtime paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvGeometry {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub group_size: usize,
    pub head_dim: usize,
    pub layout: KvCacheLayout,
}

impl KvGeometry {
    #[inline]
    pub fn from_architecture(arch: &Architecture) -> Self {
        let n_heads = arch.n_heads().max(1);
        let raw_n_kv = arch.n_kv_heads();
        let n_kv_heads = if raw_n_kv == 0 { n_heads } else { raw_n_kv.min(n_heads) }.max(1);

        let group_size = if n_heads.is_multiple_of(n_kv_heads) {
            n_heads / n_kv_heads
        } else {
            1
        };

        let head_dim = arch.d_model() / n_heads;
        let layout = infer_layout_from_prepare(arch, n_heads, n_kv_heads).unwrap_or({
            if n_kv_heads < n_heads {
                KvCacheLayout::CompactKvHeads
            } else {
                KvCacheLayout::ExpandedHeads
            }
        });

        let geometry = Self {
            n_heads,
            n_kv_heads,
            group_size,
            head_dim,
            layout,
        };
        tracing::trace!(
            n_heads = geometry.n_heads,
            n_kv_heads = geometry.n_kv_heads,
            group_size = geometry.group_size,
            head_dim = geometry.head_dim,
            cache_heads = geometry.cache_heads(),
            layout = ?geometry.layout,
            "Resolved KV geometry from architecture"
        );
        geometry
    }

    #[inline]
    pub const fn cache_heads(&self) -> usize {
        match self.layout {
            KvCacheLayout::ExpandedHeads => self.n_heads,
            KvCacheLayout::CompactKvHeads => self.n_kv_heads,
        }
    }

    #[inline]
    pub fn kv_head_for_query_head(&self, query_head: usize) -> usize {
        match self.layout {
            KvCacheLayout::ExpandedHeads => query_head,
            KvCacheLayout::CompactKvHeads => query_head / self.group_size.max(1),
        }
    }

    #[inline]
    pub fn per_token_bytes_f16(&self, n_layers: usize) -> usize {
        // K + V (2), cache_heads, head_dim, F16 bytes (2)
        n_layers
            .saturating_mul(2)
            .saturating_mul(self.cache_heads())
            .saturating_mul(self.head_dim)
            .saturating_mul(2)
    }

    /// Validate/cache geometry from a concrete KV tensor.
    pub fn from_cache_tensor(arch: &Architecture, tensor: &TensorArg, expected_capacity: usize) -> Result<Self, MetalError> {
        let base = Self::from_architecture(arch);

        if tensor.dtype != crate::tensor::Dtype::F16 {
            return Err(MetalError::InvalidShape(format!(
                "KV cache tensor must be F16, got {:?}",
                tensor.dtype
            )));
        }

        if tensor.dims.len() != 3 {
            return Err(MetalError::InvalidShape(format!(
                "KV cache tensor must be rank-3 [heads, seq, head_dim], got {:?}",
                tensor.dims
            )));
        }

        let cache_heads = tensor.dims[0];
        let seq = tensor.dims[1];
        let head_dim = tensor.dims[2];

        if seq != expected_capacity {
            return Err(MetalError::InvalidShape(format!(
                "KV cache capacity mismatch: got {}, expected {}",
                seq, expected_capacity
            )));
        }
        if head_dim != base.head_dim {
            return Err(MetalError::InvalidShape(format!(
                "KV cache head_dim mismatch: got {}, expected {}",
                head_dim, base.head_dim
            )));
        }

        let layout = if cache_heads == base.n_heads {
            KvCacheLayout::ExpandedHeads
        } else if cache_heads == base.n_kv_heads {
            KvCacheLayout::CompactKvHeads
        } else {
            return Err(MetalError::InvalidShape(format!(
                "KV cache heads mismatch: got {}, expected {} (expanded) or {} (compact)",
                cache_heads, base.n_heads, base.n_kv_heads
            )));
        };

        let geometry = Self { layout, ..base };
        tracing::trace!(
            dims = ?tensor.dims,
            strides = ?tensor.strides,
            expected_capacity,
            n_heads = geometry.n_heads,
            n_kv_heads = geometry.n_kv_heads,
            group_size = geometry.group_size,
            cache_heads = geometry.cache_heads(),
            layout = ?geometry.layout,
            "Validated KV tensor geometry"
        );
        Ok(geometry)
    }
}

fn infer_layout_from_prepare(arch: &Architecture, n_heads: usize, n_kv_heads: usize) -> Option<KvCacheLayout> {
    let mut bindings = TensorBindings::new();
    for (k, v) in &arch.params {
        if let Some(u) = v.as_usize() {
            bindings.set_int_global(k, u);
        }
    }

    for tensor in &arch.prepare.tensors {
        if tensor.storage != StorageClass::KvCache || tensor.dims.is_empty() {
            continue;
        }
        let first_dim = &tensor.dims[0];
        if first_dim.vars().iter().any(|name| bindings.get_int_global(name.as_ref()).is_none()) {
            continue;
        }
        let heads = first_dim.eval(&bindings);
        if heads == n_heads {
            tracing::trace!(
                tensor_name = %tensor.name,
                first_dim_heads = heads,
                n_heads,
                n_kv_heads,
                "KV layout inferred from prepare tensor dims: expanded"
            );
            return Some(KvCacheLayout::ExpandedHeads);
        }
        if heads == n_kv_heads {
            tracing::trace!(
                tensor_name = %tensor.name,
                first_dim_heads = heads,
                n_heads,
                n_kv_heads,
                "KV layout inferred from prepare tensor dims: compact"
            );
            return Some(KvCacheLayout::CompactKvHeads);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{ArchValue, Architecture};

    fn mock_arch() -> Architecture {
        let mut params = rustc_hash::FxHashMap::default();
        params.insert("d_model".to_string(), ArchValue::USize(3072));
        params.insert("n_heads".to_string(), ArchValue::USize(24));
        params.insert("n_kv_heads".to_string(), ArchValue::USize(8));
        params.insert("n_layers".to_string(), ArchValue::USize(28));
        params.insert("ff_dim".to_string(), ArchValue::USize(8192));
        params.insert("vocab_size".to_string(), ArchValue::USize(128256));
        params.insert("max_seq_len".to_string(), ArchValue::USize(8192));
        params.insert("rope_base".to_string(), ArchValue::F32(500000.0));
        params.insert("rms_eps".to_string(), ArchValue::F32(1e-5));

        Architecture {
            params,
            tensor_names: Default::default(),
            metadata_keys: Default::default(),
            prepare: Default::default(),
            weight_bindings: Vec::new(),
            forward: Vec::new(),
        }
    }

    #[test]
    fn kv_geometry_compact_defaults_for_gqa() {
        let g = KvGeometry::from_architecture(&mock_arch());
        assert_eq!(g.layout, KvCacheLayout::CompactKvHeads);
        assert_eq!(g.n_heads, 24);
        assert_eq!(g.n_kv_heads, 8);
        assert_eq!(g.group_size, 3);
        assert_eq!(g.head_dim, 128);
        assert_eq!(g.cache_heads(), 8);
    }
}
