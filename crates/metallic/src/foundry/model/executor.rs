//! CompiledModel executor for running inference.
//!
//! Interprets the ModelSpec execution plan (DSL) by calling Step::execute().

use half::f16;
use rustc_hash::FxHashMap;

use super::builder::WeightBundle;
use crate::{
    error::MetalError, foundry::{
        Foundry, spec::{ModelSpec, TensorBindings}
    }, gguf::{GGUFDataType, tensor_info::GGUFRawTensor}, types::TensorArg
};

/// A compiled, ready-to-run model.
///
/// Created via `ModelBuilder::build()`, this struct holds everything
/// needed to run inference. The forward pass is executed by iterating
/// through the DSL steps defined in the ModelSpec.
pub struct CompiledModel {
    spec: ModelSpec,
    weights: WeightBundle,
}

impl CompiledModel {
    /// Create a new CompiledModel from spec and weights.
    pub(crate) fn new(spec: ModelSpec, weights: WeightBundle) -> Result<Self, MetalError> {
        if let Some(gguf_arch) = weights.architecture() {
            tracing::debug!("Loading model: spec='{}' gguf_arch='{}'", spec.name, gguf_arch);
        }

        tracing::info!("CompiledModel created with {} forward steps", spec.architecture.forward.len());

        Ok(Self { spec, weights })
    }

    /// Get the model name from the spec.
    pub fn name(&self) -> &str {
        &self.spec.name
    }

    /// Get the architecture configuration.
    pub fn architecture(&self) -> &crate::foundry::spec::Architecture {
        &self.spec.architecture
    }

    /// Get access to the underlying weights bundle.
    pub fn weights(&self) -> &WeightBundle {
        &self.weights
    }

    /// Get access to the underlying GGUF model for tensor materialization.
    pub fn gguf_model(&self) -> &crate::gguf::model_loader::GGUFModel {
        self.weights.gguf_model()
    }

    /// Create a tokenizer from the model's GGUF metadata.
    pub fn tokenizer(&self) -> Result<crate::Tokenizer, MetalError> {
        crate::Tokenizer::from_gguf_metadata(self.weights.gguf_model().get_metadata())
    }

    /// Prepare tensor bindings by:
    /// 1. Setting config globals (n_layers, d_model, etc.)
    /// 2. Materializing weight tensors from GGUF using logical name resolution
    /// 3. Allocating intermediate buffers for activations
    pub fn prepare_bindings(&self, foundry: &mut Foundry) -> Result<TensorBindings, MetalError> {
        let mut bindings = TensorBindings::new();
        let _gguf = self.weights.gguf_model();
        let arch = &self.spec.architecture;
        let tensor_names = &arch.tensor_names;

        // 1. Set config globals for DSL variable interpolation
        bindings.set_global("n_layers", arch.n_layers.to_string());
        bindings.set_global("d_model", arch.d_model.to_string());
        bindings.set_global("n_heads", arch.n_heads.to_string());
        bindings.set_global("n_kv_heads", arch.n_kv_heads.to_string());
        bindings.set_global("ff_dim", arch.ff_dim.to_string());
        bindings.set_global("vocab_size", arch.vocab_size.to_string());
        let kv_dim = arch.d_model / arch.n_heads * arch.n_kv_heads;
        bindings.set_global("kv_dim", kv_dim.to_string());

        // Build a set of available GGUF tensor names for resolution
        let available: FxHashMap<String, ()> = self.weights.tensor_names().map(|name| (name.clone(), ())).collect();

        // 2. Resolve and bind global weight tensors
        let global_keys = ["embedding", "output_weight", "final_norm", "rope_cos", "rope_sin"];
        for key in global_keys {
            if let Some(gguf_name) = tensor_names.resolve(key, None, &available) {
                self.bind_gguf_tensor(&mut bindings, foundry, &gguf_name, key)?;
            }
        }

        // 3. Resolve and bind per-layer weight tensors
        // Non-FFN weights use regular binding
        let regular_layer_keys = ["layer.attn_norm", "layer.ffn_norm"];
        // FFN and Attn weights use canonical k-block-major conversion for GemvCanonical
        let canonical_layer_keys = [
            "layer.ffn_gate",
            "layer.ffn_up",
            "layer.ffn_down",
            "layer.attn_q",
            "layer.attn_k",
            "layer.attn_v",
            "layer.attn_output",
        ];
        for i in 0..arch.n_layers {
            for key in regular_layer_keys {
                if let Some(gguf_name) = tensor_names.resolve(key, Some(i), &available) {
                    let logical_name = format!("{key}_{i}");
                    self.bind_gguf_tensor(&mut bindings, foundry, &gguf_name, &logical_name)?;
                }
            }
            for key in canonical_layer_keys {
                if let Some(gguf_name) = tensor_names.resolve(key, Some(i), &available) {
                    let logical_name = format!("{key}_{i}");
                    let (expected_k, expected_n) = match key {
                        "layer.attn_q" => (arch.d_model, arch.d_model),
                        "layer.attn_k" => (arch.d_model, kv_dim),
                        "layer.attn_v" => (arch.d_model, kv_dim),
                        "layer.attn_output" => (arch.d_model, arch.d_model),
                        "layer.ffn_gate" => (arch.d_model, arch.ff_dim),
                        "layer.ffn_up" => (arch.d_model, arch.ff_dim),
                        "layer.ffn_down" => (arch.ff_dim, arch.d_model),
                        _ => {
                            return Err(MetalError::InvalidShape(format!(
                                "Unhandled canonical weight key '{}'; update executor canonical binding map",
                                key
                            )));
                        }
                    };
                    self.bind_gguf_tensor_canonical(&mut bindings, foundry, &gguf_name, &logical_name, expected_k, expected_n)?;
                }
            }
        }

        // 3b. Resolve and bind per-layer bias tensors (fallback to zero if missing)
        let kv_dim = arch.d_model / arch.n_heads * arch.n_kv_heads;
        let mut zero_cache: FxHashMap<usize, TensorArg> = FxHashMap::default();
        let bias_specs = [
            ("layer.attn_q_bias", arch.d_model),
            ("layer.attn_k_bias", kv_dim),
            ("layer.attn_v_bias", kv_dim),
            ("layer.ffn_gate_bias", arch.ff_dim),
            ("layer.ffn_up_bias", arch.ff_dim),
            ("layer.ffn_down_bias", arch.d_model),
        ];

        for i in 0..arch.n_layers {
            for (key, size) in bias_specs {
                let logical_name = format!("{key}_{i}");
                if let Some(gguf_name) = tensor_names.resolve(key, Some(i), &available) {
                    self.bind_gguf_tensor(&mut bindings, foundry, &gguf_name, &logical_name)?;
                } else {
                    let zero = if let Some(tensor) = zero_cache.get(&size) {
                        tensor.clone()
                    } else {
                        let tensor = self.zero_tensor_arg(foundry, size)?;
                        zero_cache.insert(size, tensor.clone());
                        tensor
                    };
                    bindings.insert(logical_name, zero);
                }
            }
        }

        // 4. Allocate intermediate buffers
        // These are named buffers used by the forward pass for activations
        // For now, we use seq_len=1 (single token inference), batch=1
        let batch = 1;
        let seq_len = 1; // Will be updated per-forward based on input

        // 1D buffers for general intermediates
        self.allocate_intermediate(&mut bindings, foundry, "hidden", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, foundry, "norm_out", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, foundry, "proj_out", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, foundry, "residual_1", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, foundry, "ffn_norm_out", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, foundry, "gate", arch.ff_dim)?;
        self.allocate_intermediate(&mut bindings, foundry, "up", arch.ff_dim)?;
        self.allocate_intermediate(&mut bindings, foundry, "ffn_out", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, foundry, "final_norm_out", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, foundry, "logits", arch.vocab_size)?;

        // 3D buffers for SDPA (batch, seq_len, dim)
        let kv_dim = arch.d_model / arch.n_heads * arch.n_kv_heads;
        self.allocate_intermediate_3d(&mut bindings, foundry, "q", batch, seq_len, arch.d_model)?;
        self.allocate_intermediate_3d(&mut bindings, foundry, "k", batch, seq_len, kv_dim)?;
        self.allocate_intermediate_3d(&mut bindings, foundry, "v", batch, seq_len, kv_dim)?;
        let head_dim = arch.d_model / arch.n_heads;
        self.allocate_intermediate_3d(&mut bindings, foundry, "q_heads", batch * arch.n_heads, seq_len, head_dim)?;
        self.allocate_intermediate_3d(&mut bindings, foundry, "k_heads", batch * arch.n_kv_heads, seq_len, head_dim)?;
        self.allocate_intermediate_3d(&mut bindings, foundry, "v_heads", batch * arch.n_kv_heads, seq_len, head_dim)?;
        self.allocate_intermediate_3d(&mut bindings, foundry, "q_rot", batch, seq_len, arch.d_model)?;
        self.allocate_intermediate_3d(&mut bindings, foundry, "k_rot", batch, seq_len, kv_dim)?;
        // Expanded K/V buffers for GQA (after RepeatKvHeads, same dim as Q)
        // Must be sized for MAX sequence length because they hold repeated history
        let max_seq_len_for_slice = 2048;
        let expanded_dim = batch * max_seq_len_for_slice * arch.d_model;
        self.allocate_intermediate(&mut bindings, foundry, "k_expanded", expanded_dim)?;
        self.allocate_intermediate(&mut bindings, foundry, "v_expanded", expanded_dim)?;
        // SDPA output is per-step (seq_len=1) in incremental decode; keep this compact.
        self.allocate_intermediate(&mut bindings, foundry, "attn_out", arch.d_model)?;

        // KV slice buffers for cache reads (sized for max sequence to avoid reallocation)
        // These hold the sliced cache [n_kv_heads, current_seq_len, head_dim]
        // We allocate to max size and track actual usage via globals
        let max_seq_len_for_slice = 2048;
        let kv_slice_dim = arch.n_kv_heads * max_seq_len_for_slice * head_dim;
        self.allocate_intermediate(&mut bindings, foundry, "k_slice", kv_slice_dim)?;
        self.allocate_intermediate(&mut bindings, foundry, "v_slice", kv_slice_dim)?;

        // 5. Create a "zero" buffer for unused bias/residual slots
        self.allocate_intermediate(&mut bindings, foundry, "zero", 1)?;

        // 6. Allocate KV cache for autoregressive generation
        // Shape: [n_kv_heads, max_seq_len, head_dim] per layer
        // We'll create one k_cache and v_cache per layer, named k_cache_0, k_cache_1, etc.
        let max_seq_len = 2048; // Maximum context length
        let head_dim = arch.d_model / arch.n_heads;
        for layer_idx in 0..arch.n_layers {
            let k_cache_name = format!("k_cache_{}", layer_idx);
            let v_cache_name = format!("v_cache_{}", layer_idx);
            // KV cache: [n_kv_heads, max_seq_len, head_dim]
            self.allocate_kv_cache(&mut bindings, foundry, &k_cache_name, arch.n_kv_heads, max_seq_len, head_dim)?;
            self.allocate_kv_cache(&mut bindings, foundry, &v_cache_name, arch.n_kv_heads, max_seq_len, head_dim)?;
        }
        // Store max_seq_len as a global for kernels to use
        bindings.set_global("max_seq_len", max_seq_len.to_string());
        bindings.set_global("head_dim", head_dim.to_string());
        bindings.set_global("n_kv_heads", arch.n_kv_heads.to_string());
        bindings.set_global("n_heads", arch.n_heads.to_string());

        // 7. Compute and bind RoPE cos/sin caches
        // These are precomputed based on model config since they're not in GGUF
        self.compute_and_bind_rope_caches(&mut bindings, foundry, arch)?;

        tracing::info!("Prepared {} bindings (weights + intermediates + RoPE + KV cache)", bindings.len());
        Ok(bindings)
    }

    /// Compute and bind RoPE cos/sin cache tables.
    ///
    /// RoPE caches are not stored in GGUF, so we compute them based on arch config.
    fn compute_and_bind_rope_caches(
        &self,
        bindings: &mut TensorBindings,
        foundry: &mut Foundry,
        arch: &crate::foundry::spec::Architecture,
    ) -> Result<(), MetalError> {
        use half::f16;
        use objc2_metal::MTLDevice;

        let head_dim = arch.d_model / arch.n_heads;
        let dim_half = head_dim / 2;
        let max_seq_len = arch.max_seq_len.min(4096); // Cap at reasonable size for memory
        let rope_base = arch.rope_base;

        // Compute cos and sin tables
        let table_size = max_seq_len * dim_half;
        let mut cos_data: Vec<f16> = vec![f16::ZERO; table_size];
        let mut sin_data: Vec<f16> = vec![f16::ZERO; table_size];

        for pos in 0..max_seq_len {
            for i in 0..dim_half {
                let idx = pos * dim_half + i;
                let exponent = (2 * i) as f32 / head_dim as f32;
                let inv_freq = 1.0f32 / rope_base.powf(exponent);
                let angle = pos as f32 * inv_freq;
                cos_data[idx] = f16::from_f32(angle.cos());
                sin_data[idx] = f16::from_f32(angle.sin());
            }
        }

        // Allocate and fill cos buffer
        let cos_byte_size = table_size * 2; // F16 = 2 bytes
        let cos_buffer = foundry
            .device
            .0
            .newBufferWithLength_options(cos_byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed("Failed to allocate rope_cos buffer".into()))?;

        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = cos_buffer.contents().as_ptr() as *mut f16;
            std::ptr::copy_nonoverlapping(cos_data.as_ptr(), ptr, table_size);
        }

        let cos_tensor = TensorArg::from_buffer(
            crate::types::MetalBuffer(cos_buffer),
            crate::tensor::Dtype::F16,
            vec![max_seq_len, dim_half],
            vec![dim_half, 1],
        );
        bindings.insert("rope_cos".to_string(), cos_tensor);

        // Allocate and fill sin buffer
        let sin_buffer = foundry
            .device
            .0
            .newBufferWithLength_options(cos_byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed("Failed to allocate rope_sin buffer".into()))?;

        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = sin_buffer.contents().as_ptr() as *mut f16;
            std::ptr::copy_nonoverlapping(sin_data.as_ptr(), ptr, table_size);
        }

        let sin_tensor = TensorArg::from_buffer(
            crate::types::MetalBuffer(sin_buffer),
            crate::tensor::Dtype::F16,
            vec![max_seq_len, dim_half],
            vec![dim_half, 1],
        );
        bindings.insert("rope_sin".to_string(), sin_tensor);

        tracing::debug!("Computed RoPE caches: [{}, {}] (rope_base={})", max_seq_len, dim_half, rope_base);
        Ok(())
    }

    /// Bind a GGUF tensor to bindings under a logical name.
    fn bind_gguf_tensor(
        &self,
        bindings: &mut TensorBindings,
        foundry: &mut Foundry,
        gguf_name: &str,
        logical_name: &str,
    ) -> Result<(), MetalError> {
        let gguf = self.weights.gguf_model();

        if let Some(tensor) = gguf.get_tensor(gguf_name) {
            let dtype = crate::gguf::model_loader::GGUFModel::map_dtype(tensor.data_type());
            let dims: Vec<usize> = tensor.dims().to_vec();
            let strides = crate::foundry::tensor::compute_strides(&dims);

            let (buffer, dtype) = if dtype == crate::tensor::Dtype::F32 {
                // F32 weights need downcast to F16
                let view = tensor
                    .raw_view(&gguf.gguf_file)
                    .map_err(|e| MetalError::InvalidShape(format!("GGUF raw view error: {:?}", e)))?;

                let f32_slice: &[f32] = match view {
                    GGUFRawTensor::F32(values) => values,
                    GGUFRawTensor::Bytes(raw, GGUFDataType::F32) => bytemuck::try_cast_slice(raw)
                        .map_err(|_| MetalError::InvalidShape(format!("Invalid F32 bytes for '{}'", gguf_name)))?,
                    _ => {
                        return Err(MetalError::InvalidShape(format!(
                            "Unsupported GGUF dtype {:?} for F32 downcast: {}",
                            tensor.data_type(),
                            gguf_name
                        )));
                    }
                };

                let mut f16_data: Vec<f16> = Vec::with_capacity(f32_slice.len());
                for &v in f32_slice {
                    f16_data.push(f16::from_f32(v));
                }

                use objc2_metal::MTLDevice;
                let byte_size = f16_data.len() * std::mem::size_of::<f16>();
                let buffer = foundry
                    .device
                    .0
                    .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, gguf_name)))?;

                unsafe {
                    use objc2_metal::MTLBuffer;
                    let ptr = buffer.contents().as_ptr() as *mut f16;
                    std::ptr::copy_nonoverlapping(f16_data.as_ptr(), ptr, f16_data.len());
                }

                (crate::types::MetalBuffer(buffer), crate::tensor::Dtype::F16)
            } else {
                let buffer = tensor
                    .materialize_to_device(&gguf.gguf_file, &foundry.device)
                    .map_err(|e| MetalError::InvalidShape(format!("GGUF materialization error: {:?}", e)))?;
                (buffer, dtype)
            };

            let tensor_arg = TensorArg::from_buffer(buffer, dtype, dims, strides);

            // Bind under both gguf_name (for direct access) and logical_name (for DSL refs)
            bindings.insert(gguf_name.to_string(), tensor_arg.clone());
            bindings.insert(logical_name.to_string(), tensor_arg);

            tracing::trace!("Bound '{}' -> '{}'", logical_name, gguf_name);
        }

        Ok(())
    }

    /// Bind a GGUF tensor to bindings in canonical k-block-major layout.
    /// Used for 2D weight matrices with GemvCanonical kernel.
    fn bind_gguf_tensor_canonical(
        &self,
        bindings: &mut TensorBindings,
        foundry: &mut Foundry,
        gguf_name: &str,
        logical_name: &str,
        expected_k: usize,
        expected_n: usize,
    ) -> Result<(), MetalError> {
        const WEIGHTS_PER_BLOCK: usize = 32;

        let gguf = self.weights.gguf_model();

        if let Some(tensor) = gguf.get_tensor(gguf_name) {
            let dims: Vec<usize> = tensor.dims().to_vec();

            if dims.len() != 2 {
                // Non-2D tensors fall back to regular binding
                return self.bind_gguf_tensor(bindings, foundry, gguf_name, logical_name);
            }

            if expected_k == 0 || expected_n == 0 {
                return Err(MetalError::InvalidShape(format!(
                    "Canonical weight '{}' expects non-zero (K,N), got K={} N={}",
                    gguf_name, expected_k, expected_n
                )));
            }

            // GGUF weights can be stored as either NK or KN depending on metadata, and GGUF dims
            // can appear in either order. We avoid guessing by using expected (K,N) from the model arch.
            let layout_hint = gguf.layout_hint();

            // Validate GGUF dims match expected (up to ordering).
            if !((dims[0] == expected_k && dims[1] == expected_n) || (dims[0] == expected_n && dims[1] == expected_k)) {
                return Err(MetalError::InvalidShape(format!(
                    "Canonical weight '{}' dims {:?} do not match expected (K,N)=({}, {})",
                    gguf_name, dims, expected_k, expected_n
                )));
            }

            let blocks_per_k = (expected_k + WEIGHTS_PER_BLOCK - 1) / WEIGHTS_PER_BLOCK;
            let canonical_len = blocks_per_k * expected_n * WEIGHTS_PER_BLOCK;

            // Load source data as F16
            let view = tensor
                .raw_view(&gguf.gguf_file)
                .map_err(|e| MetalError::InvalidShape(format!("GGUF raw view error: {:?}", e)))?;

            let src_f16: Vec<f16> = match view {
                GGUFRawTensor::F32(values) => values.iter().map(|&v| f16::from_f32(v)).collect(),
                GGUFRawTensor::F16(values) => values.to_vec(),
                GGUFRawTensor::Bytes(raw, GGUFDataType::F32) => {
                    let f32_slice: &[f32] = bytemuck::try_cast_slice(raw)
                        .map_err(|_| MetalError::InvalidShape(format!("Invalid F32 bytes for '{}'", gguf_name)))?;
                    f32_slice.iter().map(|&v| f16::from_f32(v)).collect()
                }
                GGUFRawTensor::Bytes(raw, GGUFDataType::F16) => {
                    let f16_slice: &[f16] = bytemuck::try_cast_slice(raw)
                        .map_err(|_| MetalError::InvalidShape(format!("Invalid F16 bytes for '{}'", gguf_name)))?;
                    f16_slice.to_vec()
                }
                _ => {
                    return Err(MetalError::InvalidShape(format!(
                        "Unsupported GGUF dtype {:?} for canonical conversion: {}",
                        tensor.data_type(),
                        gguf_name
                    )));
                }
            };

            // Convert NK row-major to canonical k-block-major layout
            // Layout: for each k-block, store N columns of WEIGHTS_PER_BLOCK elements each
            let mut canonical_data = vec![f16::ZERO; canonical_len];

            for out_idx in 0..expected_n {
                for block in 0..blocks_per_k {
                    let k_base = block * WEIGHTS_PER_BLOCK;
                    let dst_base = (block * expected_n + out_idx) * WEIGHTS_PER_BLOCK;
                    let remaining = expected_k.saturating_sub(k_base);

                    if remaining >= WEIGHTS_PER_BLOCK {
                        for i in 0..WEIGHTS_PER_BLOCK {
                            let k_idx = k_base + i;
                            let src_idx = match layout_hint {
                                // NK row-major: rows=N, cols=K
                                crate::gguf::model_loader::GGUFLayoutHint::Nk => out_idx * expected_k + k_idx,
                                // KN row-major: rows=K, cols=N
                                crate::gguf::model_loader::GGUFLayoutHint::Kn => k_idx * expected_n + out_idx,
                            };
                            canonical_data[dst_base + i] = src_f16[src_idx];
                        }
                    } else if remaining > 0 {
                        for i in 0..remaining {
                            let k_idx = k_base + i;
                            let src_idx = match layout_hint {
                                crate::gguf::model_loader::GGUFLayoutHint::Nk => out_idx * expected_k + k_idx,
                                crate::gguf::model_loader::GGUFLayoutHint::Kn => k_idx * expected_n + out_idx,
                            };
                            canonical_data[dst_base + i] = src_f16[src_idx];
                        }
                        // Padding already zero-initialized
                    }
                }
            }

            // Allocate GPU buffer
            use objc2_metal::MTLDevice;
            let byte_size = canonical_data.len() * std::mem::size_of::<f16>();
            let buffer = foundry
                .device
                .0
                .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    MetalError::OperationFailed(format!("Failed to allocate {} bytes for canonical '{}'", byte_size, gguf_name))
                })?;

            unsafe {
                use objc2_metal::MTLBuffer;
                let ptr = buffer.contents().as_ptr() as *mut f16;
                std::ptr::copy_nonoverlapping(canonical_data.as_ptr(), ptr, canonical_data.len());
            }

            // Use canonical layout dims: [canonical_len] (flat buffer)
            let tensor_arg = TensorArg::from_buffer(
                crate::types::MetalBuffer(buffer),
                crate::tensor::Dtype::F16,
                vec![canonical_len],
                vec![1],
            );

            bindings.insert(gguf_name.to_string(), tensor_arg.clone());
            bindings.insert(logical_name.to_string(), tensor_arg);

            tracing::trace!(
                "Bound canonical '{}' -> '{}' (K={}, N={}, blocks={}, layout_hint={:?})",
                logical_name,
                gguf_name,
                expected_k,
                expected_n,
                blocks_per_k,
                layout_hint
            );
        }

        Ok(())
    }

    fn zero_tensor_arg(&self, foundry: &mut Foundry, size: usize) -> Result<TensorArg, MetalError> {
        if size == 0 {
            return Err(MetalError::InvalidShape("zero_tensor_arg size must be > 0".into()));
        }

        use objc2_metal::MTLDevice;
        let byte_size = size * std::mem::size_of::<f16>();
        let buffer = foundry
            .device
            .0
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for zero buffer", byte_size)))?;

        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, byte_size);
        }

        Ok(TensorArg::from_buffer(
            crate::types::MetalBuffer(buffer),
            crate::tensor::Dtype::F16,
            vec![size],
            vec![1],
        ))
    }

    /// Allocate an intermediate buffer for activations.
    fn allocate_intermediate(
        &self,
        bindings: &mut TensorBindings,
        foundry: &mut Foundry,
        name: &str,
        size: usize,
    ) -> Result<(), MetalError> {
        use objc2_metal::MTLDevice;

        // Skip if already bound
        if bindings.contains(name) {
            return Ok(());
        }

        // Allocate F16 buffer (2 bytes per element)
        let byte_size = size * 2;
        let buffer = foundry
            .device
            .0
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        let tensor_arg = TensorArg::from_buffer(crate::types::MetalBuffer(buffer), crate::tensor::Dtype::F16, vec![size], vec![1]);

        bindings.insert(name.to_string(), tensor_arg);
        tracing::trace!("Allocated intermediate '{}' ({} elements)", name, size);

        Ok(())
    }

    /// Allocate a 3D intermediate buffer for SDPA-style tensors.
    fn allocate_intermediate_3d(
        &self,
        bindings: &mut TensorBindings,
        foundry: &mut Foundry,
        name: &str,
        batch: usize,
        seq_len: usize,
        dim: usize,
    ) -> Result<(), MetalError> {
        use objc2_metal::MTLDevice;

        // Skip if already bound
        if bindings.contains(name) {
            return Ok(());
        }

        let total_elements = batch * seq_len * dim;
        let byte_size = total_elements * 2; // F16 = 2 bytes
        let buffer = foundry
            .device
            .0
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        let tensor_arg = TensorArg::from_buffer(
            crate::types::MetalBuffer(buffer),
            crate::tensor::Dtype::F16,
            vec![batch, seq_len, dim],
            crate::foundry::tensor::compute_strides(&[batch, seq_len, dim]),
        );

        bindings.insert(name.to_string(), tensor_arg);
        tracing::trace!("Allocated 3D intermediate '{}' [{}, {}, {}]", name, batch, seq_len, dim);

        Ok(())
    }

    /// Allocate a KV cache buffer for attention caching.
    /// Shape: [n_heads, max_seq_len, head_dim]
    fn allocate_kv_cache(
        &self,
        bindings: &mut TensorBindings,
        foundry: &mut Foundry,
        name: &str,
        n_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Result<(), MetalError> {
        use objc2_metal::MTLDevice;

        // Skip if already bound
        if bindings.contains(name) {
            return Ok(());
        }

        let total_elements = n_heads * max_seq_len * head_dim;
        let byte_size = total_elements * 2; // F16 = 2 bytes
        let buffer = foundry
            .device
            .0
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        let tensor_arg = TensorArg::from_buffer(
            crate::types::MetalBuffer(buffer),
            crate::tensor::Dtype::F16,
            vec![n_heads, max_seq_len, head_dim],
            crate::foundry::tensor::compute_strides(&[n_heads, max_seq_len, head_dim]),
        );

        bindings.insert(name.to_string(), tensor_arg);
        tracing::trace!("Allocated KV cache '{}' [{}, {}, {}]", name, n_heads, max_seq_len, head_dim);

        Ok(())
    }

    /// Run a single forward step by executing all DSL steps.
    ///
    /// Each step in `spec.architecture.forward` is executed via `Step::execute()`.
    pub fn forward(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        for step in &self.spec.architecture.forward {
            step.execute(foundry, bindings)?;
        }
        Ok(())
    }

    /// Generate multiple tokens autoregressively.
    ///
    /// # Arguments
    /// * `foundry` - The foundry instance for kernel execution
    /// * `prompt_tokens` - Initial token IDs for the prompt
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `stop_tokens` - Token IDs that signal end of generation
    /// * `temperature` - Sampling temperature (higher = more random)
    /// * `top_k` - Top-K sampling parameter
    /// * `top_p` - Top-P (nucleus) sampling parameter
    pub fn generate(
        &self,
        foundry: &mut Foundry,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        stop_tokens: &[u32],
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) -> Result<Vec<u32>, MetalError> {
        self.generate_with_seed(
            foundry,
            prompt_tokens,
            max_new_tokens,
            stop_tokens,
            temperature,
            top_k,
            top_p,
            42u32,
        )
    }

    /// Generate multiple tokens autoregressively with an explicit base seed for sampling.
    ///
    /// The seed is advanced once per generated token (`seed + step`) to avoid pathological repetition
    /// while keeping the output deterministic for parity testing.
    pub fn generate_with_seed_streaming<F>(
        &self,
        foundry: &mut Foundry,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        stop_tokens: &[u32],
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
        mut callback: F,
    ) -> Result<Vec<u32>, MetalError>
    where
        F: FnMut(u32) -> Result<bool, MetalError>,
    {
        if prompt_tokens.is_empty() {
            return Err(MetalError::InvalidShape("generate requires non-empty prompt_tokens".into()));
        }

        let mut generated = Vec::with_capacity(max_new_tokens);
        let arch = &self.spec.architecture;

        // Prepare bindings (weights + intermediates + globals)
        let mut bindings = self.prepare_bindings(foundry)?;

        let d_model = arch.d_model;
        let n_heads = arch.n_heads;
        let n_kv_heads = arch.n_kv_heads;
        let head_dim = d_model / n_heads;
        let ff_dim = arch.ff_dim;

        // Allocate a single-token input_ids buffer and overwrite each step (matches DSL tests).
        let input_buffer = self.allocate_u32_buffer(foundry, "input_ids", 1)?;
        let input_tensor = TensorArg::from_buffer(input_buffer.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);
        bindings.insert("input_ids".to_string(), input_tensor);

        // Keep max_seq_len consistent with KV cache allocation.
        bindings.set_global("max_seq_len", "2048".to_string());

        let set_globals_for_pos = |bindings: &mut TensorBindings, pos: usize| {
            let seq_len = 1usize;
            let kv_seq_len = pos + seq_len;

            bindings.set_global("seq_len", "1".to_string());
            bindings.set_global("position_offset", pos.to_string());
            bindings.set_global("kv_seq_len", kv_seq_len.to_string());

            bindings.set_global("total_elements_q", (n_heads * head_dim).to_string());
            bindings.set_global("total_elements_k", (n_kv_heads * head_dim).to_string());
            bindings.set_global("total_elements_hidden", d_model.to_string());
            bindings.set_global("total_elements_ffn", ff_dim.to_string());
            bindings.set_global("total_elements_slice", (n_kv_heads * kv_seq_len * head_dim).to_string());
            bindings.set_global("total_elements_repeat", (n_heads * kv_seq_len * head_dim).to_string());
            bindings.set_global("total_elements_write", (n_kv_heads * head_dim).to_string());
        };

        // Allocate output buffer for sampled token
        let sample_output_buffer = self.allocate_u32_buffer(foundry, "_sample_output", 1)?;
        let sample_output = TensorArg::from_buffer(sample_output_buffer.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);

        let greedy = temperature <= 0.0 || !temperature.is_finite() || top_k == 0;
        let vocab_size = arch.vocab_size as u32;

        fn argmax_logits(logits: &TensorArg) -> Result<u32, MetalError> {
            use half::f16;

            use crate::types::KernelArg;

            let dims = logits.dims();
            if dims.is_empty() {
                return Err(MetalError::InvalidShape("logits must have at least one dimension".into()));
            }
            let last_dim = *dims.last().unwrap();
            if last_dim == 0 {
                return Ok(0);
            }

            let total = dims.iter().product::<usize>();
            if total < last_dim {
                return Err(MetalError::InvalidShape("logits buffer shorter than last dimension".into()));
            }
            let start = total - last_dim;

            let buffer = logits.buffer();
            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            let mut found = false;

            unsafe {
                use objc2_metal::MTLBuffer;
                match logits.dtype() {
                    crate::tensor::Dtype::F16 => {
                        let ptr = buffer.contents().as_ptr() as *const f16;
                        let slice = std::slice::from_raw_parts(ptr, total);
                        for (offset, &raw) in slice[start..].iter().enumerate() {
                            let value = raw.to_f32();
                            if !value.is_finite() {
                                continue;
                            }
                            let token_idx = offset as u32;
                            if !found || value > best_val || (value == best_val && token_idx > best_idx) {
                                found = true;
                                best_val = value;
                                best_idx = token_idx;
                            }
                        }
                    }
                    crate::tensor::Dtype::F32 => {
                        let ptr = buffer.contents().as_ptr() as *const f32;
                        let slice = std::slice::from_raw_parts(ptr, total);
                        for (offset, &value) in slice[start..].iter().enumerate() {
                            if !value.is_finite() {
                                continue;
                            }
                            let token_idx = offset as u32;
                            if !found || value > best_val || (value == best_val && token_idx > best_idx) {
                                found = true;
                                best_val = value;
                                best_idx = token_idx;
                            }
                        }
                    }
                    other => {
                        return Err(MetalError::InvalidShape(format!("Unsupported logits dtype for argmax: {other:?}")));
                    }
                }
            }

            Ok(best_idx)
        }

        // Prefill KV cache: run each prompt token at its absolute position (seq_len=1).
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            unsafe {
                use objc2_metal::MTLBuffer;
                let ptr = input_buffer.0.contents().as_ptr() as *mut u32;
                *ptr = token;
            }
            set_globals_for_pos(&mut bindings, pos);
            self.forward(foundry, &mut bindings)?;
        }

        // Now autoregressive decode: sample from last prompt-token logits, then step forward per token.
        for step in 0..max_new_tokens {
            // Get logits buffer from bindings
            let logits = bindings.get("logits")?;

            let next_token = if greedy {
                argmax_logits(&logits)?
            } else {
                // Create and run SampleTopK kernel
                let sample_kernel = crate::metals::sampling::SampleTopK::new(
                    &logits,
                    &sample_output,
                    vocab_size,
                    top_k,
                    top_p,
                    temperature,
                    seed.wrapping_add(step as u32),
                );
                foundry.run(&sample_kernel)?;

                // Read sampled token from output buffer
                unsafe {
                    use objc2_metal::MTLBuffer;
                    let ptr = sample_output_buffer.0.contents().as_ptr() as *const u32;
                    *ptr
                }
            };

            if stop_tokens.contains(&next_token) {
                break;
            }

            generated.push(next_token);
            if !callback(next_token)? {
                break;
            }

            // Advance one token at absolute position (prompt_len + step).
            let pos = prompt_tokens.len() + step;

            // Update input_ids buffer for this step (just the new token)
            unsafe {
                use objc2_metal::MTLBuffer;
                let ptr = input_buffer.0.contents().as_ptr() as *mut u32;
                *ptr = next_token;
            }

            set_globals_for_pos(&mut bindings, pos);
            self.forward(foundry, &mut bindings)?;
        }

        Ok(generated)
    }

    /// Generate multiple tokens autoregressively with an explicit base seed for sampling.
    ///
    /// The seed is advanced once per generated token (`seed + step`) to avoid pathological repetition
    /// while keeping the output deterministic for parity testing.
    pub fn generate_with_seed(
        &self,
        foundry: &mut Foundry,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        stop_tokens: &[u32],
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
    ) -> Result<Vec<u32>, MetalError> {
        self.generate_with_seed_streaming(
            foundry,
            prompt_tokens,
            max_new_tokens,
            stop_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            |_| Ok(true),
        )
    }

    /// Convenience helper: apply the model's chat template (when available), tokenize, and generate.
    ///
    /// Returns only the newly generated token ids (not including the prompt tokens).
    pub fn generate_from_prompt(
        &self,
        foundry: &mut Foundry,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) -> Result<Vec<u32>, MetalError> {
        let tokenizer = self.tokenizer()?;
        let prompt_tokens = tokenizer.encode_single_turn_chat_prompt(prompt)?;
        let eos = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);
        self.generate(foundry, &prompt_tokens, max_new_tokens, &[eos], temperature, top_k, top_p)
    }

    /// Convenience helper: generate and decode newly generated text (not including the prompt).
    pub fn generate_text_from_prompt(
        &self,
        foundry: &mut Foundry,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) -> Result<String, MetalError> {
        let tokenizer = self.tokenizer()?;
        let tokens = self.generate_from_prompt(foundry, prompt, max_new_tokens, temperature, top_k, top_p)?;
        tokenizer.decode(&tokens)
    }

    /// Allocate a U32 buffer for tokens.
    fn allocate_u32_buffer(&self, foundry: &mut Foundry, name: &str, count: usize) -> Result<crate::types::MetalBuffer, MetalError> {
        use objc2_metal::MTLDevice;

        let byte_size = count * 4; // u32 = 4 bytes
        let buffer = foundry
            .device
            .0
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        Ok(crate::types::MetalBuffer(buffer))
    }

    // Keep old method for backward compatibility, delegating to new one
    #[deprecated(note = "Use prepare_bindings instead")]
    pub fn prepare_weight_bindings(&self, foundry: &mut Foundry) -> Result<TensorBindings, MetalError> {
        self.prepare_bindings(foundry)
    }
}
