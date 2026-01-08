//! CompiledModel executor for running inference.
//!
//! Interprets the ModelSpec execution plan (DSL) by calling Step::execute().

use half::f16;
use rustc_hash::FxHashMap;

use super::builder::WeightBundle;
use crate::{
    error::MetalError, foundry::{
        Foundry, spec::{
            ModelSpec, TensorBindings, compiled::{CompiledStep, FastBindings, SymbolTable}
        }
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
    /// Optimized execution steps (compiled from DSL)
    compiled_steps: Vec<Box<dyn CompiledStep>>,
    /// Symbol table mapping tensor names to integer indices for fast lookup
    symbol_table: SymbolTable,
}

impl CompiledModel {
    /// Optimized execution steps (compiled from DSL)
    pub fn compiled_steps(&self) -> &[Box<dyn CompiledStep>] {
        &self.compiled_steps
    }

    /// Symbol table mapping tensor names to integer indices for fast lookup
    pub fn symbol_table(&self) -> &SymbolTable {
        &self.symbol_table
    }

    /// Create a new CompiledModel from spec and weights.
    pub(crate) fn new(spec: ModelSpec, weights: WeightBundle) -> Result<Self, MetalError> {
        if let Some(gguf_arch) = weights.architecture() {
            tracing::debug!("Loading model: spec='{}' gguf_arch='{}'", spec.name, gguf_arch);
        }

        tracing::info!("Compiling model with {} forward DSL steps...", spec.architecture.forward.len());

        // Compiler setup
        let mut symbols = SymbolTable::new();
        let mut resolver = TensorBindings::new();
        let arch = &spec.architecture;

        // Set config globals for DSL variable interpolation (needed for Repeat unrolling, etc.)
        resolver.set_global("n_layers", arch.n_layers.to_string());
        resolver.set_global("d_model", arch.d_model.to_string());
        resolver.set_global("n_heads", arch.n_heads.to_string());
        resolver.set_global("n_kv_heads", arch.n_kv_heads.to_string());
        resolver.set_global("ff_dim", arch.ff_dim.to_string());
        resolver.set_global("vocab_size", arch.vocab_size.to_string());
        let head_dim = arch.d_model / arch.n_heads;
        resolver.set_global("head_dim", head_dim.to_string());
        let kv_dim = head_dim * arch.n_kv_heads;
        resolver.set_global("kv_dim", kv_dim.to_string());

        // Compile steps
        let mut compiled_steps = Vec::new();
        for step in &spec.architecture.forward {
            compiled_steps.extend(step.compile(&mut resolver, &mut symbols));
        }

        tracing::info!(
            "CompiledModel ready: {} compiled steps, {} symbols",
            compiled_steps.len(),
            symbols.len()
        );

        Ok(Self {
            spec,
            weights,
            compiled_steps,
            symbol_table: symbols,
        })
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
    pub fn prepare_bindings(&self, foundry: &mut Foundry) -> Result<(TensorBindings, FastBindings), MetalError> {
        let mut bindings = TensorBindings::new();
        let mut fast_bindings = FastBindings::new(self.symbol_table.len());

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
        let head_dim = arch.d_model / arch.n_heads;
        bindings.set_global("head_dim", head_dim.to_string());
        let kv_dim = head_dim * arch.n_kv_heads;
        bindings.set_global("kv_dim", kv_dim.to_string());

        // Build a set of available GGUF tensor names for resolution
        let available: FxHashMap<String, ()> = self.weights.tensor_names().map(|name| (name.clone(), ())).collect();

        // 2. Resolve and bind global weight tensors
        let global_keys = ["embedding", "final_norm", "rope_cos", "rope_sin"];
        for key in global_keys {
            if let Some(gguf_name) = tensor_names.resolve(key, None, &available) {
                self.bind_gguf_tensor(&mut bindings, &mut fast_bindings, foundry, &gguf_name, key)?;
            }
        }

        // 2b. Bind output_weight (LM Head) using canonical layout for GemvCanonical
        if let Some(gguf_name) = tensor_names.resolve("output_weight", None, &available) {
            self.bind_gguf_tensor_canonical(
                &mut bindings,
                &mut fast_bindings,
                foundry,
                &gguf_name,
                "output_weight",
                arch.d_model,
                arch.vocab_size,
            )?;
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
                    self.bind_gguf_tensor(&mut bindings, &mut fast_bindings, foundry, &gguf_name, &logical_name)?;
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
                    self.bind_gguf_tensor_canonical(
                        &mut bindings,
                        &mut fast_bindings,
                        foundry,
                        &gguf_name,
                        &logical_name,
                        expected_k,
                        expected_n,
                    )?;
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
                    self.bind_gguf_tensor(&mut bindings, &mut fast_bindings, foundry, &gguf_name, &logical_name)?;
                } else {
                    let zero = if let Some(tensor) = zero_cache.get(&size) {
                        tensor.clone()
                    } else {
                        let tensor = self.zero_tensor_arg(foundry, size)?;
                        zero_cache.insert(size, tensor.clone());
                        tensor
                    };
                    self.insert_binding(&mut bindings, &mut fast_bindings, logical_name, zero);
                }
            }
        }

        // 4. Allocate intermediate buffers
        // These are named buffers used by the forward pass for activations
        // For now, we use seq_len=1 (single token inference), batch=1
        let batch = 1;
        let seq_len = 1; // Will be updated per-forward based on input

        // 1D buffers for general intermediates
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "hidden", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "norm_out", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "proj_out", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "residual_1", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "ffn_norm_out", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "gate", arch.ff_dim)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "up", arch.ff_dim)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "ffn_out", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "final_norm_out", arch.d_model)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "logits", arch.vocab_size)?;

        // 3D buffers for SDPA (batch, seq_len, dim)
        let kv_dim = arch.d_model / arch.n_heads * arch.n_kv_heads;
        self.allocate_intermediate_3d(&mut bindings, &mut fast_bindings, foundry, "q", batch, seq_len, arch.d_model)?;
        self.allocate_intermediate_3d(&mut bindings, &mut fast_bindings, foundry, "k", batch, seq_len, kv_dim)?;
        self.allocate_intermediate_3d(&mut bindings, &mut fast_bindings, foundry, "v", batch, seq_len, kv_dim)?;
        let head_dim = arch.d_model / arch.n_heads;
        self.allocate_intermediate_3d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "q_heads",
            batch * arch.n_heads,
            seq_len,
            head_dim,
        )?;
        self.allocate_intermediate_3d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "k_heads",
            batch * arch.n_kv_heads,
            seq_len,
            head_dim,
        )?;
        self.allocate_intermediate_3d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "v_heads",
            batch * arch.n_kv_heads,
            seq_len,
            head_dim,
        )?;
        self.allocate_intermediate_3d(&mut bindings, &mut fast_bindings, foundry, "q_rot", batch, seq_len, arch.d_model)?;
        self.allocate_intermediate_3d(&mut bindings, &mut fast_bindings, foundry, "k_rot", batch, seq_len, kv_dim)?;
        // Expanded K/V buffers for GQA (after RepeatKvHeads, same dim as Q)
        // Must be sized for MAX sequence length because they hold repeated history
        let max_seq_len_for_slice = 2048;
        let expanded_dim = batch * max_seq_len_for_slice * arch.d_model;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "k_expanded", expanded_dim)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "v_expanded", expanded_dim)?;
        // SDPA output is per-step (seq_len=1) in incremental decode; keep this compact.
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "attn_out", arch.d_model)?;

        // KV slice buffers for cache reads (sized for max sequence to avoid reallocation)
        // These hold the sliced cache [n_kv_heads, current_seq_len, head_dim]
        // We allocate to max size and track actual usage via globals
        let max_seq_len_for_slice = 2048;
        let kv_slice_dim = arch.n_kv_heads * max_seq_len_for_slice * head_dim;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "k_slice", kv_slice_dim)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "v_slice", kv_slice_dim)?;

        // 5. Create a "zero" buffer for unused bias/residual slots
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "zero", 1)?;

        // 6. Allocate KV cache for autoregressive generation
        // Shape: [n_kv_heads, max_seq_len, head_dim] per layer
        // We'll create one k_cache and v_cache per layer, named k_cache_0, k_cache_1, etc.
        let max_seq_len = 2048; // Maximum context length
        let head_dim = arch.d_model / arch.n_heads;
        for layer_idx in 0..arch.n_layers {
            let k_cache_name = format!("k_cache_{}", layer_idx);
            let v_cache_name = format!("v_cache_{}", layer_idx);
            // KV cache: [n_kv_heads, max_seq_len, head_dim]
            self.allocate_kv_cache(
                &mut bindings,
                &mut fast_bindings,
                foundry,
                &k_cache_name,
                arch.n_kv_heads,
                max_seq_len,
                head_dim,
            )?;
            self.allocate_kv_cache(
                &mut bindings,
                &mut fast_bindings,
                foundry,
                &v_cache_name,
                arch.n_kv_heads,
                max_seq_len,
                head_dim,
            )?;
        }
        // Store max_seq_len as a global for kernels to use
        bindings.set_global("max_seq_len", max_seq_len.to_string());
        bindings.set_global("head_dim", head_dim.to_string());
        bindings.set_global("n_kv_heads", arch.n_kv_heads.to_string());
        bindings.set_global("n_heads", arch.n_heads.to_string());

        // 7. Compute and bind RoPE cos/sin caches
        // These are precomputed based on model config since they're not in GGUF
        self.compute_and_bind_rope_caches(&mut bindings, &mut fast_bindings, foundry, arch)?;

        tracing::info!("Prepared {} bindings (weights + intermediates + RoPE + KV cache)", bindings.len());
        Ok((bindings, fast_bindings))
    }

    /// Compute and bind RoPE cos/sin cache tables.
    ///
    /// RoPE caches are not stored in GGUF, so we compute them based on arch config.
    /// Compute and bind RoPE cos/sin cache tables.
    ///
    /// RoPE caches are not stored in GGUF, so we compute them based on arch config.
    fn compute_and_bind_rope_caches(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
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
        self.insert_binding(bindings, fast_bindings, "rope_cos".to_string(), cos_tensor);

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
        self.insert_binding(bindings, fast_bindings, "rope_sin".to_string(), sin_tensor);

        tracing::debug!("Computed RoPE caches: [{}, {}] (rope_base={})", max_seq_len, dim_half, rope_base);
        Ok(())
    }

    /// Bind a GGUF tensor to bindings under a logical name.
    fn bind_gguf_tensor(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
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
            self.insert_binding(bindings, fast_bindings, gguf_name.to_string(), tensor_arg.clone());
            self.insert_binding(bindings, fast_bindings, logical_name.to_string(), tensor_arg);

            tracing::trace!("Bound '{}' -> '{}'", logical_name, gguf_name);
        }

        Ok(())
    }

    /// Bind a GGUF tensor to bindings in canonical k-block-major layout.
    /// Used for 2D weight matrices with GemvCanonical kernel.
    fn bind_gguf_tensor_canonical(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
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
                return self.bind_gguf_tensor(bindings, fast_bindings, foundry, gguf_name, logical_name);
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

            self.insert_binding(bindings, fast_bindings, gguf_name.to_string(), tensor_arg.clone());
            self.insert_binding(bindings, fast_bindings, logical_name.to_string(), tensor_arg);

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
        fast_bindings: &mut FastBindings,
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

        self.insert_binding(bindings, fast_bindings, name.to_string(), tensor_arg);
        tracing::trace!("Allocated intermediate '{}' ({} elements)", name, size);

        Ok(())
    }

    /// Allocate a 3D intermediate buffer for SDPA-style tensors.
    fn allocate_intermediate_3d(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
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

        self.insert_binding(bindings, fast_bindings, name.to_string(), tensor_arg);
        tracing::trace!("Allocated 3D intermediate '{}' [{}, {}, {}]", name, batch, seq_len, dim);

        Ok(())
    }

    /// Helper to insert a tensor into both string and fast bindings
    fn insert_binding(&self, bindings: &mut TensorBindings, fast_bindings: &mut FastBindings, name: String, tensor: TensorArg) {
        if let Some(id) = self.symbol_table.get(&name) {
            fast_bindings.set(id, tensor.clone());
        }
        bindings.insert(name, tensor);
    }

    /// Allocate a KV cache buffer for attention caching.
    /// Shape: [n_heads, max_seq_len, head_dim]
    fn allocate_kv_cache(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
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

        self.insert_binding(bindings, fast_bindings, name.to_string(), tensor_arg);
        tracing::trace!("Allocated KV cache '{}' [{}, {}, {}]", name, n_heads, max_seq_len, head_dim);

        Ok(())
    }

    /// Run a single forward step by executing all DSL steps.
    ///
    /// Each step in `spec.architecture.forward` is executed via `Step::execute()`.
    /// Run a single forward step by executing all compiled steps.
    pub fn forward(&self, foundry: &mut Foundry, bindings: &mut TensorBindings, fast_bindings: &FastBindings) -> Result<(), MetalError> {
        // If we are already capturing (e.g. batched prompt processing), don't start a new capture.
        let nested_capture = foundry.is_capturing();

        if !nested_capture {
            // Start a new command buffer for this forward pass (token)
            // Reuse the existing command buffer if one is active but not capturing?
            // No, start_capture() handles creation.
            foundry.start_capture()?;
        }

        for step in &self.compiled_steps {
            step.execute(foundry, fast_bindings, bindings)?;
        }

        if !nested_capture {
            // Commit but do NOT wait - standard async dispatch
            // Note: If we are in autoregressive loop, caller might need to wait if reading back to CPU.
            // But forward() itself should be async.
            foundry.end_capture()?;
        }

        Ok(())
    }

    /// Run a single forward step by executing DSL steps (uncompiled/interpreted).
    ///
    /// Unlike `forward()` which uses pre-compiled steps, this method executes the
    /// original `Step::execute()` method on each step in `spec.architecture.forward`.
    /// This allows runtime modification of variables like `n_layers` via `bindings.set_global()`.
    pub fn forward_uncompiled(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        // Start a new command buffer for this forward pass (token)
        foundry.start_capture()?;

        for step in &self.spec.architecture.forward {
            if step.name() == "Sample" {
                continue;
            }
            step.execute(foundry, bindings)?;
        }

        // Commit and wait for the token to complete
        let cmd_buffer = foundry.end_capture()?;
        {
            use objc2_metal::MTLCommandBuffer as _;
            cmd_buffer.waitUntilCompleted();
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
        let (mut bindings, mut fast_bindings) = self.prepare_bindings(foundry)?;

        let d_model = arch.d_model;
        let n_heads = arch.n_heads;
        let n_kv_heads = arch.n_kv_heads;
        let head_dim = d_model / n_heads;
        let ff_dim = arch.ff_dim;

        // Allocate a single large input buffer for the entire prompt to avoid reallocation
        // and allow batched processing (different offsets).
        let prompt_len = prompt_tokens.len();
        let full_input_buffer = self.allocate_u32_buffer(foundry, "input_ids_full", prompt_len)?;

        // Write all prompt tokens to the shared buffer
        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = full_input_buffer.0.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(prompt_tokens.as_ptr(), ptr, prompt_len);
        }

        // Keep max_seq_len consistent with KV cache allocation.
        bindings.set_global("max_seq_len", "2048".to_string());

        // Pre-compute static globals ONCE (avoid String allocations per token)
        // These don't change between tokens:
        let static_total_elements_q = (n_heads * head_dim).to_string();
        let static_total_elements_k = (n_kv_heads * head_dim).to_string();
        let static_total_elements_hidden = d_model.to_string();
        let static_total_elements_ffn = ff_dim.to_string();
        let static_total_elements_write = (n_kv_heads * head_dim).to_string();

        bindings.set_global("seq_len", "1".to_string());
        bindings.set_global("total_elements_q", static_total_elements_q);
        bindings.set_global("total_elements_k", static_total_elements_k);
        bindings.set_global("total_elements_hidden", static_total_elements_hidden);
        bindings.set_global("total_elements_ffn", static_total_elements_ffn);
        bindings.set_global("total_elements_write", static_total_elements_write);

        // Only position-dependent globals need updating per token
        // Use int_globals to avoid String allocation on every token
        let update_pos_globals = |bindings: &mut TensorBindings, pos: usize| {
            let kv_seq_len = pos + 1; // seq_len is always 1

            // Set integer globals for all steps (manual Sdpa and auto-generated Rope/KvCacheWrite)
            // DynamicValue::resolve now checks int_globals for u32/usize types efficiently
            bindings.set_int_global("position_offset", pos);
            bindings.set_int_global("kv_seq_len", kv_seq_len);
            bindings.set_int_global("total_elements_slice", n_kv_heads * kv_seq_len * head_dim);
            bindings.set_int_global("total_elements_repeat", n_heads * kv_seq_len * head_dim);
        };

        let greedy = temperature <= 0.0 || !temperature.is_finite() || top_k == 0;
        let vocab_size = arch.vocab_size as u32;

        // Prefill KV cache, batched in chunks to reduce overhead.
        // We update the 'input_ids' binding to point to the correct offset in full_input_buffer.
        // Larger chunk size for prefill since we don't need intermediate CPU syncs.
       
        //QUESTION: Would Having GEMM available for Foundry allow us to prefill faster and improve further than basic batching like this in dispatch?
        const PREFILL_CHUNK_SIZE: usize = 512;

        // Cache the input_ids key to avoid String allocation per token
        let input_ids_key = "input_ids".to_string();

        // Check for debug sync flag to disable batched capture for isolation
        let debug_sync = std::env::var("METALLIC_DEBUG_FORWARD_SYNC").is_ok();

        for chunk in prompt_tokens.chunks(PREFILL_CHUNK_SIZE).enumerate() {
            let (chunk_idx, chunk_tokens) = chunk;

            let base_pos = chunk_idx * PREFILL_CHUNK_SIZE;

            // Start capture for this chunk (unless debugging)
            if !debug_sync {
                foundry.start_capture()?;
            }

            for (i, _) in chunk_tokens.iter().enumerate() {
                let pos = base_pos + i;

                // Update bindings to point to the specific token in the large buffer
                let mut tensor_input = TensorArg::from_buffer(full_input_buffer.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);
                tensor_input.offset = pos * 4; // 4 bytes per u32
                self.insert_binding(&mut bindings, &mut fast_bindings, input_ids_key.clone(), tensor_input);

                update_pos_globals(&mut bindings, pos);
                self.forward(foundry, &mut bindings, &fast_bindings)?;
            }

            // Commit and wait for the chunk

            if !debug_sync {
                foundry.end_capture()?;
            }
        }

        // Now autoregressive decode: sample from last prompt-token logits, then step forward per token.
        // We reuse the first slot of full_input_buffer as a scratch space for generated tokens.
        let single_input_buffer = full_input_buffer; // reuse ownership

        // Ensure input_ids is bound to the single input buffer (offset 0) for the generated tokens
        let input_ids_key = "input_ids".to_string();
        {
            let mut tensor_input = TensorArg::from_buffer(single_input_buffer.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);
            tensor_input.offset = 0;
            self.insert_binding(&mut bindings, &mut fast_bindings, input_ids_key.clone(), tensor_input);
        }
        
        // BATCHING: We batch multiple tokens into a single command buffer to amortize submission overhead.
        // We keep tokens on GPU (copying Sample -> Input) and only sync when the batch is full.
        const BATCH_SIZE: usize = 16;
        let mut step_output_buffers = Vec::with_capacity(BATCH_SIZE);
        for i in 0..BATCH_SIZE {
            step_output_buffers.push(self.allocate_u32_buffer(foundry, &format!("sample_out_{}", i), 1)?);
        }

        let mut pending_count = 0;

        for step in 0..max_new_tokens {
            let batch_idx = pending_count;

            // Start capture at the beginning of a batch
            if batch_idx == 0 {
                foundry.start_capture()?;
            }

            // 1. Get logits from previous forward pass (or prefill)
            let logits = bindings.get("logits")?;

            // 2. Sample (Greedy or Random) using GPU kernel
            // We use SampleTopK for both to keep execution on GPU. Greedy is just top_k=1.
            let effective_top_k = if greedy { 1 } else { top_k };
            let sample_out = &step_output_buffers[batch_idx];
            
            // Create sample kernel with destination buffer
            let sample_kernel = crate::metals::sampling::SampleTopK::new(
                &logits,
                &TensorArg::from_buffer(sample_out.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]),
                vocab_size,
                effective_top_k,
                top_p,
                temperature,
                seed.wrapping_add(step as u32),
            );
            foundry.run(&sample_kernel)?;

            // 3. Copy sampled token to input_ids buffer for next step
            // We copy from the specific sample output to the single shared input buffer (offset 0)
            // Use compute kernel instead of blit to keep encoder active (batching optimization)
            let copy_kernel = crate::metals::tensor::CopyU32::new(
                TensorArg::from_buffer(sample_out.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]),
                TensorArg::from_buffer(single_input_buffer.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]),
                1,
            );
            foundry.run(&copy_kernel)?;

            // 4. Update globals for the NEXT forward pass
            // We are about to run forward for position `prompt_len + step` (the token we just sampled)
            update_pos_globals(&mut bindings, prompt_len + step);

            // 5. Run Forward (predicts next token's logits)
            self.forward(foundry, &mut bindings, &fast_bindings)?;

            pending_count += 1;

            // 6. If batch is full or we are done, sync and process
            if pending_count >= BATCH_SIZE || step == max_new_tokens - 1 {
                let cmd = foundry.end_capture()?;
                // Sync CPU to read back the tokens
                objc2_metal::MTLCommandBuffer::waitUntilCompleted(&*cmd);

                // Process the batch results
                let mut batch_done = false;
                for i in 0..pending_count {
                    let token_buf = &step_output_buffers[i];
                    let token = unsafe {
                        use objc2_metal::MTLBuffer;
                        let ptr = token_buf.0.contents().as_ptr() as *const u32;
                        *ptr
                    };

                    generated.push(token);
                    
                    if stop_tokens.contains(&token) {
                        batch_done = true;
                        break;
                    }
                    if !callback(token)? {
                        batch_done = true;
                        break;
                    }
                }
                pending_count = 0;
                
                if batch_done {
                    break;
                }
            }
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
    /// Get the compiled symbol ID for a tensor name.
    pub fn symbol_id(&self, name: &str) -> Option<usize> {
        self.symbol_table.get(name)
    }

    // Keep old method for backward compatibility, delegating to new one
    #[deprecated(note = "Use prepare_bindings instead")]
    pub fn prepare_weight_bindings(&self, foundry: &mut Foundry) -> Result<TensorBindings, MetalError> {
        self.prepare_bindings(foundry).map(|(b, _)| b)
    }
}
