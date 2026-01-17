//! CompiledModel executor for running inference.
//!
//! Interprets the ModelSpec execution plan (DSL) by calling Step::execute().

use std::{cell::RefCell, sync::OnceLock};

use half::f16;
use rustc_hash::FxHashMap;

use super::builder::WeightBundle;
use crate::{
    Foundry, error::MetalError, metals::sampling::SampleTopK, policy::{WeightLayout, resolve_policy}, spec::{
        ModelSpec, TensorBindings, compiled::{CompiledStep, FastBindings, SymbolTable}
    }, types::TensorArg
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
    /// Reusable execution session (weights/intermediates + small persistent buffers).
    session: RefCell<Option<ModelSession>>,
}

struct ModelSession {
    bindings: TensorBindings,
    fast_bindings: FastBindings,
    input_ids_full: crate::types::MetalBuffer,
    input_ids_capacity: usize,
    sample_out_buffers: Vec<crate::types::MetalBuffer>,
}

impl CompiledModel {
    const METALLIC_IGNORE_EOS_STOP_ENV: &'static str = "METALLIC_IGNORE_EOS_STOP";
    // Foundry currently caps runtime max sequence length to avoid huge persistent allocations
    // when the spec supports very long contexts (e.g. 32k+). This value must be used consistently
    // for all KV-related buffers and for the `max_seq_len` global to keep indexing correct.
    const RUNTIME_MAX_SEQ_LEN_CAP: usize = 2048;

    #[inline]
    fn runtime_max_seq_len(arch: &crate::spec::Architecture) -> usize {
        arch.max_seq_len.min(Self::RUNTIME_MAX_SEQ_LEN_CAP).max(1)
    }

    #[inline]
    fn ignore_eos_stop_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            let Ok(value) = std::env::var(Self::METALLIC_IGNORE_EOS_STOP_ENV) else {
                return false;
            };
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return false;
            }
            let lowered = trimmed.to_ascii_lowercase();
            !matches!(lowered.as_str(), "0" | "false" | "no" | "off")
        })
    }

    fn read_prefill_usize(var: &str) -> Option<usize> {
        std::env::var(var).ok().and_then(|v| v.trim().parse::<usize>().ok())
    }

    fn prefill_config() -> (usize, usize) {
        // Defaults chosen to be "big enough to matter" but not explode memory (logits = max*V).
        const DEFAULT_MAX_PREFILL_CHUNK: usize = 32;
        const DEFAULT_PREFILL_CHUNK_SIZE: usize = 32;
        const MAX_ALLOWED: usize = 512;

        let mut max_prefill_chunk = Self::read_prefill_usize("METALLIC_MAX_PREFILL_CHUNK").unwrap_or(DEFAULT_MAX_PREFILL_CHUNK);
        let mut prefill_chunk_size = Self::read_prefill_usize("METALLIC_PREFILL_CHUNK_SIZE").unwrap_or(DEFAULT_PREFILL_CHUNK_SIZE);

        max_prefill_chunk = max_prefill_chunk.clamp(1, MAX_ALLOWED);
        prefill_chunk_size = prefill_chunk_size.clamp(1, MAX_ALLOWED);

        // Allocation must cover the largest runtime M.
        if prefill_chunk_size > max_prefill_chunk {
            max_prefill_chunk = prefill_chunk_size;
        }

        (max_prefill_chunk, prefill_chunk_size)
    }

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
            session: RefCell::new(None),
        })
    }

    /// Get the model name from the spec.
    pub fn name(&self) -> &str {
        &self.spec.name
    }

    /// Get the architecture configuration.
    pub fn architecture(&self) -> &crate::spec::Architecture {
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

    pub fn initialize_session(&self, foundry: &mut Foundry) -> Result<(), MetalError> {
        let mut session = self.session.borrow_mut();
        if session.is_some() {
            return Ok(());
        }

        let (mut bindings, mut fast_bindings) = self.prepare_bindings(foundry)?;
        let arch = self.architecture();

        let max_seq_len = Self::runtime_max_seq_len(arch);
        bindings.set_int_global("max_seq_len", max_seq_len);
        bindings.set_global("max_seq_len", max_seq_len.to_string());

        // Allocate a single prompt/input buffer sized for max_seq_len to avoid per-generation allocations.
        let input_ids_full = self.allocate_u32_buffer(foundry, "input_ids_full", max_seq_len)?;

        // Seed `input_ids` with a valid buffer. We overwrite it per-step below to point at the sampled-token buffer.
        {
            let mut tensor_input = TensorArg::from_buffer(input_ids_full.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);
            tensor_input.offset = 0;
            self.set_binding(&mut bindings, &mut fast_bindings, "input_ids", tensor_input);
        }

        // Pre-allocate decode sample-output buffers to avoid tiny allocations in the hot path.
        let default_decode_batch_size = bindings
            .get("output_weight")
            .ok()
            .map(|w| if w.dtype == crate::tensor::Dtype::F16 { 16 } else { 64 })
            .unwrap_or(64);

        let decode_batch_size = {
            const MAX: usize = 256;
            let parsed = std::env::var("METALLIC_FOUNDRY_DECODE_BATCH_SIZE")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok());
            parsed.unwrap_or(default_decode_batch_size).clamp(1, MAX)
        };

        let mut sample_out_buffers = Vec::with_capacity(decode_batch_size);
        for i in 0..decode_batch_size {
            sample_out_buffers.push(self.allocate_u32_buffer(foundry, &format!("sample_out_{i}"), 1)?);
        }

        // Default globals for decode (m=1, seq_len=1).
        let d_model = arch.d_model;
        let n_heads = arch.n_heads;
        let n_kv_heads = arch.n_kv_heads;
        let head_dim = d_model / n_heads;
        let ff_dim = arch.ff_dim;
        bindings.set_int_global("m", 1);
        bindings.set_int_global("seq_len", 1);
        bindings.set_int_global("total_elements_hidden", d_model);
        bindings.set_int_global("total_elements_q", n_heads * head_dim);
        bindings.set_int_global("total_elements_k", n_kv_heads * head_dim);
        bindings.set_int_global("total_elements_write", n_kv_heads * head_dim);
        bindings.set_int_global("total_elements_ffn", ff_dim);

        *session = Some(ModelSession {
            bindings,
            fast_bindings,
            input_ids_full,
            input_ids_capacity: max_seq_len,
            sample_out_buffers,
        });
        Ok(())
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
        let max_seq_len = Self::runtime_max_seq_len(arch);

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

        // 2b. Bind output_weight (LM Head) as NK row-major ([Vocab, DModel]) to enable GEMM prefill.
        if let Some(gguf_name) = tensor_names.resolve("output_weight", None, &available) {
            self.bind_gguf_tensor(&mut bindings, &mut fast_bindings, foundry, &gguf_name, "output_weight")?;
        }

        // 3. Resolve and bind per-layer weight tensors
        // Non-FFN weights use regular binding
        let regular_layer_keys = ["layer.attn_norm", "layer.ffn_norm"];
        // Canonical weights (blocked) are needed for FusedQkv.
        let canonical_layer_keys = ["layer.attn_q", "layer.attn_k", "layer.attn_v"];
        // NK row-major weights are GEMM-friendly for MatMul (transpose_b=true).
        let nk_layer_keys = ["layer.attn_output", "layer.ffn_gate", "layer.ffn_up", "layer.ffn_down"];
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

            for key in nk_layer_keys {
                if let Some(gguf_name) = tensor_names.resolve(key, Some(i), &available) {
                    let logical_name = format!("{key}_{i}");

                    self.bind_gguf_tensor(&mut bindings, &mut fast_bindings, foundry, &gguf_name, &logical_name)?;
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
        // Pre-allocate for max batch size to support batched prefill (M>1)
        // Tune via:
        // - METALLIC_MAX_PREFILL_CHUNK (allocation)
        // - METALLIC_PREFILL_CHUNK_SIZE (runtime chunking; may raise allocation if larger)
        let (max_prefill_chunk, _prefill_chunk_size) = Self::prefill_config();
        bindings.set_int_global("max_prefill_chunk", max_prefill_chunk);
        let batch = 1;

        // 2D buffers for general intermediates (max batch rows).
        // IMPORTANT: keep the last dim as the feature dim so steps that infer K/N from dims stay correct.
        self.allocate_intermediate_2d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "hidden",
            max_prefill_chunk,
            arch.d_model,
        )?;
        self.allocate_intermediate_2d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "norm_out",
            max_prefill_chunk,
            arch.d_model,
        )?;
        self.allocate_intermediate_2d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "proj_out",
            max_prefill_chunk,
            arch.d_model,
        )?;
        self.allocate_intermediate_2d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "residual_1",
            max_prefill_chunk,
            arch.d_model,
        )?;
        self.allocate_intermediate_2d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "ffn_norm_out",
            max_prefill_chunk,
            arch.d_model,
        )?;
        self.allocate_intermediate_2d(&mut bindings, &mut fast_bindings, foundry, "gate", max_prefill_chunk, arch.ff_dim)?;
        self.allocate_intermediate_2d(&mut bindings, &mut fast_bindings, foundry, "up", max_prefill_chunk, arch.ff_dim)?;
        self.allocate_intermediate_2d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "ffn_out",
            max_prefill_chunk,
            arch.d_model,
        )?;
        self.allocate_intermediate_2d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "final_norm_out",
            max_prefill_chunk,
            arch.d_model,
        )?;
        self.allocate_intermediate_2d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "logits",
            max_prefill_chunk,
            arch.vocab_size,
        )?;
        self.allocate_intermediate_2d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "attn_out",
            max_prefill_chunk,
            arch.d_model,
        )?;

        // 3D buffers for SDPA (batch, seq_len, dim)
        // For prefill, seq_len dimension can be up to max_prefill_chunk
        let kv_dim = arch.d_model / arch.n_heads * arch.n_kv_heads;
        self.allocate_intermediate_3d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "q",
            batch,
            max_prefill_chunk,
            arch.d_model,
        )?;
        self.allocate_intermediate_3d(&mut bindings, &mut fast_bindings, foundry, "k", batch, max_prefill_chunk, kv_dim)?;
        self.allocate_intermediate_3d(&mut bindings, &mut fast_bindings, foundry, "v", batch, max_prefill_chunk, kv_dim)?;
        let head_dim = arch.d_model / arch.n_heads;
        self.allocate_intermediate_3d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "q_heads",
            batch * arch.n_heads,
            max_prefill_chunk,
            head_dim,
        )?;
        self.allocate_intermediate_3d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "k_heads",
            batch * arch.n_kv_heads,
            max_prefill_chunk,
            head_dim,
        )?;
        self.allocate_intermediate_3d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "v_heads",
            batch * arch.n_kv_heads,
            max_prefill_chunk,
            head_dim,
        )?;
        self.allocate_intermediate_3d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "q_rot",
            batch,
            max_prefill_chunk,
            arch.d_model,
        )?;
        self.allocate_intermediate_3d(
            &mut bindings,
            &mut fast_bindings,
            foundry,
            "k_rot",
            batch,
            max_prefill_chunk,
            kv_dim,
        )?;
        // Expanded K/V buffers for GQA (after RepeatKvHeads, same dim as Q).
        // Must be sized for the runtime max sequence length because they hold repeated history.
        let expanded_dim = batch * max_seq_len * arch.d_model;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "k_expanded", expanded_dim)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "v_expanded", expanded_dim)?;

        // KV slice buffers for cache reads (sized for runtime max sequence length to avoid reallocation)
        // These hold the sliced cache [n_kv_heads, current_seq_len, head_dim]
        // We allocate to max size and track actual usage via globals
        let kv_slice_dim = arch.n_kv_heads * max_seq_len * head_dim;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "k_slice", kv_slice_dim)?;
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "v_slice", kv_slice_dim)?;

        // 5. Create a "zero" buffer for unused bias/residual slots
        self.allocate_intermediate(&mut bindings, &mut fast_bindings, foundry, "zero", 1)?;

        // 6. Allocate KV cache for autoregressive generation
        // Context stores KV already repeated to n_heads (so decode avoids a RepeatKvHeads dispatch).
        // Shape: [n_heads, max_seq_len, head_dim] per layer
        // We'll create one k_cache and v_cache per layer, named k_cache_0, k_cache_1, etc.
        let head_dim = arch.d_model / arch.n_heads;
        for layer_idx in 0..arch.n_layers {
            let k_cache_name = format!("k_cache_{}", layer_idx);
            let v_cache_name = format!("v_cache_{}", layer_idx);
            // KV cache: [n_heads, max_seq_len, head_dim]
            self.allocate_kv_cache(
                &mut bindings,
                &mut fast_bindings,
                foundry,
                &k_cache_name,
                arch.n_heads,
                max_seq_len,
                head_dim,
            )?;
            self.allocate_kv_cache(
                &mut bindings,
                &mut fast_bindings,
                foundry,
                &v_cache_name,
                arch.n_heads,
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
        arch: &crate::spec::Architecture,
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
        self.bind_gguf_tensor_with_layout(bindings, fast_bindings, foundry, gguf_name, logical_name, WeightLayout::RowMajor)
    }

    /// Bind a GGUF tensor to bindings in canonical k-block-major layout.
    /// Used for 2D weight matrices with GemvCanonical kernel.
    #[allow(clippy::too_many_arguments)]
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
        self.bind_gguf_tensor_with_layout(
            bindings,
            fast_bindings,
            foundry,
            gguf_name,
            logical_name,
            WeightLayout::Canonical { expected_k, expected_n },
        )
    }

    /// Helper to bind a GGUF tensor using the new QuantizationPolicy system.
    fn bind_gguf_tensor_with_layout(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        gguf_name: &str,
        logical_name: &str,
        layout: WeightLayout,
    ) -> Result<(), MetalError> {
        let gguf = self.weights.gguf_model();
        let tensor_info = gguf
            .get_tensor(gguf_name)
            .ok_or_else(|| MetalError::InputNotFound(format!("Tensor '{}' not found in GGUF", gguf_name)))?;

        let policy = resolve_policy(tensor_info.data_type());

        // Load weights using the policy (handles Q8 splitting, F32 downcast, canonical, etc.)
        let loaded_tensors = policy
            .load_weights(foundry, gguf, gguf_name, logical_name, layout)
            .map_err(|e| MetalError::OperationFailed(format!("Policy load failed for '{}': {}", gguf_name, e)))?;

        for (name, tensor_arg) in loaded_tensors {
            self.insert_binding(bindings, fast_bindings, name.clone(), tensor_arg.clone());

            if name == logical_name {
                // Also alias the GGUF name to the same tensor arg
                self.insert_binding(bindings, fast_bindings, gguf_name.to_string(), tensor_arg);
                tracing::trace!("Bound '{}' -> '{}' using policy {}", logical_name, gguf_name, policy.name());
            } else {
                tracing::trace!("Bound derived '{}' using policy {}", name, policy.name());
            }
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

    /// Allocate a 2D intermediate buffer for activations (row-major).
    fn allocate_intermediate_2d(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(), MetalError> {
        use objc2_metal::MTLDevice;

        if rows == 0 || cols == 0 {
            return Err(MetalError::InvalidShape(format!(
                "allocate_intermediate_2d '{name}' requires rows>0 and cols>0"
            )));
        }

        // Skip if already bound
        if bindings.contains(name) {
            return Ok(());
        }

        let total_elements = rows * cols;
        let byte_size = total_elements * 2; // F16
        let buffer = foundry
            .device
            .0
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        let tensor_arg = TensorArg::from_buffer(
            crate::types::MetalBuffer(buffer),
            crate::tensor::Dtype::F16,
            vec![rows, cols],
            crate::tensor::compute_strides(&[rows, cols]),
        );

        self.insert_binding(bindings, fast_bindings, name.to_string(), tensor_arg);
        tracing::trace!("Allocated 2D intermediate '{}' [{}, {}]", name, rows, cols);

        Ok(())
    }

    /// Allocate a 3D intermediate buffer for SDPA-style tensors.
    #[allow(clippy::too_many_arguments)]
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
            crate::tensor::compute_strides(&[batch, seq_len, dim]),
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

    /// Helper to update an existing binding without allocating a new String key.
    /// Falls back to insert if the binding isn't present (should be rare on hot paths).
    fn set_binding(&self, bindings: &mut TensorBindings, fast_bindings: &mut FastBindings, name: &str, tensor: TensorArg) {
        if let Some(id) = self.symbol_table.get(name) {
            fast_bindings.set(id, tensor.clone());
        }
        bindings.set_binding(name, tensor);
    }

    /// Allocate a KV cache buffer for attention caching.
    /// Shape: [n_heads, max_seq_len, head_dim]
    #[allow(clippy::too_many_arguments)]
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

        // KV caches are hot-read on GPU every decode step; prefer private storage.
        // Shared is only useful for debugging (CPU visibility) and is slower on GPU.
        let storage_mode = if std::env::var("METALLIC_KV_CACHE_SHARED").is_ok() {
            objc2_metal::MTLResourceOptions::StorageModeShared
        } else {
            objc2_metal::MTLResourceOptions::StorageModePrivate
        };
        let buffer = foundry
            .device
            .0
            .newBufferWithLength_options(byte_size, storage_mode)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        let tensor_arg = TensorArg::from_buffer(
            crate::types::MetalBuffer(buffer),
            crate::tensor::Dtype::F16,
            vec![n_heads, max_seq_len, head_dim],
            crate::tensor::compute_strides(&[n_heads, max_seq_len, head_dim]),
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
        let profiling_per_kernel = crate::instrument::foundry_per_kernel_profiling_enabled();

        if !nested_capture && !profiling_per_kernel {
            // Start a new command buffer for this forward pass (token)
            // Reuse the existing command buffer if one is active but not capturing?
            // No, start_capture() handles creation.
            foundry.start_capture()?;
        }

        for step in &self.compiled_steps {
            step.execute(foundry, fast_bindings, bindings, &self.symbol_table)?;
        }

        if !nested_capture && !profiling_per_kernel {
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
        F: FnMut(u32, std::time::Duration, std::time::Duration) -> Result<bool, MetalError>,
    {
        if prompt_tokens.is_empty() {
            return Err(MetalError::InvalidShape("generate requires non-empty prompt_tokens".into()));
        }

        let mut generated = Vec::with_capacity(max_new_tokens);
        let arch = &self.spec.architecture;

        let setup_start = std::time::Instant::now();

        // Reuse the prepared session (weights + intermediates + persistent buffers) instead of
        // materializing everything per generation.
        self.initialize_session(foundry)?;
        let mut session_guard = self.session.borrow_mut();
        let session = session_guard
            .as_mut()
            .ok_or_else(|| MetalError::OperationFailed("Foundry session missing after initialization".into()))?;
        let bindings = &mut session.bindings;
        let fast_bindings = &mut session.fast_bindings;

        let d_model = arch.d_model;
        let n_heads = arch.n_heads;
        let n_kv_heads = arch.n_kv_heads;
        let head_dim = d_model / n_heads;
        let ff_dim = arch.ff_dim;

        let prompt_len = prompt_tokens.len();
        if prompt_len > session.input_ids_capacity {
            return Err(MetalError::InvalidShape(format!(
                "Prompt length {prompt_len} exceeds max_seq_len {}",
                session.input_ids_capacity
            )));
        }

        // Write all prompt tokens to the shared buffer
        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = session.input_ids_full.0.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(prompt_tokens.as_ptr(), ptr, prompt_len);
        }

        // Defaults for single-token inference (MatMul dispatch). Batched prefill overrides these.
        bindings.set_int_global("m", 1);
        bindings.set_int_global("seq_len", 1);
        bindings.set_int_global("total_elements_hidden", d_model);
        bindings.set_int_global("total_elements_q", n_heads * head_dim);
        bindings.set_int_global("total_elements_k", n_kv_heads * head_dim);
        bindings.set_int_global("total_elements_write", n_kv_heads * head_dim);
        bindings.set_int_global("total_elements_ffn", ff_dim);

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

        // Prefill KV cache.
        //
        // Fast path is batched prefill (M>1) in large chunks. However, we've observed correctness
        // issues when the *final* chunk is very small (e.g. 32 + 2 tokens). To keep performance
        // while avoiding that tail pathology, we rebalance the chunk size so the prompt splits
        // into similarly-sized chunks (e.g. 34 @ 32 => 17 + 17) instead of a tiny tail.
        //
        // Set `METALLIC_DISABLE_BATCHED_PREFILL=1` to force fully sequential prefill for isolation.
        let profiling_per_kernel = crate::instrument::foundry_per_kernel_profiling_enabled();
        let disable_batched_prefill = profiling_per_kernel || std::env::var("METALLIC_DISABLE_BATCHED_PREFILL").is_ok();

        // We use "prefill_chunk_size" as the vector width (m) for batched prefill and as a capture
        // batching knob in sequential mode.
        let max_prefill_chunk = bindings.get_int_global("max_prefill_chunk").unwrap_or(32).max(1);
        let (_, mut prefill_chunk_size) = Self::prefill_config();
        prefill_chunk_size = prefill_chunk_size.min(max_prefill_chunk).max(1);

        let input_ids_key = "input_ids";

        // Check for debug sync flag to disable batched capture for isolation
        let debug_sync = std::env::var("METALLIC_DEBUG_FORWARD_SYNC").is_ok();

        let prefill_start = std::time::Instant::now();
        let setup_duration = prefill_start.duration_since(setup_start);
        let mut last_prefill_m = 1usize;

        if !disable_batched_prefill {
            // === BATCHED PREFILL (m>1) ===
            // In debug-sync mode, isolate each chunk into its own synchronous command buffer.
            if !debug_sync && !profiling_per_kernel {
                foundry.start_capture()?;
            }

            let rebalance_chunk_size = |prompt_len: usize, requested: usize, max_allowed: usize| -> usize {
                let requested = requested.max(1).min(max_allowed.max(1));
                if prompt_len <= 1 {
                    return 1;
                }

                let chunks = prompt_len.div_ceil(requested);
                let balanced = prompt_len.div_ceil(chunks);
                balanced.max(1).min(max_allowed)
            };

            let chunk_size = rebalance_chunk_size(prompt_len, prefill_chunk_size, max_prefill_chunk);

            for (chunk_idx, chunk_tokens) in prompt_tokens.chunks(chunk_size).enumerate() {
                let m = chunk_tokens.len();
                last_prefill_m = m;
                let base_pos = chunk_idx * chunk_size;

                if debug_sync && !profiling_per_kernel {
                    foundry.start_capture()?;
                }

                bindings.set_int_global("m", m);
                bindings.set_int_global("seq_len", m);
                bindings.set_int_global("position_offset", base_pos);
                bindings.set_int_global("kv_seq_len", base_pos + m);

                bindings.set_int_global("total_elements_hidden", m * d_model);
                bindings.set_int_global("total_elements_q", m * n_heads * head_dim);
                bindings.set_int_global("total_elements_k", m * n_kv_heads * head_dim);
                bindings.set_int_global("total_elements_write", m * n_kv_heads * head_dim);
                bindings.set_int_global("total_elements_ffn", m * ff_dim);
                bindings.set_int_global("total_elements_slice", n_kv_heads * (base_pos + m) * head_dim);
                bindings.set_int_global("total_elements_repeat", n_heads * (base_pos + m) * head_dim);

                let mut tensor_input = TensorArg::from_buffer(session.input_ids_full.clone(), crate::tensor::Dtype::U32, vec![m], vec![1]);
                tensor_input.offset = base_pos * 4;
                self.set_binding(bindings, fast_bindings, input_ids_key, tensor_input);

                self.forward(foundry, bindings, &*fast_bindings)?;

                if debug_sync && !profiling_per_kernel {
                    let cmd = foundry.end_capture()?;
                    objc2_metal::MTLCommandBuffer::waitUntilCompleted(&*cmd);
                }
            }

            if !debug_sync && !profiling_per_kernel {
                let cmd = foundry.end_capture()?;
                objc2_metal::MTLCommandBuffer::waitUntilCompleted(&*cmd);
            }
        } else {
            // === SEQUENTIAL PREFILL (seq_len=1) ===
            // Capture the whole prefill to amortize submission overhead, but execute each token
            // with the same semantics as the decode loop.
            bindings.set_int_global("m", 1);
            bindings.set_int_global("seq_len", 1);
            bindings.set_int_global("total_elements_hidden", d_model);
            bindings.set_int_global("total_elements_q", n_heads * head_dim);
            bindings.set_int_global("total_elements_k", n_kv_heads * head_dim);
            bindings.set_int_global("total_elements_write", n_kv_heads * head_dim);
            bindings.set_int_global("total_elements_ffn", ff_dim);

            for (chunk_idx, chunk_tokens) in prompt_tokens.chunks(prefill_chunk_size).enumerate() {
                let base_pos = chunk_idx * prefill_chunk_size;

                if !profiling_per_kernel && (debug_sync || chunk_idx == 0) {
                    foundry.start_capture()?;
                }

                for (i, _token_id) in chunk_tokens.iter().enumerate() {
                    let pos = base_pos + i;
                    update_pos_globals(bindings, pos);

                    let mut tensor_input =
                        TensorArg::from_buffer(session.input_ids_full.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);
                    tensor_input.offset = pos * 4;
                    self.set_binding(bindings, fast_bindings, input_ids_key, tensor_input);

                    self.forward(foundry, bindings, &*fast_bindings)?;
                }

                if debug_sync && !profiling_per_kernel {
                    let cmd = foundry.end_capture()?;
                    objc2_metal::MTLCommandBuffer::waitUntilCompleted(&*cmd);
                }
            }

            if !debug_sync && !profiling_per_kernel {
                let cmd = foundry.end_capture()?;
                objc2_metal::MTLCommandBuffer::waitUntilCompleted(&*cmd);
            }
        }

        let prefill_duration = prefill_start.elapsed();

        // Reset to M=1 for autoregressive decode
        bindings.set_int_global("m", 1);
        bindings.set_int_global("seq_len", 1);
        bindings.set_int_global("total_elements_hidden", d_model);
        bindings.set_int_global("total_elements_q", n_heads * head_dim);
        bindings.set_int_global("total_elements_k", n_kv_heads * head_dim);
        bindings.set_int_global("total_elements_write", n_kv_heads * head_dim);
        bindings.set_int_global("total_elements_ffn", ff_dim);

        // Now autoregressive decode: sample from last prompt-token logits, then step forward per token.
        // Reuse the session input buffer as a valid fallback input_ids buffer (we overwrite binding to sampled-token
        // buffers during decode).
        let single_input_buffer = session.input_ids_full.clone();

        let default_decode_batch_size = bindings
            .get("output_weight")
            .ok()
            .map(|w| if w.dtype == crate::tensor::Dtype::F16 { 16 } else { 64 })
            .unwrap_or(64);

        let decode_batch_size = || -> usize {
            const MAX: usize = 256;
            let parsed = std::env::var("METALLIC_FOUNDRY_DECODE_BATCH_SIZE")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok());
            parsed.unwrap_or(default_decode_batch_size).clamp(1, MAX)
        };

        // Ensure input_ids is bound to the single input buffer (offset 0) for the generated tokens
        let input_ids_key = "input_ids";
        // Seed `input_ids` with any valid U32 buffer; we overwrite it per-step below to point at the
        // sampled-token buffer (avoids a dedicated CopyU32 kernel each step).
        {
            let mut tensor_input = TensorArg::from_buffer(single_input_buffer.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);
            tensor_input.offset = 0;
            self.set_binding(bindings, fast_bindings, input_ids_key, tensor_input);
        }

        // BATCHING: We batch multiple tokens into a single command buffer to amortize submission overhead.
        // We keep tokens on GPU (copying Sample -> Input) and only sync when the batch is full.
        let batch_size = if profiling_per_kernel { 1 } else { decode_batch_size() };
        if batch_size > session.sample_out_buffers.len() {
            return Err(MetalError::InvalidShape(format!(
                "Decode batch_size {batch_size} exceeds session capacity {}. This typically means METALLIC_FOUNDRY_DECODE_BATCH_SIZE changed after model load.",
                session.sample_out_buffers.len()
            )));
        }

        let mut pending_count = 0;
        let ignore_eos_stop = Self::ignore_eos_stop_enabled();
        let emit_host_metrics = crate::instrument::foundry_metrics_enabled();
        let mut batch_encode_start: Option<std::time::Instant> = None;

        for step in 0..max_new_tokens {
            let batch_idx = pending_count;

            // Start capture at the beginning of a batch
            if batch_idx == 0 && !profiling_per_kernel {
                foundry.start_capture()?;
                if emit_host_metrics {
                    batch_encode_start = Some(std::time::Instant::now());
                }
            }

            // 1. Get logits from previous forward pass (or prefill)
            let logits = bindings.get("logits")?;

            let mut logits_arg = logits.clone();

            // If sampling from prefill result (step 0), offset to the last token if batch > 1
            if step == 0 && last_prefill_m > 1 {
                logits_arg.offset += (last_prefill_m - 1) * (vocab_size as usize) * 2; // F16 bytes
            }

            // 2. Sample (Greedy or Random) using GPU kernel
            // We use SampleTopK for both to keep execution on GPU. Greedy is just top_k=1.
            let effective_top_k = if greedy { 1 } else { top_k };
            let sample_out = &session.sample_out_buffers[batch_idx];

            // Create sample kernel with destination buffer
            let sample_kernel = SampleTopK::new(
                &logits_arg,
                &TensorArg::from_buffer(sample_out.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]),
                vocab_size,
                effective_top_k,
                top_p,
                temperature,
                seed.wrapping_add(step as u32),
            );
            foundry.run(&sample_kernel)?;

            // 3. Feed the sampled token directly into the next forward pass by rebinding `input_ids`
            // to the sampled-token buffer. This avoids an extra CopyU32 dispatch per step.
            self.set_binding(
                bindings,
                fast_bindings,
                input_ids_key,
                TensorArg::from_buffer(sample_out.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]),
            );

            // 4. Update globals for the NEXT forward pass
            // We are about to run forward for position `prompt_len + step` (the token we just sampled)
            update_pos_globals(bindings, prompt_len + step);

            // 5. Run Forward (predicts next token's logits)
            self.forward(foundry, bindings, &*fast_bindings)?;

            pending_count += 1;

            // 6. If batch is full or we are done, sync and process
            if pending_count >= batch_size || step == max_new_tokens - 1 {
                if !profiling_per_kernel {
                    if emit_host_metrics {
                        if let Some(start) = batch_encode_start.take() {
                            let encode_duration = start.elapsed();
                            if !encode_duration.is_zero() {
                                metallic_instrumentation::record_metric_async!(
                                    metallic_instrumentation::MetricEvent::InternalKernelCompleted {
                                        parent_op_name: "foundry_decode".to_string(),
                                        internal_kernel_name: "batch_encode_total".to_string(),
                                        duration_us: encode_duration.as_micros().min(u128::from(u64::MAX)) as u64,
                                    }
                                );
                            }
                        }
                    }

                    let end_capture_start = std::time::Instant::now();
                    let cmd = foundry.end_capture()?;
                    let end_capture_duration = end_capture_start.elapsed();
                    if emit_host_metrics && !end_capture_duration.is_zero() {
                        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::InternalKernelCompleted {
                            parent_op_name: "foundry_decode".to_string(),
                            internal_kernel_name: "end_capture".to_string(),
                            duration_us: end_capture_duration.as_micros().min(u128::from(u64::MAX)) as u64,
                        });
                    }

                    // Sync CPU to read back the tokens.
                    let wait_start = std::time::Instant::now();
                    objc2_metal::MTLCommandBuffer::waitUntilCompleted(&*cmd);
                    let wait_duration = wait_start.elapsed();
                    if emit_host_metrics && !wait_duration.is_zero() {
                        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::InternalKernelCompleted {
                            parent_op_name: "foundry_decode".to_string(),
                            internal_kernel_name: "cb_wait".to_string(),
                            duration_us: wait_duration.as_micros().min(u128::from(u64::MAX)) as u64,
                        });
                    }
                } else {
                    foundry.synchronize()?;
                }

                // Process the batch results
                let process_start = std::time::Instant::now();
                let mut batch_done = false;
                for buffer in session.sample_out_buffers.iter().take(pending_count) {
                    let token_buf = buffer;
                    let token = unsafe {
                        use objc2_metal::MTLBuffer;
                        let ptr = token_buf.0.contents().as_ptr() as *const u32;
                        *ptr
                    };

                    generated.push(token);

                    if !ignore_eos_stop && stop_tokens.contains(&token) {
                        batch_done = true;
                        break;
                    }
                    if !callback(token, prefill_duration, setup_duration)? {
                        batch_done = true;
                        break;
                    }
                }
                let process_duration = process_start.elapsed();
                if emit_host_metrics && !process_duration.is_zero() {
                    metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::InternalKernelCompleted {
                        parent_op_name: "foundry_decode".to_string(),
                        internal_kernel_name: "batch_process".to_string(),
                        duration_us: process_duration.as_micros().min(u128::from(u64::MAX)) as u64,
                    });
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
    #[allow(clippy::too_many_arguments)]
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
            |_, _, _| Ok(true),
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

#[cfg(test)]
mod tests {
    #[test]
    fn test_rebalanced_prefill_chunk_size_avoids_tiny_tail() {
        let rebalance = |prompt_len: usize, requested: usize, max_allowed: usize| -> usize {
            let requested = requested.max(1).min(max_allowed.max(1));
            if prompt_len <= 1 {
                return 1;
            }
            let chunks = prompt_len.div_ceil(requested);
            let balanced = prompt_len.div_ceil(chunks);
            balanced.max(1).min(max_allowed)
        };

        // Repro shape: 34 with requested 32 becomes 17+17 (no 2-token tail).
        assert_eq!(rebalance(34, 32, 32), 17);
        // Already balanced.
        assert_eq!(rebalance(1000, 32, 32), 32);
        // Single chunk.
        assert_eq!(rebalance(31, 32, 32), 31);
        // Degenerate.
        assert_eq!(rebalance(0, 32, 32), 1);
        assert_eq!(rebalance(1, 32, 32), 1);
    }
}
