use crate::gguf::{GGUFValue, model_loader::GGUFModel};
use crate::metallic::{
    Context, MetalError,
    models::{LoadableModel, Qwen25, Qwen25Config},
};

/// Implement the LoadableModel trait so Qwen25 can be created from a GGUFModel
impl LoadableModel for super::Qwen25 {
    fn load_from_gguf(gguf_model: &GGUFModel, ctx: &mut Context) -> Result<Self, MetalError> {
        // Build config heuristically from metadata (fallbacks kept consistent with qwen25::new)
        let d_model = gguf_model
            .get_metadata_u32_or(
                &["qwen2.embedding_length", "qwen2.d_model", "model.d_model"],
                896,
            ) as usize;
        let ff_dim = gguf_model
            .get_metadata_u32_or(
                &["qwen2.feed_forward_length", "qwen2.ff_dim", "model.ff_dim"],
                4864,
            ) as usize;
        let n_heads = gguf_model
            .get_metadata_u32_or(
                &["qwen2.attention.head_count", "qwen2.n_heads", "model.n_heads"],
                14,
            ) as usize;
        let n_kv_heads = gguf_model
            .get_metadata_u32_or(
                &["qwen2.attention.head_count_kv", "qwen2.n_kv_heads", "model.n_kv_heads"],
                2,
            ) as usize;
        let n_layers = gguf_model
            .get_metadata_u32_or(&["qwen2.block_count", "qwen2.n_layers", "model.n_layers"], 24)
            as usize;
        let seq_len = gguf_model.get_metadata_u32_or(&["qwen2.context_length", "model.context_length"], 32768) as usize;
        let rope_freq_base = gguf_model.get_metadata_f32_or(
            &["qwen2.rope.freq_base", "qwen2.rope_theta"],
            1_000_000.0,
        );
        let rms_eps = gguf_model.get_metadata_f32_or(
            &["qwen2.attention.layer_norm_rms_epsilon", "qwen2.rms_norm_eps", "qwen2.rope.rms_eps"],
            1e-6,
        );

        // Determine vocabulary size:
        // 1) Prefer explicit `vocab_size` or `model.vocab_size` metadata
        // 2) Otherwise, if GGUF provides `tokenizer.ggml.tokens` (an array), use its length
        // 3) Fallback to legacy default (32000)
        let metadata = gguf_model.metadata();
        let vocab_size = if let Some(GGUFValue::U32(v)) = metadata
            .entries
            .get("vocab_size")
            .or_else(|| metadata.entries.get("model.vocab_size"))
        {
            *v as usize
        } else if let Some(GGUFValue::Array(arr)) = metadata.entries.get("tokenizer.ggml.tokens") {
            // Use tokenizer token count when available to avoid token-id out-of-range errors.
            arr.len()
        } else {
            151936usize // vocab_size from qwen25 config.json
        };

        let cfg = Qwen25Config {
            n_layers,
            d_model,
            ff_dim,
            n_heads,
            n_kv_heads,
            seq_len,
            vocab_size,
            rope_freq_base,
            rms_eps,
        };

        // Instantiate Qwen25 with default-initialized weights
        let mut qwen = Qwen25::new(cfg, ctx)?;

        // Helper: attempt to copy from gguf tensor into dst.
        // Fast-path when shapes match, and a linear fallback when only total element counts match.
        // We avoid transposing rectangular matrices since GGUF already has the correct layout for our matmuls.
        fn try_copy(
            gguf_model: &GGUFModel,
            tensor_info: &crate::gguf::GGUTensorInfo,
            dst: &mut crate::metallic::Tensor,
        ) -> Result<(), MetalError> {
            let expected = dst.len();
            let actual = gguf_model.tensor_element_count(tensor_info);
            if actual != expected {
                return Err(MetalError::DimensionMismatch { expected, actual });
            }

            gguf_model
                .copy_tensor_to_f32(tensor_info, dst.as_mut_slice())
                .map_err(|e| MetalError::InvalidOperation(format!(
                    "failed to copy tensor '{}' into destination: {}",
                    tensor_info.name, e
                )))
        }

        // layer index extractor (searches for common patterns)
        fn parse_layer_index(name: &str) -> Option<usize> {
            let patterns = ["layers.", "layer.", "blocks.", "block.", "layer_", "layers_"];
            let lname = name.to_lowercase();
            for pat in patterns.iter() {
                if let Some(pos) = lname.find(pat) {
                    let start = pos + pat.len();
                    let mut digits = String::new();
                    for ch in lname[start..].chars() {
                        if ch.is_ascii_digit() {
                            digits.push(ch);
                        } else {
                            break;
                        }
                    }
                    if !digits.is_empty()
                        && let Ok(idx) = digits.parse::<usize>()
                    {
                        return Some(idx);
                    }
                }
            }
            // fallback: first run of digits
            let mut found = String::new();
            for ch in lname.chars() {
                if ch.is_ascii_digit() {
                    found.push(ch);
                } else if !found.is_empty() {
                    break;
                }
            }
            if !found.is_empty()
                && let Ok(idx) = found.parse::<usize>()
            {
                return Some(idx);
            }
            None
        }

        // Iterate gguf tensors and map into Qwen25 where names match heuristics.
        // This mapping is intentionally permissive to handle different exporter naming schemes.
        for tensor_info in gguf_model.tensor_infos() {
            let name = &tensor_info.name;
            let lname = name.to_lowercase();
            let len = gguf_model.tensor_element_count(tensor_info);

            // Embedding and weight tying for token embeddings
            if ((lname.contains("token") && lname.contains("emb"))
                || lname.contains("tok_emb")
                || lname.contains("tokembedding"))
                && len == qwen.embed_weight.len()
            {
                try_copy(gguf_model, tensor_info, &mut qwen.embed_weight)?;
                continue;
            }

            // Output / lm_head
            if (lname.contains("lm_head")
                || lname.contains("lmhead")
                || (lname.contains("output") && lname.contains("weight")))
                && len == qwen.output_weight.len()
            {
                try_copy(gguf_model, tensor_info, &mut qwen.output_weight)?;
                continue;
            }

            // Final norm (output_norm)
            if (lname.contains("output_norm")
                || lname.contains("final_norm")
                || (lname == "norm.weight" && len == qwen.final_norm_gamma.len()))
            {
                try_copy(gguf_model, tensor_info, &mut qwen.final_norm_gamma)?;
                continue;
            }

            // Handle weight tying - if this is the token embedding and we haven't set output weights yet,
            // use it as the output weight
            if lname.contains("token_embd") && lname.contains("weight") {
                let output_slice = qwen.output_weight.as_slice();
                let is_output_unset = output_slice.iter().all(|&x| x == 0.0);

                if is_output_unset {
                    if len == qwen.output_weight.len() {
                        if let Err(e) = try_copy(gguf_model, tensor_info, &mut qwen.output_weight) {
                            println!("MAPPING -> token_embd.weight copy to output_weight failed: {:?}", e);
                        }
                    } else {
                        println!("MAPPING -> token_embd.weight size mismatch: {} vs {}", len, qwen.output_weight.len());
                    }
                }
                continue;
            }

            // Per-layer parameters
            if let Some(layer_idx) = parse_layer_index(&lname) {
                if layer_idx >= qwen.blocks.len() {
                    continue;
                }
                let block = &mut qwen.blocks[layer_idx];

                // Attention projections (many exporters use different names)
                if (lname.contains("wq")
                    || lname.contains("attn.q")
                    || lname.contains("attn_q")
                    || lname.contains("q_proj.weight")
                    || lname.contains("query.weight")
                    || lname.contains("q.weight")
                    || lname.contains("attention.query.weight"))
                    && len == block.attn_q_weight.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.attn_q_weight)?;
                    continue;
                }

                if (lname.contains("wk")
                    || lname.contains("attn.k")
                    || lname.contains("attn_k")
                    || lname.contains("k_proj.weight")
                    || lname.contains("key.weight")
                    || lname.contains("k.weight")
                    || lname.contains("attention.key.weight"))
                    && len == block.attn_k_weight.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.attn_k_weight)?;
                    continue;
                }

                if (lname.contains("wv")
                    || lname.contains("attn.v")
                    || lname.contains("attn_v")
                    || lname.contains("v_proj.weight")
                    || lname.contains("value.weight")
                    || lname.contains("v.weight")
                    || lname.contains("attention.value.weight"))
                    && len == block.attn_v_weight.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.attn_v_weight)?;
                    continue;
                }

                if (lname.contains("wo")
                    || lname.contains("attn_out")
                    || lname.contains("attn.out")
                    || lname.contains("out_proj")
                    || lname.contains("o.weight")
                    || lname.contains("out.weight")
                    || lname.contains("attention.output.weight"))
                    && len == block.attn_out_weight.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.attn_out_weight)?;
                    continue;
                }

                if (lname.contains("attn.q.bias")
                    || lname.contains("attn_q.bias")
                    || lname.contains("attention.query.bias"))
                    && len == block.attn_q_bias.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.attn_q_bias)?;
                    continue;
                }

                if (lname.contains("attn.k.bias")
                    || lname.contains("attn_k.bias")
                    || lname.contains("attention.key.bias"))
                    && len == block.attn_k_bias.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.attn_k_bias)?;
                    continue;
                }

                if (lname.contains("attn.v.bias")
                    || lname.contains("attn_v.bias")
                    || lname.contains("attention.value.bias"))
                    && len == block.attn_v_bias.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.attn_v_bias)?;
                    continue;
                }

                if (lname.contains("mlp.gate_proj.weight")
                    || lname.contains("ffn.gate")
                    || lname.contains("ffn_gate")
                    || lname.contains("gate.weight")
                    || lname.contains("wg.weight")
                    || lname.contains("w1.weight"))
                    && len == block.ffn_gate.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.ffn_gate)?;
                    continue;
                }

                if (lname.contains("mlp.up_proj.weight")
                    || lname.contains("ffn.up")
                    || lname.contains("ffn_up")
                    || lname.contains("up.weight")
                    || lname.contains("w3.weight")
                    || lname.contains("fc1.weight"))
                    && len == block.ffn_up.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.ffn_up)?;
                    continue;
                }

                if (lname.contains("mlp.down_proj.weight")
                    || lname.contains("ffn.down")
                    || lname.contains("ffn_down")
                    || lname.contains("down.weight")
                    || lname.contains("w2.weight")
                    || lname.contains("fc2.weight")
                    || lname.contains("wo.weight"))
                    && len == block.ffn_down.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.ffn_down)?;
                    continue;
                }

                if (lname.contains("mlp.gate_proj.bias")
                    || lname.contains("ffn.gate.bias")
                    || lname.contains("ffn_gate.bias")
                    || lname.contains("gate.bias")
                    || lname.contains("w1.bias")
                    || lname.contains("wg.bias"))
                    && len == block.ffn_gate_bias.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.ffn_gate_bias)?;
                    continue;
                }

                if (lname.contains("mlp.up_proj.bias")
                    || lname.contains("ffn.up.bias")
                    || lname.contains("ffn_up.bias")
                    || lname.contains("up.bias")
                    || lname.contains("w3.bias"))
                    && len == block.ffn_up_bias.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.ffn_up_bias)?;
                    continue;
                }

                if (lname.contains("mlp.down_proj.bias")
                    || lname.contains("ffn.down.bias")
                    || lname.contains("ffn_down.bias")
                    || lname.contains("down.bias")
                    || lname.contains("w2.bias")
                    || lname.contains("b2.bias")
                    || lname.contains("fc2.bias")
                    || lname.contains("wo.bias"))
                    && len == block.ffn_down_bias.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.ffn_down_bias)?;
                    continue;
                }

                if ((lname.contains("attn_norm") && lname.contains("gamma"))
                    || lname.contains("attn.g")
                    || lname.contains("attn_norm.weight")
                    || lname.contains("attention.layernorm.weight")
                    || lname.contains("ln_attn.weight"))
                    && len == block.attn_norm_gamma.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.attn_norm_gamma)?;
                    continue;
                }

                if ((lname.contains("proj_norm") && lname.contains("gamma"))
                    || lname.contains("proj.g")
                    || lname.contains("ffn_norm.weight")
                    || lname.contains("ffn.layernorm.weight")
                    || lname.contains("ln_ffn.weight"))
                    && len == block.ffn_norm_gamma.len()
                {
                    try_copy(gguf_model, tensor_info, &mut block.ffn_norm_gamma)?;
                    continue;
                }
            }
        }

        Ok(qwen)
    }
}
