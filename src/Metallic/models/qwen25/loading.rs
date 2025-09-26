use crate::gguf::{GGUFValue, model_loader::GGUFModel};
use crate::metallic::{
    Context, MetalError,
    models::{LoadableModel, Qwen25, Qwen25Config},
};

/// Implement the LoadableModel trait so Qwen25 can be created from a GGUFModel
impl LoadableModel for super::Qwen25 {
    fn load_from_gguf(gguf_model: &GGUFModel, ctx: &mut Context) -> Result<Self, MetalError> {
        // Build config heuristically from metadata (fallbacks kept consistent with qwen25::new)
        let d_model = gguf_model.get_metadata_u32_or(&["qwen2.d_model", "model.d_model"], 896) as usize;
        let ff_dim = gguf_model.get_metadata_u32_or(&["qwen2.ff_dim", "model.ff_dim"], 4864) as usize;
        let n_heads = gguf_model.get_metadata_u32_or(&["qwen2.n_heads", "model.n_heads"], 14) as usize;
        let n_kv_heads = gguf_model.get_metadata_u32_or(&["qwen2.n_kv_heads", "model.n_kv_heads"], 2) as usize;
        let n_layers = gguf_model.get_metadata_u32_or(&["qwen2.n_layers", "model.n_layers"], 24) as usize;
        let seq_len = gguf_model.get_metadata_u32_or(&["qwen2.context_length", "model.context_length"], 32768) as usize;
        let rope_freq_base = gguf_model.get_metadata_f32_or(&["qwen2.rope_theta", "qwen2.rope.freq_base"], 1_000_000.0);
        let rms_eps = gguf_model.get_metadata_f32_or(&["qwen2.rms_norm_eps", "qwen2.rope.rms_eps"], 1e-6);

        // Determine vocabulary size:
        // 1) Prefer explicit `vocab_size` or `model.vocab_size` metadata
        // 2) Otherwise, if GGUF provides `tokenizer.ggml.tokens` (an array), use its length
        // 3) Fallback to legacy default (32000)
        let vocab_size = if let Some(GGUFValue::U32(v)) = gguf_model
            .metadata
            .entries
            .get("vocab_size")
            .or_else(|| gguf_model.metadata.entries.get("model.vocab_size"))
        {
            *v as usize
        } else if let Some(GGUFValue::Array(arr)) = gguf_model.metadata.entries.get("tokenizer.ggml.tokens") {
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
        fn try_copy(src: &crate::metallic::Tensor, dst: &mut crate::metallic::Tensor) -> Result<(), MetalError> {
            // Basic size check
            if src.len() != dst.len() {
                return Err(MetalError::DimensionMismatch {
                    expected: dst.len(),
                    actual: src.len(),
                });
            }

            // Exact-shape fast path
            if src.dims == dst.dims {
                let s = src.as_slice();
                let d = dst.as_mut_slice();
                d.copy_from_slice(s);
                return Ok(());
            }

            // Fallback linear copy for other shapes
            let s = src.as_slice();
            let d = dst.as_mut_slice();
            d.copy_from_slice(s);
            Ok(())
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
        for (name, tensor) in &gguf_model.tensors {
            let lname = name.to_lowercase();

            // Embedding and weight tying for token embeddings
            if ((lname.contains("token") && lname.contains("emb")) || lname.contains("tok_emb") || lname.contains("tokembedding"))
                && tensor.len() == qwen.embed_weight.len()
            {
                // For this model, the data is already laid out for [vocab, d_model] despite GGUF dims; linear copy
                let src_slice = tensor.as_slice();
                let dst_slice = qwen.embed_weight.as_mut_slice();
                dst_slice.copy_from_slice(src_slice);
                //println!("MAPPING DEBUG: Linear copy for token_embd to [vocab, d_model]");

                continue;
            }

            // Output / lm_head
            if (lname.contains("lm_head") || lname.contains("lmhead") || (lname.contains("output") && lname.contains("weight")))
                && tensor.len() == qwen.output_weight.len()
            {
                try_copy(tensor, &mut qwen.output_weight).expect("succesfull copy");
                continue;
            }

            // Final norm (output_norm)
            if lname.contains("output_norm")
                || lname.contains("final_norm")
                || lname == "norm.weight" && tensor.len() == qwen.final_norm_gamma.len()
            {
                try_copy(tensor, &mut qwen.final_norm_gamma).expect("successful copy");
                continue;
            }

            // Handle weight tying - if this is the token embedding and we haven't set output weights yet,
            // use it as the output weight
            if lname.contains("token_embd") && lname.contains("weight") {
                // Check if output_weight is still all zeros (not set)
                let output_slice = qwen.output_weight.as_slice();
                let is_output_unset = output_slice.iter().all(|&x| x == 0.0);

                if is_output_unset {
                    // The GGUF token_embd.weight has shape [d_model, vocab_size] which matches our output_weight shape
                    if tensor.len() == qwen.output_weight.len() {
                        match try_copy(tensor, &mut qwen.output_weight) {
                            Ok(()) => {
                                //println!("MAPPING -> Applied weight tying: token_embd.weight -> output_weight");
                            }
                            Err(e) => {
                                println!("MAPPING -> token_embd.weight copy to output_weight failed: {:?}", e);
                            }
                        }
                    } else {
                        println!(
                            "MAPPING -> token_embd.weight size mismatch: {} vs {}",
                            tensor.len(),
                            qwen.output_weight.len()
                        );
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
                // Query
                if (lname.contains("wq")
                    || lname.contains("attn.q")
                    || lname.contains("attn_q")
                    || lname.contains("q_proj.weight")
                    || lname.contains("query.weight")
                    || lname.contains("q.weight")
                    || lname.contains("attention.query.weight"))
                    && tensor.len() == block.attn_q_weight.len()
                {
                    try_copy(tensor, &mut block.attn_q_weight).expect("succesfull copy");
                    continue;
                }

                // Key
                if (lname.contains("wk")
                    || lname.contains("attn.k")
                    || lname.contains("attn_k")
                    || lname.contains("k_proj.weight")
                    || lname.contains("key.weight")
                    || lname.contains("k.weight")
                    || lname.contains("attention.key.weight"))
                    && tensor.len() == block.attn_k_weight.len()
                {
                    try_copy(tensor, &mut block.attn_k_weight).expect("successful copy");
                    continue;
                }

                // Value
                if (lname.contains("wv")
                    || lname.contains("attn.v")
                    || lname.contains("attn_v")
                    || lname.contains("v_proj.weight")
                    || lname.contains("value.weight")
                    || lname.contains("v.weight")
                    || lname.contains("attention.value.weight"))
                    && tensor.len() == block.attn_v_weight.len()
                {
                    try_copy(tensor, &mut block.attn_v_weight).expect("succesfull copy");
                    continue;
                }
                // Attention output / projection (wo, outproj, o_proj)
                if (lname.contains("wo")
                    || lname.contains("attn_out")
                    || lname.contains("attn.out")
                    || lname.contains("out_proj")
                    || lname.contains("o.weight")
                    || lname.contains("out.weight")
                    || lname.contains("attention.output.weight"))
                    && tensor.len() == block.attn_out_weight.len()
                {
                    try_copy(tensor, &mut block.attn_out_weight).expect("succesfull copy");
                    continue;
                }

                // Attention biases (optional)
                if (lname.contains("attn.q.bias") || lname.contains("attn_q.bias") || lname.contains("attention.query.bias"))
                    && tensor.len() == block.attn_q_bias.len()
                {
                    try_copy(tensor, &mut block.attn_q_bias).expect("succesfull copy");
                    continue;
                }

                if (lname.contains("attn.k.bias") || lname.contains("attn_k.bias") || lname.contains("attention.key.bias"))
                    && tensor.len() == block.attn_k_bias.len()
                {
                    try_copy(tensor, &mut block.attn_k_bias).expect("succesfull copy");
                    continue;
                }
                if (lname.contains("attn.v.bias") || lname.contains("attn_v.bias") || lname.contains("attention.value.bias"))
                    && tensor.len() == block.attn_v_bias.len()
                {
                    try_copy(tensor, &mut block.attn_v_bias).expect("succesfull copy");
                    continue;
                }
                if (lname.contains("attn.v.bias") || lname.contains("attn_v.bias") || lname.contains("attention.value.bias"))
                    && tensor.len() == block.attn_v_bias.len()
                {
                    try_copy(tensor, &mut block.attn_v_bias).expect("succesfull copy");
                    continue;
                }

                // FFN down/gate/up (common aliases) - Updated for correct SwiGLU mapping
                // gate_proj (for SiLU activation)
                if (lname.contains("mlp.gate_proj.weight")
                    || lname.contains("ffn.gate")
                    || lname.contains("ffn_gate")
                    || lname.contains("gate.weight")
                    || lname.contains("wg.weight")
                    || lname.contains("w1.weight"))
                    && tensor.len() == block.ffn_gate.len()
                {
                    if tensor.dims == block.ffn_gate.dims() {
                        try_copy(tensor, &mut block.ffn_gate).expect("successful copy for ffn_gate (exact match)");
                    } else if tensor.dims.len() == 2
                        && block.ffn_gate.dims().len() == 2
                        && tensor.dims[0] == block.ffn_gate.dims()[1]
                        && tensor.dims[1] == block.ffn_gate.dims()[0]
                    {
                        // GGUF metadata reports dims swapped, but data is already laid out row-major in
                        // [ff_dim, d_model]; copy directly without transposing to preserve values.
                        let src = tensor.as_slice();
                        let dst = block.ffn_gate.as_mut_slice();
                        dst.copy_from_slice(src);
                    } else {
                        try_copy(tensor, &mut block.ffn_gate).expect("successful linear copy fallback for ffn_gate");
                    }
                    continue;
                }

                // up_proj (for element-wise multiplication)
                if (lname.contains("mlp.up_proj.weight")
                    || lname.contains("ffn.up")
                    || lname.contains("ffn_up")
                    || lname.contains("up.weight")
                    || lname.contains("w3.weight"))
                    && tensor.len() == block.ffn_up.len()
                {
                    if tensor.dims == block.ffn_up.dims() {
                        try_copy(tensor, &mut block.ffn_up).expect("successful copy for ffn_up (exact match)");
                    } else if tensor.dims.len() == 2
                        && block.ffn_up.dims().len() == 2
                        && tensor.dims[0] == block.ffn_up.dims()[1]
                        && tensor.dims[1] == block.ffn_up.dims()[0]
                    {
                        let src = tensor.as_slice();
                        let dst = block.ffn_up.as_mut_slice();
                        dst.copy_from_slice(src);
                    } else {
                        try_copy(tensor, &mut block.ffn_up).expect("successful linear copy fallback for ffn_up");
                    }
                    continue;
                }

                // down_proj (final projection)
                if (lname.contains("mlp.down_proj.weight")
                    || lname.contains("ffn.down")
                    || lname.contains("ffn_down")
                    || lname.contains("down.weight")
                    || lname.contains("w2.weight")
                    || lname.contains("fc2.weight")
                    || lname.contains("wo.weight"))
                    && tensor.len() == block.ffn_down.len()
                {
                    if tensor.dims == block.ffn_down.dims() {
                        try_copy(tensor, &mut block.ffn_down).expect("successful copy for ffn_down (exact match)");
                    } else if tensor.dims.len() == 2
                        && block.ffn_down.dims().len() == 2
                        && tensor.dims[0] == block.ffn_down.dims()[1]
                        && tensor.dims[1] == block.ffn_down.dims()[0]
                    {
                        let src = tensor.as_slice();
                        let dst = block.ffn_down.as_mut_slice();
                        dst.copy_from_slice(src);
                        //println!("MAPPING DEBUG: FFN down loaded via direct copy with swapped dims metadata");
                    } else {
                        try_copy(tensor, &mut block.ffn_down).expect("successful linear copy fallback for ffn_down");
                    }
                    continue;
                }

                // FFN biases (gate/up/down) - map biases into block fields when present
                if lname.contains("mlp.gate_proj.bias")
                    || lname.contains("ffn.gate.bias")
                    || lname.contains("ffn_gate.bias")
                    || lname.contains("gate.bias")
                    || lname.contains("w1.bias")
                    || lname.contains("wg.bias")
                {
                    if tensor.len() == block.ffn_gate_bias.len() {
                        try_copy(tensor, &mut block.ffn_gate_bias).expect("successful copy of gate bias");
                    }
                    continue;
                }
                if lname.contains("mlp.up_proj.bias")
                    || lname.contains("ffn.up.bias")
                    || lname.contains("ffn_up.bias")
                    || lname.contains("up.bias")
                    || lname.contains("w3.bias")
                {
                    if tensor.len() == block.ffn_up_bias.len() {
                        try_copy(tensor, &mut block.ffn_up_bias).expect("successful copy of up bias");
                    }
                    continue;
                }
                if lname.contains("mlp.down_proj.bias")
                    || lname.contains("ffn.down.bias")
                    || lname.contains("ffn_down.bias")
                    || lname.contains("down.bias")
                    || lname.contains("w2.bias")
                    || lname.contains("b2.bias")
                    || lname.contains("fc2.bias")
                    || lname.contains("wo.bias")
                {
                    if tensor.len() == block.ffn_down_bias.len() {
                        try_copy(tensor, &mut block.ffn_down_bias).expect("successful copy of down bias");
                    }
                    continue;
                }

                // Norm gammas (RMSNorm)
                if ((lname.contains("attn_norm") && lname.contains("gamma"))
                    || lname.contains("attn.g")
                    || lname.contains("attn_norm.weight")
                    || lname.contains("attention.layernorm.weight")
                    || lname.contains("ln_attn.weight"))
                    && tensor.len() == block.attn_norm_gamma.len()
                {
                    try_copy(tensor, &mut block.attn_norm_gamma).expect("succesfull copy");
                    continue;
                }

                // (old duplicate FFN bias handling removed)

                // Norm gammas (RMSNorm)
                if ((lname.contains("attn_norm") && lname.contains("gamma"))
                    || lname.contains("attn.g")
                    || lname.contains("attn_norm.weight")
                    || lname.contains("attention.layernorm.weight")
                    || lname.contains("ln_attn.weight"))
                    && tensor.len() == block.attn_norm_gamma.len()
                {
                    try_copy(tensor, &mut block.attn_norm_gamma).expect("succesfull copy");
                    continue;
                }
                if ((lname.contains("proj_norm") && lname.contains("gamma"))
                    || lname.contains("proj.g")
                    || lname.contains("ffn_norm.weight")
                    || lname.contains("ffn.layernorm.weight")
                    || lname.contains("ln_ffn.weight"))
                    && tensor.len() == block.ffn_norm_gamma.len()
                {
                    try_copy(tensor, &mut block.ffn_norm_gamma).expect("succesfull copy");
                    continue;
                }
            }
        }

        Ok(qwen)
    }
}
