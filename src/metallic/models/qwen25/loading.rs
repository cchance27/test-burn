use crate::gguf::{GGUFValue, model_loader::GGUFModel};
use crate::metallic::{
    Context, Dtype, MetalError, Tensor, TensorElement,
    models::{LoadableModel, Qwen25, Qwen25Config},
};
use std::{any::TypeId, borrow::Cow};

fn tensor_data_as_f32<'a, T: TensorElement>(tensor: &'a Tensor<T>) -> Cow<'a, [f32]> {
    if T::DTYPE == Dtype::F32 {
        debug_assert_eq!(std::mem::size_of::<T::Scalar>(), std::mem::size_of::<f32>());
        let slice = tensor.as_slice();
        let ptr = slice.as_ptr();
        let len = slice.len();
        let f32_slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, len) };
        Cow::Borrowed(f32_slice)
    } else {
        Cow::Owned(tensor.as_slice().iter().copied().map(T::to_f32).collect())
    }
}

fn copy_f32_into_tensor<TDst: TensorElement>(src: &[f32], dst: &mut Tensor<TDst>) -> Result<(), MetalError> {
    if src.len() != dst.len() {
        return Err(MetalError::DimensionMismatch {
            expected: dst.len(),
            actual: src.len(),
        });
    }

    TDst::copy_from_f32_slice(src, dst.as_mut_slice());
    Ok(())
}

fn copy_tensor_data_into_slice<TSrc: TensorElement, TDst: TensorElement>(
    src: &Tensor<TSrc>,
    dst_slice: &mut [TDst::Scalar],
) -> Result<(), MetalError> {
    if src.len() != dst_slice.len() {
        return Err(MetalError::DimensionMismatch {
            expected: dst_slice.len(),
            actual: src.len(),
        });
    }

    if TypeId::of::<TSrc::Scalar>() == TypeId::of::<TDst::Scalar>()
        && std::mem::size_of::<TSrc::Scalar>() == std::mem::size_of::<TDst::Scalar>()
    {
        let src_slice = src.as_slice();
        unsafe {
            let typed_src = std::slice::from_raw_parts(src_slice.as_ptr() as *const TDst::Scalar, src_slice.len());
            dst_slice.copy_from_slice(typed_src);
        }
        return Ok(());
    }

    match (TSrc::DTYPE, TDst::DTYPE) {
        (Dtype::F32, _) => {
            let src_slice = src.as_slice();
            debug_assert_eq!(std::mem::size_of::<TSrc::Scalar>(), std::mem::size_of::<f32>());
            let src_f32 = unsafe { std::slice::from_raw_parts(src_slice.as_ptr() as *const f32, src_slice.len()) };
            TDst::copy_from_f32_slice(src_f32, dst_slice);
            Ok(())
        }
        (_, Dtype::F32) => {
            let src_slice = src.as_slice();
            debug_assert_eq!(std::mem::size_of::<TDst::Scalar>(), std::mem::size_of::<f32>());
            let dst_f32 = unsafe { std::slice::from_raw_parts_mut(dst_slice.as_mut_ptr() as *mut f32, dst_slice.len()) };
            for (dst, value) in dst_f32.iter_mut().zip(src_slice.iter().copied()) {
                *dst = TSrc::to_f32(value);
            }
            Ok(())
        }
        _ => {
            let src_slice = tensor_data_as_f32(src);
            TDst::copy_from_f32_slice(src_slice.as_ref(), dst_slice);
            Ok(())
        }
    }
}

fn copy_tensor_into<TSrc: TensorElement, TDst: TensorElement>(src: &Tensor<TSrc>, dst: &mut Tensor<TDst>) -> Result<(), MetalError> {
    if src.len() != dst.len() {
        return Err(MetalError::DimensionMismatch {
            expected: dst.len(),
            actual: src.len(),
        });
    }

    let dst_slice = dst.as_mut_slice();
    copy_tensor_data_into_slice::<TSrc, TDst>(src, dst_slice)
}

fn pack_weight_transposed_into_fused_slice<TDst: TensorElement>(
    src_slice: &[f32],
    src_dims: &[usize],
    dst_slice: &mut [TDst::Scalar],
    dst_dims: &[usize],
    dst_col_offset: usize,
) -> Result<(), MetalError> {
    if src_dims.len() != 2 {
        return Err(MetalError::InvalidShape(format!("Expected 2D weight tensor, got {:?}", src_dims)));
    }

    if dst_dims.len() != 2 {
        return Err(MetalError::InvalidShape(format!("Fused weight must be 2D, got {:?}", dst_dims)));
    }

    let fused_rows = dst_dims[0];
    let fused_cols = dst_dims[1];

    let (out_features, in_features) = if src_dims[1] == fused_rows {
        (src_dims[0], src_dims[1])
    } else if src_dims[0] == fused_rows {
        (src_dims[1], src_dims[0])
    } else {
        return Err(MetalError::InvalidShape(format!(
            "Unable to map weight {:?} into fused layout with {} rows",
            src_dims, fused_rows
        )));
    };

    if dst_col_offset + out_features > fused_cols {
        return Err(MetalError::InvalidShape(format!(
            "Column range [{}..{}) exceeds fused weight width {}",
            dst_col_offset,
            dst_col_offset + out_features,
            fused_cols
        )));
    }

    for out_idx in 0..out_features {
        for in_idx in 0..in_features {
            let src_index = out_idx * in_features + in_idx;
            let dst_row = in_idx;
            let dst_col = dst_col_offset + out_idx;
            let dst_index = dst_row * fused_cols + dst_col;
            dst_slice[dst_index] = TDst::from_f32(src_slice[src_index]);
        }
    }

    Ok(())
}

fn copy_weight_transposed_into_fused<TSrc: TensorElement, TDst: TensorElement>(
    src: &Tensor<TSrc>,
    dst: &mut Tensor<TDst>,
    dst_col_offset: usize,
) -> Result<(), MetalError> {
    let dst_dims = dst.dims().to_vec();
    let src_slice = tensor_data_as_f32(src);
    pack_weight_transposed_into_fused_slice::<TDst>(src_slice.as_ref(), src.dims(), dst.as_mut_slice(), &dst_dims, dst_col_offset)
}

fn copy_fused_gate_up_weight<TSrc: TensorElement, TDst: TensorElement>(
    src: &Tensor<TSrc>,
    dst: &mut Tensor<TDst>,
) -> Result<(), MetalError> {
    if src.len() != dst.len() {
        return Err(MetalError::DimensionMismatch {
            expected: dst.len(),
            actual: src.len(),
        });
    }

    let src_dims = src.dims().to_vec();
    let dst_dims = dst.dims().to_vec();

    if src_dims == dst_dims {
        return copy_tensor_into(src, dst);
    }

    if src_dims.len() == 2 && dst_dims.len() == 2 && src_dims[0] == dst_dims[1] && src_dims[1] == dst_dims[0] {
        let src_slice = tensor_data_as_f32(src);
        return pack_weight_transposed_into_fused_slice::<TDst>(src_slice.as_ref(), &src_dims, dst.as_mut_slice(), &dst_dims, 0);
    }

    copy_tensor_into(src, dst)
}

fn copy_bias_into_fused<TSrc: TensorElement, TDst: TensorElement>(
    src: &Tensor<TSrc>,
    dst: &mut Tensor<TDst>,
    dst_offset: usize,
) -> Result<(), MetalError> {
    if dst_offset + src.len() > dst.len() {
        return Err(MetalError::DimensionMismatch {
            expected: dst.len(),
            actual: dst_offset + src.len(),
        });
    }

    let end = dst_offset + src.len();
    let dst_slice = &mut dst.as_mut_slice()[dst_offset..end];
    copy_tensor_data_into_slice::<TSrc, TDst>(src, dst_slice)
}

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

fn load_tensor_into_model<T: TensorElement>(lname: &str, tensor: &Tensor<T>, qwen: &mut Qwen25<T>) -> Result<(), MetalError> {
    // Embedding weight (token embeddings)
    if ((lname.contains("token") && lname.contains("emb")) || lname.contains("tok_emb") || lname.contains("tokembedding"))
        && tensor.len() == qwen.embed_weight.len()
    {
        copy_tensor_into(tensor, &mut qwen.embed_weight)?;
        return Ok(());
    }

    // Output / lm_head weights
    if (lname.contains("lm_head") || lname.contains("lmhead") || (lname.contains("output") && lname.contains("weight")))
        && tensor.len() == qwen.output_weight.len()
    {
        copy_tensor_into(tensor, &mut qwen.output_weight)?;
        return Ok(());
    }

    // Final normalization gamma
    if lname.contains("output_norm")
        || lname.contains("final_norm")
        || lname == "norm.weight" && tensor.len() == qwen.final_norm_gamma.len()
    {
        copy_tensor_into(tensor, &mut qwen.final_norm_gamma)?;
        return Ok(());
    }

    // Weight tying from token embedding to output weight if still unset
    if lname.contains("token_embd") && lname.contains("weight") {
        let output_slice = qwen.output_weight.as_slice();
        let is_output_unset = output_slice.iter().all(|&x| T::to_f32(x) == 0.0);

        if is_output_unset && tensor.len() == qwen.output_weight.len() {
            copy_tensor_into(tensor, &mut qwen.output_weight)?;
        } else if is_output_unset {
            return Err(MetalError::InvalidOperation(format!(
                "MAPPING -> token_embd.weight size mismatch: {} vs {}",
                tensor.len(),
                qwen.output_weight.len()
            )));
        }
        return Ok(());
    }

    if let Some(layer_idx) = parse_layer_index(lname) {
        if layer_idx >= qwen.blocks.len() {
            return Ok(());
        }

        let block = &mut qwen.blocks[layer_idx];

        let d_model_layer = qwen.config.d_model;
        let kv_dim = block.kv_dim;
        let q_offset = 0;
        let k_offset = d_model_layer;
        let v_offset = d_model_layer + kv_dim;

        if (lname.contains("wq")
            || lname.contains("attn.q")
            || lname.contains("attn_q")
            || lname.contains("q_proj.weight")
            || lname.contains("query.weight")
            || lname.contains("q.weight")
            || lname.contains("attention.query.weight"))
            && tensor.len() == d_model_layer * d_model_layer
        {
            copy_weight_transposed_into_fused(tensor, &mut block.attn_qkv_weight, q_offset)?;
            return Ok(());
        }

        if (lname.contains("wk")
            || lname.contains("attn.k")
            || lname.contains("attn_k")
            || lname.contains("k_proj.weight")
            || lname.contains("key.weight")
            || lname.contains("k.weight")
            || lname.contains("attention.key.weight"))
            && tensor.len() == kv_dim * d_model_layer
        {
            copy_weight_transposed_into_fused(tensor, &mut block.attn_qkv_weight, k_offset)?;
            return Ok(());
        }

        if (lname.contains("wv")
            || lname.contains("attn.v")
            || lname.contains("attn_v")
            || lname.contains("v_proj.weight")
            || lname.contains("value.weight")
            || lname.contains("v.weight")
            || lname.contains("attention.value.weight"))
            && tensor.len() == kv_dim * d_model_layer
        {
            copy_weight_transposed_into_fused(tensor, &mut block.attn_qkv_weight, v_offset)?;
            return Ok(());
        }

        if (lname.contains("wo")
            || lname.contains("attn_out")
            || lname.contains("attn.out")
            || lname.contains("out_proj")
            || lname.contains("o.weight")
            || lname.contains("out.weight")
            || lname.contains("attention.output.weight"))
            && tensor.len() == block.attn_out_weight.len()
        {
            copy_tensor_into(tensor, &mut block.attn_out_weight)?;
            return Ok(());
        }

        if (lname.contains("attn.q.bias") || lname.contains("attn_q.bias") || lname.contains("attention.query.bias"))
            && tensor.len() == d_model_layer
        {
            copy_bias_into_fused(tensor, &mut block.attn_qkv_bias, q_offset)?;
            return Ok(());
        }

        if (lname.contains("attn.k.bias") || lname.contains("attn_k.bias") || lname.contains("attention.key.bias"))
            && tensor.len() == kv_dim
        {
            copy_bias_into_fused(tensor, &mut block.attn_qkv_bias, k_offset)?;
            return Ok(());
        }

        if (lname.contains("attn.v.bias") || lname.contains("attn_v.bias") || lname.contains("attention.value.bias"))
            && tensor.len() == kv_dim
        {
            copy_bias_into_fused(tensor, &mut block.attn_qkv_bias, v_offset)?;
            return Ok(());
        }

        if ((lname.contains("fused") && lname.contains("gate") && lname.contains("up"))
            || lname.contains("gate_up_proj.weight")
            || lname.contains("gateup_proj.weight")
            || lname.contains("gate_up.weight")
            || lname.contains("gateup.weight")
            || lname.contains("ffn_gate_up"))
            && tensor.len() == block.ffn_gate_up_weight.len()
        {
            copy_fused_gate_up_weight(tensor, &mut block.ffn_gate_up_weight)?;
            return Ok(());
        }

        if (lname.contains("mlp.gate_proj.weight")
            || lname.contains("ffn.gate")
            || lname.contains("ffn_gate")
            || lname.contains("gate.weight")
            || lname.contains("wg.weight")
            || lname.contains("w1.weight"))
            && tensor.len() == block.ffn_gate.len()
        {
            if tensor.dims() == block.ffn_gate.dims() {
                copy_tensor_into(tensor, &mut block.ffn_gate)?;
            } else if tensor.dims().len() == 2
                && block.ffn_gate.dims().len() == 2
                && tensor.dims()[0] == block.ffn_gate.dims()[1]
                && tensor.dims()[1] == block.ffn_gate.dims()[0]
            {
                let src = tensor_data_as_f32(tensor);
                copy_f32_into_tensor(src.as_ref(), &mut block.ffn_gate)?;
            } else {
                copy_tensor_into(tensor, &mut block.ffn_gate)?;
            }
            return Ok(());
        }

        if (lname.contains("mlp.up_proj.weight")
            || lname.contains("ffn.up")
            || lname.contains("ffn_up")
            || lname.contains("up.weight")
            || lname.contains("w3.weight"))
            && tensor.len() == block.ffn_up.len()
        {
            if tensor.dims() == block.ffn_up.dims() {
                copy_tensor_into(tensor, &mut block.ffn_up)?;
            } else if tensor.dims().len() == 2
                && block.ffn_up.dims().len() == 2
                && tensor.dims()[0] == block.ffn_up.dims()[1]
                && tensor.dims()[1] == block.ffn_up.dims()[0]
            {
                let src = tensor_data_as_f32(tensor);
                copy_f32_into_tensor(src.as_ref(), &mut block.ffn_up)?;
            } else {
                copy_tensor_into(tensor, &mut block.ffn_up)?;
            }
            return Ok(());
        }

        if (lname.contains("mlp.down_proj.weight")
            || lname.contains("ffn.down")
            || lname.contains("ffn_down")
            || lname.contains("down.weight")
            || lname.contains("w2.weight")
            || lname.contains("fc2.weight")
            || lname.contains("wo.weight"))
            && tensor.len() == block.ffn_down.len()
        {
            if tensor.dims() == block.ffn_down.dims() {
                copy_tensor_into(tensor, &mut block.ffn_down)?;
            } else if tensor.dims().len() == 2
                && block.ffn_down.dims().len() == 2
                && tensor.dims()[0] == block.ffn_down.dims()[1]
                && tensor.dims()[1] == block.ffn_down.dims()[0]
            {
                let src = tensor_data_as_f32(tensor);
                copy_f32_into_tensor(src.as_ref(), &mut block.ffn_down)?;
            } else {
                copy_tensor_into(tensor, &mut block.ffn_down)?;
            }
            return Ok(());
        }

        if lname.contains("mlp.gate_proj.bias")
            || lname.contains("ffn.gate.bias")
            || lname.contains("ffn_gate.bias")
            || lname.contains("gate.bias")
            || lname.contains("w1.bias")
            || lname.contains("wg.bias")
        {
            if tensor.len() == block.ffn_gate_bias.len() {
                copy_tensor_into(tensor, &mut block.ffn_gate_bias)?;
            }
            return Ok(());
        }

        if lname.contains("mlp.up_proj.bias")
            || lname.contains("ffn.up.bias")
            || lname.contains("ffn_up.bias")
            || lname.contains("up.bias")
            || lname.contains("w3.bias")
        {
            if tensor.len() == block.ffn_up_bias.len() {
                copy_tensor_into(tensor, &mut block.ffn_up_bias)?;
            }
            return Ok(());
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
                copy_tensor_into(tensor, &mut block.ffn_down_bias)?;
            }
            return Ok(());
        }

        if ((lname.contains("attn_norm") && lname.contains("gamma"))
            || lname.contains("attn.g")
            || lname.contains("attn_norm.weight")
            || lname.contains("attention.layernorm.weight")
            || lname.contains("ln_attn.weight"))
            && tensor.len() == block.attn_norm_gamma.len()
        {
            copy_tensor_into(tensor, &mut block.attn_norm_gamma)?;
            return Ok(());
        }

        if ((lname.contains("proj_norm") && lname.contains("gamma"))
            || lname.contains("proj.g")
            || lname.contains("ffn_norm.weight")
            || lname.contains("ffn.layernorm.weight")
            || lname.contains("ln_ffn.weight"))
            && tensor.len() == block.ffn_norm_gamma.len()
        {
            copy_tensor_into(tensor, &mut block.ffn_norm_gamma)?;
            return Ok(());
        }
    }

    // If we reach here, tensor was not handled specifically.
    Ok(())
}

/// Implement the LoadableModel trait so Qwen25 can be created from a GGUFModel
impl<T: TensorElement> LoadableModel<T> for Qwen25<T> {
    fn load_from_gguf(gguf_model: &GGUFModel, ctx: &mut Context<T>) -> Result<Self, MetalError> {
        let cfg = extract_config_from_ggufmodel(gguf_model);

        // Instantiate Qwen25 with default-initialized weights
        let mut qwen = Qwen25::new(cfg, ctx)?;

        for (name, descriptor) in &gguf_model.tensors {
            let lname = name.to_lowercase();
            let materialized = gguf_model.materialize_tensor::<T>(name, &*ctx).map_err(|err| {
                MetalError::InvalidOperation(format!(
                    "Failed to materialize tensor '{}' (dtype={:?}): {err}",
                    name,
                    descriptor.data_type()
                ))
            })?;

            load_tensor_into_model(&lname, &materialized, &mut qwen)
                .map_err(|err| MetalError::InvalidOperation(format!("Failed to load tensor '{}' into model: {err}", name)))?;
        }

        Ok(qwen)
    }
}

fn extract_config_from_ggufmodel(gguf_model: &GGUFModel) -> Qwen25Config {
    // Build config heuristically from metadata (fallbacks kept consistent with qwen25::new)
    // Get our metadata sizes with variations and defaults
    let d_model = gguf_model.get_metadata_u32_or(
        &[
            "qwen2.d_model",
            "model.d_model",
            "qwen2.embedding_length",
            "model.hidden_size",
            "llama.embedding_length",
        ],
        896,
    ) as usize;
    let ff_dim = gguf_model.get_metadata_u32_or(
        &[
            "qwen2.ff_dim",
            "model.ff_dim",
            "qwen2.feed_forward_length",
            "model.intermediate_size",
        ],
        4864,
    ) as usize;
    let n_heads = gguf_model.get_metadata_u32_or(
        &[
            "qwen2.n_heads",
            "model.n_heads",
            "qwen2.attention.head_count",
            "model.num_attention_heads",
        ],
        14,
    ) as usize;
    let n_kv_heads = gguf_model.get_metadata_u32_or(
        &[
            "qwen2.n_kv_heads",
            "model.n_kv_heads",
            "qwen2.attention.head_count_kv",
            "model.num_key_value_heads",
        ],
        2,
    ) as usize;
    let n_layers = gguf_model.get_metadata_u32_or(
        &["qwen2.n_layers", "model.n_layers", "qwen2.block_count", "model.num_hidden_layers"],
        24,
    ) as usize;
    let seq_len = gguf_model.get_metadata_u32_or(&["qwen2.context_length", "model.context_length"], 32768) as usize;
    let rope_freq_base = gguf_model.get_metadata_f32_or(&["qwen2.rope_theta", "qwen2.rope.freq_base", "model.rope.freq_base"], 1_000_000.0);
    let rms_eps = gguf_model.get_metadata_f32_or(
        &[
            "qwen2.rms_norm_eps",
            "qwen2.rope.rms_eps",
            "qwen2.attention.layer_norm_rms_epsilon",
            "model.rms_norm_eps",
        ],
        1e-6,
    );

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

    Qwen25Config {
        n_layers,
        d_model,
        ff_dim,
        n_heads,
        n_kv_heads,
        seq_len,
        vocab_size,
        rope_freq_base,
        rms_eps,
    }
}

#[cfg(test)]
mod tests {
    use super::pack_weight_transposed_into_fused_slice;
    use crate::metallic::{F32Element, MetalError};

    #[test]
    fn pack_weight_transposes_row_major_layouts() {
        let src_dims = vec![3, 2];
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let fused_dims = vec![2, 3];
        let mut fused = vec![0.0; fused_dims[0] * fused_dims[1]];

        pack_weight_transposed_into_fused_slice::<F32Element>(&src, &src_dims, &mut fused, &fused_dims, 0).unwrap();

        assert_eq!(fused, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn pack_weight_transposes_column_major_exports() {
        let src_dims = vec![2, 3];
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let fused_dims = vec![2, 3];
        let mut fused = vec![0.0; fused_dims[0] * fused_dims[1]];

        pack_weight_transposed_into_fused_slice::<F32Element>(&src, &src_dims, &mut fused, &fused_dims, 0).unwrap();

        assert_eq!(fused, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn pack_weight_errors_when_rows_mismatch_fused_input() {
        let src_dims = vec![4, 3];
        let src = vec![0.0; 12];
        let fused_dims = vec![2, 6];
        let mut fused = vec![0.0; fused_dims[0] * fused_dims[1]];

        let err = pack_weight_transposed_into_fused_slice::<F32Element>(&src, &src_dims, &mut fused, &fused_dims, 0)
            .expect_err("expected invalid shape error");

        match err {
            MetalError::InvalidShape(_) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }
}
