//! Parity test: DSL-based inference vs Context<T>-based inference.
//!
//! This test runs both forward pass implementations on the same model
//! and compares intermediate tensors to identify divergence points.

use half::f16;
use metallic_context::{
    Context, F16Element, MetalError, Tensor, TensorElement, context::{QkvWeights, RepeatKvWorkspaceKind}, kernels::{
        backend_registry::KernelBackendKind, elemwise_add::{BroadcastElemwiseAddInplaceOp, BroadcastElemwiseAddOp}, elemwise_mul::ElemwiseMulOp, kv_rearrange::KvRearrangeOp, repeat_kv_heads::RepeatKvHeadsOp, rmsnorm::RMSNormOp, rope::RoPEOp, silu::SiluOp, swiglu::SwiGLUOp
    }, models::Qwen25, tensor::{QuantizedTensor, TensorType}
};
use metallic_foundry::{Foundry, model::ModelBuilder, spec::TensorBindings};
use serial_test::serial;

const MODEL_SPEC_PATH: &str = "../../models/qwen25.json";
const GGUF_PATH: &str = "../../models/qwen2.5-coder-0.5b-instruct-fp16.gguf";

type LegacyStepResult = Result<(Vec<f16>, Vec<f16>, Option<Vec<(usize, Vec<f16>)>>), MetalError>;

/// Compare two f16 slices and return (max_diff, avg_diff, first_mismatch_idx)
fn compare_f16_slices(a: &[f16], b: &[f16], tolerance: f32) -> (f32, f32, Option<usize>) {
    if a.len() != b.len() {
        eprintln!("Length mismatch: {} vs {}", a.len(), b.len());
        return (f32::INFINITY, f32::INFINITY, Some(0));
    }

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut first_mismatch = None;

    for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va.to_f32() - vb.to_f32()).abs();
        sum_diff += diff;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tolerance && first_mismatch.is_none() {
            first_mismatch = Some(i);
        }
    }

    let avg_diff = if a.is_empty() { 0.0 } else { sum_diff / a.len() as f32 };
    (max_diff, avg_diff, first_mismatch)
}

/// Read f16 data from a TensorArg buffer
fn read_f16_buffer(arg: &metallic_foundry::types::TensorArg) -> Vec<f16> {
    use metallic_foundry::types::KernelArg;
    let buffer = arg.buffer();
    let offset = arg.offset(); // offset is in bytes
    let len = arg.dims().iter().product::<usize>();
    unsafe {
        use objc2_metal::MTLBuffer;
        let ptr = (buffer.contents().as_ptr() as *const u8).add(offset) as *const f16;
        std::slice::from_raw_parts(ptr, len).to_vec()
    }
}

fn tensor_to_f16_vec<T: TensorElement>(tensor: &Tensor<T>) -> Vec<f16> {
    tensor.as_slice().iter().map(|v| f16::from_f32(T::to_f32(*v))).collect()
}

fn tensor_to_f16_vec_strided<T: TensorElement>(tensor: &Tensor<T>) -> Result<Vec<f16>, MetalError> {
    let data = tensor.try_to_vec().unwrap();
    Ok(data.into_iter().map(|v| f16::from_f32(T::to_f32(v))).collect())
}

fn argmax_f16(values: &[f16]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, value) in values.iter().enumerate() {
        let v = value.to_f32();
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    best_idx as u32
}

fn f16_canonical_enabled_from_env() -> bool {
    std::env::var("METALLIC_F16_CANONICAL_GEMM")
        .ok()
        .map(|s| s.trim() != "0")
        .unwrap_or(true)
}

fn legacy_forward_blocks_only<T: TensorElement>(
    model: &Qwen25<T>,
    input: &Tensor<T>,
    ctx: &mut Context<T>,
) -> Result<Tensor<T>, MetalError> {
    let dims = input.dims();
    if dims.len() != 3 {
        return Err(MetalError::InvalidShape(format!(
            "pre-norm forward expects input with 3 dims [batch, seq, d_model], got {:?}",
            dims
        )));
    }
    let batch = dims[0];
    let seq = dims[1];
    let d_model = dims[2];
    if d_model != model.config.d_model {
        return Err(MetalError::InvalidShape(format!(
            "Input d_model {} does not match config.d_model {}",
            d_model, model.config.d_model
        )));
    }
    if seq > model.config.seq_len {
        return Err(MetalError::InvalidShape(format!(
            "Input seq {} exceeds configured seq_len {}",
            seq, model.config.seq_len
        )));
    }

    let mut x = input.clone();
    let f16_canonical = f16_canonical_enabled_from_env();
    let n_heads = model.config.n_heads;
    let n_kv_heads = model.config.n_kv_heads;
    let head_dim = d_model / n_heads;

    for (layer_idx, block) in model.blocks.iter().enumerate() {
        let resid_attn = x.clone();

        // RMSNorm before Attention
        let x_normed_attn = ctx
            .call::<RMSNormOp>((x, block.attn_norm_gamma.clone(), d_model as u32), None)
            .unwrap();

        // QKV GEMMs
        let m = batch * seq;
        let kv_dim = block.kv_dim;
        let kv_head_dim = kv_dim / n_kv_heads;
        let x_flat = x_normed_attn.reshape(vec![m, d_model]).unwrap();
        let (q_mat, k_mat, v_mat) = if f16_canonical {
            if let (Some(wq), Some(wk), Some(wv)) = (&block.attn_q_weight_canon, &block.attn_k_weight_canon, &block.attn_v_weight_canon) {
                let q_bias = block.attn_qkv_bias.slice(0..d_model).unwrap();
                let k_bias = block.attn_qkv_bias.slice(d_model..(d_model + kv_dim)).unwrap();
                let v_bias = block.attn_qkv_bias.slice((d_model + kv_dim)..(d_model + 2 * kv_dim)).unwrap();

                ctx.qkv(
                    &x_flat,
                    QkvWeights::DenseCanonical {
                        wq,
                        wk,
                        wv,
                        q_bias: &q_bias,
                        k_bias: &k_bias,
                        v_bias: &v_bias,
                    },
                )
                .unwrap()
            } else if let Some(w_qkv) = &block.attn_qkv_weight {
                ctx.qkv(
                    &x_flat,
                    QkvWeights::Dense {
                        fused_weight: w_qkv,
                        fused_bias: &block.attn_qkv_bias,
                        d_model,
                        kv_dim,
                    },
                )
                .unwrap()
            } else {
                return Err(MetalError::InvalidOperation("Missing attn_qkv_weight for F32 fallback".to_string()));
            }
        } else if let Some(w_qkv) = &block.attn_qkv_weight {
            ctx.qkv(
                &x_flat,
                QkvWeights::Dense {
                    fused_weight: w_qkv,
                    fused_bias: &block.attn_qkv_bias,
                    d_model,
                    kv_dim,
                },
            )
            .unwrap()
        } else {
            return Err(MetalError::InvalidOperation(
                "Missing attn_qkv_weight for F32 fallback in qkv".to_string(),
            ));
        };

        // KV Head Rearrangement
        let q_heads = ctx
            .call::<KvRearrangeOp>(
                (
                    q_mat.clone(),
                    d_model as u32,
                    head_dim as u32,
                    n_heads as u32,
                    n_heads as u32,
                    head_dim as u32,
                    seq as u32,
                ),
                None,
            )
            .unwrap();
        let k_heads = ctx
            .call::<KvRearrangeOp>(
                (
                    k_mat.clone(),
                    kv_dim as u32,
                    kv_head_dim as u32,
                    n_kv_heads as u32,
                    n_kv_heads as u32,
                    kv_head_dim as u32,
                    seq as u32,
                ),
                None,
            )
            .unwrap();
        let v_heads = ctx
            .call::<KvRearrangeOp>(
                (
                    v_mat.clone(),
                    kv_dim as u32,
                    kv_head_dim as u32,
                    n_kv_heads as u32,
                    n_kv_heads as u32,
                    kv_head_dim as u32,
                    seq as u32,
                ),
                None,
            )
            .unwrap();

        // RoPE for Q and K
        let q_heads_after_rope = ctx
            .call::<RoPEOp>(
                (
                    q_heads,
                    model.rope_cos_cache.clone(),
                    model.rope_sin_cache.clone(),
                    head_dim as u32,
                    seq as u32,
                    0,
                ),
                None,
            )
            .unwrap();
        let k_heads_after_rope = ctx
            .call::<RoPEOp>(
                (
                    k_heads,
                    model.rope_cos_cache.clone(),
                    model.rope_sin_cache.clone(),
                    kv_head_dim as u32,
                    seq as u32,
                    0,
                ),
                None,
            )
            .unwrap();

        // Repeat K and V to match Q head count for SDPA (GQA)
        let group_size = n_heads / n_kv_heads;
        let prefer_shared = ctx.backend_registry().select_sdpa(KernelBackendKind::Legacy).backend == KernelBackendKind::Graph;
        let k_repeated = ctx
            .call::<RepeatKvHeadsOp>(
                (
                    k_heads_after_rope,
                    group_size as u32,
                    batch as u32,
                    n_kv_heads as u32,
                    n_heads as u32,
                    seq as u32,
                    kv_head_dim as u32,
                    seq as u32,
                    layer_idx as u32,
                    RepeatKvWorkspaceKind::Key,
                    prefer_shared,
                ),
                None,
            )
            .unwrap();
        let v_repeated = ctx
            .call::<RepeatKvHeadsOp>(
                (
                    v_heads,
                    group_size as u32,
                    batch as u32,
                    n_kv_heads as u32,
                    n_heads as u32,
                    seq as u32,
                    kv_head_dim as u32,
                    seq as u32,
                    layer_idx as u32,
                    RepeatKvWorkspaceKind::Value,
                    prefer_shared,
                ),
                None,
            )
            .unwrap();

        // SDPA (causal mask enabled)
        let attn_out_heads = ctx
            .scaled_dot_product_attention(&q_heads_after_rope, &k_repeated, &v_repeated, true)
            .unwrap();

        // Attention output projection
        let attn_out_reshaped = attn_out_heads
            .reshape(vec![batch, n_heads, seq, head_dim])
            .unwrap()
            .permute(&[0, 2, 1, 3], ctx)
            .unwrap()
            .reshape(vec![batch, seq, d_model])
            .unwrap();

        let attn_out = if let Some(q8) = &block.attn_out_weight_q8 {
            let a = attn_out_reshaped.reshape(vec![m, d_model]).unwrap();
            ctx.matmul(&a, &TensorType::Quant(QuantizedTensor::Q8_0(q8)), false, true, None, None, None)
                .unwrap()
                .reshape(vec![batch, seq, d_model])
                .unwrap()
        } else if f16_canonical {
            if let Some(canon) = &block.attn_out_weight_canon {
                ctx.matmul(
                    &attn_out_reshaped.reshape(vec![m, d_model]).unwrap(),
                    &TensorType::DenseCanonical(canon),
                    false,
                    false,
                    None,
                    None,
                    None,
                )
                .unwrap()
                .reshape(vec![batch, seq, d_model])
                .unwrap()
            } else if let Some(w) = &block.attn_out_weight {
                ctx.matmul(
                    &attn_out_reshaped.reshape(vec![m, d_model]).unwrap(),
                    &TensorType::Dense(w),
                    false,
                    true,
                    None,
                    None,
                    None,
                )
                .unwrap()
                .reshape(vec![batch, seq, d_model])
                .unwrap()
            } else {
                return Err(MetalError::InvalidOperation("Missing attn_out_weight for F32 fallback".to_string()));
            }
        } else if let Some(w) = &block.attn_out_weight {
            ctx.matmul(
                &attn_out_reshaped.reshape(vec![m, d_model]).unwrap(),
                &TensorType::Dense(w),
                false,
                true,
                None,
                None,
                None,
            )
            .unwrap()
            .reshape(vec![batch, seq, d_model])
            .unwrap()
        } else {
            return Err(MetalError::InvalidOperation("Missing attn_out_weight for F32 fallback".to_string()));
        };

        x = resid_attn.add_elem(&attn_out, ctx).unwrap();

        // MLP block
        let resid_mlp = x.clone();
        let x_normed_mlp = ctx
            .call::<RMSNormOp>((x, block.ffn_norm_gamma.clone(), d_model as u32), None)
            .unwrap();
        let x_normed_mlp_flat = x_normed_mlp.reshape(vec![m, d_model]).unwrap();

        let ffn_output_flat = if f16_canonical {
            if let (Some(g), Some(u), Some(d)) = (&block.ffn_gate_canon, &block.ffn_up_canon, &block.ffn_down_canon) {
                let hidden = ctx
                    .swiglu(
                        &x_normed_mlp_flat,
                        &TensorType::DenseCanonical(g),
                        &TensorType::DenseCanonical(u),
                        Some(&block.ffn_gate_bias),
                        Some(&block.ffn_up_bias),
                    )
                    .unwrap();
                ctx.matmul(&hidden, &TensorType::DenseCanonical(d), false, false, None, None, None)
                    .unwrap()
            } else {
                ctx.call::<SwiGLUOp>(
                    (
                        &x_normed_mlp_flat,
                        block.ffn_gate.as_ref().unwrap(),
                        &block.ffn_gate_bias,
                        block.ffn_up.as_ref().unwrap(),
                        &block.ffn_up_bias,
                        block.ffn_down.as_ref().unwrap(),
                        &block.ffn_down_bias,
                        block.ffn_gate_up_weight.as_ref(),
                    ),
                    None,
                )
                .unwrap()
            }
        } else {
            ctx.call::<SwiGLUOp>(
                (
                    &x_normed_mlp_flat,
                    block.ffn_gate.as_ref().unwrap(),
                    &block.ffn_gate_bias,
                    block.ffn_up.as_ref().unwrap(),
                    &block.ffn_up_bias,
                    block.ffn_down.as_ref().unwrap(),
                    &block.ffn_down_bias,
                    block.ffn_gate_up_weight.as_ref(),
                ),
                None,
            )
            .unwrap()
        };
        let ffn_output = ffn_output_flat.reshape(vec![batch, seq, d_model]).unwrap();
        x = resid_mlp.add_elem(&ffn_output, ctx).unwrap();
    }

    Ok(x)
}

fn legacy_sdpa_layer0<T: TensorElement>(model: &Qwen25<T>, input: &Tensor<T>, ctx: &mut Context<T>) -> Result<Tensor<T>, MetalError> {
    let dims = input.dims();
    if dims.len() != 3 {
        return Err(MetalError::InvalidShape(format!(
            "SDPA probe expects input with 3 dims [batch, seq, d_model], got {:?}",
            dims
        )));
    }
    let batch = dims[0];
    let seq = dims[1];
    let d_model = dims[2];
    let block = model
        .blocks
        .first()
        .ok_or_else(|| MetalError::InvalidShape("Model has no transformer blocks for SDPA probe".to_string()))
        .unwrap();

    let n_heads = model.config.n_heads;
    let n_kv_heads = model.config.n_kv_heads;
    let head_dim = d_model / n_heads;
    let kv_dim = block.kv_dim;
    let kv_head_dim = kv_dim / n_kv_heads;
    let f16_canonical = f16_canonical_enabled_from_env();

    let x_normed_attn = ctx
        .call::<RMSNormOp>((input.clone(), block.attn_norm_gamma.clone(), d_model as u32), None)
        .unwrap();
    let m = batch * seq;
    let x_flat = x_normed_attn.reshape(vec![m, d_model]).unwrap();

    let (q_mat, k_mat, v_mat) = if f16_canonical {
        if let (Some(wq), Some(wk), Some(wv)) = (&block.attn_q_weight_canon, &block.attn_k_weight_canon, &block.attn_v_weight_canon) {
            let q_bias = block.attn_qkv_bias.slice(0..d_model).unwrap();
            let k_bias = block.attn_qkv_bias.slice(d_model..(d_model + kv_dim)).unwrap();
            let v_bias = block.attn_qkv_bias.slice((d_model + kv_dim)..(d_model + 2 * kv_dim)).unwrap();

            ctx.qkv(
                &x_flat,
                QkvWeights::DenseCanonical {
                    wq,
                    wk,
                    wv,
                    q_bias: &q_bias,
                    k_bias: &k_bias,
                    v_bias: &v_bias,
                },
            )
            .unwrap()
        } else if let Some(w_qkv) = &block.attn_qkv_weight {
            ctx.qkv(
                &x_flat,
                QkvWeights::Dense {
                    fused_weight: w_qkv,
                    fused_bias: &block.attn_qkv_bias,
                    d_model,
                    kv_dim,
                },
            )
            .unwrap()
        } else {
            return Err(MetalError::InvalidOperation("Missing attn_qkv_weight for F32 fallback".to_string()));
        }
    } else if let Some(w_qkv) = &block.attn_qkv_weight {
        ctx.qkv(
            &x_flat,
            QkvWeights::Dense {
                fused_weight: w_qkv,
                fused_bias: &block.attn_qkv_bias,
                d_model,
                kv_dim,
            },
        )
        .unwrap()
    } else {
        return Err(MetalError::InvalidOperation(
            "Missing attn_qkv_weight for F32 fallback in qkv".to_string(),
        ));
    };

    let q_heads = ctx
        .call::<KvRearrangeOp>(
            (
                q_mat.clone(),
                d_model as u32,
                head_dim as u32,
                n_heads as u32,
                n_heads as u32,
                head_dim as u32,
                seq as u32,
            ),
            None,
        )
        .unwrap();
    let k_heads = ctx
        .call::<KvRearrangeOp>(
            (
                k_mat.clone(),
                kv_dim as u32,
                kv_head_dim as u32,
                n_kv_heads as u32,
                n_kv_heads as u32,
                kv_head_dim as u32,
                seq as u32,
            ),
            None,
        )
        .unwrap();
    let v_heads = ctx
        .call::<KvRearrangeOp>(
            (
                v_mat.clone(),
                kv_dim as u32,
                kv_head_dim as u32,
                n_kv_heads as u32,
                n_kv_heads as u32,
                kv_head_dim as u32,
                seq as u32,
            ),
            None,
        )
        .unwrap();

    let q_heads_after_rope = ctx
        .call::<RoPEOp>(
            (
                q_heads,
                model.rope_cos_cache.clone(),
                model.rope_sin_cache.clone(),
                head_dim as u32,
                seq as u32,
                0,
            ),
            None,
        )
        .unwrap();
    let k_heads_after_rope = ctx
        .call::<RoPEOp>(
            (
                k_heads,
                model.rope_cos_cache.clone(),
                model.rope_sin_cache.clone(),
                kv_head_dim as u32,
                seq as u32,
                0,
            ),
            None,
        )
        .unwrap();

    let group_size = n_heads / n_kv_heads;
    let prefer_shared = ctx.backend_registry().select_sdpa(KernelBackendKind::Legacy).backend == KernelBackendKind::Graph;
    let k_repeated = ctx
        .call::<RepeatKvHeadsOp>(
            (
                k_heads_after_rope,
                group_size as u32,
                batch as u32,
                n_kv_heads as u32,
                n_heads as u32,
                seq as u32,
                kv_head_dim as u32,
                seq as u32,
                0,
                RepeatKvWorkspaceKind::Key,
                prefer_shared,
            ),
            None,
        )
        .unwrap();
    let v_repeated = ctx
        .call::<RepeatKvHeadsOp>(
            (
                v_heads,
                group_size as u32,
                batch as u32,
                n_kv_heads as u32,
                n_heads as u32,
                seq as u32,
                kv_head_dim as u32,
                seq as u32,
                0,
                RepeatKvWorkspaceKind::Value,
                prefer_shared,
            ),
            None,
        )
        .unwrap();

    ctx.scaled_dot_product_attention(&q_heads_after_rope, &k_repeated, &v_repeated, true)
}

struct LegacyLayer0Outputs {
    attn_out: Vec<f16>,
    q_rot: Vec<f16>,      // Q after RoPE (before SDPA)
    k_expanded: Vec<f16>, // K after repeat (before SDPA)
    v_expanded: Vec<f16>, // V after repeat (before SDPA)
    q_proj: Vec<f16>,     // Q projection output
    k_proj: Vec<f16>,     // K projection output
    v_proj: Vec<f16>,     // V projection output
    ffn_norm_out: Vec<f16>,
    gate: Vec<f16>,
    swiglu_out: Vec<f16>,
    proj_out: Vec<f16>,
    residual_1: Vec<f16>,
    ffn_out: Vec<f16>,
    hidden: Vec<f16>,
}

fn legacy_layer0_outputs<T: TensorElement>(
    model: &Qwen25<T>,
    input: &Tensor<T>,
    ctx: &mut Context<T>,
) -> Result<LegacyLayer0Outputs, MetalError> {
    let transpose_for = |dims: &[usize], expected_k: usize, name: &str| -> Result<bool, MetalError> {
        if dims.len() != 2 {
            return Err(MetalError::InvalidShape(format!(
                "{} weight expected 2D dims, got {:?}",
                name, dims
            )));
        }
        if dims[0] == expected_k {
            Ok(false)
        } else if dims[1] == expected_k {
            Ok(true)
        } else {
            Err(MetalError::DimensionMismatch {
                expected: expected_k,
                actual: dims[0],
            })
        }
    };
    let dims = input.dims();
    if dims.len() != 3 {
        return Err(MetalError::InvalidShape(format!(
            "layer0 probe expects input with 3 dims [batch, seq, d_model], got {:?}",
            dims
        )));
    }
    let batch = dims[0];
    let seq = dims[1];
    let d_model = dims[2];
    let block = model
        .blocks
        .first()
        .ok_or_else(|| MetalError::InvalidShape("Model has no transformer blocks".to_string()))
        .unwrap();

    let n_heads = model.config.n_heads;
    let n_kv_heads = model.config.n_kv_heads;
    let head_dim = d_model / n_heads;
    let kv_dim = block.kv_dim;
    let kv_head_dim = kv_dim / n_kv_heads;
    let f16_canonical = f16_canonical_enabled_from_env();

    let resid_attn = input.clone();
    let x_normed_attn = ctx
        .call::<RMSNormOp>((input.clone(), block.attn_norm_gamma.clone(), d_model as u32), None)
        .unwrap();
    let m = batch * seq;
    let x_flat = x_normed_attn.reshape(vec![m, d_model]).unwrap();

    let (q_mat, k_mat, v_mat) = if f16_canonical {
        if let (Some(wq), Some(wk), Some(wv)) = (&block.attn_q_weight_canon, &block.attn_k_weight_canon, &block.attn_v_weight_canon) {
            let q_bias = block.attn_qkv_bias.slice(0..d_model).unwrap();
            let k_bias = block.attn_qkv_bias.slice(d_model..(d_model + kv_dim)).unwrap();
            let v_bias = block.attn_qkv_bias.slice((d_model + kv_dim)..(d_model + 2 * kv_dim)).unwrap();
            ctx.qkv(
                &x_flat,
                QkvWeights::DenseCanonical {
                    wq,
                    wk,
                    wv,
                    q_bias: &q_bias,
                    k_bias: &k_bias,
                    v_bias: &v_bias,
                },
            )
            .unwrap()
        } else if let Some(w_qkv) = &block.attn_qkv_weight {
            ctx.qkv(
                &x_flat,
                QkvWeights::Dense {
                    fused_weight: w_qkv,
                    fused_bias: &block.attn_qkv_bias,
                    d_model,
                    kv_dim,
                },
            )
            .unwrap()
        } else {
            return Err(MetalError::InvalidOperation("Missing attn_qkv_weight for F32 fallback".to_string()));
        }
    } else if let Some(w_qkv) = &block.attn_qkv_weight {
        ctx.qkv(
            &x_flat,
            QkvWeights::Dense {
                fused_weight: w_qkv,
                fused_bias: &block.attn_qkv_bias,
                d_model,
                kv_dim,
            },
        )
        .unwrap()
    } else {
        return Err(MetalError::InvalidOperation(
            "Missing attn_qkv_weight for F32 fallback in qkv".to_string(),
        ));
    };

    let q_heads = ctx
        .call::<KvRearrangeOp>(
            (
                q_mat.clone(),
                d_model as u32,
                head_dim as u32,
                n_heads as u32,
                n_heads as u32,
                head_dim as u32,
                seq as u32,
            ),
            None,
        )
        .unwrap();
    let k_heads = ctx
        .call::<KvRearrangeOp>(
            (
                k_mat.clone(),
                kv_dim as u32,
                kv_head_dim as u32,
                n_kv_heads as u32,
                n_kv_heads as u32,
                kv_head_dim as u32,
                seq as u32,
            ),
            None,
        )
        .unwrap();
    let v_heads = ctx
        .call::<KvRearrangeOp>(
            (
                v_mat.clone(),
                kv_dim as u32,
                kv_head_dim as u32,
                n_kv_heads as u32,
                n_kv_heads as u32,
                kv_head_dim as u32,
                seq as u32,
            ),
            None,
        )
        .unwrap();

    let q_heads_after_rope = ctx
        .call::<RoPEOp>(
            (
                q_heads,
                model.rope_cos_cache.clone(),
                model.rope_sin_cache.clone(),
                head_dim as u32,
                seq as u32,
                0,
            ),
            None,
        )
        .unwrap();
    let k_heads_after_rope = ctx
        .call::<RoPEOp>(
            (
                k_heads,
                model.rope_cos_cache.clone(),
                model.rope_sin_cache.clone(),
                kv_head_dim as u32,
                seq as u32,
                0,
            ),
            None,
        )
        .unwrap();

    let group_size = n_heads / n_kv_heads;
    let prefer_shared = ctx.backend_registry().select_sdpa(KernelBackendKind::Legacy).backend == KernelBackendKind::Graph;
    let k_repeated = ctx
        .call::<RepeatKvHeadsOp>(
            (
                k_heads_after_rope,
                group_size as u32,
                batch as u32,
                n_kv_heads as u32,
                n_heads as u32,
                seq as u32,
                kv_head_dim as u32,
                seq as u32,
                0,
                RepeatKvWorkspaceKind::Key,
                prefer_shared,
            ),
            None,
        )
        .unwrap();
    let v_repeated = ctx
        .call::<RepeatKvHeadsOp>(
            (
                v_heads,
                group_size as u32,
                batch as u32,
                n_kv_heads as u32,
                n_heads as u32,
                seq as u32,
                kv_head_dim as u32,
                seq as u32,
                0,
                RepeatKvWorkspaceKind::Value,
                prefer_shared,
            ),
            None,
        )
        .unwrap();

    let attn_out_heads = ctx
        .scaled_dot_product_attention(&q_heads_after_rope, &k_repeated, &v_repeated, true)
        .unwrap();
    let attn_out_reshaped = attn_out_heads
        .reshape(vec![batch, n_heads, seq, head_dim])
        .unwrap()
        .permute(&[0, 2, 1, 3], ctx)
        .unwrap()
        .reshape(vec![batch, seq, d_model])
        .unwrap();

    let proj_out = if let Some(q8) = &block.attn_out_weight_q8 {
        let a = attn_out_reshaped.reshape(vec![m, d_model]).unwrap();
        ctx.matmul(&a, &TensorType::Quant(QuantizedTensor::Q8_0(q8)), false, true, None, None, None)
            .unwrap()
            .reshape(vec![batch, seq, d_model])
            .unwrap()
    } else if f16_canonical {
        if let Some(canon) = &block.attn_out_weight_canon {
            ctx.matmul(
                &attn_out_reshaped.reshape(vec![m, d_model]).unwrap(),
                &TensorType::DenseCanonical(canon),
                false,
                false,
                None,
                None,
                None,
            )
            .unwrap()
            .reshape(vec![batch, seq, d_model])
            .unwrap()
        } else if let Some(w) = &block.attn_out_weight {
            ctx.matmul(
                &attn_out_reshaped.reshape(vec![m, d_model]).unwrap(),
                &TensorType::Dense(w),
                false,
                true,
                None,
                None,
                None,
            )
            .unwrap()
            .reshape(vec![batch, seq, d_model])
            .unwrap()
        } else {
            return Err(MetalError::InvalidOperation("Missing attn_out_weight for F32 fallback".to_string()));
        }
    } else if let Some(w) = &block.attn_out_weight {
        ctx.matmul(
            &attn_out_reshaped.reshape(vec![m, d_model]).unwrap(),
            &TensorType::Dense(w),
            false,
            true,
            None,
            None,
            None,
        )
        .unwrap()
        .reshape(vec![batch, seq, d_model])
        .unwrap()
    } else {
        return Err(MetalError::InvalidOperation("Missing attn_out_weight for F32 fallback".to_string()));
    };

    let residual_1 = resid_attn.add_elem(&proj_out, ctx).unwrap();
    let resid_mlp = residual_1.clone();
    let x_normed_mlp = ctx
        .call::<RMSNormOp>((residual_1, block.ffn_norm_gamma.clone(), d_model as u32), None)
        .unwrap();
    let x_normed_mlp_flat = x_normed_mlp.reshape(vec![m, d_model]).unwrap();

    let (gate_proj, _up_proj, swiglu_out, ffn_output_flat) = if f16_canonical {
        if let (Some(g), Some(u), Some(d)) = (&block.ffn_gate_canon, &block.ffn_up_canon, &block.ffn_down_canon) {
            let gate = ctx
                .matmul(&x_normed_mlp_flat, &TensorType::DenseCanonical(g), false, false, None, None, None)
                .unwrap();
            let up = ctx
                .matmul(&x_normed_mlp_flat, &TensorType::DenseCanonical(u), false, false, None, None, None)
                .unwrap();
            let gate_bias = ctx
                .call::<BroadcastElemwiseAddOp>((gate.clone(), block.ffn_gate_bias.clone()), None)
                .unwrap();
            let up_bias = ctx
                .call::<BroadcastElemwiseAddOp>((up.clone(), block.ffn_up_bias.clone()), None)
                .unwrap();
            let silu_out = ctx.call::<SiluOp>(gate_bias, None).unwrap();
            let mul_out = ctx.call::<ElemwiseMulOp>((silu_out, up_bias), None).unwrap();
            let down = ctx
                .matmul(&mul_out, &TensorType::DenseCanonical(d), false, false, None, None, None)
                .unwrap();
            (gate, up, mul_out, down)
        } else {
            let gate = ctx
                .matmul(
                    &x_normed_mlp_flat,
                    &TensorType::Dense(block.ffn_gate.as_ref().unwrap()),
                    false,
                    transpose_for(block.ffn_gate.as_ref().unwrap().dims(), d_model, "ffn_gate").unwrap(),
                    None,
                    None,
                    None,
                )
                .unwrap();
            let up = ctx
                .matmul(
                    &x_normed_mlp_flat,
                    &TensorType::Dense(block.ffn_up.as_ref().unwrap()),
                    false,
                    transpose_for(block.ffn_up.as_ref().unwrap().dims(), d_model, "ffn_up").unwrap(),
                    None,
                    None,
                    None,
                )
                .unwrap();
            let gate_bias = ctx
                .call::<BroadcastElemwiseAddOp>((gate.clone(), block.ffn_gate_bias.clone()), None)
                .unwrap();
            let up_bias = ctx
                .call::<BroadcastElemwiseAddOp>((up.clone(), block.ffn_up_bias.clone()), None)
                .unwrap();
            let silu_out = ctx.call::<SiluOp>(gate_bias, None).unwrap();
            let mul_out = ctx.call::<ElemwiseMulOp>((silu_out, up_bias), None).unwrap();
            let down = ctx
                .matmul(
                    &mul_out,
                    &TensorType::Dense(block.ffn_down.as_ref().unwrap()),
                    false,
                    transpose_for(block.ffn_down.as_ref().unwrap().dims(), block.ffn_gate_bias.len(), "ffn_down").unwrap(),
                    None,
                    None,
                    None,
                )
                .unwrap();
            (gate, up, mul_out, down)
        }
    } else {
        let gate = ctx
            .matmul(
                &x_normed_mlp_flat,
                &TensorType::Dense(block.ffn_gate.as_ref().unwrap()),
                false,
                transpose_for(block.ffn_gate.as_ref().unwrap().dims(), d_model, "ffn_gate").unwrap(),
                None,
                None,
                None,
            )
            .unwrap();
        let up = ctx
            .matmul(
                &x_normed_mlp_flat,
                &TensorType::Dense(block.ffn_up.as_ref().unwrap()),
                false,
                transpose_for(block.ffn_up.as_ref().unwrap().dims(), d_model, "ffn_up").unwrap(),
                None,
                None,
                None,
            )
            .unwrap();
        let gate_bias = ctx
            .call::<BroadcastElemwiseAddOp>((gate.clone(), block.ffn_gate_bias.clone()), None)
            .unwrap();
        let up_bias = ctx
            .call::<BroadcastElemwiseAddOp>((up.clone(), block.ffn_up_bias.clone()), None)
            .unwrap();
        let silu_out = ctx.call::<SiluOp>(gate_bias, None).unwrap();
        let mul_out = ctx.call::<ElemwiseMulOp>((silu_out, up_bias), None).unwrap();
        let down = ctx
            .matmul(
                &mul_out,
                &TensorType::Dense(block.ffn_down.as_ref().unwrap()),
                false,
                transpose_for(block.ffn_down.as_ref().unwrap().dims(), block.ffn_gate_bias.len(), "ffn_down").unwrap(),
                None,
                None,
                None,
            )
            .unwrap();
        (gate, up, mul_out, down)
    };
    let ffn_output_flat = ctx
        .call::<BroadcastElemwiseAddInplaceOp>((ffn_output_flat, block.ffn_down_bias.clone()), None)
        .unwrap();
    let ffn_out = ffn_output_flat.reshape(vec![batch, seq, d_model]).unwrap();
    let hidden = resid_mlp.add_elem(&ffn_out, ctx).unwrap();

    ctx.synchronize();
    Ok(LegacyLayer0Outputs {
        attn_out: tensor_to_f16_vec(&attn_out_reshaped),
        q_rot: tensor_to_f16_vec(&q_heads_after_rope),
        k_expanded: tensor_to_f16_vec(&k_repeated),
        v_expanded: tensor_to_f16_vec(&v_repeated),
        q_proj: tensor_to_f16_vec(&q_mat),
        k_proj: tensor_to_f16_vec(&k_mat),
        v_proj: tensor_to_f16_vec(&v_mat),
        ffn_norm_out: tensor_to_f16_vec(&x_normed_mlp),
        gate: tensor_to_f16_vec(&gate_proj),
        swiglu_out: tensor_to_f16_vec(&swiglu_out),
        proj_out: tensor_to_f16_vec(&proj_out),
        residual_1: tensor_to_f16_vec(&resid_mlp),
        ffn_out: tensor_to_f16_vec(&ffn_out),
        hidden: tensor_to_f16_vec(&hidden),
    })
}

#[test]
#[serial]
#[ignore = "requires qwen2.5 model file"]
fn test_dsl_vs_context_embedding_parity() -> Result<(), MetalError> {
    // =========================================================================
    // STEP 1: Load model via Context<T> (legacy)
    // =========================================================================
    let mut ctx = Context::<F16Element>::new().unwrap();
    let gguf_file = metallic_context::gguf::GGUFFile::load_mmap_and_get_metadata(GGUF_PATH)
        .map_err(|e| MetalError::InvalidShape(format!("GGUF load failed: {}", e)))
        .unwrap();
    let loader = metallic_context::gguf::model_loader::GGUFModelLoader::new(gguf_file);
    let gguf_model = loader
        .load_model()
        .map_err(|e| MetalError::InvalidShape(format!("GGUF model failed: {}", e)))
        .unwrap();
    let legacy_model: Qwen25<F16Element> = gguf_model
        .instantiate(&mut ctx)
        .map_err(|e| MetalError::InvalidShape(format!("Model instantiate failed: {}", e)))
        .unwrap();

    // =========================================================================
    // STEP 2: Load model via DSL (new)
    // =========================================================================
    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let mut foundry = Foundry::new().unwrap();
    let dsl_model = ModelBuilder::new()
        .with_spec_file(&spec_path)
        .unwrap()
        .with_gguf(GGUF_PATH)
        .unwrap()
        .build(&mut foundry)
        .unwrap();
    let (mut bindings, _fast_bindings) = dsl_model.prepare_bindings(&mut foundry).unwrap();

    // =========================================================================
    // STEP 3: Create tokenizer and encode a test prompt
    // =========================================================================
    let tokenizer = dsl_model.tokenizer().unwrap();
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt).unwrap();
    eprintln!("Test prompt: '{}' -> tokens: {:?}", prompt, tokens);

    // =========================================================================
    // STEP 4: Run embedding via Legacy Context<T>
    // =========================================================================
    let legacy_embedded = legacy_model.embed(&tokens, &mut ctx).unwrap();
    ctx.synchronize();
    let legacy_embed_data: Vec<f16> = legacy_embedded.as_slice().iter().map(|v| f16::from_f32(v.to_f32())).collect();
    eprintln!(
        "Legacy embedding shape: {:?}, first 5: {:?}",
        legacy_embedded.dims(),
        &legacy_embed_data[..5.min(legacy_embed_data.len())]
    );

    // =========================================================================
    // STEP 5: Run embedding via DSL
    // =========================================================================
    // Set up input_ids for DSL
    let input_buffer = {
        use objc2_metal::MTLDevice;
        let byte_size = tokens.len() * 4;
        let buf = foundry
            .device
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate input buffer");
        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, tokens.len());
        }
        metallic_foundry::types::MetalBuffer::from_retained(buf)
    };
    let input_tensor =
        metallic_foundry::types::TensorArg::from_buffer(input_buffer, metallic_foundry::tensor::Dtype::U32, vec![tokens.len()], vec![1]);
    bindings.insert("input_ids".to_string(), input_tensor);
    bindings.set_global("seq_len", tokens.len().to_string());
    bindings.set_global("position_offset", "0".to_string());

    // CRITICAL: Set total_elements_hidden for embedding kernel dispatch
    let d_model = 896; // From qwen2.5 spec
    let total_elements_hidden = tokens.len() * d_model;
    bindings.set_global("total_elements_hidden", total_elements_hidden.to_string());
    eprintln!(
        "Set total_elements_hidden = {} (seq_len={}, d_model={})",
        total_elements_hidden,
        tokens.len(),
        d_model
    );

    // Run only embedding step (first step in forward)
    let arch = dsl_model.architecture();
    if !arch.forward.is_empty() {
        // Execute just the embedding step
        let embedding_step = &arch.forward[0];
        embedding_step.execute(&mut foundry, &mut bindings).unwrap();
    }

    // Read back the hidden tensor
    let hidden_arg = bindings.get("hidden").unwrap();
    let dsl_embed_data = read_f16_buffer(&hidden_arg);
    eprintln!(
        "DSL embedding shape: {:?}, first 5: {:?}",
        hidden_arg.dims(),
        &dsl_embed_data[..5.min(dsl_embed_data.len())]
    );

    // =========================================================================
    // STEP 6: Compare embeddings
    // =========================================================================
    let (max_diff, avg_diff, first_mismatch) = compare_f16_slices(&legacy_embed_data, &dsl_embed_data, 1e-3);
    eprintln!(
        "Embedding comparison: max_diff={}, avg_diff={}, first_mismatch={:?}",
        max_diff, avg_diff, first_mismatch
    );

    assert!(max_diff < 1e-2, "Embedding parity failed: max_diff={}", max_diff);
    eprintln!("✅ Embedding parity PASSED");

    Ok(())
}

#[test]
#[serial]
#[ignore = "requires qwen2.5 model file"]
fn test_dsl_vs_context_generation_greedy_parity() -> Result<(), MetalError> {
    eprintln!("\n=== DSL vs Context Greedy Generation Parity Test ===\n");

    let mut ctx = Context::<F16Element>::new().unwrap();
    let gguf_file = metallic_context::gguf::GGUFFile::load_mmap_and_get_metadata(GGUF_PATH)
        .map_err(|e| MetalError::InvalidShape(format!("GGUF load failed: {}", e)))
        .unwrap();
    let loader = metallic_context::gguf::model_loader::GGUFModelLoader::new(gguf_file);
    let gguf_model = loader
        .load_model()
        .map_err(|e| MetalError::InvalidShape(format!("GGUF model failed: {}", e)))
        .unwrap();
    let legacy_model: Qwen25<F16Element> = gguf_model
        .instantiate(&mut ctx)
        .map_err(|e| MetalError::InvalidShape(format!("Model instantiate failed: {}", e)))
        .unwrap();

    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let mut foundry = Foundry::new().unwrap();
    let dsl_model = ModelBuilder::new()
        .with_spec_file(&spec_path)
        .unwrap()
        .with_gguf(GGUF_PATH)
        .unwrap()
        .build(&mut foundry)
        .unwrap();
    let (mut bindings, mut fast_bindings) = dsl_model.prepare_bindings(&mut foundry).unwrap();

    let tokenizer = dsl_model.tokenizer().unwrap();
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt).unwrap();
    eprintln!("Prompt: '{}' -> tokens: {:?}", prompt, tokens);

    let max_new_tokens = 16usize;

    let kv_capacity = tokens.len() + max_new_tokens;
    let legacy_n_heads = legacy_model.config.n_heads;
    let legacy_n_kv_heads = legacy_model.config.n_kv_heads;
    let legacy_d_model = legacy_model.config.d_model;
    let kv_dim = legacy_d_model * legacy_n_kv_heads / legacy_n_heads;
    let kv_head_dim = kv_dim / legacy_n_kv_heads;
    for layer_idx in 0..legacy_model.config.n_layers {
        ctx.alloc_kv_cache(layer_idx, kv_capacity, legacy_n_heads, kv_head_dim).unwrap();
    }

    let arch = dsl_model.architecture();
    let d_model = arch.d_model;
    let n_heads = arch.n_heads;
    let n_kv_heads = arch.n_kv_heads;
    let ff_dim = arch.ff_dim;
    let head_dim = d_model / n_heads;

    let input_buffer = {
        use objc2_metal::MTLDevice;
        let buf = foundry
            .device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate input buffer");
        metallic_foundry::types::MetalBuffer::from_retained(buf)
    };
    let input_tensor =
        metallic_foundry::types::TensorArg::from_buffer(input_buffer.clone(), metallic_foundry::tensor::Dtype::U32, vec![1], vec![1]);
    // Sync input_ids to both bindings and fast_bindings for compiled step execution
    bindings.insert("input_ids".to_string(), input_tensor.clone());
    if let Some(id) = dsl_model.symbol_id("input_ids") {
        fast_bindings.set(id, input_tensor);
    }

    #[allow(clippy::too_many_arguments)]
    fn run_dsl_step(
        foundry: &mut Foundry,
        bindings: &mut TensorBindings,
        fast_bindings: &mut metallic_foundry::spec::compiled::FastBindings,
        dsl_model: &metallic_foundry::model::CompiledModel,
        input_buffer: &metallic_foundry::types::MetalBuffer,
        token: u32,
        pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        d_model: usize,
        ff_dim: usize,
    ) -> Result<(Vec<f16>, Vec<f16>), MetalError> {
        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = input_buffer.contents().as_ptr() as *mut u32;
            *ptr = token;
        }

        let seq_len = 1usize;
        let kv_seq_len = pos + seq_len;
        bindings.set_global("seq_len", seq_len.to_string());
        bindings.set_global("position_offset", pos.to_string());
        bindings.set_global("kv_seq_len", kv_seq_len.to_string());
        bindings.set_global("total_elements_q", (n_heads * seq_len * head_dim).to_string());
        bindings.set_global("total_elements_k", (n_kv_heads * seq_len * head_dim).to_string());
        bindings.set_global("total_elements_hidden", (seq_len * d_model).to_string());
        bindings.set_global("total_elements_ffn", (seq_len * ff_dim).to_string());
        bindings.set_global("total_elements_slice", (n_kv_heads * kv_seq_len * head_dim).to_string());
        bindings.set_global("total_elements_repeat", (n_heads * kv_seq_len * head_dim).to_string());
        bindings.set_global("total_elements_write", (n_kv_heads * seq_len * head_dim).to_string());

        // Also set int_globals for fast lookup
        bindings.set_int_global("position_offset", pos);
        bindings.set_int_global("kv_seq_len", kv_seq_len);
        bindings.set_int_global("total_elements_slice", n_kv_heads * kv_seq_len * head_dim);
        bindings.set_int_global("total_elements_repeat", n_heads * kv_seq_len * head_dim);

        // Use compiled forward path instead of interpreted step.execute()
        dsl_model.forward(foundry, bindings, fast_bindings).unwrap();

        let logits = bindings.get("logits").unwrap();
        let hidden = bindings.get("final_norm_out").or_else(|_| bindings.get("hidden")).unwrap();
        Ok((read_f16_buffer(&logits), read_f16_buffer(&hidden)))
    }

    let enable_legacy_v_cache_compare = std::env::var("METALLIC_PARITY_COMPARE_V_CACHE")
        .ok()
        .map(|v| v.trim() == "1")
        .unwrap_or(false);
    let cache_layer_list = if enable_legacy_v_cache_compare {
        std::env::var("METALLIC_PARITY_COMPARE_V_CACHE_LAYERS")
            .ok()
            .and_then(|raw| {
                let raw = raw.trim();
                if raw.eq_ignore_ascii_case("all") {
                    return Some((0..legacy_model.config.n_layers).collect());
                }
                let mut layers = Vec::new();
                for chunk in raw.split(',') {
                    let chunk = chunk.trim();
                    if chunk.is_empty() {
                        continue;
                    }
                    let layer = chunk.parse::<usize>().ok().unwrap();
                    if layer < legacy_model.config.n_layers {
                        layers.push(layer);
                    }
                }
                if layers.is_empty() { None } else { Some(layers) }
            })
            .or_else(|| Some(vec![0]))
    } else {
        None
    };

    let mut run_legacy = |token: u32, pos: usize| -> LegacyStepResult {
        let embedded = legacy_model.embed(&[token], &mut ctx).unwrap();
        let (hidden, _) = legacy_model.forward_step(&embedded, pos, &mut ctx).unwrap();
        let logits = legacy_model.output(&hidden, &mut ctx).unwrap();
        let mut v_cache_layers = None;
        if pos == 0 {
            ctx.synchronize();
            if let Some(layers) = cache_layer_list.as_ref() {
                let mut snapshots = Vec::with_capacity(layers.len());
                for &layer_idx in layers {
                    let cache_entry = ctx
                        .kv_caches()
                        .get(&layer_idx)
                        .ok_or_else(|| MetalError::InvalidOperation(format!("KV cache for layer {} not found", layer_idx)))
                        .unwrap();
                    let cache = cache_entry.v.clone();
                    let legacy_view = ctx.kv_cache_history_view(&cache, pos + 1).unwrap().0;
                    snapshots.push((layer_idx, tensor_to_f16_vec_strided(&legacy_view).unwrap()));
                }
                v_cache_layers = Some(snapshots);
            }
        }
        ctx.synchronize();
        Ok((tensor_to_f16_vec(&logits), tensor_to_f16_vec(&hidden), v_cache_layers))
    };

    let mut next_token = None;
    for (pos, token) in tokens.iter().copied().enumerate() {
        let (legacy_logits, legacy_hidden, legacy_v_cache_layers) = run_legacy(token, pos).unwrap();
        let (dsl_logits, dsl_hidden) = run_dsl_step(
            &mut foundry,
            &mut bindings,
            &mut fast_bindings,
            &dsl_model,
            &input_buffer,
            token,
            pos,
            n_heads,
            n_kv_heads,
            head_dim,
            d_model,
            ff_dim,
        )
        .unwrap();

        let (hidden_max, hidden_avg, hidden_mismatch) = compare_f16_slices(&legacy_hidden, &dsl_hidden, 0.5);
        eprintln!(
            "[prompt pos {}] hidden diff: max_diff={:.4}, avg_diff={:.6}, mismatch={:?}",
            pos, hidden_max, hidden_avg, hidden_mismatch
        );

        if pos == 0 {
            if let Ok(attn_out) = bindings.get("attn_out") {
                let attn_data = read_f16_buffer(&attn_out);
                if let Ok(v_expanded) = bindings.get("v_expanded") {
                    let v_data = read_f16_buffer(&v_expanded);
                    let expected_len = n_heads * (pos + 1) * head_dim;
                    let len = expected_len.min(attn_data.len()).min(v_data.len());
                    if len > 0 {
                        let (sdpa_max, sdpa_avg, sdpa_mismatch) = compare_f16_slices(&attn_data[..len], &v_data[..len], 0.01);
                        eprintln!(
                            "[prompt pos {}] sdpa vs v_expanded: max_diff={:.4}, avg_diff={:.6}, mismatch={:?}",
                            pos, sdpa_max, sdpa_avg, sdpa_mismatch
                        );
                    }
                }
            }

            if enable_legacy_v_cache_compare
                && let (Ok(v_tensor), Ok(v_cache_tensor)) = (bindings.get("v_heads"), bindings.get("v_cache_0"))
            {
                let v_data = read_f16_buffer(&v_tensor);
                let v_cache_data = read_f16_buffer(&v_cache_tensor);
                let cache_stride = v_cache_tensor.dims().get(1).copied().unwrap_or(0);
                for kv_head in 0..n_kv_heads {
                    let v_offset = kv_head * head_dim;
                    let cache_offset = (kv_head * cache_stride + pos) * head_dim;
                    if v_offset + head_dim <= v_data.len() && cache_offset + head_dim <= v_cache_data.len() {
                        let (v_max, v_avg, v_mismatch) = compare_f16_slices(
                            &v_data[v_offset..v_offset + head_dim],
                            &v_cache_data[cache_offset..cache_offset + head_dim],
                            0.01,
                        );
                        eprintln!(
                            "[prompt pos {}] dsl v vs v_cache head {}: max_diff={:.4}, avg_diff={:.6}, mismatch={:?}",
                            pos, kv_head, v_max, v_avg, v_mismatch
                        );
                    }
                }
            }

            if let Some(legacy_layers) = legacy_v_cache_layers {
                let group_size = n_heads / n_kv_heads;
                for (layer_idx, legacy_v) in legacy_layers {
                    let cache_name = format!("v_cache_{}", layer_idx);
                    let Ok(v_cache_tensor) = bindings.get(&cache_name) else {
                        continue;
                    };
                    let v_cache_data = read_f16_buffer(&v_cache_tensor);
                    let cache_stride = v_cache_tensor.dims().get(1).copied().unwrap_or(0);
                    let mut layer_max = 0.0f32;
                    let mut layer_avg_sum = 0.0f32;
                    for kv_head in 0..n_kv_heads {
                        let legacy_head = kv_head * group_size;
                        let legacy_offset = legacy_head * head_dim;
                        let cache_offset = (kv_head * cache_stride + pos) * head_dim;
                        if legacy_offset + head_dim <= legacy_v.len() && cache_offset + head_dim <= v_cache_data.len() {
                            let (v_max, v_avg, _) = compare_f16_slices(
                                &legacy_v[legacy_offset..legacy_offset + head_dim],
                                &v_cache_data[cache_offset..cache_offset + head_dim],
                                0.01,
                            );
                            layer_max = layer_max.max(v_max);
                            layer_avg_sum += v_avg;
                            if layer_idx == 0 {
                                eprintln!(
                                    "[prompt pos {}] legacy v_cache head {} vs dsl v_cache head {}: max_diff={:.4}, avg_diff={:.6}",
                                    pos, legacy_head, kv_head, v_max, v_avg
                                );
                            }
                        }
                    }
                    let layer_avg = layer_avg_sum / n_kv_heads as f32;
                    eprintln!(
                        "[prompt pos {}] legacy v_cache vs dsl v_cache layer {}: max_diff={:.4}, avg_diff={:.6}",
                        pos, layer_idx, layer_max, layer_avg
                    );
                }
            }
        }

        let (max_diff, avg_diff, first_mismatch) = compare_f16_slices(&legacy_logits, &dsl_logits, 0.5);
        eprintln!(
            "[prompt pos {}] logits diff: max_diff={:.4}, avg_diff={:.6}, mismatch={:?}",
            pos, max_diff, avg_diff, first_mismatch
        );

        let legacy_argmax = argmax_f16(&legacy_logits);
        let dsl_argmax = argmax_f16(&dsl_logits);
        if legacy_argmax != dsl_argmax || max_diff > 0.5 {
            return Err(MetalError::InvalidShape(format!(
                "Prompt parity failed at pos {pos}: legacy_argmax={legacy_argmax} dsl_argmax={dsl_argmax} max_diff={max_diff:.4} hidden_max={hidden_max:.4}"
            )));
        }

        if pos + 1 == tokens.len() {
            next_token = Some(legacy_argmax);
        }
    }

    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut current_token = next_token
        .ok_or_else(|| MetalError::InvalidShape("Missing next token after prompt".to_string()))
        .unwrap();
    let mut pos = tokens.len();

    for step in 0..max_new_tokens {
        generated.push(current_token);

        let (legacy_logits, legacy_hidden, _legacy_v_cache) = run_legacy(current_token, pos).unwrap();
        let (dsl_logits, dsl_hidden) = run_dsl_step(
            &mut foundry,
            &mut bindings,
            &mut fast_bindings,
            &dsl_model,
            &input_buffer,
            current_token,
            pos,
            n_heads,
            n_kv_heads,
            head_dim,
            d_model,
            ff_dim,
        )
        .unwrap();

        let (hidden_max, hidden_avg, hidden_mismatch) = compare_f16_slices(&legacy_hidden, &dsl_hidden, 0.5);
        eprintln!(
            "[gen step {} @pos {}] hidden diff: max_diff={:.4}, avg_diff={:.6}, mismatch={:?}",
            step, pos, hidden_max, hidden_avg, hidden_mismatch
        );

        let (max_diff, avg_diff, first_mismatch) = compare_f16_slices(&legacy_logits, &dsl_logits, 0.5);
        eprintln!(
            "[gen step {} @pos {}] logits diff: max_diff={:.4}, avg_diff={:.6}, mismatch={:?}",
            step, pos, max_diff, avg_diff, first_mismatch
        );

        let legacy_argmax = argmax_f16(&legacy_logits);
        let dsl_argmax = argmax_f16(&dsl_logits);
        if legacy_argmax != dsl_argmax || max_diff > 0.5 {
            return Err(MetalError::InvalidShape(format!(
                "Generation parity failed at step {step} (pos {pos}): legacy_argmax={legacy_argmax} dsl_argmax={dsl_argmax} max_diff={max_diff:.4} hidden_max={hidden_max:.4}"
            )));
        }

        current_token = legacy_argmax;
        pos += 1;
    }

    eprintln!("Generated tokens ({}): {:?}", generated.len(), generated);
    let decoded = tokenizer.decode(&generated).unwrap_or_else(|_| "<decode failed>".to_string());
    eprintln!("Generated text:\n---\n{}\n---", decoded);
    eprintln!("✅ Greedy generation parity PASSED");

    Ok(())
}

#[test]
#[serial]
#[ignore = "requires qwen2.5 model file"]
fn test_dsl_vs_context_pre_norm_parity() -> Result<(), MetalError> {
    eprintln!("\n=== DSL vs Context Pre-Final-Norm Parity Test ===\n");

    let mut ctx = Context::<F16Element>::new().unwrap();
    let gguf_file = metallic_context::gguf::GGUFFile::load_mmap_and_get_metadata(GGUF_PATH)
        .map_err(|e| MetalError::InvalidShape(format!("GGUF load failed: {}", e)))
        .unwrap();
    let loader = metallic_context::gguf::model_loader::GGUFModelLoader::new(gguf_file);
    let gguf_model = loader
        .load_model()
        .map_err(|e| MetalError::InvalidShape(format!("GGUF model failed: {}", e)))
        .unwrap();
    let legacy_model: Qwen25<F16Element> = gguf_model
        .instantiate(&mut ctx)
        .map_err(|e| MetalError::InvalidShape(format!("Model instantiate failed: {}", e)))
        .unwrap();

    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let mut foundry = Foundry::new().unwrap();
    let dsl_model = ModelBuilder::new()
        .with_spec_file(&spec_path)
        .unwrap()
        .with_gguf(GGUF_PATH)
        .unwrap()
        .build(&mut foundry)
        .unwrap();
    let (mut bindings, _fast_bindings) = dsl_model.prepare_bindings(&mut foundry).unwrap();

    let tokenizer = dsl_model.tokenizer().unwrap();
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt).unwrap();
    eprintln!("Prompt: '{}' -> tokens: {:?}", prompt, tokens);

    // Legacy embedding + blocks-only forward (pre-final norm)
    let legacy_embedded = legacy_model.embed(&tokens, &mut ctx).unwrap();
    let legacy_hidden_pre = legacy_forward_blocks_only(&legacy_model, &legacy_embedded, &mut ctx).unwrap();
    ctx.synchronize();
    let legacy_hidden_data = tensor_to_f16_vec(&legacy_hidden_pre);

    // DSL forward through embedding + repeat
    let input_buffer = {
        use objc2_metal::MTLDevice;
        let byte_size = tokens.len() * 4;
        let buf = foundry
            .device
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate input buffer");
        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, tokens.len());
        }
        metallic_foundry::types::MetalBuffer::from_retained(buf)
    };
    let input_tensor =
        metallic_foundry::types::TensorArg::from_buffer(input_buffer, metallic_foundry::tensor::Dtype::U32, vec![tokens.len()], vec![1]);
    bindings.insert("input_ids".to_string(), input_tensor);

    let arch = dsl_model.architecture();
    let seq_len = tokens.len();
    let d_model = arch.d_model;
    let n_heads = arch.n_heads;
    let n_kv_heads = arch.n_kv_heads;
    let ff_dim = arch.ff_dim;
    let head_dim = d_model / n_heads;
    let position_offset = 0usize;
    let kv_seq_len = position_offset + seq_len;

    bindings.set_global("seq_len", seq_len.to_string());
    bindings.set_global("position_offset", position_offset.to_string());
    bindings.set_global("kv_seq_len", kv_seq_len.to_string());
    bindings.set_global("total_elements_q", (n_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_k", (n_kv_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_hidden", (seq_len * d_model).to_string());
    bindings.set_global("total_elements_ffn", (seq_len * ff_dim).to_string());
    bindings.set_global("total_elements_slice", (n_kv_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_repeat", (n_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_write", (n_kv_heads * seq_len * head_dim).to_string());

    for step in arch.forward.iter() {
        step.execute(&mut foundry, &mut bindings).unwrap();
        if step.name() == "Repeat" {
            break;
        }
    }

    let dsl_hidden = bindings.get("hidden").unwrap();
    let dsl_hidden_data = read_f16_buffer(&dsl_hidden);

    let (max_diff, avg_diff, first_mismatch) = compare_f16_slices(&legacy_hidden_data, &dsl_hidden_data, 0.1);
    eprintln!(
        "Pre-final-norm comparison: max_diff={:.4}, avg_diff={:.6}, mismatch={:?}",
        max_diff, avg_diff, first_mismatch
    );

    assert!(max_diff < 0.5, "Pre-final-norm parity failed: max_diff={}", max_diff);
    eprintln!("✅ Pre-final-norm parity PASSED");

    Ok(())
}

#[test]
#[serial]
#[ignore = "requires qwen2.5 model file"]
fn test_dsl_vs_context_sdpa_layer0_parity() -> Result<(), MetalError> {
    eprintln!("\n=== DSL vs Context SDPA Layer0 Parity Test ===\n");

    let mut ctx = Context::<F16Element>::new().unwrap();
    let gguf_file = metallic_context::gguf::GGUFFile::load_mmap_and_get_metadata(GGUF_PATH)
        .map_err(|e| MetalError::InvalidShape(format!("GGUF load failed: {}", e)))
        .unwrap();
    let loader = metallic_context::gguf::model_loader::GGUFModelLoader::new(gguf_file);
    let gguf_model = loader
        .load_model()
        .map_err(|e| MetalError::InvalidShape(format!("GGUF model failed: {}", e)))
        .unwrap();
    let legacy_model: Qwen25<F16Element> = gguf_model
        .instantiate(&mut ctx)
        .map_err(|e| MetalError::InvalidShape(format!("Model instantiate failed: {}", e)))
        .unwrap();

    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let mut foundry = Foundry::new().unwrap();
    let dsl_model = ModelBuilder::new()
        .with_spec_file(&spec_path)
        .unwrap()
        .with_gguf(GGUF_PATH)
        .unwrap()
        .build(&mut foundry)
        .unwrap();
    let (mut bindings, _fast_bindings) = dsl_model.prepare_bindings(&mut foundry).unwrap();

    let tokenizer = dsl_model.tokenizer().unwrap();
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt).unwrap();
    eprintln!("Prompt: '{}' -> tokens: {:?}", prompt, tokens);

    // Legacy attention-only (layer 0) SDPA output
    let legacy_embedded = legacy_model.embed(&tokens, &mut ctx).unwrap();
    let legacy_attn_out = legacy_sdpa_layer0(&legacy_model, &legacy_embedded, &mut ctx).unwrap();
    ctx.synchronize();
    let legacy_attn_data = tensor_to_f16_vec(&legacy_attn_out);

    // DSL run with n_layers=1 to capture layer0 attn_out
    bindings.set_global("n_layers", "1");

    let input_buffer = {
        use objc2_metal::MTLDevice;
        let byte_size = tokens.len() * 4;
        let buf = foundry
            .device
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate input buffer");
        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, tokens.len());
        }
        metallic_foundry::types::MetalBuffer::from_retained(buf)
    };
    let input_tensor =
        metallic_foundry::types::TensorArg::from_buffer(input_buffer, metallic_foundry::tensor::Dtype::U32, vec![tokens.len()], vec![1]);
    bindings.insert("input_ids".to_string(), input_tensor);

    let arch = dsl_model.architecture();
    let seq_len = tokens.len();
    let d_model = arch.d_model;
    let n_heads = arch.n_heads;
    let n_kv_heads = arch.n_kv_heads;
    let ff_dim = arch.ff_dim;
    let head_dim = d_model / n_heads;
    let position_offset = 0usize;
    let kv_seq_len = position_offset + seq_len;

    bindings.set_global("seq_len", seq_len.to_string());
    bindings.set_global("position_offset", position_offset.to_string());
    bindings.set_global("kv_seq_len", kv_seq_len.to_string());
    bindings.set_global("total_elements_q", (n_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_k", (n_kv_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_hidden", (seq_len * d_model).to_string());
    bindings.set_global("total_elements_ffn", (seq_len * ff_dim).to_string());
    bindings.set_global("total_elements_slice", (n_kv_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_repeat", (n_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_write", (n_kv_heads * seq_len * head_dim).to_string());

    for step in arch.forward.iter() {
        step.execute(&mut foundry, &mut bindings).unwrap();
        if step.name() == "Repeat" {
            break;
        }
    }

    let dsl_attn_out = bindings.get("attn_out").unwrap();
    let dsl_attn_data = read_f16_buffer(&dsl_attn_out);

    let expected_len = legacy_attn_data.len();
    if dsl_attn_data.len() < expected_len {
        return Err(MetalError::InvalidShape(format!(
            "DSL attn_out length {} is smaller than expected {}",
            dsl_attn_data.len(),
            expected_len
        )));
    }
    let dsl_attn_slice = &dsl_attn_data[..expected_len];

    let (max_diff, avg_diff, first_mismatch) = compare_f16_slices(&legacy_attn_data, dsl_attn_slice, 0.1);
    eprintln!(
        "SDPA layer0 comparison: max_diff={:.4}, avg_diff={:.6}, mismatch={:?}",
        max_diff, avg_diff, first_mismatch
    );

    assert!(max_diff < 0.5, "SDPA layer0 parity failed: max_diff={}", max_diff);
    eprintln!("✅ SDPA layer0 parity PASSED");

    Ok(())
}

#[test]
#[serial]
#[ignore = "requires qwen2.5 model file"]
fn test_dsl_vs_context_layer0_block_parity() -> Result<(), MetalError> {
    eprintln!("\n=== DSL vs Context Layer0 Block Parity Test ===\n");

    let mut ctx = Context::<F16Element>::new().unwrap();
    let gguf_file = metallic_context::gguf::GGUFFile::load_mmap_and_get_metadata(GGUF_PATH)
        .map_err(|e| MetalError::InvalidShape(format!("GGUF load failed: {}", e)))
        .unwrap();
    let loader = metallic_context::gguf::model_loader::GGUFModelLoader::new(gguf_file);
    let gguf_model = loader
        .load_model()
        .map_err(|e| MetalError::InvalidShape(format!("GGUF model failed: {}", e)))
        .unwrap();
    let legacy_model: Qwen25<F16Element> = gguf_model
        .instantiate(&mut ctx)
        .map_err(|e| MetalError::InvalidShape(format!("Model instantiate failed: {}", e)))
        .unwrap();

    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let mut foundry = Foundry::new().unwrap();
    let dsl_model = ModelBuilder::new()
        .with_spec_file(&spec_path)
        .unwrap()
        .with_gguf(GGUF_PATH)
        .unwrap()
        .build(&mut foundry)
        .unwrap();
    let (mut bindings, _fast_bindings) = dsl_model.prepare_bindings(&mut foundry).unwrap();

    let tokenizer = dsl_model.tokenizer().unwrap();
    let tokens = tokenizer.encode("Hello").unwrap();
    let token = tokens
        .first()
        .copied()
        .ok_or_else(|| MetalError::InvalidShape("No tokens".to_string()))
        .unwrap();

    let input_buffer = {
        use objc2_metal::MTLDevice;
        let buf = foundry
            .device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate input buffer");
        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = buf.contents().as_ptr() as *mut u32;
            *ptr = token;
        }
        metallic_foundry::types::MetalBuffer::from_retained(buf)
    };
    let input_tensor =
        metallic_foundry::types::TensorArg::from_buffer(input_buffer, metallic_foundry::tensor::Dtype::U32, vec![1], vec![1]);
    bindings.insert("input_ids".to_string(), input_tensor);

    let arch = dsl_model.architecture();
    let seq_len = 1usize;
    let d_model = arch.d_model;
    let n_heads = arch.n_heads;
    let n_kv_heads = arch.n_kv_heads;
    let ff_dim = arch.ff_dim;
    let head_dim = d_model / n_heads;
    let kv_seq_len = seq_len;

    bindings.set_global("n_layers", "1");
    bindings.set_global("seq_len", seq_len.to_string());
    bindings.set_global("position_offset", "0");
    bindings.set_global("kv_seq_len", kv_seq_len.to_string());
    bindings.set_global("total_elements_q", (n_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_k", (n_kv_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_hidden", (seq_len * d_model).to_string());
    bindings.set_global("total_elements_ffn", (seq_len * ff_dim).to_string());
    bindings.set_global("total_elements_slice", (n_kv_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_repeat", (n_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_write", (n_kv_heads * seq_len * head_dim).to_string());

    for step in arch.forward.iter() {
        step.execute(&mut foundry, &mut bindings).unwrap();
        if step.name() == "Repeat" {
            break;
        }
    }

    let legacy_embedded = legacy_model.embed(&[token], &mut ctx).unwrap();
    let legacy_out = legacy_layer0_outputs(&legacy_model, &legacy_embedded, &mut ctx).unwrap();

    let ffn_norm_out = read_f16_buffer(&bindings.get("ffn_norm_out").unwrap());
    let ffn_norm_gamma = read_f16_buffer(&bindings.get("layer.ffn_norm_0").unwrap());
    let gate = read_f16_buffer(&bindings.get("gate").unwrap());
    let swiglu_out = read_f16_buffer(&bindings.get("up").unwrap());
    let proj_out = read_f16_buffer(&bindings.get("proj_out").unwrap());
    let residual_1 = read_f16_buffer(&bindings.get("residual_1").unwrap());
    let ffn_out = read_f16_buffer(&bindings.get("ffn_out").unwrap());
    let hidden = read_f16_buffer(&bindings.get("hidden").unwrap());

    let legacy_ffn_norm_gamma = tensor_to_f16_vec(&legacy_model.blocks[0].ffn_norm_gamma);
    let (gamma_max, gamma_avg, _) = compare_f16_slices(&legacy_ffn_norm_gamma, &ffn_norm_gamma, 0.0);
    let (norm_max, norm_avg, _) = compare_f16_slices(&legacy_out.ffn_norm_out, &ffn_norm_out, 0.1);
    let (gate_max, gate_avg, _) = compare_f16_slices(&legacy_out.gate, &gate, 0.1);
    let (swiglu_max, swiglu_avg, _) = compare_f16_slices(&legacy_out.swiglu_out, &swiglu_out, 0.1);
    let (proj_max, proj_avg, _) = compare_f16_slices(&legacy_out.proj_out, &proj_out, 0.1);
    let (res_max, res_avg, _) = compare_f16_slices(&legacy_out.residual_1, &residual_1, 0.1);
    let (ffn_max, ffn_avg, _) = compare_f16_slices(&legacy_out.ffn_out, &ffn_out, 0.1);
    let (hid_max, hid_avg, _) = compare_f16_slices(&legacy_out.hidden, &hidden, 0.1);

    // Compare attn_out (SDPA output) - DSL stores in "attn_out"
    // DSL buffer is pre-allocated at max_seq_len×d_model but only seq_len×d_model elements are valid
    let dsl_attn_out_full = read_f16_buffer(&bindings.get("attn_out").unwrap());
    let attn_out_full_len = dsl_attn_out_full.len();
    let attn_out_valid_len = seq_len * d_model;
    let dsl_attn_out = if attn_out_full_len > attn_out_valid_len {
        // Extract only valid portion based on current seq_len
        dsl_attn_out_full[..attn_out_valid_len].to_vec()
    } else {
        dsl_attn_out_full
    };
    eprintln!(
        "  Legacy attn_out len: {}, DSL attn_out len: {} (from {} total buffer)",
        legacy_out.attn_out.len(),
        dsl_attn_out.len(),
        attn_out_full_len
    );
    eprintln!(
        "  Legacy attn_out first 5: {:?}",
        &legacy_out.attn_out[..5.min(legacy_out.attn_out.len())]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>()
    );
    eprintln!(
        "  DSL attn_out first 5: {:?}",
        &dsl_attn_out[..5.min(dsl_attn_out.len())]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>()
    );
    let (attn_max, attn_avg, _) = if legacy_out.attn_out.len() == dsl_attn_out.len() {
        compare_f16_slices(&legacy_out.attn_out, &dsl_attn_out, 0.1)
    } else {
        eprintln!(
            "  ⚠️ Length mismatch after extraction: legacy={}, dsl={}",
            legacy_out.attn_out.len(),
            dsl_attn_out.len()
        );
        (f32::INFINITY, f32::INFINITY, Some(0))
    };
    eprintln!("Layer0 attn_out diff: max_diff={:.4}, avg_diff={:.6}", attn_max, attn_avg);

    // Compare Q heads and Q rotated to trace divergence source
    // --- Legacy vs DSL K/V Projection Comparison ---
    if let (Ok(dsl_k_proj_arg), Ok(dsl_v_proj_arg)) = (bindings.get("k"), bindings.get("v")) {
        let dsl_k_proj = read_f16_buffer(&dsl_k_proj_arg);
        let dsl_v_proj = read_f16_buffer(&dsl_v_proj_arg);

        let (max_diff_k, avg_diff_k, first_k) = compare_f16_slices(&legacy_out.k_proj, &dsl_k_proj, 0.1);
        let (max_diff_v, avg_diff_v, first_v) = compare_f16_slices(&legacy_out.v_proj, &dsl_v_proj, 0.1);

        eprintln!("\n--- Legacy vs DSL K/V Projection Comparison ---");
        eprintln!(
            "  k_proj: Legacy len={}, DSL len={}, max_diff={:.6}, avg_diff={:.6}",
            legacy_out.k_proj.len(),
            dsl_k_proj.len(),
            max_diff_k,
            avg_diff_k
        );
        if max_diff_k > 0.1 {
            eprintln!("    Legacy first 5: {:?}", &legacy_out.k_proj[..5.min(legacy_out.k_proj.len())]);
            eprintln!("    DSL first 5:    {:?}", &dsl_k_proj[..5.min(dsl_k_proj.len())]);
            eprintln!("    First mismatch: {:?}", first_k);
        }

        eprintln!(
            "  v_proj: Legacy len={}, DSL len={}, max_diff={:.6}, avg_diff={:.6}",
            legacy_out.v_proj.len(),
            dsl_v_proj.len(),
            max_diff_v,
            avg_diff_v
        );
        if max_diff_v > 0.1 {
            eprintln!("    Legacy first 5: {:?}", &legacy_out.v_proj[..5.min(legacy_out.v_proj.len())]);
            eprintln!("    DSL first 5:    {:?}", &dsl_v_proj[..5.min(dsl_v_proj.len())]);
            eprintln!("    First mismatch: {:?}", first_v);
        }
    } else {
        eprintln!("\n--- Legacy vs DSL K/V Projection Comparison ---");
        eprintln!("  ⚠️ Could not retrieve 'k' or 'v' from bindings");
        // fallback to printing all bindings if possible, or just error
        // bindings might be large to print, so maybe just say failed.
        // Try printing debug
        // eprintln!("Bindings: {:?}", bindings);
    }

    // --- Weight Analysis ---
    eprintln!("\n--- K/V Weight Analysis ---");
    let block = legacy_model.blocks.first().unwrap();

    // Check attn_k weight
    if let Ok(dsl_k_weight) = bindings.get("layer.attn_k_0") {
        let dsl_k = read_f16_buffer(&dsl_k_weight);
        let legacy_k = if let Some(w) = &block.attn_k_weight_canon {
            // In canonical mode, weight might be transposed or packed
            tensor_to_f16_vec(&w.data)
        } else {
            // Fallback for fused weights if needed, but for now just empty
            vec![]
        };

        if !legacy_k.is_empty() {
            eprintln!("  Legacy attn_k len: {}", legacy_k.len());
            eprintln!("  DSL attn_k len:    {}", dsl_k.len());

            // Compare first few elements to check if they match (accounting for potential transpose)
            let (max_diff, _, _) = compare_f16_slices(&legacy_k, &dsl_k, 0.1);
            eprintln!("  attn_k weight diff: max_diff={:.6}", max_diff);
            if max_diff > 0.1 {
                eprintln!("    Legacy first 5: {:?}", &legacy_k[..5.min(legacy_k.len())]);
                eprintln!("    DSL first 5:    {:?}", &dsl_k[..5.min(dsl_k.len())]);

                // Perform full unswizzled comparison
                // Legacy: [blocks_per_k, N, 32] interleaved k-block-major
                // DSL: [N, K] row-major (GGUF)
                // We assume N=128, K=896 based on problem description
                let n_dim = 128; // kv_dim
                let k_dim = 896; // d_model
                let weights_per_block = 32;

                let mut mismatches = 0;
                let mut max_w_diff = f16::from_f32(0.0);

                if legacy_k.len() == dsl_k.len() && legacy_k.len() == n_dim * k_dim {
                    eprintln!("  Verifying unswizzled weights (N={}, K={})...", n_dim, k_dim);
                    for n in 0..n_dim {
                        for k in 0..k_dim {
                            // DSL index (Row Major N, K)
                            let dsl_idx = n * k_dim + k;

                            // Legacy index (Interleaved K-Block Major)
                            // block_idx = k / 32
                            // inner_idx = k % 32
                            // global_block_idx = block * N + n
                            // flat_idx = global_block_idx * 32 + inner_idx
                            let block = k / weights_per_block;
                            let inner = k % weights_per_block;
                            let legacy_idx = (block * n_dim + n) * weights_per_block + inner;

                            let d_val = dsl_k[dsl_idx];
                            let l_val = legacy_k[legacy_idx];
                            let diff = (d_val.to_f32() - l_val.to_f32()).abs();
                            if diff > max_w_diff.to_f32() {
                                max_w_diff = f16::from_f32(diff);
                            }
                            if diff > 0.01 {
                                mismatches += 1;
                                if mismatches <= 5 {
                                    eprintln!(
                                        "    Mismatch at (n={}, k={}): DSL={:?} Legacy={:?} Diff={:?}",
                                        n, k, d_val, l_val, diff
                                    );
                                }
                            }
                        }
                    }
                    eprintln!(
                        "  Unswizzled Weight Comparison: max_diff={:?}, mismatches={}",
                        max_w_diff, mismatches
                    );
                } else {
                    eprintln!("  ⚠️ skipping unswizzle check due to shape mismatch or hardcoded dims");
                }
            }
        } else {
            eprintln!("  ⚠️ Legacy attn_k weight not found in expected fields");
        }
    } else {
        eprintln!("  ⚠️ Could not retrieve 'layer.attn_k_0' from bindings");
    }

    // Check attn_k bias
    eprintln!("\n--- K Bias Analysis ---");
    if let Ok(dsl_k_bias_arg) = bindings.get("layer.attn_k_bias_0") {
        let dsl_k_bias = read_f16_buffer(&dsl_k_bias_arg);

        // Legacy K bias is in attn_qkv_bias at offset d_model
        let d_model = dsl_model.architecture().d_model;
        let k_offset = d_model;
        let kv_dim = 128; // Hardcoded derived from shape

        let legacy_bias_full = tensor_to_f16_vec(&block.attn_qkv_bias);
        if legacy_bias_full.len() > k_offset + kv_dim {
            let legacy_k_bias = &legacy_bias_full[k_offset..k_offset + kv_dim];
            let (max_diff, _, _) = compare_f16_slices(legacy_k_bias, &dsl_k_bias, 0.1);
            eprintln!("  K bias comparison: max_diff={:.6}", max_diff);
            if max_diff > 0.1 {
                eprintln!("    Legacy K bias first 5: {:?}", &legacy_k_bias[..5]);
                eprintln!("    DSL K bias first 5:    {:?}", &dsl_k_bias[..5.min(dsl_k_bias.len())]);
            }
        } else {
            eprintln!("  ⚠️ Legacy bias tensor too short: {}", legacy_bias_full.len());
        }
    } else {
        eprintln!("  ⚠️ Could not retrieve 'layer.attn_k_bias_0' from bindings");
    }

    eprintln!("\n--- Q/K/V Path Analysis ---");

    // Compare Q projection output (before KvRearrange)
    let dsl_q_proj = read_f16_buffer(&bindings.get("q").unwrap());
    eprintln!(
        "  DSL q (before rearrange) len: {}, first 5: {:?}",
        dsl_q_proj.len(),
        &dsl_q_proj[..5.min(dsl_q_proj.len())].iter().map(|x| x.to_f32()).collect::<Vec<_>>()
    );

    // Check if the anomaly is in q_proj or introduced by KvRearrange
    let dsl_q_heads = read_f16_buffer(&bindings.get("q_heads").unwrap());
    let dsl_q_rot = read_f16_buffer(&bindings.get("q_rot").unwrap());
    eprintln!(
        "  DSL q_heads len: {}, first 5: {:?}",
        dsl_q_heads.len(),
        &dsl_q_heads[..5.min(dsl_q_heads.len())]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>()
    );
    eprintln!(
        "  DSL q_rot len: {}, first 5: {:?}",
        &dsl_q_rot.len(),
        &dsl_q_rot[..5.min(dsl_q_rot.len())].iter().map(|x| x.to_f32()).collect::<Vec<_>>()
    );

    // Find max absolute value in q_proj vs q_heads to confirm if anomaly comes from GEMV
    let q_proj_max = dsl_q_proj.iter().map(|x| x.to_f32().abs()).fold(0.0f32, f32::max);
    let q_heads_max = dsl_q_heads.iter().map(|x| x.to_f32().abs()).fold(0.0f32, f32::max);
    eprintln!("  q (proj) max abs: {:.4}, q_heads max abs: {:.4}", q_proj_max, q_heads_max);

    // Compare Legacy vs DSL SDPA inputs
    eprintln!("\n--- Legacy vs DSL SDPA Input Comparison ---");

    // Q comparison (after RoPE)
    let dsl_q_rot_slice = &dsl_q_rot[..d_model.min(dsl_q_rot.len())];
    let legacy_q_rot_slice = &legacy_out.q_rot[..d_model.min(legacy_out.q_rot.len())];
    let (q_max, q_avg, _) = compare_f16_slices(legacy_q_rot_slice, dsl_q_rot_slice, 0.1);
    eprintln!(
        "  Q_rot: Legacy len={}, DSL len={}, max_diff={:.6}, avg_diff={:.6}",
        legacy_out.q_rot.len(),
        dsl_q_rot.len(),
        q_max,
        q_avg
    );
    eprintln!(
        "    Legacy first 5: {:?}",
        &legacy_out.q_rot[..5.min(legacy_out.q_rot.len())]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>()
    );

    // K comparison (after repeat)
    let dsl_k_expanded = read_f16_buffer(&bindings.get("k_expanded").unwrap());
    let k_compare_len = d_model.min(dsl_k_expanded.len()).min(legacy_out.k_expanded.len());
    let (k_max, k_avg, _) = compare_f16_slices(&legacy_out.k_expanded[..k_compare_len], &dsl_k_expanded[..k_compare_len], 0.1);
    eprintln!(
        "  K_expanded: Legacy len={}, DSL len={}, max_diff={:.6}, avg_diff={:.6}",
        legacy_out.k_expanded.len(),
        dsl_k_expanded.len(),
        k_max,
        k_avg
    );
    eprintln!(
        "    Legacy first 5: {:?}",
        &legacy_out.k_expanded[..5.min(legacy_out.k_expanded.len())]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>()
    );
    eprintln!(
        "    DSL first 5: {:?}",
        &dsl_k_expanded[..5.min(dsl_k_expanded.len())]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>()
    );

    // V comparison (after repeat)
    let dsl_v_expanded = read_f16_buffer(&bindings.get("v_expanded").unwrap());
    let v_compare_len = d_model.min(dsl_v_expanded.len()).min(legacy_out.v_expanded.len());
    let (v_max, v_avg, _) = compare_f16_slices(&legacy_out.v_expanded[..v_compare_len], &dsl_v_expanded[..v_compare_len], 0.1);
    eprintln!(
        "  V_expanded: Legacy len={}, DSL len={}, max_diff={:.6}, avg_diff={:.6}",
        legacy_out.v_expanded.len(),
        dsl_v_expanded.len(),
        v_max,
        v_avg
    );
    eprintln!(
        "    Legacy first 5: {:?}",
        &legacy_out.v_expanded[..5.min(legacy_out.v_expanded.len())]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>()
    );
    eprintln!(
        "    DSL first 5: {:?}",
        &dsl_v_expanded[..5.min(dsl_v_expanded.len())]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>()
    );

    // Identify divergence source
    if q_max < 0.001 && k_max < 0.001 && v_max < 0.001 {
        eprintln!("  ✅ Q/K/V inputs match! Divergence must be in SDPA kernel itself");
    } else {
        eprintln!("  ❌ Q/K/V inputs differ - investigating projection weights/bias");
    }

    // Investigate Q weight and bias binding
    eprintln!("\n--- Q Weight/Bias Investigation ---");
    let q_weight = bindings.get("layer.attn_q_0").unwrap();
    let q_bias = bindings.get("layer.attn_q_bias_0").unwrap();
    eprintln!("  Q weight dims: {:?}, dtype: {:?}", q_weight.dims(), q_weight.dtype());
    eprintln!("  Q bias dims: {:?}, dtype: {:?}", q_bias.dims(), q_bias.dtype());

    let q_bias_data = read_f16_buffer(&q_bias);
    let q_bias_max = q_bias_data.iter().map(|x| x.to_f32().abs()).fold(0.0f32, f32::max);
    eprintln!(
        "  Q bias first 5: {:?}",
        &q_bias_data[..5.min(q_bias_data.len())]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>()
    );
    eprintln!("  Q bias max abs: {:.4}", q_bias_max);

    // Check norm_out (input to Q GEMV)
    let norm_out_data = read_f16_buffer(&bindings.get("norm_out").unwrap());
    let norm_out_max = norm_out_data.iter().map(|x| x.to_f32().abs()).fold(0.0f32, f32::max);
    eprintln!(
        "  norm_out (GEMV input) first 5: {:?}",
        &norm_out_data[..5.min(norm_out_data.len())]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>()
    );
    eprintln!("  norm_out max abs: {:.4}", norm_out_max);

    // Compare with Legacy Q bias
    eprintln!("\n--- Legacy Q Bias Comparison ---");
    let legacy_q_bias = &legacy_model.blocks[0].attn_qkv_bias;
    let legacy_q_bias_slice = legacy_q_bias.as_slice();
    let legacy_q_bias_first5: Vec<f32> = legacy_q_bias_slice[..5.min(legacy_q_bias_slice.len())]
        .iter()
        .map(|x: &f16| x.to_f32())
        .collect();
    let legacy_q_bias_max = legacy_q_bias_slice.iter().map(|x: &f16| x.to_f32().abs()).fold(0.0f32, f32::max);
    eprintln!(
        "  Legacy attn_qkv_bias dims: {:?}, len: {}",
        legacy_q_bias.dims(),
        legacy_q_bias.len()
    );
    eprintln!(
        "  Legacy Q bias first 5 (from attn_qkv_bias[0..d_model]): {:?}",
        &legacy_q_bias_first5
    );
    eprintln!("  Legacy attn_qkv_bias max abs: {:.4}", legacy_q_bias_max);

    eprintln!("Layer0 ffn_norm_out diff: max_diff={:.4}, avg_diff={:.6}", norm_max, norm_avg);
    eprintln!("Layer0 ffn_norm_gamma diff: max_diff={:.4}, avg_diff={:.6}", gamma_max, gamma_avg);
    eprintln!("Layer0 gate diff: max_diff={:.4}, avg_diff={:.6}", gate_max, gate_avg);
    eprintln!("Layer0 swiglu_out diff: max_diff={:.4}, avg_diff={:.6}", swiglu_max, swiglu_avg);
    eprintln!("Layer0 proj_out diff: max_diff={:.4}, avg_diff={:.6}", proj_max, proj_avg);
    eprintln!("Layer0 residual_1 diff: max_diff={:.4}, avg_diff={:.6}", res_max, res_avg);
    eprintln!("Layer0 ffn_out diff: max_diff={:.4}, avg_diff={:.6}", ffn_max, ffn_avg);
    eprintln!("Layer0 hidden diff: max_diff={:.4}, avg_diff={:.6}", hid_max, hid_avg);

    // =========================================================================
    // Cross-path RmsNorm test: Run DSL RmsNorm on Legacy's residual_1
    // =========================================================================
    eprintln!("\n--- Cross-Path RmsNorm Analysis ---");

    // Compute RMS values for comparison
    let compute_rms = |data: &[f16]| -> f32 {
        let sum_sq: f32 = data.iter().map(|x| x.to_f32().powi(2)).sum();
        (sum_sq / data.len() as f32 + 1e-6).sqrt()
    };

    let legacy_rms = compute_rms(&legacy_out.residual_1);
    let dsl_rms = compute_rms(&residual_1);
    eprintln!("  Legacy residual_1 RMS: {:.6}", legacy_rms);
    eprintln!("  DSL residual_1 RMS: {:.6}", dsl_rms);
    eprintln!("  RMS diff: {:.6}", (legacy_rms - dsl_rms).abs());

    // Run DSL RmsNorm kernel on Legacy's residual_1 data
    use metallic_foundry::{
        metals::rmsnorm::{RmsNorm as RmsNormKernel, RmsNormParamsResolved}, storage::Pooled, tensor::{F16, Tensor as FoundryTensor}
    };

    let legacy_res1_tensor = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![d_model],
        metallic_foundry::tensor::TensorInit::CopyFrom(&legacy_out.residual_1),
    )
    .unwrap();
    let cross_output =
        FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![d_model], metallic_foundry::tensor::TensorInit::Uninitialized).unwrap();

    let cross_params = RmsNormParamsResolved {
        feature_dim: d_model as u32,
        total_elements: d_model as u32,
    };
    let cross_kernel = RmsNormKernel::new(
        &metallic_foundry::types::TensorArg::from_tensor(&legacy_res1_tensor),
        &metallic_foundry::types::TensorArg::from_tensor(&cross_output),
        &bindings.get("layer.ffn_norm_0").unwrap(),
        cross_params,
    );
    foundry.run(&cross_kernel).unwrap();

    let cross_result = FoundryTensor::to_vec(&cross_output, &foundry);
    let cross_f16: Vec<f16> = cross_result.into_iter().collect();

    // Compare: DSL kernel on Legacy input vs Legacy output
    let (cross_max, cross_avg, _) = compare_f16_slices(&legacy_out.ffn_norm_out, &cross_f16, 0.01);
    eprintln!(
        "  DSL RmsNorm on Legacy input vs Legacy output: max_diff={:.4}, avg_diff={:.6}",
        cross_max, cross_avg
    );

    // Compare: DSL kernel on Legacy input vs DSL's ffn_norm_out
    let (cross_dsl_max, cross_dsl_avg, _) = compare_f16_slices(&ffn_norm_out, &cross_f16, 0.01);
    eprintln!(
        "  DSL RmsNorm on Legacy input vs DSL ffn_norm_out: max_diff={:.4}, avg_diff={:.6}",
        cross_dsl_max, cross_dsl_avg
    );

    if cross_max < 0.01 {
        eprintln!("  ✅ DSL RmsNorm kernel matches Legacy when given same input!");
        eprintln!("  → Divergence is due to INPUT difference (residual_1), not kernel");
    } else {
        eprintln!("  ❌ DSL RmsNorm kernel differs from Legacy even on same input");
        eprintln!("  → Need to investigate kernel implementation difference");
    }

    // =========================================================================
    // Cross-path FFN Gate GEMV test: Run DSL GemvColMajor on Legacy's ffn_norm_out
    // =========================================================================
    eprintln!("\n--- Cross-Path FFN Gate GEMV Analysis ---");

    // Create input tensor from Legacy's ffn_norm_out
    let legacy_ffn_norm_tensor = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![1, d_model],
        metallic_foundry::tensor::TensorInit::CopyFrom(&legacy_out.ffn_norm_out),
    )
    .unwrap();
    let cross_gate_output = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![4864], // ff_dim for qwen2.5-0.5b
        metallic_foundry::tensor::TensorInit::Uninitialized,
    )
    .unwrap();

    // Run GemvV2 (RowMajor) with same weights as DSL uses
    use std::sync::Arc;

    use metallic_foundry::{
        compound::stages::Layout, metals::gemv::{GemvStrategy, GemvV2Args, get_gemv_v2_kernel, warp_dispatch_config}, policy::f16::PolicyF16
    };

    let gate_weight = bindings.get("layer.ffn_gate_0").unwrap();
    let gate_dims = gate_weight.dims();
    eprintln!("  Gate weight dims: {:?}", gate_dims);

    // Compare Legacy vs DSL gate weight
    let dsl_gate_weight_data = read_f16_buffer(&gate_weight);
    if let Some(legacy_gate) = &legacy_model.blocks[0].ffn_gate {
        let legacy_gate_slice = legacy_gate.as_slice();
        eprintln!(
            "  Legacy gate weight len: {}, DSL gate weight len: {}",
            legacy_gate_slice.len(),
            dsl_gate_weight_data.len()
        );
        if legacy_gate_slice.len() == dsl_gate_weight_data.len() {
            let (w_max, w_avg, _) = compare_f16_slices(legacy_gate_slice, &dsl_gate_weight_data, 0.01);
            eprintln!("  Gate weight diff: max_diff={:.4}, avg_diff={:.6}", w_max, w_avg);
        }
        eprintln!(
            "  Legacy gate first 5: {:?}",
            &legacy_gate_slice[..5.min(legacy_gate_slice.len())]
                .iter()
                .map(|x: &f16| x.to_f32())
                .collect::<Vec<_>>()
        );
        eprintln!(
            "  DSL gate first 5: {:?}",
            &dsl_gate_weight_data[..5.min(dsl_gate_weight_data.len())]
                .iter()
                .map(|x: &f16| x.to_f32())
                .collect::<Vec<_>>()
        );
    } else {
        eprintln!("  Legacy ffn_gate is None (using canonical.unwrap())");
        if let Some(legacy_canon) = &legacy_model.blocks[0].ffn_gate_canon {
            eprintln!("  Legacy ffn_gate_canon dims: {:?}", legacy_canon.logical_dims);
        }
    }

    // Skip cross-path GEMV test for canonical weights (1D layout)
    // Cross-path test was designed for row-major 2D weights
    if gate_dims.len() < 2 {
        eprintln!("  [Skipping cross-path test - weights are in canonical 1D format]");
        eprintln!("  → DSL now uses GemvCanonical for FFN projections");
        return Ok(());
    }

    // RowMajor: dims[0] = K, dims[1] = N (GGUF stores [K, N])
    let k_dim = gate_dims[0] as u32;
    let n_dim = gate_dims[1] as u32;

    let zero_buf = bindings.get("zero").unwrap();

    let args_gate = GemvV2Args {
        weights: gate_weight.clone(),
        scale_bytes: gate_weight.clone(), // Dummy (F16)
        input: metallic_foundry::types::TensorArg::from_tensor(&legacy_ffn_norm_tensor),
        output: metallic_foundry::types::TensorArg::from_tensor(&cross_gate_output),
        k_dim,
        n_dim,
        weights_per_block: 32,  // Unused for F16
        bias: zero_buf.clone(), // Zero bias
        has_bias: 0,
        alpha: 1.0,
        residual: metallic_foundry::types::TensorArg::from_tensor(&cross_gate_output),
        has_residual: 0,
        beta: 0.0,
    };

    let kernel = get_gemv_v2_kernel(Arc::new(PolicyF16), Layout::RowMajor, GemvStrategy::Vectorized);
    foundry.run(&kernel.bind(args_gate, warp_dispatch_config(n_dim))).unwrap();

    let cross_gate_result = FoundryTensor::to_vec(&cross_gate_output, &foundry);
    let cross_gate_f16: Vec<f16> = cross_gate_result.into_iter().collect();

    // Compare with Legacy gate output
    let (gate_cross_max, gate_cross_avg, _) = compare_f16_slices(&legacy_out.gate, &cross_gate_f16, 0.1);
    eprintln!(
        "  DSL GemvColMajor on Legacy ffn_norm_out vs Legacy gate: max_diff={:.4}, avg_diff={:.6}",
        gate_cross_max, gate_cross_avg
    );

    if gate_cross_max < 0.1 {
        eprintln!("  ✅ FFN Gate GEMV matches Legacy when given same input!");
        eprintln!("  → Gate divergence comes from ffn_norm_out input difference");
    } else {
        eprintln!("  ❌ FFN Gate GEMV differs from Legacy even on same input!");
        eprintln!("  → Need to investigate GEMV kernel or weight layout");
    }

    Ok(())
}

#[test]
#[serial]
#[ignore = "requires qwen2.5 model file"]
fn test_dsl_vs_context_full_forward_parity() -> Result<(), MetalError> {
    eprintln!("\n=== DSL vs Context Full Forward Parity Test ===\n");

    // =========================================================================
    // STEP 1: Load both models
    // =========================================================================
    let mut ctx = Context::<F16Element>::new().unwrap();
    let legacy_gguf_file = metallic_context::gguf::GGUFFile::load_mmap_and_get_metadata(GGUF_PATH)
        .map_err(|e| MetalError::InvalidShape(format!("GGUF load failed: {}", e)))
        .unwrap();
    let legacy_loader = metallic_context::gguf::model_loader::GGUFModelLoader::new(legacy_gguf_file);
    let legacy_gguf_model = legacy_loader
        .load_model()
        .map_err(|e| MetalError::InvalidShape(format!("GGUF model failed: {}", e)))
        .unwrap();
    let legacy_model: Qwen25<F16Element> = legacy_gguf_model
        .instantiate(&mut ctx)
        .map_err(|e| MetalError::InvalidShape(format!("Model instantiate failed: {}", e)))
        .unwrap();

    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let mut foundry = Foundry::new().unwrap();
    let dsl_model = ModelBuilder::new()
        .with_spec_file(&spec_path)
        .unwrap()
        .with_gguf(GGUF_PATH)
        .unwrap()
        .build(&mut foundry)
        .unwrap();
    let (mut bindings, _fast_bindings) = dsl_model.prepare_bindings(&mut foundry).unwrap();

    eprintln!("✅ Both models loaded");

    // =========================================================================
    // STEP 2: Encode prompt
    // =========================================================================
    let tokenizer = dsl_model.tokenizer().unwrap();
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt).unwrap();
    eprintln!("Prompt: '{}' -> tokens: {:?}", prompt, tokens);

    // =========================================================================
    // STEP 3: Run Legacy full forward
    // =========================================================================
    eprintln!("\n--- Running Legacy Forward ---");
    let legacy_embedded = legacy_model.embed(&tokens, &mut ctx).unwrap();
    eprintln!("Legacy embedding shape: {:?}", legacy_embedded.dims());

    let legacy_hidden_pre = legacy_forward_blocks_only(&legacy_model, &legacy_embedded, &mut ctx).unwrap();
    ctx.synchronize();
    let legacy_hidden = legacy_model.forward(&legacy_embedded, &mut ctx).unwrap();
    ctx.synchronize();
    eprintln!("Legacy hidden shape: {:?}", legacy_hidden.dims());

    let legacy_logits = legacy_model.output(&legacy_hidden, &mut ctx).unwrap();
    ctx.synchronize();
    eprintln!("Legacy logits shape: {:?}", legacy_logits.dims());

    let legacy_logits_data: Vec<f16> = legacy_logits.as_slice().iter().map(|v: &f16| f16::from_f32(v.to_f32())).collect();
    let legacy_hidden_pre_data = tensor_to_f16_vec(&legacy_hidden_pre);
    let legacy_hidden_data = tensor_to_f16_vec(&legacy_hidden);
    eprintln!(
        "Legacy logits first 10: {:?}",
        &legacy_logits_data[..10.min(legacy_logits_data.len())]
    );

    // =========================================================================
    // STEP 4: Run DSL full forward
    // =========================================================================
    eprintln!("\n--- Running DSL Forward ---");

    // Set up input_ids
    let input_buffer = {
        use objc2_metal::MTLDevice;
        let byte_size = tokens.len() * 4;
        let buf = foundry
            .device
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate input buffer");
        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, tokens.len());
        }
        metallic_foundry::types::MetalBuffer::from_retained(buf)
    };
    let input_tensor =
        metallic_foundry::types::TensorArg::from_buffer(input_buffer, metallic_foundry::tensor::Dtype::U32, vec![tokens.len()], vec![1]);
    bindings.insert("input_ids".to_string(), input_tensor);

    // Set all required globals
    let arch = dsl_model.architecture();
    let seq_len = tokens.len();
    let d_model = arch.d_model;
    let n_heads = arch.n_heads;
    let n_kv_heads = arch.n_kv_heads;
    let ff_dim = arch.ff_dim;
    let head_dim = d_model / n_heads;
    let position_offset = 0usize;
    let kv_seq_len = position_offset + seq_len;

    bindings.set_global("seq_len", seq_len.to_string());
    bindings.set_global("position_offset", position_offset.to_string());
    bindings.set_global("kv_seq_len", kv_seq_len.to_string());
    bindings.set_global("total_elements_q", (n_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_k", (n_kv_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_hidden", (seq_len * d_model).to_string());
    bindings.set_global("total_elements_ffn", (seq_len * ff_dim).to_string());
    bindings.set_global("total_elements_slice", (n_kv_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_repeat", (n_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_write", (n_kv_heads * seq_len * head_dim).to_string());
    eprintln!("Set globals: seq_len={}, total_elements_hidden={}", seq_len, seq_len * d_model);

    // Execute DSL steps one by one and compare after each
    eprintln!("\n--- Step-by-step DSL execution ---");
    for (step_idx, step) in arch.forward.iter().enumerate() {
        let step_name = step.name();
        eprintln!("[Step {}] {}", step_idx, step_name);

        step.execute(&mut foundry, &mut bindings).unwrap();

        // After embedding, compare with legacy embedding
        if step_name == "Embedding" {
            let dsl_hidden = bindings.get("hidden").unwrap();
            let dsl_hidden_data = read_f16_buffer(&dsl_hidden);
            let legacy_embed_data: Vec<f16> = legacy_embedded.as_slice().iter().map(|v: &f16| f16::from_f32(v.to_f32())).collect();

            let (max_diff, avg_diff, first_mismatch) = compare_f16_slices(&legacy_embed_data, &dsl_hidden_data, 0.001);
            eprintln!(
                "  → hidden: max_diff={:.6}, avg_diff={:.8}, mismatch={:?}",
                max_diff, avg_diff, first_mismatch
            );
            if max_diff > 0.01 {
                eprintln!("  ❌ DIVERGENCE at Embedding!");
                eprintln!("  Legacy[0..5]: {:?}", &legacy_embed_data[..5.min(legacy_embed_data.len())]);
                eprintln!("  DSL[0..5]: {:?}", &dsl_hidden_data[..5.min(dsl_hidden_data.len())]);
                return Err(MetalError::InvalidShape("Embedding divergence".into()));
            }
            eprintln!("  ✅ Embedding matches");
        }

        // After all layers (Repeat), this is *pre-final RMSNorm* in the DSL path.
        // Legacy `forward` returns *post-final RMSNorm*, so a direct comparison here
        // will always show a diff. We log stats but compare after the final RmsNorm step.
        if step_name == "Repeat" {
            let dsl_hidden = bindings.get("hidden").unwrap();
            let dsl_hidden_data = read_f16_buffer(&dsl_hidden);
            let (max_diff, avg_diff, first_mismatch) = compare_f16_slices(&legacy_hidden_pre_data, &dsl_hidden_data, 0.1);
            eprintln!(
                "  → hidden after layers (pre-final norm): max_diff={:.4}, avg_diff={:.6}, mismatch={:?}",
                max_diff, avg_diff, first_mismatch
            );
            if max_diff > 1.0 {
                eprintln!("  ❌ DIVERGENCE in transformer layers (pre-final norm)!");
                eprintln!(
                    "  Legacy[0..5]: {:?}",
                    &legacy_hidden_pre_data[..5.min(legacy_hidden_pre_data.len())]
                );
                eprintln!("  DSL[0..5]: {:?}", &dsl_hidden_data[..5.min(dsl_hidden_data.len())]);
            }
        }

        // After final RmsNorm, compare with legacy hidden
        if step_name == "RmsNorm" {
            let dsl_final = match bindings.get("final_norm_out") {
                Ok(arg) => arg,
                Err(_) => bindings.get("hidden").unwrap(),
            };
            let dsl_hidden_data = read_f16_buffer(&dsl_final);
            let (max_diff, avg_diff, first_mismatch) = compare_f16_slices(&legacy_hidden_data, &dsl_hidden_data, 0.1);
            eprintln!(
                "  → hidden after final norm: max_diff={:.4}, avg_diff={:.6}, mismatch={:?}",
                max_diff, avg_diff, first_mismatch
            );
            if max_diff > 1.0 {
                eprintln!("  ❌ DIVERGENCE after final RMSNorm!");
                eprintln!("  Legacy[0..5]: {:?}", &legacy_hidden_data[..5.min(legacy_hidden_data.len())]);
                eprintln!("  DSL[0..5]: {:?}", &dsl_hidden_data[..5.min(dsl_hidden_data.len())]);
            }
        }
    }

    eprintln!("\n--- DSL step-by-step completed ---");

    // Read logits for final comparison
    let dsl_logits = bindings.get("logits").unwrap();
    let dsl_logits_data = read_f16_buffer(&dsl_logits);
    eprintln!("DSL logits shape: {:?}", dsl_logits.dims());
    eprintln!("DSL logits first 10: {:?}", &dsl_logits_data[..10.min(dsl_logits_data.len())]);

    // =========================================================================
    // STEP 5: Compare logits
    // =========================================================================
    eprintln!("\n--- Comparing Logits ---");
    let (max_diff, avg_diff, first_mismatch) = compare_f16_slices(&legacy_logits_data, &dsl_logits_data, 0.1);
    eprintln!(
        "Logits comparison: max_diff={:.4}, avg_diff={:.6}, first_mismatch={:?}",
        max_diff, avg_diff, first_mismatch
    );

    if max_diff > 0.5 {
        eprintln!("⚠️ Logits divergence detected - need to check intermediate stages");

        // Check hidden state after forward (before output projection)
        let dsl_hidden = bindings.get("hidden").unwrap();
        let dsl_hidden_data = read_f16_buffer(&dsl_hidden);
        let legacy_hidden_data: Vec<f16> = legacy_hidden.as_slice().iter().map(|v: &f16| f16::from_f32(v.to_f32())).collect();

        let (hidden_max, hidden_avg, hidden_mismatch) = compare_f16_slices(&legacy_hidden_data, &dsl_hidden_data, 0.01);
        eprintln!(
            "Hidden state comparison: max_diff={:.4}, avg_diff={:.6}, first_mismatch={:?}",
            hidden_max, hidden_avg, hidden_mismatch
        );
    } else {
        eprintln!("✅ Full forward parity PASSED");
    }

    Ok(())
}

/// Comprehensive block-level parity test comparing DSL vs Legacy at each sub-step.
/// This test runs layer 0 through both paths and compares intermediate outputs.
#[test]
#[serial]
#[ignore = "requires qwen2.5 model file"]
fn test_full_block_step_parity() -> Result<(), MetalError> {
    eprintln!("\n========================================");
    eprintln!("  BLOCK-LEVEL DSL vs LEGACY PARITY TEST");
    eprintln!("========================================\n");

    // =========================================================================
    // STEP 1: Load both models
    // =========================================================================
    eprintln!("[1/4] Loading Legacy model...");
    let mut ctx = Context::<F16Element>::new().unwrap();
    let legacy_gguf_file = metallic_context::gguf::GGUFFile::load_mmap_and_get_metadata(GGUF_PATH)
        .map_err(|e| MetalError::InvalidShape(format!("GGUF load failed: {}", e)))
        .unwrap();
    let legacy_loader = metallic_context::gguf::model_loader::GGUFModelLoader::new(legacy_gguf_file);
    let legacy_gguf_model = legacy_loader
        .load_model()
        .map_err(|e| MetalError::InvalidShape(format!("GGUF model failed: {}", e)))
        .unwrap();
    let legacy_model: Qwen25<F16Element> = legacy_gguf_model
        .instantiate(&mut ctx)
        .map_err(|e| MetalError::InvalidShape(format!("Model instantiate failed: {}", e)))
        .unwrap();

    eprintln!("[2/4] Loading DSL model...");
    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let mut foundry = Foundry::new().unwrap();
    let dsl_model = ModelBuilder::new()
        .with_spec_file(&spec_path)
        .unwrap()
        .with_gguf(GGUF_PATH)
        .unwrap()
        .build(&mut foundry)
        .unwrap();
    let (mut bindings, mut fast_bindings) = dsl_model.prepare_bindings(&mut foundry).unwrap();

    // =========================================================================
    // STEP 2: Create shared input (single token for consistent shapes)
    // =========================================================================
    eprintln!("[3/4] Creating test input...");
    let tokenizer = dsl_model.tokenizer().unwrap();
    let prompt = "H"; // Single character for single token
    let tokens = tokenizer.encode(prompt).unwrap();
    let tokens = &tokens[0..1]; // Ensure single token
    eprintln!("  Prompt: '{}' -> tokens: {:?}", prompt, tokens);

    let d_model = legacy_model.config.d_model;
    let _batch = 1;
    let seq_len = tokens.len();

    // Get embeddings from legacy model
    let legacy_input = legacy_model.embed(tokens, &mut ctx).unwrap();
    ctx.synchronize();
    let legacy_embed_data = tensor_to_f16_vec(&legacy_input);

    // =========================================================================
    // STEP 3: Run Legacy layer 0
    // =========================================================================
    eprintln!("[4/4] Running layer 0 forward passes...\n");
    let legacy_out = legacy_layer0_outputs(&legacy_model, &legacy_input, &mut ctx).unwrap();

    // =========================================================================
    // STEP 4: Run DSL layer 0
    // =========================================================================
    let arch = dsl_model.architecture();
    if arch.d_model != d_model {
        return Err(MetalError::InvalidShape(format!(
            "DSL d_model {} != legacy d_model {}",
            arch.d_model, d_model
        )));
    }
    let n_heads = arch.n_heads;
    let n_kv_heads = arch.n_kv_heads;
    let ff_dim = arch.ff_dim;
    let head_dim = d_model / n_heads;

    let position_offset = 0usize;
    let kv_seq_len = position_offset + seq_len;

    // Force Repeat to run a single layer so intermediate buffers reflect layer 0.
    bindings.set_global("n_layers", "1".to_string());

    // Required globals for dynamic dispatch in the qwen25.json spec.
    bindings.set_global("seq_len", seq_len.to_string());
    bindings.set_global("position_offset", position_offset.to_string());
    bindings.set_global("kv_seq_len", kv_seq_len.to_string());
    bindings.set_global("max_seq_len", "2048".to_string());
    bindings.set_global("total_elements_q", (n_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_k", (n_kv_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_hidden", (seq_len * d_model).to_string());
    bindings.set_global("total_elements_ffn", (seq_len * ff_dim).to_string());
    bindings.set_global("total_elements_slice", (n_kv_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_repeat", (n_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_write", (n_kv_heads * seq_len * head_dim).to_string());

    // Set up input_ids for DSL
    let input_buffer = {
        use objc2_metal::MTLDevice;
        let byte_size = tokens.len() * 4;
        let buf = foundry
            .device
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate input buffer");
        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, tokens.len());
        }
        metallic_foundry::types::MetalBuffer::from_retained(buf)
    };
    let input_tensor =
        metallic_foundry::types::TensorArg::from_buffer(input_buffer, metallic_foundry::tensor::Dtype::U32, vec![tokens.len()], vec![1]);

    // Bind input_ids to both binding types
    if let Some(id) = dsl_model.symbol_id("input_ids") {
        fast_bindings.set(id, input_tensor.clone());
    }
    bindings.insert("input_ids".to_string(), input_tensor);

    // Embedding parity (run embedding step directly)
    {
        let embedding_step = arch
            .forward
            .first()
            .ok_or_else(|| MetalError::InvalidShape("DSL architecture.forward is empty".into()))
            .unwrap();
        if embedding_step.name() != "Embedding" {
            return Err(MetalError::InvalidShape(format!(
                "Expected first DSL step to be Embedding, got '{}'",
                embedding_step.name()
            )));
        }

        embedding_step.execute(&mut foundry, &mut bindings).unwrap();
        let dsl_hidden = bindings.get("hidden").unwrap();
        let dsl_hidden_data = read_f16_buffer(&dsl_hidden);
        let (max_diff, avg_diff, first_mismatch) = compare_f16_slices(&legacy_embed_data, &dsl_hidden_data, 0.0);
        eprintln!(
            "Embedding parity: max_diff={:.6} avg_diff={:.8} mismatch={:?}",
            max_diff, avg_diff, first_mismatch
        );
        eprintln!("   Legacy Hidden[0..5]: {:?}", &legacy_embed_data[..5]);
        eprintln!("   DSL Hidden[0..5]:    {:?}", &dsl_hidden_data[..5]);
        if max_diff != 0.0 {
            return Err(MetalError::InvalidShape("Embedding parity FAILED".into()));
        }
    }

    // Run full DSL forward pass (embedding + layer 0)
    // Use forward_uncompiled so the n_layers=1 override works (compiled steps are pre-unrolled)
    dsl_model.forward_uncompiled(&mut foundry, &mut bindings).unwrap();
    ctx.synchronize();

    // =========================================================================
    // STEP 5: Compare intermediate outputs at layer 0
    // =========================================================================
    eprintln!("--- STEP-BY-STEP COMPARISON ---\n");

    // Check attn_q weight binding
    if let Ok(attn_q) = bindings.get("layer.attn_q_0") {
        let attn_q_data = read_f16_buffer(&attn_q);
        eprintln!(
            "DSL attn_q_0 dims: {:?}, first 5: {:?}",
            attn_q.dims(),
            &attn_q_data[..5.min(attn_q_data.len())]
        );
        let non_zero = attn_q_data.iter().filter(|v| v.to_f32().abs() > 1e-6).count();
        eprintln!("DSL attn_q_0 non-zero values: {} / {}\n", non_zero, attn_q_data.len());
    } else {
        eprintln!("⚠️ DSL attn_q_0 binding not found!\n");
    }

    let tolerance = 0.01f32;
    let strict_tolerance = 0.001f32;
    let mut all_passed = true;

    let mut compare = |name: &str, dsl_name: &str, legacy: &[f16], strict: bool| {
        let tol = if strict { strict_tolerance } else { tolerance };
        match bindings.get(dsl_name) {
            Ok(arg) => {
                let dsl = read_f16_buffer(&arg);
                if dsl.len() < legacy.len() {
                    eprintln!(
                        "❌ FAIL {:30} length mismatch: legacy_len={} dsl_len={}",
                        name,
                        legacy.len(),
                        dsl.len()
                    );
                    all_passed = false;
                    return;
                }
                let dsl_view = &dsl[..legacy.len()];
                let (max_diff, avg_diff, first_mismatch) = compare_f16_slices(legacy, dsl_view, tol);
                let passed = max_diff <= tol;
                let status = if passed { "✅ PASS" } else { "❌ FAIL" };
                eprintln!(
                    "{} {:30} max_diff={:.6} avg_diff={:.8} mismatch={:?}",
                    status, name, max_diff, avg_diff, first_mismatch
                );
                if !passed {
                    eprintln!("   Legacy[0..5]: {:?}", &legacy[..5.min(legacy.len())]);
                    eprintln!("   DSL[0..5]:    {:?}", &dsl_view[..5.min(dsl_view.len())]);
                    all_passed = false;
                }
            }
            Err(e) => {
                eprintln!("⚠️  SKIP {:30} binding '{}' not found: {}", name, dsl_name, e);
            }
        }
    };

    // Q/K/V projection outputs (before rearrange)
    compare("Q projection", "q", &legacy_out.q_proj, false);
    if legacy_out.q_proj.len() > 666 {
        eprintln!("DEBUG Legacy OutQ index 666: {:?}", legacy_out.q_proj[666]);
    }
    compare("K projection", "k", &legacy_out.k_proj, false);
    compare("V projection", "v", &legacy_out.v_proj, false);

    // After RoPE
    compare("Q after RoPE", "q_rot", &legacy_out.q_rot, false);

    // After RepeatKvHeads
    compare("K expanded", "k_expanded", &legacy_out.k_expanded, false);
    compare("V expanded", "v_expanded", &legacy_out.v_expanded, false);

    // Attention output
    compare("Attention output", "attn_out", &legacy_out.attn_out, false);

    // Attention output projection + residual add is fused in the DSL spec:
    // `residual_1 = attn_proj(attn_out) + hidden`
    compare("Attn+Residual (residual_1)", "residual_1", &legacy_out.residual_1, false);

    // FFN norm + gate/up + SwiGLU is fused in the DSL spec (writes into `up`).
    compare("SwiGLU output", "up", &legacy_out.swiglu_out, false);

    // Final hidden state
    compare("Block output (hidden)", "hidden", &legacy_out.hidden, false);

    eprintln!("\n--- SUMMARY ---");
    if all_passed {
        eprintln!("✅ ALL COMPARISONS PASSED");
    } else {
        eprintln!("❌ SOME COMPARISONS FAILED - see above for details");
    }

    Ok(())
}
