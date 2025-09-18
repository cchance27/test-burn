use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::{Bool, Float, Shape, Tensor};

pub fn scaled_dot_product_attention<B: Backend>(
    query: Tensor<B, 3, Float>,
    key: Tensor<B, 3, Float>,
    value: Tensor<B, 3, Float>,
    mask: Option<Tensor<B, 3, Bool>>,
    is_causal: bool,
) -> Tensor<B, 3, Float> {
    let d_k = query.dims()[2] as f32;
    let mut attn = query.clone().matmul(key.clone().swap_dims(1, 2)) / d_k.sqrt();

    let device = query.device();

    if is_causal {
        let [batch, seq_q, _] = query.dims();
        let seq_k = key.dims()[1];
        let temp: Tensor<B, 2, Float> =
            Tensor::<B, 2, Float>::ones(Shape::new([seq_q, seq_k]), &device).triu(1);
        let causal_mask: Tensor<B, 2, Bool> = temp.greater_elem(0.0f32);
        let causal_mask = causal_mask.unsqueeze_dim(0).repeat(&[batch, 1, 1]);
        attn = attn.mask_fill(causal_mask, f32::NEG_INFINITY);
    }
    if let Some(m) = mask {
        attn = attn.mask_fill(m, f32::NEG_INFINITY);
    }

    let attn = softmax(attn, 2);

    attn.matmul(value)
}
