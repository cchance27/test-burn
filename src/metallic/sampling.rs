use crate::metallic::TensorElement;
use rand::RngCore;

/// Workspace reused across sampling invocations to avoid per-token allocations.
#[derive(Default)]
pub struct SamplerBuffers {
    pub scaled: Vec<f32>,
    pub indices: Vec<usize>,
}

/// Compute the effective top-k value by clamping the requested `top_k` to the
/// vocabulary size and enforcing a minimum of one candidate.
#[inline]
pub fn effective_top_k(top_k: usize, vocab_len: usize) -> usize {
    top_k.max(1).min(vocab_len)
}

/// Sample from logits using top-k and top-p (nucleus) sampling with a fixed RNG draw.
///
/// This is the core implementation shared across CPU and GPU paths. The caller
/// provides a deterministic `random_u32` to avoid consuming RNG state inside the
/// routine, which allows the GPU kernel to share the same randomness and keeps
/// unit tests deterministic.
pub fn sample_top_k_top_p_with_random_value<T: TensorElement>(
    logits: &[T::Scalar],
    top_k: usize,
    top_p: f32,
    temperature: f32,
    random_u32: u32,
    buffers: &mut SamplerBuffers,
) -> usize {
    let mut fallback_idx = 0usize;
    let mut fallback_found = false;
    let mut fallback_val = f32::NEG_INFINITY;
    for (i, &raw) in logits.iter().enumerate() {
        let val = T::to_f32(raw);
        if val.is_finite() && (!fallback_found || val > fallback_val || (val == fallback_val && i > fallback_idx)) {
            fallback_idx = i;
            fallback_val = val;
            fallback_found = true;
        }
    }

    if temperature <= 0.0 || !temperature.is_finite() {
        return if fallback_found { fallback_idx } else { 0 };
    }

    if logits.is_empty() {
        return 0;
    }

    if top_k == 0 {
        return if fallback_found { fallback_idx } else { 0 };
    }

    let effective_top_k = effective_top_k(top_k, logits.len());

    let mut top_k_candidates: Vec<_> = logits.iter().enumerate().map(|(i, &raw)| {
        let val = T::to_f32(raw);
        let scaled_val = val / temperature;
        (scaled_val, i)
    }).filter(|(s, _)| s.is_finite()).collect();

    if top_k_candidates.len() > effective_top_k {
        top_k_candidates.select_nth_unstable_by(effective_top_k - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap().then_with(|| b.1.cmp(&a.1))
        });
        top_k_candidates.truncate(effective_top_k);
    }

    top_k_candidates.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap().then_with(|| b.1.cmp(&a.1))
    });

    let scaled = &mut buffers.scaled;
    scaled.clear();
    scaled.extend(top_k_candidates.iter().map(|(s, _)| *s));

    let indices = &mut buffers.indices;
    indices.clear();
    indices.extend(top_k_candidates.iter().map(|(_, i)| *i));

    if indices.is_empty() {
        return if fallback_found { fallback_idx } else { 0 };
    }

    let mut has_positive = false;
    let max_val = scaled[0];
    let mut total = 0.0f32;
    for val in scaled.iter_mut() {
        if val.is_finite() {
            let mut exp_val = (*val - max_val).exp();
            if exp_val > 1e10 {
                exp_val = 1e10;
            } else if exp_val < 1e-10 {
                exp_val = 0.0;
            }
            *val = exp_val;
        } else {
            *val = 0.0;
        }
        total += *val;
        has_positive |= *val > 0.0;
    }

    if !has_positive || total <= 0.0 || total.is_infinite() || total.is_nan() {
        return if fallback_found { fallback_idx } else { 0 };
    }

    let normalized_top_p = if top_p.is_finite() { top_p.clamp(0.0, 1.0) } else { 1.0 };
    let mut cutoff = indices.len() - 1;
    if normalized_top_p <= 0.0 {
        cutoff = 0;
    } else if normalized_top_p < 1.0 {
        let mut cum = 0.0f32;
        let threshold = normalized_top_p * total;
        for (i, &weight) in scaled.iter().enumerate() {
            cum += weight;
            cutoff = i;
            if cum >= threshold || cum.is_infinite() || cum.is_nan() {
                break;
            }
        }
    }

    scaled.truncate(cutoff + 1);
    indices.truncate(cutoff + 1);

    let shortlist_total: f32 = scaled.iter().sum();
    if shortlist_total <= 0.0 || shortlist_total.is_infinite() || shortlist_total.is_nan() {
        return indices.first().copied().unwrap_or(if fallback_found { fallback_idx } else { 0 });
    }

    for weight in scaled.iter_mut() {
        *weight /= shortlist_total;
    }

    let r = (random_u32 as f32) / (u32::MAX as f32);
    let mut acc = 0.0f32;
    for (&idx, &prob) in indices.iter().zip(scaled.iter()) {
        acc += prob;
        if r <= acc || acc.is_infinite() || acc.is_nan() {
            return idx;
        }
    }

    indices.last().copied().unwrap_or(if fallback_found { fallback_idx } else { 0 })
}

/// Convenience wrapper that sources randomness from the thread-local RNG.
pub fn sample_top_k_top_p<T: TensorElement>(
    logits: &[T::Scalar],
    top_k: usize,
    top_p: f32,
    temperature: f32,
    buffers: &mut SamplerBuffers,
) -> usize {
    let mut rng = rand::rng();
    let random_u32 = rng.next_u32();
    sample_top_k_top_p_with_random_value::<T>(logits, top_k, top_p, temperature, random_u32, buffers)
}
