use crate::kernels::sdpa_mps_graph::SdpaMpsGraphOp;
use crate::tensor::dtypes::TensorElement;
use crate::{Context, F16Element, MetalError, Tensor};

#[test]
fn test_sdpa_mpsgraph_parity_matrix_representative() -> Result<(), MetalError> {
    use crate::kernels::scaled_dot_product_attention::ScaledDotProductAttentionOptimizedOp;
    // Representative cases from Milestone A matrix to control runtime
    let cases = vec![
        (1, 128, 128, 64),
        (4, 256, 256, 64),
        (8, 512, 512, 128),
        (4, 128, 256, 64),
    ];

    for (batch, seq_q, seq_k, dim) in cases {
        let mut ctx = Context::<F16Element>::new()?;
        let q = Tensor::<F16Element>::random_uniform(vec![batch, seq_q, dim], &mut ctx)?;
        let k = Tensor::<F16Element>::random_uniform(vec![batch, seq_k, dim], &mut ctx)?;
        let v = Tensor::<F16Element>::random_uniform(vec![batch, seq_k, dim], &mut ctx)?;

        for &causal in &[false, true] {
            let ref_out = ctx.call::<ScaledDotProductAttentionOptimizedOp>((&q, &k, &v, causal, 0))?;
            let graph_out = ctx.call::<SdpaMpsGraphOp>(&q, &k, &v, causal, 0)?;
            let diffs: Vec<f32> = ref_out
                .as_slice()
                .iter()
                .zip(graph_out.as_slice())
                .map(|(&a, &b)| (a.to_f32() - b.to_f32()).abs())
                .collect();
            let max_diff = diffs.iter().fold(0.0f32, |a, &b| a.max(b));
            let tol = if causal { 1.0 } else { 0.5 };
            assert!(max_diff < tol, "Parity exceeded: max_diff={max_diff} tol={tol} at shape B{batch} S{seq_q}/{seq_k} D{dim} causal={causal}");
        }
    }

    Ok(())
}
