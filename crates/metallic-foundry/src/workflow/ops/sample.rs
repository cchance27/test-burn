use crate::{
    error::MetalError, metals::sampling::SampleTopK, types::{MetalResourceOptions, TensorArg}, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext, spec::Param
    }
};

pub(crate) struct SampleOp {
    logits_var: String,
    output_var: String,
    temperature: Param<f32>,
    top_k: Param<u32>,
    top_p: Param<f32>,
    seed: Param<u32>,
}

impl SampleOp {
    pub(crate) fn new(
        logits_var: String,
        output_var: String,
        temperature: Param<f32>,
        top_k: Param<u32>,
        top_p: Param<f32>,
        seed: Param<u32>,
    ) -> Self {
        Self {
            logits_var,
            output_var,
            temperature,
            top_k,
            top_p,
            seed,
        }
    }
}

impl WorkflowOp for SampleOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let logits_arg = ctx
            .values
            .get(&self.logits_var)
            .and_then(|v| v.as_tensor())
            .ok_or_else(|| MetalError::InvalidOperation(format!("SampleOp missing logits variable '{}' (Tensor)", self.logits_var)))?;

        let temp = ctx.resolve_param_f32(&self.temperature)?;
        let top_k = ctx.resolve_param_u32(&self.top_k)?;
        let top_p = ctx.resolve_param_f32(&self.top_p)?;
        let seed = ctx.resolve_param_u32(&self.seed)?;

        // Allocate a small 1-token buffer for the result.
        // DEBT: Reuse this buffer across executions if possible.
        let out_buffer = ctx
            .foundry
            .device
            .new_buffer(4, MetalResourceOptions::StorageModeShared)
            .expect("Failed to allocate sample output buffer");
        let out_arg = TensorArg::from_buffer(out_buffer.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);

        let vocab_size = logits_arg.dims()[logits_arg.dims().len() - 1] as u32;
        let kernel = SampleTopK::new(logits_arg, &out_arg, vocab_size, top_k, top_p, temp, seed);

        ctx.foundry.run(&kernel)?;

        // Synchronize and read back
        let token = out_buffer.read_scalar::<u32>();
        ctx.values.insert(self.output_var.clone(), Value::U32(token));

        Ok(WorkflowOpOutcome::Continue)
    }
}
