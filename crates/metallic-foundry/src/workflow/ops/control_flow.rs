use crate::{
    error::MetalError, types::TensorArg, workflow::{
        ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext, spec::Param
    }
};

pub(crate) struct IfOp {
    condition: String,
    then_ops: Vec<Box<dyn WorkflowOp>>,
    else_ops: Vec<Box<dyn WorkflowOp>>,
}

impl IfOp {
    pub(crate) fn new(condition: String, then_ops: Vec<Box<dyn WorkflowOp>>, else_ops: Vec<Box<dyn WorkflowOp>>) -> Self {
        Self {
            condition,
            then_ops,
            else_ops,
        }
    }
}

impl WorkflowOp for IfOp {
    fn begin_run(&mut self, ctx: &mut WorkflowExecutionContext<'_>) -> Result<(), MetalError> {
        for op in &mut self.then_ops {
            op.begin_run(ctx)?;
        }
        for op in &mut self.else_ops {
            op.begin_run(ctx)?;
        }
        Ok(())
    }

    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let cond_val = ctx
            .values
            .get(&self.condition)
            .ok_or_else(|| MetalError::InvalidOperation(format!("IfOp missing condition variable '{}'", self.condition)))?;

        let should_run_then = match cond_val {
            crate::workflow::Value::Bool(b) => *b,
            crate::workflow::Value::U32(v) => *v != 0,
            crate::workflow::Value::Usize(v) => *v != 0,
            _ => {
                return Err(MetalError::InvalidOperation(format!(
                    "IfOp condition '{}' is not a boolean or integer",
                    self.condition
                )));
            }
        };

        let ops = if should_run_then { &mut self.then_ops } else { &mut self.else_ops };

        for op in ops {
            match op.execute(ctx, on_token)? {
                WorkflowOpOutcome::Continue => {}
                outcome => return Ok(outcome),
            }
        }

        Ok(WorkflowOpOutcome::Continue)
    }
}

pub(crate) struct WhileOp {
    condition: String,
    max_iterations: Option<Param<usize>>,
    body_ops: Vec<Box<dyn WorkflowOp>>,
}

impl WhileOp {
    pub(crate) fn new(condition: String, max_iterations: Option<Param<usize>>, body_ops: Vec<Box<dyn WorkflowOp>>) -> Self {
        Self {
            condition,
            max_iterations,
            body_ops,
        }
    }
}

impl WorkflowOp for WhileOp {
    fn begin_run(&mut self, ctx: &mut WorkflowExecutionContext<'_>) -> Result<(), MetalError> {
        for op in &mut self.body_ops {
            op.begin_run(ctx)?;
        }
        Ok(())
    }

    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let max_iters = if let Some(p) = &self.max_iterations {
            Some(ctx.resolve_param_usize(p)?)
        } else {
            None
        };

        let mut iter = 0;
        loop {
            if let Some(max) = max_iters {
                if iter >= max {
                    break;
                }
            }

            // If the condition is a variable name, we lookup the variable.
            let cond_val = ctx
                .values
                .get(&self.condition)
                .ok_or_else(|| MetalError::InvalidOperation(format!("WhileOp missing condition variable '{}'", self.condition)))?;

            let should_run = match cond_val {
                crate::workflow::Value::Bool(b) => *b,
                crate::workflow::Value::U32(v) => *v != 0,
                crate::workflow::Value::Usize(v) => *v != 0,
                _ => {
                    return Err(MetalError::InvalidOperation(format!(
                        "WhileOp condition '{}' is not a boolean or integer",
                        self.condition
                    )));
                }
            };

            if !should_run {
                break;
            }

            let mut break_loop = false;
            for op in &mut self.body_ops {
                match op.execute(ctx, on_token)? {
                    WorkflowOpOutcome::Continue => {}
                    WorkflowOpOutcome::Return => return Ok(WorkflowOpOutcome::Return),
                    WorkflowOpOutcome::Break => {
                        break_loop = true;
                        break;
                    }
                    WorkflowOpOutcome::LoopContinue => {
                        // Stop executing body, proceed to next iteration
                        break;
                    }
                }
            }

            if break_loop {
                break;
            }

            iter += 1;
        }

        Ok(WorkflowOpOutcome::Continue)
    }
}

fn ignore_eos_stop() -> bool {
    static IGNORE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *IGNORE.get_or_init(|| std::env::var("METALLIC_IGNORE_EOS_STOP").is_ok_and(|v| v != "0"))
}

fn default_decode_batch_size() -> usize {
    static DEFAULT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEFAULT.get_or_init(|| {
        const MAX: usize = 256;
        std::env::var("METALLIC_FOUNDRY_DECODE_BATCH_SIZE")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, MAX)
    })
}

pub(crate) struct WhileBatchedOp {
    condition: String,
    max_iterations: Option<Param<usize>>,
    batch_size: Option<Param<usize>>,
    unsafe_allow_overshoot: bool,
    token_var: String,
    stream_channel: Option<String>,
    output_tokens: String,
    eos_token: Param<u32>,
    body_ops: Vec<Box<dyn WorkflowOp>>,
}

impl WhileBatchedOp {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        condition: String,
        max_iterations: Option<Param<usize>>,
        batch_size: Option<Param<usize>>,
        unsafe_allow_overshoot: bool,
        token_var: String,
        stream_channel: Option<String>,
        output_tokens: String,
        eos_token: Param<u32>,
        body_ops: Vec<Box<dyn WorkflowOp>>,
    ) -> Self {
        Self {
            condition,
            max_iterations,
            batch_size,
            unsafe_allow_overshoot,
            token_var,
            stream_channel,
            output_tokens,
            eos_token,
            body_ops,
        }
    }
}

impl WorkflowOp for WhileBatchedOp {
    fn begin_run(&mut self, ctx: &mut WorkflowExecutionContext<'_>) -> Result<(), MetalError> {
        for op in &mut self.body_ops {
            op.begin_run(ctx)?;
        }
        Ok(())
    }

    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let max_iters = if let Some(p) = &self.max_iterations {
            Some(ctx.resolve_param_usize(p)?)
        } else {
            None
        };

        let mut batch_size = if let Some(p) = &self.batch_size {
            ctx.resolve_param_usize(p)?
        } else {
            default_decode_batch_size()
        };
        batch_size = batch_size.max(1);

        // Internal knobs for downstream ops (e.g. SampleOp) to avoid scalar readback in-batch.
        ctx.values
            .insert("_internal.decode_batch_size".to_string(), crate::workflow::Value::Usize(batch_size));

        let eos = ctx.resolve_param_u32(&self.eos_token)?;
        let stop_on_eos = !ignore_eos_stop();

        let mut stream_reader = if let Some(name) = self.stream_channel.as_deref() {
            let chan =
                ctx.values.get(name).and_then(|v| v.as_channel_u32()).cloned().ok_or_else(|| {
                    MetalError::InvalidOperation(format!("WhileBatchedOp stream_channel '{}' missing (channel_u32)", name))
                })?;
            if (chan.capacity as usize) < batch_size {
                return Err(MetalError::InvalidOperation(format!(
                    "WhileBatchedOp stream_channel '{}' capacity {} < batch_size {}",
                    name, chan.capacity, batch_size
                )));
            }
            Some(crate::workflow::ChannelU32Reader::new(chan))
        } else {
            None
        };
        let mut drained_tokens: Vec<u32> = Vec::with_capacity(batch_size);

        // Guardrail: `batch_size > 1` with EOS stopping enabled can compute "overshoot" tokens into KV.
        // This is fine for throughput runs (`METALLIC_IGNORE_EOS_STOP=1`) but unsafe for multi-turn reuse.
        if stop_on_eos && batch_size > 1 && !self.unsafe_allow_overshoot {
            return Err(MetalError::InvalidOperation(
                "Invalid workflow: while_batched with batch_size>1 while EOS stopping is enabled can cause KV overshoot. Set METALLIC_IGNORE_EOS_STOP=1, use batch_size=1, or set while_batched.unsafe_allow_overshoot=true if you accept this behavior."
                    .into(),
            ));
        }

        let mut iter = 0usize;
        'outer: loop {
            if let Some(max) = max_iters {
                if iter >= max {
                    break;
                }
            }

            // Condition is still a variable lookup (bool/int), matching WhileOp semantics.
            let cond_val = ctx
                .values
                .get(&self.condition)
                .ok_or_else(|| MetalError::InvalidOperation(format!("WhileBatchedOp missing condition variable '{}'", self.condition)))?;

            let should_run = match cond_val {
                crate::workflow::Value::Bool(b) => *b,
                crate::workflow::Value::U32(v) => *v != 0,
                crate::workflow::Value::Usize(v) => *v != 0,
                _ => {
                    return Err(MetalError::InvalidOperation(format!(
                        "WhileBatchedOp condition '{}' is not a boolean or integer",
                        self.condition
                    )));
                }
            };
            if !should_run {
                break;
            }

            let remaining = max_iters.map(|m| m.saturating_sub(iter)).unwrap_or(batch_size);
            let chunk = batch_size.min(remaining).max(1);

            if !ctx.foundry.is_capturing() {
                ctx.foundry.start_capture()?;
            }

            enum TokenRef {
                Host(u32),
                Tensor(TensorArg),
            }
            let mut tokens: Vec<TokenRef> = if stream_reader.is_some() {
                Vec::new()
            } else {
                Vec::with_capacity(chunk)
            };

            for batch_idx in 0..chunk {
                ctx.values
                    .insert("_internal.decode_batch_idx".to_string(), crate::workflow::Value::Usize(batch_idx));

                for op in &mut self.body_ops {
                    match op.execute(ctx, on_token)? {
                        WorkflowOpOutcome::Continue => {}
                        WorkflowOpOutcome::Return => return Ok(WorkflowOpOutcome::Return),
                        WorkflowOpOutcome::Break => break 'outer,
                        WorkflowOpOutcome::LoopContinue => break,
                    }
                }

                // In non-stream mode, capture the per-iteration token TensorArg without synchronizing.
                if stream_reader.is_none() {
                    if let Some(tok) = ctx.values.get(&self.token_var).and_then(|v| v.as_u32()) {
                        tokens.push(TokenRef::Host(tok));
                    } else if let Some(t) = ctx.values.get(&self.token_var).and_then(|v| v.as_tensor()) {
                        tokens.push(TokenRef::Tensor(t.clone()));
                    } else {
                        return Err(MetalError::InvalidOperation(format!(
                            "WhileBatchedOp missing token variable '{}' (u32 or Tensor u32[1])",
                            self.token_var
                        )));
                    }
                }

                iter = iter.saturating_add(1);
                if let Some(max) = max_iters {
                    if iter >= max {
                        break;
                    }
                }
            }

            let cmd = ctx.foundry.end_capture()?;
            cmd.wait_until_completed();

            // Emit tokens in-order, trimming at EOS (if enabled). Note that any overshoot tokens
            // are still computed into the model KV cache; this is intended for single-turn flows.
            drained_tokens.clear();
            if let Some(r) = stream_reader.as_mut() {
                r.drain_into(&mut drained_tokens)?;
                for token in drained_tokens.iter().copied() {
                    if stop_on_eos && token == eos {
                        break 'outer;
                    }

                    // Append to output list
                    if let Some(val) = ctx.values.get_mut(&self.output_tokens) {
                        if let crate::workflow::Value::TokensU32(vec) = val {
                            vec.push(token);
                        } else {
                            return Err(MetalError::InvalidOperation(format!(
                                "WhileBatchedOp output '{}' is not a TokensU32 list",
                                self.output_tokens
                            )));
                        }
                    } else {
                        ctx.values
                            .insert(self.output_tokens.clone(), crate::workflow::Value::TokensU32(vec![token]));
                    }

                    let prefill_us = ctx.read_usize("_internal.prefill_us").unwrap_or(0);
                    let setup_us = ctx.read_usize("_internal.setup_us").unwrap_or(0);
                    let should_continue = on_token(
                        token,
                        std::time::Duration::from_micros(prefill_us as u64),
                        std::time::Duration::from_micros(setup_us as u64),
                        None,
                    )?;
                    if !should_continue {
                        break 'outer;
                    }
                }
            } else {
                for t in tokens {
                    let token: u32 = match t {
                        TokenRef::Host(v) => v,
                        TokenRef::Tensor(t) => {
                            let buf = t
                                .buffer
                                .as_ref()
                                .ok_or_else(|| MetalError::InvalidOperation("WhileBatchedOp token tensor missing buffer".into()))?;
                            buf.read_scalar::<u32>()
                        }
                    };

                    if stop_on_eos && token == eos {
                        break 'outer;
                    }

                    // Append to output list
                    if let Some(val) = ctx.values.get_mut(&self.output_tokens) {
                        if let crate::workflow::Value::TokensU32(vec) = val {
                            vec.push(token);
                        } else {
                            return Err(MetalError::InvalidOperation(format!(
                                "WhileBatchedOp output '{}' is not a TokensU32 list",
                                self.output_tokens
                            )));
                        }
                    } else {
                        ctx.values
                            .insert(self.output_tokens.clone(), crate::workflow::Value::TokensU32(vec![token]));
                    }

                    let prefill_us = ctx.read_usize("_internal.prefill_us").unwrap_or(0);
                    let setup_us = ctx.read_usize("_internal.setup_us").unwrap_or(0);
                    let should_continue = on_token(
                        token,
                        std::time::Duration::from_micros(prefill_us as u64),
                        std::time::Duration::from_micros(setup_us as u64),
                        None,
                    )?;
                    if !should_continue {
                        break 'outer;
                    }
                }
            }
        }

        // Clean up internal hints (best-effort; keep DX clean when debugging state dumps).
        ctx.values.remove("_internal.decode_batch_idx");
        ctx.values.remove("_internal.decode_batch_size");

        Ok(WorkflowOpOutcome::Continue)
    }
}

pub(crate) struct BreakOp;

impl WorkflowOp for BreakOp {
    fn execute(
        &mut self,
        _ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        Ok(WorkflowOpOutcome::Break)
    }
}

pub(crate) struct ContinueOp;

impl WorkflowOp for ContinueOp {
    fn execute(
        &mut self,
        _ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        Ok(WorkflowOpOutcome::LoopContinue)
    }
}
