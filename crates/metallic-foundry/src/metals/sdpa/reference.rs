use std::sync::Arc;

use half::f16;
use serde::{Deserialize, Serialize};

use crate::{
    Foundry, MetalError, constants, metals::{
        gemm::step::{GemmParams, GemmV2Args, gemm_dispatch_config, get_gemm_kernel}, mma::stages::TileConfig, softmax::{SoftmaxV2SdpaBatchedArgs, get_softmax_v2_sdpa_batched_kernel}
    }, policy::activation::Activation, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Fallback/Reference SDPA op (materialized GEMM->Softmax->GEMM path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdpaReferenceStep {
    pub q: Ref,
    pub k: Ref,
    pub v: Ref,
    pub output: Ref,
    #[serde(default)]
    pub causal: bool,
    pub query_offset: DynamicValue<u32>,
    pub n_heads: DynamicValue<u32>,
    pub head_dim: DynamicValue<u32>,
    pub kv_seq_len: DynamicValue<u32>,
    #[serde(default)]
    pub m: DynamicValue<u32>,
    #[serde(default)]
    pub kv_head_major: bool,
}

#[derive(Debug, Clone)]
pub struct CompiledSdpaReferenceStep {
    pub step: SdpaReferenceStep,
    pub q_idx: usize,
    pub k_idx: usize,
    pub v_idx: usize,
    pub output_idx: usize,
}

#[typetag::serde(name = "SdpaReference")]
impl Step for SdpaReferenceStep {
    fn name(&self) -> &'static str {
        "SdpaReference"
    }

    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = FastBindings::new(symbols.len());

        for (name, symbol_id) in symbols.iter() {
            if let Ok(tensor) = bindings.get(name) {
                fast_bindings.set(*symbol_id, tensor);
            }
        }

        for step in compiled {
            step.execute(foundry, &fast_bindings, bindings, &symbols)?;
        }

        Ok(())
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let q_idx = symbols.get_or_create(bindings.interpolate(self.q.0.clone()));
        let k_idx = symbols.get_or_create(bindings.interpolate(self.k.0.clone()));
        let v_idx = symbols.get_or_create(bindings.interpolate(self.v.0.clone()));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));

        vec![Box::new(CompiledSdpaReferenceStep {
            step: self.clone(),
            q_idx,
            k_idx,
            v_idx,
            output_idx,
        })]
    }
}

impl CompiledStep for CompiledSdpaReferenceStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let q = fast_bindings.get(self.q_idx).ok_or(MetalError::InputNotFound("q".into()))?;
        let k = fast_bindings.get(self.k_idx).ok_or(MetalError::InputNotFound("k".into()))?;
        let v = fast_bindings.get(self.v_idx).ok_or(MetalError::InputNotFound("v".into()))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or(MetalError::InputNotFound("output".into()))?;

        let head_dim = self.step.head_dim.resolve(bindings);
        let kv_seq_len = self.step.kv_seq_len.resolve(bindings);
        let n_heads = self.step.n_heads.resolve(bindings);
        let q_offset_val = self.step.query_offset.resolve(bindings);
        let m = self.step.m.resolve(bindings).max(1);

        if !self.step.kv_head_major {
            return Err(MetalError::OperationNotSupported(
                "SdpaReference only supports kv_head_major=true for now".into(),
            ));
        }

        execute_sdpa_reference(
            foundry,
            q,
            k,
            v,
            output,
            n_heads,
            head_dim,
            kv_seq_len,
            q_offset_val,
            m,
            self.step.causal,
        )
    }

    fn name(&self) -> &'static str {
        "SdpaReference"
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_sdpa_reference(
    foundry: &mut Foundry,
    q: &TensorArg,
    k: &TensorArg,
    v: &TensorArg,
    output: &TensorArg,
    n_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    q_offset_val: u32,
    m: u32,
    causal: bool,
) -> Result<(), MetalError> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let d_model = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::OperationNotSupported("d_model overflow".into()))? as usize;

    let (q_batch_stride, q_row_stride) = {
        let dims = q.dims();
        let strides = q.strides();

        match dims {
            [len] if strides.len() == 1 && *len >= (m as usize) * d_model => (head_dim as i64, d_model as i32),
            [1, _m_cap, dm0] if *dm0 == d_model && strides.len() == 3 => {
                let q_head_stride = (m as i64) * (head_dim as i64);
                (q_head_stride, head_dim as i32)
            }
            [rows, dm0] if *dm0 == d_model && strides.len() == 2 && *rows >= m as usize => (head_dim as i64, d_model as i32),
            [rows, h, d] if *h == n_heads as usize && *d == head_dim as usize && strides.len() == 3 && *rows >= m as usize => {
                (strides[1] as i64, strides[0] as i32)
            }
            [h, rows, d] if *h == n_heads as usize && *d == head_dim as usize && strides.len() == 3 && *rows >= m as usize => {
                ((m as i64) * (head_dim as i64), head_dim as i32)
            }
            _ => {
                return Err(MetalError::OperationNotSupported(format!(
                    "SdpaReference Q layout unsupported: dims={dims:?} strides={strides:?} (expected token-major [m,d_model] or fixed-cap [1,m_cap,d_model] with head-major contents)"
                )));
            }
        }
    };

    let (out_batch_stride, out_row_stride) = {
        let dims = output.dims();
        let strides = output.strides();

        match dims {
            [len] if strides.len() == 1 && *len >= (m as usize) * d_model => (head_dim as i64, d_model as i32),
            [rows, dm0] if *dm0 == d_model && strides.len() == 2 && *rows >= m as usize => (head_dim as i64, d_model as i32),
            [1, rows, dm0] if *dm0 == d_model && strides.len() == 3 && *rows >= m as usize => (head_dim as i64, d_model as i32),
            [rows, h, d] if *h == n_heads as usize && *d == head_dim as usize && strides.len() == 3 && *rows >= m as usize => {
                (strides[1] as i64, strides[0] as i32)
            }
            [h, rows, d] if *h == n_heads as usize && *d == head_dim as usize && strides.len() == 3 && *rows >= m as usize => {
                ((m as i64) * (head_dim as i64), head_dim as i32)
            }
            _ => {
                return Err(MetalError::OperationNotSupported(format!(
                    "SdpaReference output layout unsupported: dims={dims:?} strides={strides:?}"
                )));
            }
        }
    };

    let tile_config = TileConfig::default();
    let (bm, _, _, _, _) = tile_config.tile_sizes();

    let scratch_layout = crate::compound::layout::TiledLayout::sdpa_scratch(n_heads, m, kv_seq_len, bm);
    let (scores_all, probs_all) = crate::metals::sdpa::scratch::get_sdpa_scratch_f16(foundry, scratch_layout)?;
    let scale_arg = constants::f16_scalar(foundry, f16::ONE)?;

    let k_seq_stride = k.dims().get(1).copied().unwrap_or(kv_seq_len as usize);
    let v_seq_stride = v.dims().get(1).copied().unwrap_or(kv_seq_len as usize);

    let qk_gemm_kernel = get_gemm_kernel(
        Arc::new(crate::policy::f16::PolicyF16),
        Arc::new(crate::policy::f16::PolicyF16),
        false,
        true,
        tile_config,
        true,
        false,
        Activation::None,
    );

    let mut qk_params = GemmParams::simple(m as i32, kv_seq_len as i32, head_dim as i32, false, true, tile_config);
    qk_params.lda = q_row_stride;
    qk_params.batch_stride_a = q_batch_stride;
    qk_params.batch_stride_b = (k_seq_stride as i64) * (head_dim as i64);
    qk_params.batch_stride_c = scratch_layout.head_stride as i64;
    qk_params.batch_stride_d = scratch_layout.head_stride as i64;

    let qk_dispatch = {
        let base = gemm_dispatch_config(&qk_params, tile_config);
        DispatchConfig {
            grid: GridSize::new(qk_params.tiles_n as usize, qk_params.tiles_m as usize, n_heads as usize),
            group: base.group,
        }
    };

    let qk_args = GemmV2Args {
        a: TensorArg::from_tensor(q),
        b: TensorArg::from_tensor(k),
        d: scores_all.clone(),
        c: scores_all.clone(),
        bias: scores_all.clone(),
        b_scales: scores_all.clone(),
        weights_per_block: 32,
        params: qk_params,
        alpha: scale,
        beta: 0.0,
        b_is_canonical: 0,
    };
    foundry.run(&qk_gemm_kernel.clone().bind_arc(qk_args, qk_dispatch))?;

    let softmax_sdpa_kernel = get_softmax_v2_sdpa_batched_kernel();
    let softmax_dispatch = DispatchConfig {
        grid: GridSize::d1((n_heads as usize) * (scratch_layout.padded_m as usize)),
        group: ThreadgroupSize::d1(256),
    };
    let softmax_args = SoftmaxV2SdpaBatchedArgs {
        input: scores_all.clone(),
        scale: scale_arg.clone(),
        output: probs_all.clone(),
        seq_k: kv_seq_len,
        causal: if causal { 1 } else { 0 },
        query_offset: q_offset_val,
        rows_per_batch: scratch_layout.padded_m,
    };
    foundry.run(&softmax_sdpa_kernel.clone().bind_arc(softmax_args, softmax_dispatch))?;

    let av_gemm_kernel = get_gemm_kernel(
        Arc::new(crate::policy::f16::PolicyF16),
        Arc::new(crate::policy::f16::PolicyF16),
        false,
        false,
        tile_config,
        false,
        false,
        Activation::None,
    );

    let mut av_params = GemmParams::simple(m as i32, head_dim as i32, kv_seq_len as i32, false, false, tile_config);
    av_params.ldc = out_row_stride;
    av_params.ldd = out_row_stride;
    av_params.batch_stride_a = scratch_layout.head_stride as i64;
    av_params.batch_stride_b = (v_seq_stride as i64) * (head_dim as i64);
    av_params.batch_stride_c = out_batch_stride;
    av_params.batch_stride_d = out_batch_stride;

    let av_dispatch = {
        let base = gemm_dispatch_config(&av_params, tile_config);
        DispatchConfig {
            grid: GridSize::new(av_params.tiles_n as usize, av_params.tiles_m as usize, n_heads as usize),
            group: base.group,
        }
    };

    let av_args = GemmV2Args {
        a: probs_all.clone(),
        b: TensorArg::from_tensor(v),
        d: TensorArg::from_tensor(output),
        c: TensorArg::from_tensor(output),
        bias: TensorArg::from_tensor(output),
        b_scales: TensorArg::from_tensor(output),
        weights_per_block: 32,
        params: av_params,
        alpha: 1.0,
        beta: 0.0,
        b_is_canonical: 0,
    };

    foundry.run(&av_gemm_kernel.clone().bind_arc(av_args, av_dispatch))?;

    Ok(())
}
