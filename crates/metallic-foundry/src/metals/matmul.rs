use serde::{Deserialize, Serialize};

use crate::{
    Foundry, MetalError, compound::Layout, metals::{
        gemm::GemmV2Step, gemv::{GemvStrategy, GemvV2Params, GemvV2UnifiedExecutionStep}
    }, policy::activation::Activation, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}
};

/// Unified MatMul step that dynamically dispatches to GEMV or GEMM.
///
/// If M=1, it uses `GemvV2UnifiedExecutionStep` for low-latency vector-matrix multiplication.
/// If M>1, it uses `GemmV2Step` for high-throughput matrix-matrix multiplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MatMulStep {
    /// A matrix [M, K] or [K, M] if transposed (though GEMM V2 usually expects M x K)
    pub a: Ref,
    /// B matrix [K, N] or [N, K] if transposed
    pub b: Ref,
    /// Output matrix [M, N]
    pub output: Ref,
    /// Optional bias vector [N]
    pub bias: Option<Ref>,
    /// Optional scales for B (if quantized)
    pub b_scales: Option<Ref>, // Maps to scale_bytes in Gemv
    /// Optional C/Residual matrix to add: Output = Alpha*A*B + Beta*C
    pub c: Option<Ref>, // Maps to residual
    pub m: DynamicValue<u32>,
    pub n: DynamicValue<u32>,
    pub k: DynamicValue<u32>,
    pub transpose_a: bool,
    pub transpose_b: bool,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    #[serde(default = "default_beta")]
    pub beta: f32,
    #[serde(default = "default_weights_per_block")]
    pub weights_per_block: u32,
    #[serde(default)]
    pub activation: Activation,
}

fn default_alpha() -> f32 {
    1.0
}

fn default_beta() -> f32 {
    0.0
}

fn default_weights_per_block() -> u32 {
    32
}

impl Default for MatMulStep {
    fn default() -> Self {
        Self {
            a: Ref("a".into()),
            b: Ref("b".into()),
            output: Ref("output".into()),
            bias: None,
            b_scales: None,
            c: None,
            m: DynamicValue::Literal(1),
            n: DynamicValue::Literal(1),
            k: DynamicValue::Literal(1),
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
            weights_per_block: 32,
            activation: Activation::None,
        }
    }
}

// =============================================================================
// Compiled Wrapper
// =============================================================================

#[derive(Debug)]
pub struct CompiledMatMulStep {
    // We hold compiled versions of both kernels (or lazily?)
    // For now, we compile both. A more advanced version might share resources.
    gemv: Vec<Box<dyn CompiledStep>>,
    gemm: Vec<Box<dyn CompiledStep>>,

    m_val: DynamicValue<u32>,
    n_val: DynamicValue<u32>,
    k_val: DynamicValue<u32>,
    transpose_a: bool,
    transpose_b: bool,
    gemm_enabled: bool,
}

impl CompiledStep for CompiledMatMulStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        // Resolve M dimension at runtime
        let m = self.m_val.resolve(bindings);

        if m == 1 || !self.gemm_enabled {
            // Dispatch GEMV
            for step in &self.gemv {
                step.execute(foundry, fast_bindings, bindings, symbols)?;
            }
        } else {
            // Dispatch GEMM
            for step in &self.gemm {
                step.execute(foundry, fast_bindings, bindings, symbols)?;
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "MatMul (Unified)"
    }

    fn perf_metadata(&self, globals: &TensorBindings) -> Option<String> {
        let m = self.m_val.resolve(globals);
        let n = self.n_val.resolve(globals);
        let k = self.k_val.resolve(globals);
        let mode = if m == 1 || !self.gemm_enabled { "gemv" } else { "gemm" };
        Some(format!(
            "MatMul (Unified) mode={mode} m={m} n={n} k={k} ta={} tb={}",
            self.transpose_a, self.transpose_b
        ))
    }
}

#[typetag::serde(name = "MatMul")]
impl Step for MatMulStep {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = FastBindings::new(symbols.len());

        // Bind all symbols
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

    fn name(&self) -> &'static str {
        "MatMul"
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        // Detect if scales are actually provided (Q8) vs just pointing to weights (FP16/Legacy)
        let effective_b_scales = self.b_scales.as_ref().filter(|s| s.0 != self.b.0).cloned();

        // Filter "zero" placeholders for bias and c
        let effective_bias = self.bias.as_ref().filter(|b| b.0 != "zero").cloned();
        let effective_c = self.c.as_ref().filter(|c| c.0 != "zero").cloned();

        // Determine layout for GEMV dispatch
        let layout = if self.transpose_b {
            Layout::RowMajor
        } else {
            Layout::Canonical {
                expected_k: 0,
                expected_n: 0,
            }
        };

        // Canonical layout has a dedicated unrolled dot stage that matches legacy Q8 performance.
        // Defaulting to `Auto` would pick the Vectorized stage for Canonical layouts, which is slower for decode-heavy Q8.
        let strategy = match layout {
            Layout::Canonical { .. } => Some(GemvStrategy::Canonical),
            _ => None, // Auto-select
        };

        let gemm_enabled =
            !(matches!(layout, Layout::Canonical { .. }) || (self.n == DynamicValue::Literal(1) && self.k == DynamicValue::Literal(1)));

        // 1. Prepare GEMV Step (for M=1) using GemvV2UnifiedExecutionStep for both layouts
        let gemv_step = GemvV2UnifiedExecutionStep {
            weights: self.b.clone(),
            input: self.a.clone(),
            output: self.output.clone(),
            bias: effective_bias.clone(),
            residual: effective_c.clone(),
            scale_bytes: effective_b_scales.clone(),
            params: GemvV2Params {
                k_dim: self.k.clone(),
                n_dim: self.n.clone(),
                weights_per_block: self.weights_per_block,
                batch: self.m.clone(), // Correctly propagate M as batch for GEMV prefill
            },
            layout,
            strategy,
            activation: self.activation,
            alpha: self.alpha,
            beta: self.beta,
            has_bias: if effective_bias.is_some() { 1 } else { 0 },
            has_residual: if effective_c.is_some() { 1 } else { 0 },
        };
        let gemv_compiled = gemv_step.compile(bindings, symbols);

        // 2. Prepare GEMM Step (for M>1)
        let gemm_compiled = if gemm_enabled {
            use crate::metals::gemm::GemmParams;
            let gemm_step = GemmV2Step {
                a: self.a.clone(),
                b: self.b.clone(),
                d: self.output.clone(),
                c: effective_c,
                bias: effective_bias,
                b_scales: effective_b_scales,
                weights_per_block: self.weights_per_block,
                alpha: self.alpha,
                beta: self.beta,
                b_is_canonical: 0,
                params: GemmParams::default(),
                m_dim: self.m.clone(),
                n_dim: self.n.clone(),
                k_dim: self.k.clone(),
                transpose_a: self.transpose_a,
                transpose_b: self.transpose_b,
                tile_config: None,
                activation: self.activation,
            };
            gemm_step.compile(bindings, symbols)
        } else {
            vec![]
        };

        vec![Box::new(CompiledMatMulStep {
            gemv: gemv_compiled,
            gemm: gemm_compiled,
            m_val: self.m.clone(),
            n_val: self.n.clone(),
            k_val: self.k.clone(),
            transpose_a: self.transpose_a,
            transpose_b: self.transpose_b,
            gemm_enabled,
        })]
    }
}

#[path = "matmul.test.rs"]
mod tests;
