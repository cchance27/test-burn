use serde::{Deserialize, Serialize};

use crate::{
    Foundry, MetalError, compound::Layout, metals::{gemm::step::GemmV2Step, gemv::step::GemvV2Step}, policy::activation::Activation, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}
};

/// Unified MatMul step that dynamically dispatches to GEMV or GEMM.
///
/// If M=1, it uses `GemvV2Step` for low-latency vector-matrix multiplication.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_step_defaults_round_trip() {
        let json = r#"{
            "a": "a",
            "b": "b",
            "output": "output",
            "m": 1,
            "n": 1,
            "k": 1,
            "transpose_a": false,
            "transpose_b": false,
            "weights_per_block": 32
        }"#;

        let step: MatMulStep = serde_json::from_str(json).unwrap();
        assert_eq!(step.alpha, 1.0);
        assert_eq!(step.beta, 0.0);

        let json2 = r#"{
            "a": "a",
            "b": "b",
            "output": "output",
            "m": 1,
            "n": 1,
            "k": 1,
            "transpose_a": false,
            "transpose_b": false,
            "alpha": 0.25,
            "beta": 0.75
        }"#;
        let step2: MatMulStep = serde_json::from_str(json2).unwrap();
        assert_eq!(step2.alpha, 0.25);
        assert_eq!(step2.beta, 0.75);
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
        // Note: bindings contains the interpolated/runtime argument values if they were dynamic
        // but DynamicValue::resolve uses bindings.globals or scopes?
        // DynamicValue itself handles literal vs ref.
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
}

#[typetag::serde(name = "MatMul")]
impl Step for MatMulStep {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        // Compile and execute (like GemvCanonicalStep)
        use crate::spec::{FastBindings, SymbolTable};

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
        // transpose_b = true (NxK) -> Layout::RowMajor (NK) -> Use GemvV2Step
        // transpose_b = false (KxN) -> Layout::Canonical -> Use GemvCanonicalStep (proven inference)
        let use_canonical = !self.transpose_b;
        let gemm_enabled = !(use_canonical || self.n == DynamicValue::Literal(1) && self.k == DynamicValue::Literal(1));

        // 1. Prepare GEMV Step (for M=1)
        let gemv_compiled: Vec<Box<dyn CompiledStep>> = if use_canonical {
            // Use GemvCanonicalStep for Canonical layout - it infers dimensions at runtime
            use crate::metals::gemv::step::{GemvCanonicalStep, GemvLegacyParams};

            let canonical_step = GemvCanonicalStep {
                matrix: self.b.clone(),             // B weights
                scale_bytes: self.b_scales.clone(), // Keep original (step filters internally)
                vector_x: self.a.clone(),           // A input
                result_y: self.output.clone(),      // Output
                bias: self.bias.clone(),            // Keep original (step filters "zero")
                residual: self.c.clone(),           // Keep original (step filters "zero")
                params: GemvLegacyParams {
                    batch: 1,
                    weights_per_block: self.weights_per_block,
                },
                alpha: self.alpha,
                beta: self.beta,
                has_bias: if effective_bias.is_some() { 1 } else { 0 },
                activation: self.activation,
            };

            canonical_step.compile(bindings, symbols)
        } else {
            // Use GemvV2Step for RowMajor layout (transpose_b = true)
            let layout = Layout::RowMajor;

            let gemv_step = GemvV2Step {
                input: self.a.clone(),
                weights: self.b.clone(),
                output: self.output.clone(),
                bias: effective_bias.clone(),
                residual: effective_c.clone(),
                scale_bytes: effective_b_scales.clone(),
                k_dim: self.k.clone(),
                n_dim: self.n.clone(),
                weights_per_block: self.weights_per_block,
                layout,
                alpha: self.alpha,
                beta: self.beta,
                strategy: None, // Auto-select
                activation: self.activation,
            };

            gemv_step.compile(bindings, symbols)
        };

        // 2. Prepare GEMM Step (for M>1)
        let gemm_compiled = if gemm_enabled {
            // Derive b_quant from B tensor dtype at runtime
            let b_name = bindings.interpolate(self.b.0.clone());
            let b_quant = bindings
                .get(&b_name)
                .map(|t| crate::policy::resolve_policy(t.dtype.into()))
                .unwrap_or_else(|_| std::sync::Arc::new(crate::policy::f16::PolicyF16) as _);

            let gemm_step = GemmV2Step {
                a: self.a.clone(),
                b: self.b.clone(),
                output: self.output.clone(),
                b_scales: effective_b_scales,
                bias: effective_bias,
                c: effective_c,
                m_dim: self.m.clone(),
                n_dim: self.n.clone(),
                k_dim: self.k.clone(),
                b_quant,
                transpose_a: self.transpose_a,
                transpose_b: self.transpose_b,
                alpha: self.alpha,
                beta: self.beta,
                weights_per_block: self.weights_per_block,
                tile_config: None, // Auto
                activation: self.activation,
            };

            gemm_step.compile(bindings, symbols)
        } else {
            Vec::new()
        };

        // Return unified wrapper
        vec![Box::new(CompiledMatMulStep {
            gemv: gemv_compiled,
            gemm: gemm_compiled,
            m_val: self.m.clone(),
            gemm_enabled,
        })]
    }
}
