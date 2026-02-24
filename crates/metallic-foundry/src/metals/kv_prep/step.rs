use serde::{Deserialize, Serialize};

use crate::{
    Foundry, MetalError, metals::common::dtype_contract::require_uniform_dtypes, spec::{CompiledStep, FastBindings, Ref, Step, SymbolTable, TensorBindings}, types::TensorArg
};

/// DSL Step for the fused KV-prep kernel.
///
/// This is implemented manually so the DSL can supply `DynamicValue` fields in `params`,
/// which are resolved at runtime into `KvPrepFusedParamsResolved` (the Metal ABI struct).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvPrepFusedStep {
    pub q: Ref,
    pub k: Ref,
    pub v: Ref,

    pub q_rot: Ref,
    pub k_cache: Ref,
    pub v_cache: Ref,

    pub cos: Ref,
    pub sin: Ref,

    pub params: super::KvPrepFusedParams,
}

#[derive(Debug)]
pub struct CompiledKvPrepFusedStep {
    pub step: KvPrepFusedStep,
    pub q_idx: usize,
    pub k_idx: usize,
    pub v_idx: usize,
    pub q_rot_idx: usize,
    pub k_cache_idx: usize,
    pub v_cache_idx: usize,
    pub cos_idx: usize,
    pub sin_idx: usize,
}

#[typetag::serde(name = "KvPrepFused")]
impl Step for KvPrepFusedStep {
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

        let q_rot_idx = symbols.get_or_create(bindings.interpolate(self.q_rot.0.clone()));
        let k_cache_idx = symbols.get_or_create(bindings.interpolate(self.k_cache.0.clone()));
        let v_cache_idx = symbols.get_or_create(bindings.interpolate(self.v_cache.0.clone()));

        let cos_idx = symbols.get_or_create(bindings.interpolate(self.cos.0.clone()));
        let sin_idx = symbols.get_or_create(bindings.interpolate(self.sin.0.clone()));
        let mut step = self.clone();
        step.params = step.params.bind_scope_literals(bindings);

        vec![Box::new(CompiledKvPrepFusedStep {
            step,
            q_idx,
            k_idx,
            v_idx,
            q_rot_idx,
            k_cache_idx,
            v_cache_idx,
            cos_idx,
            sin_idx,
        })]
    }

    fn name(&self) -> &'static str {
        "KvPrepFused"
    }
}

impl CompiledStep for CompiledKvPrepFusedStep {
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

        let q_rot = fast_bindings.get(self.q_rot_idx).ok_or(MetalError::InputNotFound("q_rot".into()))?;
        let k_cache = fast_bindings
            .get(self.k_cache_idx)
            .ok_or(MetalError::InputNotFound("k_cache".into()))?;
        let v_cache = fast_bindings
            .get(self.v_cache_idx)
            .ok_or(MetalError::InputNotFound("v_cache".into()))?;

        let cos = fast_bindings.get(self.cos_idx).ok_or(MetalError::InputNotFound("cos".into()))?;
        let sin = fast_bindings.get(self.sin_idx).ok_or(MetalError::InputNotFound("sin".into()))?;

        require_uniform_dtypes(
            "KvPrepFused",
            &[
                ("q", q.dtype),
                ("k", k.dtype),
                ("v", v.dtype),
                ("q_rot", q_rot.dtype),
                ("k_cache", k_cache.dtype),
                ("v_cache", v_cache.dtype),
                ("cos", cos.dtype),
                ("sin", sin.dtype),
            ],
        )
        .map_err(|_| {
            MetalError::OperationFailed(format!(
                "KvPrepFused mixed-policy is unsupported (q={:?}, k={:?}, v={:?}, q_rot={:?}, k_cache={:?}, v_cache={:?}, cos={:?}, sin={:?}).",
                q.dtype, k.dtype, v.dtype, q_rot.dtype, k_cache.dtype, v_cache.dtype, cos.dtype, sin.dtype
            ))
        })?;

        let params = self.step.params.resolve(bindings);

        let kernel = super::KvPrepFused {
            q: TensorArg::from_tensor(q),
            k: TensorArg::from_tensor(k),
            v: TensorArg::from_tensor(v),
            q_rot: TensorArg::from_tensor(q_rot),
            k_cache: TensorArg::from_tensor(k_cache),
            v_cache: TensorArg::from_tensor(v_cache),
            cos: TensorArg::from_tensor(cos),
            sin: TensorArg::from_tensor(sin),
            params,
        };

        foundry.run(&kernel)
    }

    fn name(&self) -> &'static str {
        "KvPrepFused"
    }
}
