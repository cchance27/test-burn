#![cfg(test)]

use std::sync::Arc;

use metallic_loader::ModelLoader;
use rustc_hash::FxHashMap;
use smallvec::smallvec;

use super::*;
use crate::{
    Foundry, generation::default_text_generation_workflow, model::{CompiledModel, ModelBuilder}, spec::ModelSpec, tensor::Dtype, workflow::{Value, WorkflowRunner}
};

#[test]
fn test_rebalanced_prefill_chunk_size_avoids_tiny_tail() {
    let rebalance = |prompt_len: usize, requested: usize, max_allowed: usize| -> usize {
        let requested = requested.max(1).min(max_allowed.max(1));
        if prompt_len <= 1 {
            return 1;
        }
        let chunks = prompt_len.div_ceil(requested);
        let balanced = prompt_len.div_ceil(chunks);
        balanced.max(1).min(max_allowed)
    };

    // Repro shape: 34 with requested 32 becomes 17+17 (no 2-token tail).
    assert_eq!(rebalance(34, 32, 32), 17);
    // Already balanced.
    assert_eq!(rebalance(1000, 32, 32), 32);
    // Single chunk.
    assert_eq!(rebalance(31, 32, 32), 31);
    // Degenerate.
    assert_eq!(rebalance(0, 32, 32), 1);
    assert_eq!(rebalance(1, 32, 32), 1);
}

#[test]
#[serial_test::serial]
fn test_model_session_kv_growth() -> Result<(), crate::error::MetalError> {
    let spec = ModelSpec::from_json(
        r#"
        {
          "name": "test-growth",
          "architecture": {
            "d_model": 128,
            "n_heads": 2,
            "n_kv_heads": 1,
            "n_layers": 1,
            "ff_dim": 256,
            "vocab_size": 100,
            "max_seq_len": 4096,
            "rope_base": 10000.0,
            "rms_eps": 0.000001,
            "prepare": {
              "rope": { "cos": "rope_cos", "sin": "rope_sin" },
              "tensors": [
                {
                  "name": "k_cache_{i}",
                  "repeat": { "count": "n_layers", "var": "i" },
                  "storage": "kv_cache",
                  "dims": ["n_heads", "max_seq_len", "d_model / n_heads"],
                  "grow_with_kv": true
                },
                {
                  "name": "v_cache_{i}",
                  "repeat": { "count": "n_layers", "var": "i" },
                  "storage": "kv_cache",
                  "dims": ["n_heads", "max_seq_len", "d_model / n_heads"],
                  "grow_with_kv": true
                }
              ]
            },
            "forward": []
          }
        }
        "#,
    )
    .map_err(|e| crate::error::MetalError::InvalidOperation(e.to_string()))?;

    // We don't need real weights for allocation testing
    let weights = super::WeightBundle::new_empty();
    let model = CompiledModel::new(spec, weights)?;

    let mut foundry = match Foundry::new() {
        Ok(foundry) => foundry,
        Err(crate::error::MetalError::DeviceNotFound) => return Ok(()),
        Err(e) => return Err(e),
    };
    model.initialize_session(&mut foundry)?;

    {
        let mut session_guard = model.session.lock();
        let session = session_guard.as_mut().unwrap();

        // Initial capacity should be 2048 (default) aligned to 128
        assert_eq!(session.context_config.allocated_capacity, 2048);

        // Grow to 3000. Geometric growth (2x) from 2048 is 4096.
        model.ensure_kv_capacity(
            &mut foundry,
            &mut session.bindings,
            &mut session.fast_bindings,
            &mut session.context_config,
            session.current_pos,
            3000,
        )?;
        assert_eq!(session.context_config.allocated_capacity, 4096);

        // Should be present in bindings
        assert!(session.bindings.contains("k_cache_0"));
        let k_cache = session.bindings.get("k_cache_0")?;
        // Shape: [n_heads, allocated_capacity, head_dim] -> [2, 4096, 64]
        assert_eq!(k_cache.dims.as_slice(), &[2, 4096, 64]);
    }

    Ok(())
}

fn growth_spec_with_kv_tensors(tensors_json: &str) -> Result<ModelSpec, crate::error::MetalError> {
    let spec_json = format!(
        r#"
        {{
          "name": "test-growth-dtype",
          "architecture": {{
            "d_model": 128,
            "n_heads": 2,
            "n_kv_heads": 1,
            "n_layers": 1,
            "ff_dim": 256,
            "vocab_size": 100,
            "max_seq_len": 4096,
            "rope_base": 10000.0,
            "rms_eps": 0.000001,
            "prepare": {{
              "rope": {{ "cos": "rope_cos", "sin": "rope_sin" }},
              "tensors": {tensors_json}
            }},
            "forward": []
          }}
        }}
        "#
    );

    ModelSpec::from_json(&spec_json).map_err(|e| crate::error::MetalError::InvalidOperation(e.to_string()))
}

fn run_growth_with_nonzero_pos(spec: ModelSpec) -> Result<Arc<CompiledModel>, crate::error::MetalError> {
    let weights = super::WeightBundle::new_empty();
    let model = Arc::new(CompiledModel::new(spec, weights)?);

    let mut foundry = match Foundry::new() {
        Ok(foundry) => foundry,
        Err(crate::error::MetalError::DeviceNotFound) => return Err(crate::error::MetalError::DeviceNotFound),
        Err(e) => return Err(e),
    };
    model.initialize_session(&mut foundry)?;

    {
        let mut session_guard = model.session.lock();
        let session = session_guard.as_mut().unwrap();
        // Force growth path to execute preservation copy logic.
        session.current_pos = 1;
        model.ensure_kv_capacity(
            &mut foundry,
            &mut session.bindings,
            &mut session.fast_bindings,
            &mut session.context_config,
            session.current_pos,
            3000,
        )?;
        assert_eq!(session.context_config.allocated_capacity, 4096);
    }

    Ok(model)
}

#[test]
#[serial_test::serial]
fn test_model_session_kv_growth_preserves_f32_cache_layout() -> Result<(), crate::error::MetalError> {
    let spec = growth_spec_with_kv_tensors(
        r#"
        [
          {
            "name": "k_cache_{i}",
            "repeat": { "count": "n_layers", "var": "i" },
            "dtype": "F32",
            "storage": "kv_cache",
            "dims": ["n_heads", "max_seq_len", "d_model / n_heads"],
            "grow_with_kv": true
          },
          {
            "name": "v_cache_{i}",
            "repeat": { "count": "n_layers", "var": "i" },
            "dtype": "F32",
            "storage": "kv_cache",
            "dims": ["n_heads", "max_seq_len", "d_model / n_heads"],
            "grow_with_kv": true
          }
        ]
        "#,
    )?;

    let model = match run_growth_with_nonzero_pos(spec) {
        Ok(v) => v,
        Err(crate::error::MetalError::DeviceNotFound) => return Ok(()),
        Err(e) => return Err(e),
    };

    let session_guard = model.session.lock();
    let session = session_guard.as_ref().unwrap();
    let k_cache = session.bindings.get("k_cache_0")?;
    assert_eq!(k_cache.dtype, Dtype::F32);
    assert_eq!(k_cache.dims.as_slice(), &[2, 4096, 64]);
    Ok(())
}

#[test]
#[serial_test::serial]
fn test_model_session_kv_growth_preserves_i8_with_scale_tensors() -> Result<(), crate::error::MetalError> {
    let spec = growth_spec_with_kv_tensors(
        r#"
        [
          {
            "name": "k_cache_data_{i}",
            "repeat": { "count": "n_layers", "var": "i" },
            "dtype": "I8",
            "storage": "kv_cache",
            "dims": ["n_kv_heads", "max_seq_len", "d_model / n_heads"],
            "grow_with_kv": true
          },
          {
            "name": "v_cache_data_{i}",
            "repeat": { "count": "n_layers", "var": "i" },
            "dtype": "I8",
            "storage": "kv_cache",
            "dims": ["n_kv_heads", "max_seq_len", "d_model / n_heads"],
            "grow_with_kv": true
          },
          {
            "name": "k_cache_scale_{i}",
            "repeat": { "count": "n_layers", "var": "i" },
            "dtype": "F16",
            "storage": "kv_cache",
            "dims": ["n_kv_heads", "max_seq_len", "(d_model / n_heads) / 32"],
            "grow_with_kv": true
          },
          {
            "name": "v_cache_scale_{i}",
            "repeat": { "count": "n_layers", "var": "i" },
            "dtype": "F16",
            "storage": "kv_cache",
            "dims": ["n_kv_heads", "max_seq_len", "(d_model / n_heads) / 32"],
            "grow_with_kv": true
          }
        ]
        "#,
    )?;

    let model = match run_growth_with_nonzero_pos(spec) {
        Ok(v) => v,
        Err(crate::error::MetalError::DeviceNotFound) => return Ok(()),
        Err(e) => return Err(e),
    };

    let session_guard = model.session.lock();
    let session = session_guard.as_ref().unwrap();
    let data = session.bindings.get("k_cache_data_0")?;
    let scales = session.bindings.get("k_cache_scale_0")?;
    assert_eq!(data.dtype, Dtype::I8);
    assert_eq!(data.dims.as_slice(), &[1, 4096, 64]);
    assert_eq!(scales.dtype, Dtype::F16);
    assert_eq!(scales.dims.as_slice(), &[1, 4096, 2]);
    Ok(())
}

#[test]
fn test_model_spec_matmul_op_deserializes() -> Result<(), Box<dyn std::error::Error>> {
    let spec = ModelSpec::from_json(
        r#"
        {
          "name": "matmul-op-compat",
          "architecture": {
            "d_model": 128,
            "n_heads": 2,
            "n_kv_heads": 1,
            "n_layers": 1,
            "ff_dim": 256,
            "vocab_size": 100,
            "max_seq_len": 256,
            "forward": [
              {
                "op": "MatMul",
                "a": "x",
                "b": "w",
                "output": "y",
                "m": 1,
                "n": 1,
                "k": 1,
                "transpose_a": false,
                "transpose_b": false
              }
            ]
          }
        }
        "#,
    )?;

    assert_eq!(spec.architecture.forward.len(), 1);
    assert_eq!(spec.architecture.forward[0].name(), "MatMul");
    Ok(())
}

#[test]
fn test_repo_model_specs_parse() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let specs = ["llama.json", "qwen25.json", "smollm3.json"];
    for file in specs {
        let path = root.join(file);
        let spec = ModelSpec::from_file(&path)?;
        assert!(
            !spec.architecture.forward.is_empty(),
            "forward graph should not be empty for {}",
            file
        );
    }
    Ok(())
}

#[test]
#[serial_test::serial]
#[ignore = "requires local GGUF and Metal device; run manually as phase gate"]
fn smoke_prefill_and_single_decode_step_with_real_model() -> Result<(), Box<dyn std::error::Error>> {
    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen25.json");
    let gguf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen2.5-coder-0.5b-instruct-fp16.gguf");

    let mut foundry = Foundry::new()?;
    let loaded_model = ModelLoader::from_file(gguf_path)?;
    let model = Arc::new(
        ModelBuilder::new()
            .with_spec_file(spec_path)?
            .with_model(loaded_model)
            .build(&mut foundry)?,
    );

    let tokenizer = model.tokenizer()?;
    let prompt_tokens = tokenizer.encode_single_turn_chat_prompt("Hello")?;

    let workflow = default_text_generation_workflow();
    let mut models: FxHashMap<String, Arc<CompiledModel>> = FxHashMap::default();
    models.insert("llm".to_string(), model.clone());
    let mut runner = WorkflowRunner::new(models);
    let mut inputs: FxHashMap<String, Value> = FxHashMap::default();
    inputs.insert("prompt_tokens".to_string(), Value::TokensU32(prompt_tokens.clone()));
    inputs.insert("max_tokens".to_string(), Value::Usize(1));
    inputs.insert("temperature".to_string(), Value::F32(0.0));
    inputs.insert("top_k".to_string(), Value::U32(1));
    inputs.insert("top_p".to_string(), Value::F32(1.0));
    inputs.insert("min_p".to_string(), Value::F32(0.0));
    inputs.insert("repeat_penalty".to_string(), Value::F32(1.0));
    inputs.insert("repeat_last_n".to_string(), Value::Usize(64));
    inputs.insert("presence_penalty".to_string(), Value::F32(0.0));
    inputs.insert("frequency_penalty".to_string(), Value::F32(0.0));
    inputs.insert("seed".to_string(), Value::U32(42));

    let mut emitted = 0usize;
    let mut generated = Vec::new();
    let _ = runner.run_streaming(&mut foundry, &workflow, inputs, |tok, _prefill, _setup, _iter| {
        emitted += 1;
        generated.push(tok);
        Ok(true)
    })?;

    assert!(generated.len() <= 1);
    assert_eq!(generated.len(), emitted);
    Ok(())
}

fn u8_arg_2d(n: usize, k: usize) -> crate::types::TensorArg {
    crate::types::TensorArg {
        buffer: None,
        offset: 0,
        dtype: Dtype::Q8_0,
        dims: smallvec![n, k],
        strides: smallvec![k, 1],
    }
}

fn u8_arg_1d(len: usize) -> crate::types::TensorArg {
    crate::types::TensorArg {
        buffer: None,
        offset: 0,
        dtype: Dtype::Q8_0,
        dims: smallvec![len],
        strides: smallvec![1],
    }
}

#[test]
fn quantized_weight_validation_delegates_to_policy() {
    let mut symbols = SymbolTable::new();
    let w_idx = symbols.get_or_create("w".to_string());

    let mut fast = FastBindings::new(symbols.len());
    // Q8 requires K divisible by 32. 31 is invalid.
    fast.set(w_idx, u8_arg_2d(64, 31));

    let err = validate_quantized_bindings(&symbols, &fast).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("must be divisible by 32"));
}

#[test]
fn quantized_weight_scales_pass_for_consistent_bindings() {
    let mut symbols = SymbolTable::new();
    let w_idx = symbols.get_or_create("w".to_string());
    let s_idx = symbols.get_or_create("w_scales".to_string());

    let mut fast = FastBindings::new(symbols.len());
    fast.set(w_idx, u8_arg_2d(64, 32));
    fast.set(s_idx, u8_arg_1d(128));

    validate_quantized_bindings(&symbols, &fast).unwrap();
}
