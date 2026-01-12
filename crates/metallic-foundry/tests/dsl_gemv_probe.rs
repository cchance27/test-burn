//! Diagnostics for GGUF weight layout vs DSL GEMV output.
//!
//! These tests are ignored by default because they require a local GGUF file and MPS.

use half::f16;
use metallic_foundry::{
    Foundry, MetalError, gguf::{GGUFDataType, GGUFFile, model_loader::GGUFModelLoader, tensor_info::GGUFRawTensor}, model::ModelBuilder, spec::ModelSpec, types::KernelArg as _
};
use rustc_hash::FxHashMap;
use serial_test::serial;

const MODEL_SPEC_PATH: &str = "../../models/qwen25.json";
const GGUF_PATH: &str = "../../models/qwen2.5-coder-0.5b-instruct-fp16.gguf";

fn gguf_available(model: &metallic_foundry::gguf::model_loader::GGUFModel) -> FxHashMap<String, ()> {
    model.tensors.keys().map(|name| (name.clone(), ())).collect()
}

fn gguf_tensor_f32(
    model: &metallic_foundry::gguf::model_loader::GGUFModel,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>, GGUFDataType), MetalError> {
    let tensor = model
        .get_tensor(name)
        .ok_or_else(|| MetalError::InvalidShape(format!("GGUF tensor '{}' not found", name)))?;
    let dims = tensor.dims().to_vec();
    if dims.len() != 2 {
        return Err(MetalError::InvalidShape(format!(
            "GGUF tensor '{}' expected 2D, got dims {:?}",
            name, dims
        )));
    }
    let data_type = tensor.data_type();
    let view = tensor
        .raw_view(&model.gguf_file)
        .map_err(|e| MetalError::InvalidShape(format!("GGUF raw view error for '{}': {:?}", name, e)))?;

    let data: Vec<f32> = match view {
        GGUFRawTensor::F16(values) => values.iter().map(|v| v.to_f32()).collect(),
        GGUFRawTensor::F32(values) => values.iter().map(|v| f16::from_f32(*v).to_f32()).collect(),
        GGUFRawTensor::Bytes(_, dtype) => {
            return Err(MetalError::InvalidShape(format!(
                "Unsupported GGUF dtype {:?} for '{}'",
                dtype, name
            )));
        }
    };

    let expected = dims.iter().product::<usize>();
    if data.len() != expected {
        return Err(MetalError::InvalidShape(format!(
            "GGUF tensor '{}' has {} elements but dims {:?} imply {}",
            name,
            data.len(),
            dims,
            expected
        )));
    }

    Ok((data, dims, data_type))
}

fn gguf_tensor_f32_any(
    model: &metallic_foundry::gguf::model_loader::GGUFModel,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>, GGUFDataType), MetalError> {
    let tensor = model
        .get_tensor(name)
        .ok_or_else(|| MetalError::InvalidShape(format!("GGUF tensor '{}' not found", name)))?;
    let dims = tensor.dims().to_vec();
    let data_type = tensor.data_type();
    let view = tensor
        .raw_view(&model.gguf_file)
        .map_err(|e| MetalError::InvalidShape(format!("GGUF raw view error for '{}': {:?}", name, e)))?;

    let data: Vec<f32> = match view {
        GGUFRawTensor::F16(values) => values.iter().map(|v| v.to_f32()).collect(),
        GGUFRawTensor::F32(values) => values.iter().map(|v| f16::from_f32(*v).to_f32()).collect(),
        GGUFRawTensor::Bytes(_, dtype) => {
            return Err(MetalError::InvalidShape(format!(
                "Unsupported GGUF dtype {:?} for '{}'",
                dtype, name
            )));
        }
    };

    let expected = dims.iter().product::<usize>();
    if data.len() != expected {
        return Err(MetalError::InvalidShape(format!(
            "GGUF tensor '{}' has {} elements but dims {:?} imply {}",
            name,
            data.len(),
            dims,
            expected
        )));
    }

    Ok((data, dims, data_type))
}

fn cpu_gemv_nk(matrix: &[f32], vector: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (row, val) in out.iter_mut().enumerate().take(n) {
        let mut acc = 0.0f32;
        let base = row * k;
        for col in 0..k {
            acc += matrix[base + col] * vector[col];
        }
        *val = acc;
    }
    out
}

fn cpu_gemv_kn(matrix: &[f32], vector: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for col in 0..n {
        let mut acc = 0.0f32;
        for row in 0..k {
            acc += matrix[row * n + col] * vector[row];
        }
        out[col] = acc;
    }
    out
}

fn slice_prefix(data: &[f32], len: usize) -> Result<&[f32], MetalError> {
    if data.len() < len {
        return Err(MetalError::InvalidShape(format!(
            "Vector length {} is smaller than expected K={}",
            data.len(),
            len
        )));
    }
    Ok(&data[..len])
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()))
}

fn avg_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let sum = a.iter().zip(b.iter()).fold(0.0f32, |acc, (x, y)| acc + (x - y).abs());
    if a.is_empty() { 0.0 } else { sum / a.len() as f32 }
}

fn read_f16_buffer(arg: &metallic_foundry::types::TensorArg) -> Vec<f16> {
    let buffer = arg.buffer();
    let len = arg.dims().iter().product::<usize>();
    unsafe {
        use objc2_metal::MTLBuffer;
        let ptr = buffer.contents().as_ptr() as *const f16;
        std::slice::from_raw_parts(ptr, len).to_vec()
    }
}

fn log_layout(name: &str, dims: &[usize], expected_n: usize, expected_k: usize) {
    let layout = if dims.len() == 2 && dims[0] == expected_n && dims[1] == expected_k {
        "NK"
    } else if dims.len() == 2 && dims[0] == expected_k && dims[1] == expected_n {
        "KN"
    } else {
        "UNKNOWN"
    };
    eprintln!(
        "{} dims={:?} expected_n={} expected_k={} layout_hint={}",
        name, dims, expected_n, expected_k, layout
    );
}

fn report_layout_diff(label: &str, cpu_nk: &[f32], cpu_kn: &[f32], dsl_out: &[f32]) -> Result<(), MetalError> {
    let max_nk = max_abs_diff(cpu_nk, dsl_out);
    let avg_nk = avg_abs_diff(cpu_nk, dsl_out);
    let max_kn = max_abs_diff(cpu_kn, dsl_out);
    let avg_kn = avg_abs_diff(cpu_kn, dsl_out);

    eprintln!("{label} NK diff: max={:.6} avg={:.6}", max_nk, avg_nk);
    eprintln!("{label} KN diff: max={:.6} avg={:.6}", max_kn, avg_kn);

    let min_diff = max_nk.min(max_kn);
    if min_diff > 1e-2 {
        return Err(MetalError::InvalidShape(format!(
            "GEMV probe mismatch for {}: min max_diff={min_diff:.6}",
            label
        )));
    }

    Ok(())
}

fn max_abs(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()))
}

#[test]
#[serial]
#[ignore = "requires qwen2.5 gguf file"]
fn test_gguf_bias_summary_qwen25() -> Result<(), MetalError> {
    let gguf_file =
        GGUFFile::load_mmap_and_get_metadata(GGUF_PATH).map_err(|e| MetalError::InvalidShape(format!("GGUF load failed: {e}")))?;
    let gguf_model = GGUFModelLoader::new(gguf_file)
        .load_model()
        .map_err(|e| MetalError::InvalidShape(format!("GGUF model failed: {e}")))?;

    let layer = 0usize;
    let layer_prefix = format!("blk.{layer}.");
    let mut bias_names: Vec<String> = gguf_model
        .tensors
        .keys()
        .filter(|name| name.contains("bias") && name.starts_with(&layer_prefix))
        .cloned()
        .collect();
    bias_names.sort();

    if bias_names.is_empty() {
        eprintln!("No bias tensors found for layer {layer}");
        return Ok(());
    }

    eprintln!("Bias summary for layer {layer}:");
    for name in bias_names {
        let (values, dims, dtype) = gguf_tensor_f32_any(&gguf_model, &name)?;
        let max = max_abs(&values);
        eprintln!("  {} dtype={:?} dims={:?} max_abs={:.6}", name, dtype, dims, max);
    }

    Ok(())
}

#[test]
#[serial]
#[ignore = "requires qwen2.5 gguf file"]
fn test_gguf_weight_layouts_qwen25() -> Result<(), MetalError> {
    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let spec = ModelSpec::from_file(&spec_path).map_err(|e| MetalError::InvalidShape(format!("Spec load failed: {e}")))?;

    let gguf_file =
        GGUFFile::load_mmap_and_get_metadata(GGUF_PATH).map_err(|e| MetalError::InvalidShape(format!("GGUF load failed: {e}")))?;
    let gguf_model = GGUFModelLoader::new(gguf_file)
        .load_model()
        .map_err(|e| MetalError::InvalidShape(format!("GGUF model failed: {e}")))?;

    let available = gguf_available(&gguf_model);
    let arch = &spec.architecture;
    let kv_dim = arch.d_model * arch.n_kv_heads / arch.n_heads;

    let layer = 0usize;
    let weights = [
        ("layer.attn_q", arch.d_model, arch.d_model),
        ("layer.attn_k", kv_dim, arch.d_model),
        ("layer.attn_v", kv_dim, arch.d_model),
        ("layer.attn_output", arch.d_model, arch.d_model),
        ("layer.ffn_gate", arch.ff_dim, arch.d_model),
        ("layer.ffn_up", arch.ff_dim, arch.d_model),
        ("layer.ffn_down", arch.d_model, arch.ff_dim),
    ];

    for (key, expected_n, expected_k) in weights {
        if let Some(name) = arch.tensor_names.resolve(key, Some(layer), &available) {
            let tensor = gguf_model
                .get_tensor(&name)
                .ok_or_else(|| MetalError::InvalidShape(format!("GGUF tensor '{}' not found", name)))?;
            log_layout(&name, tensor.dims(), expected_n, expected_k);
        } else {
            eprintln!("Missing GGUF name for key '{key}'");
        }
    }

    if let Some(name) = arch.tensor_names.resolve("output_weight", None, &available) {
        let tensor = gguf_model
            .get_tensor(&name)
            .ok_or_else(|| MetalError::InvalidShape(format!("GGUF tensor '{}' not found", name)))?;
        log_layout(&name, tensor.dims(), arch.vocab_size, arch.d_model);
    }

    Ok(())
}

#[test]
#[serial]
#[ignore = "requires qwen2.5 gguf file + MPS"]
fn test_dsl_gemv_probe_layer0() -> Result<(), MetalError> {
    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let mut foundry = Foundry::new()?;
    let dsl_model = ModelBuilder::new()
        .with_spec_file(&spec_path)?
        .with_gguf(GGUF_PATH)?
        .build(&mut foundry)?;
    let (mut bindings, _fast_bindings) = dsl_model.prepare_bindings(&mut foundry)?;

    // Only run one layer to keep intermediate buffers stable.
    bindings.set_global("n_layers", "1".to_string());

    // Tokenize input
    let tokenizer = dsl_model.tokenizer()?;
    let tokens = tokenizer.encode("Hello")?;

    // Upload input ids
    let input_buffer = {
        use objc2_metal::MTLDevice;
        let byte_size = tokens.len() * 4;
        let buf = foundry
            .device
            .newBufferWithLength_options(byte_size, objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate input buffer");
        unsafe {
            use objc2_metal::MTLBuffer;
            let ptr = buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, tokens.len());
        }
        metallic_foundry::types::MetalBuffer::from_retained(buf)
    };
    let input_tensor =
        metallic_foundry::types::TensorArg::from_buffer(input_buffer, metallic_foundry::tensor::Dtype::U32, vec![tokens.len()], vec![1]);
    bindings.insert("input_ids".to_string(), input_tensor);

    // Globals
    let arch = dsl_model.architecture();
    let seq_len = tokens.len();
    let d_model = arch.d_model;
    let n_heads = arch.n_heads;
    let n_kv_heads = arch.n_kv_heads;
    let ff_dim = arch.ff_dim;
    let head_dim = d_model / n_heads;
    let position_offset = 0usize;
    let kv_seq_len = position_offset + seq_len;

    bindings.set_global("seq_len", seq_len.to_string());
    bindings.set_global("position_offset", position_offset.to_string());
    bindings.set_global("kv_seq_len", kv_seq_len.to_string());
    bindings.set_global("total_elements_q", (n_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_k", (n_kv_heads * seq_len * head_dim).to_string());
    bindings.set_global("total_elements_hidden", (seq_len * d_model).to_string());
    bindings.set_global("total_elements_ffn", (seq_len * ff_dim).to_string());
    bindings.set_global("total_elements_slice", (n_kv_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_repeat", (n_heads * kv_seq_len * head_dim).to_string());
    bindings.set_global("total_elements_write", (n_kv_heads * seq_len * head_dim).to_string());

    // Run embedding + repeat
    for step in arch.forward.iter().take(2) {
        step.execute(&mut foundry, &mut bindings)?;
    }

    let norm_out = bindings.get("norm_out")?;
    let q_out = bindings.get("q")?;
    let k_out = bindings.get("k")?;
    let v_out = bindings.get("v")?;
    let attn_out = bindings.get("attn_out")?;
    let proj_out = bindings.get("proj_out")?;
    let ffn_norm_out = bindings.get("ffn_norm_out")?;
    let gate_out = bindings.get("gate")?;
    let up_out = bindings.get("up")?;
    let ffn_out = bindings.get("ffn_out")?;

    let norm_out_f32: Vec<f32> = read_f16_buffer(&norm_out).into_iter().map(|v| v.to_f32()).collect();
    let q_f32: Vec<f32> = read_f16_buffer(&q_out).into_iter().map(|v| v.to_f32()).collect();
    let k_f32: Vec<f32> = read_f16_buffer(&k_out).into_iter().map(|v| v.to_f32()).collect();
    let v_f32: Vec<f32> = read_f16_buffer(&v_out).into_iter().map(|v| v.to_f32()).collect();
    let attn_in_f32: Vec<f32> = read_f16_buffer(&attn_out).into_iter().map(|v| v.to_f32()).collect();
    let proj_out_f32: Vec<f32> = read_f16_buffer(&proj_out).into_iter().map(|v| v.to_f32()).collect();
    let ffn_norm_f32: Vec<f32> = read_f16_buffer(&ffn_norm_out).into_iter().map(|v| v.to_f32()).collect();
    let gate_f32: Vec<f32> = read_f16_buffer(&gate_out).into_iter().map(|v| v.to_f32()).collect();
    let up_f32: Vec<f32> = read_f16_buffer(&up_out).into_iter().map(|v| v.to_f32()).collect();
    let ffn_out_f32: Vec<f32> = read_f16_buffer(&ffn_out).into_iter().map(|v| v.to_f32()).collect();

    let available = gguf_available(dsl_model.gguf_model());
    let layer = 0usize;

    let qkv_probes = [
        ("layer.attn_q", &q_f32, &norm_out_f32),
        ("layer.attn_k", &k_f32, &norm_out_f32),
        ("layer.attn_v", &v_f32, &norm_out_f32),
    ];

    for (key, dsl_out, vector_x) in qkv_probes {
        let gguf_name = arch
            .tensor_names
            .resolve(key, Some(layer), &available)
            .ok_or_else(|| MetalError::InvalidShape(format!("Missing GGUF name for {}", key)))?;
        let (weights, dims, dtype) = gguf_tensor_f32(dsl_model.gguf_model(), &gguf_name)?;

        let n = dsl_out.len();
        let k = vector_x.len();

        eprintln!("\nProbe {} (gguf='{}' dtype={:?} dims={:?})", key, gguf_name, dtype, dims);

        let cpu_nk = cpu_gemv_nk(&weights, vector_x, n, k);
        let cpu_kn = cpu_gemv_kn(&weights, vector_x, n, k);
        report_layout_diff(key, &cpu_nk, &cpu_kn, dsl_out)?;
    }

    // Attention output projection (attn_out -> proj_out)
    let attn_out_key = "layer.attn_output";
    if let Some(attn_out_name) = arch.tensor_names.resolve(attn_out_key, Some(layer), &available) {
        let (weights, dims, dtype) = gguf_tensor_f32(dsl_model.gguf_model(), &attn_out_name)?;
        let n = proj_out_f32.len();
        let k = arch.d_model;
        let vector_x = slice_prefix(&attn_in_f32, k)?;

        eprintln!(
            "\nProbe {} (gguf='{}' dtype={:?} dims={:?})",
            attn_out_key, attn_out_name, dtype, dims
        );

        let cpu_nk = cpu_gemv_nk(&weights, vector_x, n, k);
        let cpu_kn = cpu_gemv_kn(&weights, vector_x, n, k);
        report_layout_diff(attn_out_key, &cpu_nk, &cpu_kn, &proj_out_f32)?;
    } else {
        eprintln!("Missing GGUF name for '{attn_out_key}'");
    }

    // FFN gate/up projections (ffn_norm_out -> gate/up)
    let ffn_probes = [
        ("layer.ffn_gate", &gate_f32, &ffn_norm_f32),
        ("layer.ffn_up", &up_f32, &ffn_norm_f32),
    ];

    for (key, dsl_out, vector_x) in ffn_probes {
        let gguf_name = arch
            .tensor_names
            .resolve(key, Some(layer), &available)
            .ok_or_else(|| MetalError::InvalidShape(format!("Missing GGUF name for {}", key)))?;
        let (weights, dims, dtype) = gguf_tensor_f32(dsl_model.gguf_model(), &gguf_name)?;

        let n = dsl_out.len();
        let k = vector_x.len();

        eprintln!("\nProbe {} (gguf='{}' dtype={:?} dims={:?})", key, gguf_name, dtype, dims);

        let cpu_nk = cpu_gemv_nk(&weights, vector_x, n, k);
        let cpu_kn = cpu_gemv_kn(&weights, vector_x, n, k);
        report_layout_diff(key, &cpu_nk, &cpu_kn, dsl_out)?;
    }

    // FFN down projection (swiglu output -> ffn_out)
    let ffn_down_key = "layer.ffn_down";
    if let Some(ffn_down_name) = arch.tensor_names.resolve(ffn_down_key, Some(layer), &available) {
        let (weights, dims, dtype) = gguf_tensor_f32(dsl_model.gguf_model(), &ffn_down_name)?;
        let n = ffn_out_f32.len();
        let k = up_f32.len();

        eprintln!(
            "\nProbe {} (gguf='{}' dtype={:?} dims={:?})",
            ffn_down_key, ffn_down_name, dtype, dims
        );

        let cpu_nk = cpu_gemv_nk(&weights, &up_f32, n, k);
        let cpu_kn = cpu_gemv_kn(&weights, &up_f32, n, k);
        report_layout_diff(ffn_down_key, &cpu_nk, &cpu_kn, &ffn_out_f32)?;
    } else {
        eprintln!("Missing GGUF name for '{ffn_down_key}'");
    }

    Ok(())
}
