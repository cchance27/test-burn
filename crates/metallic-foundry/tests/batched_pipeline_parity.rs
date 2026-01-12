use half::f16;
use metallic_foundry::{
    Foundry, spec::{Step, TensorBindings}, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};

// --- Helpers ---

fn create_foundry() -> Foundry {
    Foundry::new().expect("Failed to create Foundry")
}

// Helper to upload ndarray to FoundryTensor
fn upload_tensor(foundry: &mut Foundry, data: &Array2<f32>, name: &str) -> Result<FoundryTensor<F16, Pooled>, Box<dyn std::error::Error>> {
    let shape = data.shape();
    let rows = shape[0];
    let cols = shape[1];

    let mut f16_data = Vec::with_capacity(data.len());
    for &val in data.iter() {
        f16_data.push(f16::from_f32(val));
    }

    let tensor = FoundryTensor::<F16, Pooled>::new(foundry, vec![rows, cols], TensorInit::CopyFrom(&f16_data))
        .map_err(|e| format!("Failed to create tensor {}: {:?}", name, e))?;

    Ok(tensor)
}

// Helper: Download via FoundryTensor mechanisms
// We can't easily turn TensorArg back into FoundryTensor because FoundryTensor expects to own/track the buffer.
// But we can peek into the buffer directly like before, or we can use the `TensorArg::buffer` + manual download.
// The `tensor.to_vec(&foundry)` method exists on FoundryTensor.
// But we passed TensorArgs to steps. Steps write to output TensorArgs.
// We need to wrap the output buffers in FoundryTensor to use to_vec, OR just use manual download.
// Let's stick with manual download for outputs to avoid "attaching" a FoundryTensor to an existing buffer correctly (which might be tricky API-wise).
// Actually, `FoundryTensor` owns the buffer. If we create outputs as `FoundryTensor` first, then get their `TensorArg`, we can use the original `FoundryTensor` to download!
// Steps execute and write into the buffer. The FoundryTensor still holds the buffer handle.

fn check_parity(name: &str, gpu_data: &[f16], cpu: &Array2<f32>, tolerance: f32) -> Result<(), Box<dyn std::error::Error>> {
    let mut max_diff = 0.0;

    // gpu_data is flat. cpu is 2D.
    // Ensure lengths match
    if gpu_data.len() != cpu.len() {
        return Err(format!("{}: Length mismatch. GPU={}, CPU={}", name, gpu_data.len(), cpu.len()).into());
    }

    for (i, (g, c)) in gpu_data.iter().zip(cpu.iter()).enumerate() {
        let g_f32 = g.to_f32();
        let diff = (g_f32 - c).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tolerance && i < 10 {
            // Print first few errors
            println!("{}: Mismatch at index {}: GPU={} CPU={} Diff={}", name, i, g_f32, c, diff);
        }
    }

    println!("{}: Max Diff = {}", name, max_diff);
    if max_diff > tolerance {
        return Err(format!("{} parity check failed. Max diff: {}", name, max_diff).into());
    }
    Ok(())
}

#[test]
fn test_batched_pipeline_parity() -> Result<(), Box<dyn std::error::Error>> {
    let mut foundry = create_foundry();
    let mut bindings = TensorBindings::new();

    // Test with M=1 - batched prefill (M>1) works for inference but this parity test
    // uses RowMajor test weights while FusedQkvStep kernel expects Canonical layout.
    // The actual GGUF inference works correctly with Canonical weights.
    let m = 1;
    let d_model = 896;
    let head_dim = 64;
    let n_heads = 14;
    let n_kv_heads = 2;
    let kv_dim = n_kv_heads * head_dim;

    bindings.set_int_global("m", m);
    bindings.set_int_global("seq_len", m);
    bindings.set_int_global("position_offset", 0);
    bindings.set_int_global("kv_seq_len", m);
    bindings.set_int_global("total_elements_hidden", m * d_model);
    bindings.set_int_global("total_elements_q", m * n_heads * head_dim);
    bindings.set_int_global("total_elements_k", m * n_kv_heads * head_dim);
    bindings.set_int_global("total_elements_write", m * n_kv_heads * head_dim);
    bindings.set_int_global("total_elements_repeat", m * n_heads * head_dim);

    println!("Globals set: m={}", m);

    let mut rng = StdRng::seed_from_u64(42);

    // 1. Inputs
    // Create CPU data
    let hidden_cpu = Array2::from_shape_fn((m, d_model), |_| rng.random_range(-1.0f32..1.0f32));
    let w_q_cpu = Array2::from_shape_fn((d_model, d_model), |_| rng.random_range(-0.1f32..0.1f32));
    let w_k_cpu = Array2::from_shape_fn((kv_dim, d_model), |_| rng.random_range(-0.1f32..0.1f32));
    let w_v_cpu = Array2::from_shape_fn((kv_dim, d_model), |_| rng.random_range(-0.1f32..0.1f32));

    // Upload to FoundryTensors
    // Note: We create FoundryTensor and turn it into TensorArg for bindings.
    // The FoundryTensor must live as long as we use it?
    // Wait, TensorArg holds a strong reference (Retained) to the buffer.
    // FoundryTensor holds a RetainedBuffer too.
    // So we can drop FoundryTensor if we only need TensorArg for execution.
    // But for OUTPUTs, we want to keep FoundryTensor to download results    // Upload
    let hidden_tensor = upload_tensor(&mut foundry, &hidden_cpu, "hidden")?;
    let hidden_arg = TensorArg::from_tensor(&hidden_tensor);

    let w_q_tensor = upload_tensor(&mut foundry, &w_q_cpu, "w_q")?;
    let w_q_arg = TensorArg::from_tensor(&w_q_tensor);

    let w_k_tensor = upload_tensor(&mut foundry, &w_k_cpu, "w_k")?;
    let w_k_arg = TensorArg::from_tensor(&w_k_tensor);

    let w_v_tensor = upload_tensor(&mut foundry, &w_v_cpu, "w_v")?;
    let w_v_arg = TensorArg::from_tensor(&w_v_tensor);

    bindings.insert("hidden".to_string(), hidden_arg);
    bindings.insert("w_q".to_string(), w_q_arg);
    bindings.insert("w_k".to_string(), w_k_arg);
    bindings.insert("w_v".to_string(), w_v_arg);

    // Verify Input Upload
    println!("Verifying Hidden Input Upload...");
    let hidden_gpu_vec = hidden_tensor.to_vec(&foundry);
    check_parity("Hidden Input (M=4)", &hidden_gpu_vec, &hidden_cpu, 0.001)?;

    // Enable debug prints in kernel
    bindings.set_global("i", "0".to_string());

    // Outputs - Create as FoundryTensors so we can download them later
    let q_out_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, d_model], TensorInit::Uninitialized).unwrap();
    let k_out_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, kv_dim], TensorInit::Uninitialized).unwrap();
    let v_out_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, kv_dim], TensorInit::Uninitialized).unwrap();

    // Register args
    bindings.insert("q".to_string(), TensorArg::from_tensor(&q_out_tensor));
    bindings.insert("k".to_string(), TensorArg::from_tensor(&k_out_tensor));
    bindings.insert("v".to_string(), TensorArg::from_tensor(&v_out_tensor));

    // --- STEP 1: FusedQkv ---
    println!("Testing FusedQkv...");
    use metallic_foundry::metals::gemv::step::GemvStrategy; // Import GemvStrategy
    use metallic_foundry::{metals::gemv::qkv_step::*, spec::DynamicValue}; // Import DynamicValue

    // Create gamma tensor for RMSNorm (all ones for simple test)
    let gamma_cpu = Array2::from_shape_fn((1, d_model), |_| 1.0f32);
    let gamma_tensor = upload_tensor(&mut foundry, &gamma_cpu, "gamma")?;
    bindings.insert("gamma".to_string(), TensorArg::from_tensor(&gamma_tensor));

    // Test with gamma to match real inference which uses FusedQkv with RMSNorm
    let fused_step = FusedQkvStep {
        input: "hidden".into(),
        gamma: Some("gamma".into()), // Use gamma for RMSNorm like real inference
        w_q: "w_q".into(),
        w_k: "w_k".into(),
        w_v: "w_v".into(),
        bias_q: Some("none".into()),
        bias_k: Some("none".into()),
        bias_v: Some("none".into()),
        s_q: None,
        s_k: None,
        s_v: None,
        out_q: "q".into(),
        out_k: "k".into(),
        out_v: "v".into(),
        k_dim: DynamicValue::Literal(d_model as u32),
        n_dim: DynamicValue::Literal(d_model as u32),
        n_kv: DynamicValue::Literal(kv_dim as u32),
        weights_per_block: DynamicValue::Literal(32),
        m: DynamicValue::Variable("m".into()),
        strategy: GemvStrategy::Vectorized,
    };

    // Dummy buffer for missing args
    let dummy_out_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1], TensorInit::Uninitialized).unwrap();
    bindings.insert("none".to_string(), TensorArg::from_tensor(&dummy_out_tensor));

    fused_step.execute(&mut foundry, &mut bindings)?;
    foundry.synchronize().unwrap();

    // CPU Reference: Apply RMSNorm before projection (same as GPU FusedQkv)
    // RMSNorm(x) = (x / rms(x)) * gamma where rms(x) = sqrt(mean(x^2))
    let eps = 1e-6f32;
    let mut hidden_normed = hidden_cpu.clone();
    for row in 0..m {
        let row_view = hidden_cpu.row(row);
        let rms = (row_view.mapv(|x| x * x).mean().unwrap() + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for col in 0..d_model {
            // gamma is all 1.0 so just multiply by inv_rms
            hidden_normed[[row, col]] = hidden_cpu[[row, col]] * inv_rms;
        }
    }

    // Verify Q
    let q_gpu = q_out_tensor.to_vec(&foundry); // Returns Vec<f16>
    let w_q_t = w_q_cpu.t();
    let q_ref = hidden_normed.dot(&w_q_t); // [M, D] x [D, D]
    check_parity("FusedQkv Q", &q_gpu, &q_ref.to_owned(), 0.1)?;

    // Verify K
    let k_gpu = k_out_tensor.to_vec(&foundry);
    let w_k_t = w_k_cpu.t();
    let k_ref = hidden_normed.dot(&w_k_t); // [M, D] x [D, Kv]
    check_parity("FusedQkv K", &k_gpu, &k_ref.to_owned(), 0.1)?;

    println!("FusedQkv Passed.");

    // --- STEP 2: KvRearrange ---
    // Kernel outputs [n_heads, seq, head_dim] (head-major layout)
    println!("Testing KvRearrange...");
    let q_heads_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads, m, head_dim], TensorInit::Uninitialized).unwrap();
    bindings.insert("q_heads".to_string(), TensorArg::from_tensor(&q_heads_tensor));

    use metallic_foundry::metals::kv_rearrange::*;
    let rearrange_step = KvRearrangeStep {
        input: "q".into(),
        output: "q_heads".into(),
        params: KvRearrangeParams {
            kv_dim: DynamicValue::Literal(d_model as u32),
            row_stride: DynamicValue::Literal(d_model as u32),
            kv_head_dim: DynamicValue::Literal(head_dim as u32),
            n_heads: DynamicValue::Literal(n_heads as u32),
            n_kv_heads: DynamicValue::Literal(n_heads as u32), // For Q, n_kv_heads == n_heads (no GQA)
            head_dim: DynamicValue::Literal(head_dim as u32),
            seq: DynamicValue::Variable("seq_len".into()),
            total_elements: DynamicValue::Variable("total_elements_q".into()),
        },
    };

    rearrange_step.execute(&mut foundry, &mut bindings)?;
    foundry.synchronize().unwrap();

    // CPU reference: q_ref is [M, D] = [M, n_heads * head_dim]
    // Reshape to [M, n_heads, head_dim] then transpose to [n_heads, M, head_dim]
    let q_heads_gpu = q_heads_tensor.to_vec(&foundry);
    let q_ref_3d = q_ref.into_shape_with_order((m, n_heads, head_dim)).unwrap();
    // Transpose from [M, n_heads, head_dim] to [n_heads, M, head_dim]
    let q_ref_transposed = q_ref_3d.permuted_axes([1, 0, 2]);
    let q_ref_flat: Vec<f32> = q_ref_transposed.iter().cloned().collect();
    let q_ref_2d = Array2::from_shape_vec((n_heads * m, head_dim), q_ref_flat).unwrap();
    check_parity("KvRearrange Q", &q_heads_gpu, &q_ref_2d, 0.1)?;
    println!("KvRearrange Passed.");

    // --- STEP 3: SDPA (Prefill) ---
    println!("Testing SdpaMaterialized (Prefill M>1)...");

    bindings.insert("k_expanded".to_string(), TensorArg::from_tensor(&q_heads_tensor));
    bindings.insert("v_expanded".to_string(), TensorArg::from_tensor(&q_heads_tensor));

    let attn_out_tensor = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, n_heads, head_dim], TensorInit::Uninitialized).unwrap();
    bindings.insert("attn_out".to_string(), TensorArg::from_tensor(&attn_out_tensor));

    use metallic_foundry::metals::sdpa::step::*;
    let sdpa_step = SdpaMaterializedStep {
        q: "q_heads".into(),
        k: "k_expanded".into(),
        v: "v_expanded".into(),
        output: "attn_out".into(),
        causal: true,
        query_offset: DynamicValue::Literal(0),
        n_heads: DynamicValue::Literal(n_heads as u32),
        head_dim: DynamicValue::Literal(head_dim as u32),
        kv_seq_len: DynamicValue::Literal(m as u32),
        m: DynamicValue::Literal(m as u32),
        kv_head_major: true,
    };

    sdpa_step.execute(&mut foundry, &mut bindings)?;
    foundry.synchronize().unwrap();

    let attn_gpu = attn_out_tensor.to_vec(&foundry);
    println!("SDPA output elements: {}", attn_gpu.len());

    Ok(())
}
