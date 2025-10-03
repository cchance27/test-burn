use crate::metallic::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage};
use std::sync::{Mutex, OnceLock};

const FORCE_MATMUL_BACKEND_ENV: &str = "FORCE_MATMUL_BACKEND";

fn backend_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn set_backend_env_var(value: &str) {
    // SAFETY: Environment mutations are guarded by `backend_env_lock`, ensuring
    // the process environment is only modified while the mutex is held.
    unsafe {
        std::env::set_var(FORCE_MATMUL_BACKEND_ENV, value);
    }
}

fn restore_backend_env_var(previous: Option<String>) {
    // SAFETY: See `set_backend_env_var` for synchronization guarantees.
    unsafe {
        if let Some(prev) = previous {
            std::env::set_var(FORCE_MATMUL_BACKEND_ENV, prev);
        } else {
            std::env::remove_var(FORCE_MATMUL_BACKEND_ENV);
        }
    }
}

fn new_context_for_backend(backend: &str) -> Result<Context<F32Element>, MetalError> {
    let guard = backend_env_lock().lock().expect("env mutex poisoned");
    let previous = std::env::var(FORCE_MATMUL_BACKEND_ENV).ok();
    set_backend_env_var(backend);
    let ctx_result = Context::<F32Element>::new();
    restore_backend_env_var(previous);
    drop(guard);
    ctx_result
}

struct MatmulScenario {
    description: &'static str,
    a_dims: Vec<usize>,
    b_dims: Vec<usize>,
    transpose_a: bool,
    transpose_b: bool,
}

fn tensor_from_data(ctx: &Context<F32Element>, dims: &[usize], data: &[f32]) -> Result<Tensor<F32Element>, MetalError> {
    Tensor::new(dims.to_vec(), TensorStorage::Dedicated(ctx), TensorInit::CopyFrom(data))
}

fn assert_tensors_close(
    lhs: &Tensor<F32Element>,
    rhs: &Tensor<F32Element>,
    absolute_tolerance: f32,
    relative_tolerance: f32,
    context: &str,
) {
    let lhs_vals = lhs.to_vec();
    let rhs_vals = rhs.to_vec();
    assert_eq!(lhs_vals.len(), rhs_vals.len(), "{} length mismatch", context);

    for (idx, (l, r)) in lhs_vals.iter().zip(rhs_vals.iter()).enumerate() {
        let diff = (l - r).abs();
        let scale = l.abs().max(r.abs());
        let allowed = absolute_tolerance.max(relative_tolerance * scale);
        assert!(
            diff <= allowed,
            "{} mismatch at index {}: lhs={} rhs={} diff={} (> {})",
            context,
            idx,
            l,
            r,
            diff,
            allowed
        );
    }
}

fn run_matmul_backend_comparison(index: usize, scenario: &MatmulScenario) -> Result<(), MetalError> {
    let mut ctx_mps = new_context_for_backend("mps")?;
    let mut ctx_mlx = new_context_for_backend("mlx")?;

    let a_len = scenario.a_dims.iter().product::<usize>();
    let b_len = scenario.b_dims.iter().product::<usize>();

    let a_data: Vec<f32> = (0..a_len).map(|i| (i as f32) * 0.01 + 0.05 * (index as f32 + 1.0)).collect();
    let b_data: Vec<f32> = (0..b_len).map(|i| (i as f32) * -0.015 + 0.07 * (index as f32 + 1.5)).collect();

    let a_mps = tensor_from_data(&ctx_mps, &scenario.a_dims, &a_data)?;
    let b_mps = tensor_from_data(&ctx_mps, &scenario.b_dims, &b_data)?;
    let a_mlx = tensor_from_data(&ctx_mlx, &scenario.a_dims, &a_data)?;
    let b_mlx = tensor_from_data(&ctx_mlx, &scenario.b_dims, &b_data)?;

    let out_mps = ctx_mps.matmul(&a_mps, &b_mps, scenario.transpose_a, scenario.transpose_b)?;
    ctx_mps.synchronize();
    let out_mlx = ctx_mlx.matmul(&a_mlx, &b_mlx, scenario.transpose_a, scenario.transpose_b)?;
    ctx_mlx.synchronize();

    assert_tensors_close(&out_mps, &out_mlx, 1e-3, 1e-6, scenario.description);
    Ok(())
}

fn run_alpha_beta_case(index: usize, alpha: f32, beta: f32) -> Result<(), MetalError> {
    let mut ctx_mps = new_context_for_backend("mps")?;
    let mut ctx_mlx = new_context_for_backend("mlx")?;

    let m = 4;
    let k = 6;
    let n = 5;

    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.02 + 0.03 * (index as f32 + 1.0)).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * -0.025 + 0.04 * (index as f32 + 2.0)).collect();
    let c_data: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.017 - 0.05 * (index as f32 + 1.5)).collect();

    let a_mps = tensor_from_data(&ctx_mps, &[m, k], &a_data)?;
    let b_mps = tensor_from_data(&ctx_mps, &[k, n], &b_data)?;
    let c_mps = tensor_from_data(&ctx_mps, &[m, n], &c_data)?;

    let a_mlx = tensor_from_data(&ctx_mlx, &[m, k], &a_data)?;
    let b_mlx = tensor_from_data(&ctx_mlx, &[k, n], &b_data)?;
    let c_mlx = tensor_from_data(&ctx_mlx, &[m, n], &c_data)?;

    let out_mps = ctx_mps.matmul_alpha_beta(&a_mps, &b_mps, &c_mps, false, false, alpha, beta)?;
    ctx_mps.synchronize();
    let out_mlx = ctx_mlx.matmul_alpha_beta(&a_mlx, &b_mlx, &c_mlx, false, false, alpha, beta)?;
    ctx_mlx.synchronize();

    assert_tensors_close(&out_mps, &out_mlx, 1e-4, 1e-6, &format!("alpha={} beta={}", alpha, beta));
    Ok(())
}

#[test]
fn test_matmul_backends_match_qwen25_use_cases() -> Result<(), MetalError> {
    let d_model = 96;
    let hidden_dim = 192;
    let kv_dim = 32;
    let batch = 2;
    let seq = 3;
    let m = batch * seq;

    let scenarios = [
        MatmulScenario {
            description: "fused_qkv_projection",
            a_dims: vec![m, d_model],
            b_dims: vec![d_model, d_model + 2 * kv_dim],
            transpose_a: false,
            transpose_b: false,
        },
        MatmulScenario {
            description: "attention_output_weight",
            a_dims: vec![m, d_model],
            b_dims: vec![d_model, d_model],
            transpose_a: false,
            transpose_b: true,
        },
        MatmulScenario {
            description: "swiglu_gate_no_transpose",
            a_dims: vec![m, d_model],
            b_dims: vec![d_model, hidden_dim],
            transpose_a: false,
            transpose_b: false,
        },
        MatmulScenario {
            description: "swiglu_gate_transposed",
            a_dims: vec![m, d_model],
            b_dims: vec![hidden_dim, d_model],
            transpose_a: false,
            transpose_b: true,
        },
        MatmulScenario {
            description: "swiglu_down_no_transpose",
            a_dims: vec![m, hidden_dim],
            b_dims: vec![hidden_dim, d_model],
            transpose_a: false,
            transpose_b: false,
        },
        MatmulScenario {
            description: "swiglu_down_transposed",
            a_dims: vec![m, hidden_dim],
            b_dims: vec![d_model, hidden_dim],
            transpose_a: false,
            transpose_b: true,
        },
        MatmulScenario {
            description: "single_token_output",
            a_dims: vec![1, d_model],
            b_dims: vec![d_model, d_model],
            transpose_a: false,
            transpose_b: true,
        },
    ];

    for (index, scenario) in scenarios.iter().enumerate() {
        run_matmul_backend_comparison(index, scenario)?;
    }

    Ok(())
}

#[test]
fn test_matmul_alpha_beta_backends_match_extremes() -> Result<(), MetalError> {
    let cases = [(1.0_f32, 0.0_f32), (2.5_f32, 0.0_f32), (0.0_f32, 1.0_f32), (2.5_f32, -1.75_f32)];

    for (index, (alpha, beta)) in cases.into_iter().enumerate() {
        run_alpha_beta_case(index, alpha, beta)?;
    }

    Ok(())
}
