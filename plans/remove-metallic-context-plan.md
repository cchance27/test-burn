# Plan: Remove metallic-context Engine

## Overview

We have reached parity with the new metallic-foundry inference engine, and it's time to remove the legacy metallic-context engine completely. This involves:

1. Deleting the `metallic-context` crate
2. Removing all references to it throughout the codebase
3. Removing the CLI `--engine` option (since we only have one engine now)
4. Converting parity tests appropriately

## Files to Modify

### 1. Workspace Configuration

**File: [`Cargo.toml`](Cargo.toml)**
- Remove `crates/metallic-context` from workspace members
- Remove `metallic-context = { path = "crates/metallic-context", ... }` from dependencies
- Remove features `src_kernels` and `built_kernels` that reference metallic-context

### 2. metallic-foundry Crate

**File: [`crates/metallic-foundry/Cargo.toml`](crates/metallic-foundry/Cargo.toml)**
- Remove `metallic-context = { path = "../metallic-context" }` from dev-dependencies

### 3. CLI Configuration

**File: [`src/cli/config.rs`](src/cli/config.rs)**
- Remove the `Engine` enum entirely
- Remove the `engine` field from `CliConfig` struct
- Remove the `#[arg(long, value_enum, default_value_t = Engine::Context)] pub engine: Engine` line
- Update tests that reference `engine: Engine::Context`

### 4. Main Application

**File: [`src/main.rs`](src/main.rs)**
- Remove imports from `metallic_context`
- Remove the `worker_engine` variable
- Remove the entire `match worker_engine` block for `Engine::Context`
- Keep only the `Engine::Foundry` path (now the default and only path)
- Remove `report_model_weight_breakdown` function if it relies on metallic-context types
- Simplify the code to always use Foundry engine

### 5. Test Files in metallic-foundry/tests/

These tests compare Foundry vs Context and should be **DELETED**:

| File | Description |
|------|-------------|
| `dsl_vs_context_parity.rs` | Large parity test comparing DSL vs Context |
| `dsl_vs_context_generation_seed_parity.rs` | Generation seed parity |
| `dsl_vs_context_chat_prefill_parity.rs` | Chat prefill parity |
| `gemv_v2_context_parity.rs` | GEMV v2 vs Context parity |
| `softmax_v2_context_parity.rs` | Softmax v2 vs Context parity |
| `rope_v2_context_parity.rs` | RoPE v2 vs Context parity |
| `sdpa_v2_context_parity.rs` | SDPA v2 vs Context parity |
| `sdpa_v2_feature_parity.rs` | SDPA v2 feature parity |

These tests use Context as a reference but should be **CONVERTED to CPU-only parity tests**:

| File | Action |
|------|--------|
| `kv_cache.rs` | Convert to Foundry vs CPU only |
| `kv_rearrange.rs` | Convert to Foundry vs CPU only |
| `utility_kernels.rs` | Convert to Foundry vs CPU only |
| `rope.rs` | Convert to Foundry vs CPU only |
| `qkv_parity.rs` | Convert to Foundry vs CPU only |
| `rmsnorm_gemv_fused.rs` | Convert to Foundry vs CPU only |
| `fp16_fused_parity.rs` | Convert to Foundry vs CPU only |
| `bench_qwen25_matmul.rs` | Remove or convert - benchmark using Context |
| `q8_qkv_fused.rs` | Convert to Foundry vs CPU only |
| `matmul_alpha_beta_parity_test.rs` | Convert to Foundry vs CPU only |
| `fp16_transposed_parity.rs` | Convert to Foundry vs CPU only |
| `matmul.rs` | Convert to Foundry vs CPU only |
| `swiglu.rs` | Convert to Foundry vs CPU only |
| `rmsnorm.rs` | Convert to Foundry vs CPU only |
| `gemm_v2_parity.rs` | Convert to Foundry vs CPU only |
| `gemm_v2_benchmark.rs` | Remove or convert - benchmark using Context |
| `softmax_kernel_consistency_test.rs` | Convert to Foundry vs CPU only |
| `matmul_q8_m1_mlx_heuristic_test.rs` | Convert to Foundry vs CPU only |
| `q8_parity_qwen25.rs` | Convert to Foundry vs CPU only |
| `q8_proxy_smoke.rs` | Convert to Foundry vs CPU only |
| `embedding.rs` | Convert to Foundry vs CPU only |
| `repeat_kv_heads.rs` | Convert to Foundry vs CPU only |

### 6. Delete the metallic-context Crate

Delete the entire directory:
```
crates/metallic-context/
```

This includes:
- All source files
- Build scripts
- Kernels (Metal shaders)
- Tests
- Benchmarks
- Python utilities

## Execution Order

1. **Phase 1: Remove crate from workspace**
   - Update root `Cargo.toml`
   - Update `crates/metallic-foundry/Cargo.toml`

2. **Phase 2: Update CLI**
   - Remove `Engine` enum from `src/cli/config.rs`
   - Remove `engine` field from `CliConfig`

3. **Phase 3: Update main.rs**
   - Remove metallic-context imports
   - Remove Context engine code path
   - Simplify to always use Foundry

4. **Phase 4: Handle test files**
   - Delete tests that only compare Foundry vs Context
   - Convert tests that compare Foundry vs Context vs CPU to Foundry vs CPU only

5. **Phase 5: Delete metallic-context crate**
   - Delete entire `crates/metallic-context/` directory

6. **Phase 6: Verify and clean up**
   - Run `cargo build`
   - Run `cargo +nightly fmt`
   - Run `cargo clippy --fix --allow-dirty --allow-staged`
   - Run `cargo test` (for remaining tests)

## Detailed Code Changes

### src/cli/config.rs Changes

```rust
// REMOVE this enum entirely:
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum Engine {
    Context,
    Foundry,
}

// REMOVE this field from CliConfig:
#[arg(long, value_enum, default_value_t = Engine::Context)]
pub engine: Engine,
```

### src/main.rs Changes

Remove imports:
```rust
// REMOVE:
use metallic_context::{
    Context, F16Element, TensorElement, Tokenizer, gguf::{GGUFFile, model_loader::GGUFModelLoader}, kernels::{KernelBackendKind, KernelBackendOverride, KernelBackendOverrides}, profiling_state
};
```

Remove the engine matching:
```rust
// REMOVE the entire match block:
match worker_engine {
    cli::config::Engine::Context => { /* ... */ }
    cli::config::Engine::Foundry => { /* ... */ }
}
```

Keep only the Foundry path code (without the match wrapper).

## Risks and Considerations

1. **Feature flags**: The `src_kernels` and `built_kernels` features in root Cargo.toml reference metallic-context. These need to be removed or redirected.

2. **Profiling state**: `profiling_state::initialize_profiling_state_from_env()` comes from metallic-context. Need to verify if this functionality exists in metallic-foundry or needs to be migrated.

3. **Backend overrides**: The `KernelBackendKind`, `KernelBackendOverride`, `KernelBackendOverrides` types come from metallic-context. These may need to be replaced with Foundry equivalents if the CLI still supports backend selection.

4. **Test coverage**: Some tests may provide valuable coverage even if they compare against Context. Evaluate each test before deletion.

## Estimated Impact

- **Files deleted**: ~150+ files in metallic-context crate + ~8 test files
- **Files modified**: ~5-10 files
- **Lines removed**: ~100,000+ lines (entire metallic-context crate)
