# Maintainability Baseline (2026-02-22)

Scope: Rust source layout and unit-test organization, with focus on `crates/metallic-foundry`.

## Repository Baseline

- Total Rust files (`*.rs`): `272`
- Files over `500` lines: `34`
- Files over `1000` lines: `7`
- Files over `2000` lines: `3`
- Files with inline `#[cfg(test)]`/`mod tests`: `50`

## Largest Rust Files (Repo-Wide)

1. `crates/metallic-foundry/src/model/executor.rs` (`3492`)
2. `crates/metallic-macros/src/lib.rs` (`2902`)
3. `crates/metallic-foundry/src/policy/q5_k.rs` (`1507`)
4. `crates/metallic-foundry/src/tokenizer.rs` (`1454`)
5. `crates/metallic-foundry/tests/gemm_v2_parity.rs` (`1249`)
6. `crates/metallic-foundry/src/metals/flashattention/step.rs` (`1114`)
7. `crates/metallic-gui/src/state.rs` (`992`)
8. `crates/metallic-gui/src/components/input/mod.rs` (`938`)
9. `crates/metallic-foundry/src/workflow/ops/format_chat.rs` (`898`)
10. `src/tui/app.rs` (`852`)

## Foundry Source Baseline (`crates/metallic-foundry/src`)

- Total Rust files: `135`
- Files over `500` lines: `18`
- Files over `1000` lines: `5`

### Foundry Source Files Above 500 Lines

1. `crates/metallic-foundry/src/model/executor.rs` (`3492`)
2. `crates/metallic-foundry/src/policy/q5_k.rs` (`1507`)
3. `crates/metallic-foundry/src/tokenizer.rs` (`1454`)
4. `crates/metallic-foundry/src/metals/flashattention/step.rs` (`1114`)
5. `crates/metallic-foundry/src/workflow/ops/format_chat.rs` (`898`)
6. `crates/metallic-foundry/src/metals/sdpa/step.rs` (`783`)
7. `crates/metallic-foundry/src/metals/swiglu/step.rs` (`775`)
8. `crates/metallic-foundry/src/compound/mod.rs` (`752`)
9. `crates/metallic-foundry/src/types/mod.rs` (`746`)
10. `crates/metallic-foundry/src/workflow/ops/control_flow.rs` (`644`)
11. `crates/metallic-foundry/src/workflow/ops/sample.rs` (`638`)
12. `crates/metallic-foundry/src/workflow/ops/prefill.rs` (`596`)
13. `crates/metallic-foundry/src/lib.rs` (`585`)
14. `crates/metallic-foundry/src/compound/code_builder.rs` (`561`)
15. `crates/metallic-foundry/src/metals/mma/stages.rs` (`511`)
16. `crates/metallic-foundry/src/metals/gemv/qkv_step.rs` (`503`)
17. `crates/metallic-foundry/src/workflow/spec.rs` (`501`)

## Mixed Code + Unit Test Hotspots (Foundry)

These currently embed `#[cfg(test)]` modules in production files and are highest-value candidates for `.test.rs` extraction:

1. `crates/metallic-foundry/src/model/executor.rs`
2. `crates/metallic-foundry/src/policy/q5_k.rs`
3. `crates/metallic-foundry/src/tokenizer.rs`
4. `crates/metallic-foundry/src/workflow/ops/format_chat.rs`
5. `crates/metallic-foundry/src/workflow/ops/sample.rs`
6. `crates/metallic-foundry/src/workflow/ops/prefill.rs`
7. `crates/metallic-foundry/src/compound/code_builder.rs`
8. `crates/metallic-foundry/src/metals/mma/stages.rs`

## Proposed Refactor Sequence (Performance-Safe)

1. Split only at stable responsibility boundaries first (`executor`, `tokenizer`, `q5_k`).
2. Move inline unit tests to sibling `*.test.rs` files using `#[path = "..."]` module wiring and `#![cfg(test)]` in test files.
3. Preserve public API and isolate `pub use` re-exports in `mod.rs` files to avoid accidental interface leakage.
4. Run targeted test groups after each split to catch behavioral regressions early.
5. Keep per-file size trending toward `<= 500` lines unless a justified performance reason requires otherwise.

## Reassessment After Unit-Test Extraction (2026-02-22)

Inline `#[cfg(test)] mod ...` blocks were extracted into sibling `*.test.rs` files.

### Repository-Wide

- Rust files total (`*.rs`): `318` (includes new `*.test.rs` files)
- Rust files over `500` lines (all `.rs`): `31` (was `34`)
- Rust files over `1000` lines (all `.rs`): `7` (unchanged)
- Production files only (exclude `*.test.rs` and crate `tests/` trees):
  - Total: `202`
  - Over `500` lines: `25`
  - Over `1000` lines: `5`
  - Over `2000` lines: `3`

### Foundry Source (`crates/metallic-foundry/src`, excluding `*.test.rs`)

- Total files: `135`
- Over `500` lines: `15` (was `18`)
- Over `1000` lines: `4` (was `5`)

### Test-Organization Status

- Inline `#[cfg(test)] mod ...` blocks remaining in source files: `0`
- Dedicated colocated unit-test files (`*.test.rs`): `46`
