# Metallic Environment Utilities

`metallic_env` provides a shared facade around process environment variables that are used by the Metallic runtime and its instrumentation layer. The crate standardises how strongly typed descriptors are defined, offers scoped mutation guards that restore the previous state, and serialises mutations behind a global lock so concurrent tests and instrumentation components can cooperate safely.

## Getting started

Add the crate to your workspace dependencies and re-export the types you need:

```toml
[dependencies]
metallic_env = { path = "crates/metallic_env" }
```

The environment helpers rely on Rust's standard library and do not require additional setup beyond linking the crate. Because the crate manipulates process-level state, isolated setters acquire the global mutex internally while multi-step flows should prefer to hold the global `Environment::lock()` guard so the entire sequence runs within a single critical section.

## Core abstractions

### `Environment`

`Environment` exposes safe wrappers over the process environment:

* `Environment::lock()` provides a `MutexGuard` that callers can hold while performing compound environment interactions to prevent races between tests or instrumentation threads.
* `Environment::get(var)` returns the UTF-8 value if the variable is set.
* `Environment::set(var, value)` writes a UTF-8 value under the canonical key while automatically serialising the mutation.
* `Environment::remove(var)` deletes the variable entirely using the same internal lock.

`Environment` accepts any type that implements `Into<EnvVar>`. Instrumentation-specific identifiers are provided by `InstrumentEnvVar`, and additional categories can be added as described below.

### `TypedEnvVar`

`TypedEnvVar<T>` couples an `EnvVar` identifier with parse/format callbacks that convert between `String` representations and domain types.

Typical usage pattern:

```rust
use metallic_env::{Environment, METRICS_CONSOLE};

let _env_lock = Environment::lock();
let metrics_enabled = METRICS_CONSOLE.get()?;
if metrics_enabled != Some(true) {
    METRICS_CONSOLE.set(true)?;
}
```

Key capabilities:

* `TypedEnvVar::new` defines a descriptor that knows how to parse and format values.
* `TypedEnvVar::get` reads the environment, returning a typed value or `None` if unset.
* `TypedEnvVar::set` serialises the value and updates the environment.
* `TypedEnvVar::set_guard` scopes a mutation, automatically restoring the prior state on drop via `TypedEnvVarGuard`.
* `TypedEnvVar::set_guard_with_lock` mirrors `set_guard` but accepts an existing `Environment::lock()` guard so you can batch multiple mutations in a single critical section.
* `TypedEnvVar::unset_guard` removes the variable for the guard lifetime using `EnvVarGuard`.
* `TypedEnvVar::unset_guard_with_lock` mirrors `unset_guard` for callers already holding the mutex.

Errors are surfaced via `EnvVarError`, which distinguishes between parse and format failures so callers can surface actionable diagnostics.

### Instrumentation helpers

For convenience, the crate ships with ready-made descriptors and shim types for common instrumentation variables:

* `LOG_LEVEL` / `LOG_LEVEL_VAR` for configuring the tracing level.
* `METRICS_JSONL_PATH` / `METRICS_JSONL_PATH_VAR` for directing JSONL metric output.
* `METRICS_CONSOLE` / `METRICS_CONSOLE_VAR` for toggling console metric emission.

Each shim (`InstrumentLogLevel`, `InstrumentMetricsJsonlPath`, `InstrumentMetricsConsole`) exposes ergonomic `get`, `set`, `set_guard`, `set_guard_with_lock`, `unset`, `unset_guard`, and `unset_guard_with_lock` helpers while reusing the underlying typed descriptors.

## Adding new environment variable categories

Instrumentation variables live under the `InstrumentEnvVar` namespace, which is wrapped by the top-level `EnvVar` enum. When you need to introduce a new category (for example, execution configuration or feature flags):

1. **Add a new module** under `environment/` (e.g., `feature.rs`) that defines an enum similar to `InstrumentEnvVar` and provides typed descriptors and shim helpers.
2. **Extend `EnvVar`** with a new variant (e.g., `EnvVar::Feature(FeatureEnvVar)`) and implement `From<FeatureEnvVar>` and the corresponding match arm in `EnvVar::key()`.
3. **Use a consistent prefix** for keys (`METALLIC_<CATEGORY>_<NAME>`) to maintain discoverability. Avoid generic names to reduce clashes with user-defined variables.
4. **Provide defensive parsing/formatting** that fails loudly on invalid input by returning descriptive `EnvVarParseError` or `EnvVarFormatError` instances. Do not silently coerce or clamp values; report invalid data so operators can fix their configuration.
5. **Document thread-safety expectations** for any new helpers and ensure they reuse `Environment::lock()` to guard mutations.

When exposing new descriptors, follow the existing pattern of exporting both the `TypedEnvVar` constant and a shim struct with ergonomic methods. This ensures consistency for downstream consumers and makes the API easy to discover via autocomplete.

## Safety and locking requirements

All setters ultimately call into `std::env::set_var`/`remove_var`, which are marked `unsafe` because concurrent mutation is UB without synchronisation. `Environment::lock()` provides the global mutex that the crate uses internally. Callers should:

* Hold the lock before invoking sequences of `get`, `set`, or guard-producing APIs when there is a risk of concurrent access so the operations share a critical section.
* Prefer the guard helpers (`EnvVarGuard`, `TypedEnvVarGuard`) for scoped mutations to ensure the previous state is restored even if a panic occurs. When you already hold the mutex, use the `_with_lock` variants to avoid re-locking; these guards borrow the provided lock so they must drop before the caller releases the mutex.

When writing tests, always acquire the lock at the start of each case that manipulates environment variables. This mirrors the defensive patterns already used internally and prevents flaky behaviour when tests run in parallel.

## Platform considerations

The Metallic project targets Apple Silicon/Metal. Building or running the wider workspace requires Metal support; these commands will fail in environments without access to Apple GPU drivers. Run `cargo build` and other integration checks locally on supported hardware before shipping changes.

