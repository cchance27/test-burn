use rustc_hash::FxHashMap;
use tracing::warn;

/// Enumerates the available backends for a kernel family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelBackendKind {
    Legacy,
    Graph,
}

impl KernelBackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            KernelBackendKind::Legacy => "legacy",
            KernelBackendKind::Graph => "graph",
        }
    }
}

/// Override policy for backend selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KernelBackendOverride {
    #[default]
    Auto,
    Force(KernelBackendKind),
}

/// Programmatic overrides supplied by CLI/config at runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelBackendOverrides {
    pub sdpa: Option<KernelBackendOverride>,
}

/// Explains why a backend was selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendSelectionReason {
    PerOpOverride,
    GlobalOverride,
    DefaultAuto,
}

impl BackendSelectionReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            BackendSelectionReason::PerOpOverride => "per_op_override",
            BackendSelectionReason::GlobalOverride => "global_override",
            BackendSelectionReason::DefaultAuto => "default_auto",
        }
    }
}

/// Result of selecting a backend for a kernel invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendSelection {
    pub backend: KernelBackendKind,
    pub reason: BackendSelectionReason,
}

impl BackendSelection {
    pub const fn new(backend: KernelBackendKind, reason: BackendSelectionReason) -> Self {
        Self { backend, reason }
    }
}

/// Registry responsible for determining which backend should execute a kernel.
#[derive(Clone, Debug)]
pub struct KernelBackendRegistry {
    global_override: KernelBackendOverride,
    per_op_overrides: FxHashMap<&'static str, KernelBackendOverride>,
}

impl KernelBackendRegistry {
    const SDPA_OP_KEY: &'static str = "sdpa";

    pub fn new() -> Self {
        Self {
            global_override: KernelBackendOverride::Auto,
            per_op_overrides: FxHashMap::default(),
        }
    }

    /// Build a registry using environment overrides.
    pub fn from_environment() -> Self {
        let mut registry = Self::new();

        // SDPA override
        if let Ok(Some(value)) = metallic_env::FORCE_SDPA_BACKEND_VAR.get() {
            match parse_backend_override(&value) {
                Some(KernelBackendOverride::Force(kind)) => {
                    registry
                        .per_op_overrides
                        .insert(Self::SDPA_OP_KEY, KernelBackendOverride::Force(kind));
                }
                Some(KernelBackendOverride::Auto) => {
                    // Explicit AUTO clears any prior overrides.
                    registry.per_op_overrides.remove(Self::SDPA_OP_KEY);
                }
                None => {
                    warn!("Unsupported METALLIC_FORCE_SDPA_BACKEND value '{}'", value);
                }
            }
        }

        registry
    }

    /// Apply a global override that affects all operations that do not supply
    /// an explicit per-op override.
    pub fn set_global_override(&mut self, override_policy: KernelBackendOverride) {
        self.global_override = override_policy;
    }

    /// Apply a per-operation override.
    pub fn set_override(&mut self, op_key: &'static str, override_policy: KernelBackendOverride) {
        match override_policy {
            KernelBackendOverride::Auto => {
                self.per_op_overrides.remove(op_key);
            }
            KernelBackendOverride::Force(_) => {
                self.per_op_overrides.insert(op_key, override_policy);
            }
        }
    }

    /// Select a backend for the provided operation, returning both the backend
    /// and the reason that decision was made.
    pub fn select(&self, op_key: &'static str, default: KernelBackendKind) -> BackendSelection {
        if let Some(&KernelBackendOverride::Force(backend)) = self.per_op_overrides.get(op_key) {
            return BackendSelection::new(backend, BackendSelectionReason::PerOpOverride);
        }

        if let KernelBackendOverride::Force(backend) = self.global_override {
            return BackendSelection::new(backend, BackendSelectionReason::GlobalOverride);
        }

        BackendSelection::new(default, BackendSelectionReason::DefaultAuto)
    }

    /// Convenience helper for SDPA kernels.
    pub fn select_sdpa(&self, default: KernelBackendKind) -> BackendSelection {
        self.select(Self::SDPA_OP_KEY, default)
    }

    /// Retrieve the currently-configured SDPA override policy (Auto when unset).
    pub fn sdpa_override(&self) -> KernelBackendOverride {
        self.per_op_overrides
            .get(Self::SDPA_OP_KEY)
            .copied()
            .unwrap_or(KernelBackendOverride::Auto)
    }

    /// Set/clear the SDPA backend override.
    pub fn set_sdpa_override(&mut self, override_policy: KernelBackendOverride) {
        self.set_override(Self::SDPA_OP_KEY, override_policy);
    }

    /// Apply a bundle of per-kernel overrides (typically from CLI/config).
    pub fn apply_overrides(&mut self, overrides: KernelBackendOverrides) {
        if let Some(policy) = overrides.sdpa {
            self.set_sdpa_override(policy);
        }
    }
}

impl Default for KernelBackendRegistry {
    fn default() -> Self {
        Self::from_environment()
    }
}

fn parse_backend_override(value: &str) -> Option<KernelBackendOverride> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "auto" | "default" => Some(KernelBackendOverride::Auto),
        "legacy" | "metal" | "mps" => Some(KernelBackendOverride::Force(KernelBackendKind::Legacy)),
        "graph" | "mpsgraph" => Some(KernelBackendOverride::Force(KernelBackendKind::Graph)),
        other => {
            // Accept explicit dtype strings for forward compatibility (e.g. "f16_graph")
            // by parsing suffix hints. This allows future pipelines to specify policies
            // such as "graph_f16_acc_f32" without breaking older builds.
            if other.ends_with("_graph") || other.ends_with("_mpsgraph") {
                Some(KernelBackendOverride::Force(KernelBackendKind::Graph))
            } else if other.ends_with("_legacy") || other.ends_with("_metal") {
                Some(KernelBackendOverride::Force(KernelBackendKind::Legacy))
            } else {
                None
            }
        }
    }
}

// Run environment touching tests with #[serial] to avoid interference between them.
#[cfg(test)]
mod tests {
    use metallic_env::FORCE_SDPA_BACKEND_VAR;
    use serial_test::serial;

    use super::*;

    #[test]
    #[serial]
    fn sdpa_registry_defaults_to_legacy() {
        let registry = KernelBackendRegistry::new();
        let selection = registry.select_sdpa(KernelBackendKind::Legacy);
        assert_eq!(selection.backend, KernelBackendKind::Legacy);
        assert_eq!(selection.reason, BackendSelectionReason::DefaultAuto);
    }

    #[test]
    #[serial]
    fn sdpa_registry_honours_graph_env_override() {
        let _guard = FORCE_SDPA_BACKEND_VAR.set_guard("mpsgraph").unwrap();
        let registry = KernelBackendRegistry::from_environment();
        let selection = registry.select_sdpa(KernelBackendKind::Legacy);
        assert_eq!(selection.backend, KernelBackendKind::Graph);
        assert_eq!(selection.reason, BackendSelectionReason::PerOpOverride);
    }

    #[test]
    #[serial]
    fn sdpa_registry_honours_legacy_env_override() {
        let _guard = FORCE_SDPA_BACKEND_VAR.set_guard("legacy").unwrap();
        let registry = KernelBackendRegistry::from_environment();
        let selection = registry.select_sdpa(KernelBackendKind::Graph);
        assert_eq!(selection.backend, KernelBackendKind::Legacy);
        assert_eq!(selection.reason, BackendSelectionReason::PerOpOverride);
    }

    #[test]
    #[serial]
    fn sdpa_programmatic_override_prefers_cli_over_env() {
        let _guard = FORCE_SDPA_BACKEND_VAR.set_guard("auto").unwrap();
        let mut registry = KernelBackendRegistry::from_environment();
        assert_eq!(registry.sdpa_override(), KernelBackendOverride::Auto);

        registry.set_sdpa_override(KernelBackendOverride::Force(KernelBackendKind::Graph));
        let selection = registry.select_sdpa(KernelBackendKind::Legacy);
        assert_eq!(selection.backend, KernelBackendKind::Graph);
        assert_eq!(selection.reason, BackendSelectionReason::PerOpOverride);

        registry.apply_overrides(KernelBackendOverrides {
            sdpa: Some(KernelBackendOverride::Auto),
        });
        assert_eq!(registry.sdpa_override(), KernelBackendOverride::Auto);
        let defaulted = registry.select_sdpa(KernelBackendKind::Legacy);
        assert_eq!(defaulted.backend, KernelBackendKind::Legacy);
        assert_eq!(defaulted.reason, BackendSelectionReason::DefaultAuto);
    }
}
