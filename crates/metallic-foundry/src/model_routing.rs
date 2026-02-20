use std::{
    collections::BTreeMap, path::{Path, PathBuf}, sync::OnceLock
};

use metallic_loader::{LoadedModel, MetadataValue, ModelMetadata};
use regex::Regex;
use serde::Deserialize;

const DEFAULT_WORKFLOW_REL_PATH: &str = "crates/metallic-foundry/workflows/multiturn_chat.json";

#[derive(Debug, Clone)]
pub struct ResolvedModelRouting {
    pub architecture: String,
    pub spec_path: PathBuf,
    pub workflow_path: PathBuf,
    pub matched_rule: String,
}

#[derive(Debug, Deserialize)]
struct ArchitectureRegistry {
    #[serde(default)]
    default_workflow: Option<String>,
    entries: Vec<ArchitectureRule>,
}

#[derive(Debug, Deserialize)]
struct ArchitectureRule {
    #[serde(default)]
    id: Option<String>,
    architecture: String,
    spec: String,
    #[serde(default)]
    workflow: Option<String>,
    #[serde(default)]
    metadata_matches: BTreeMap<String, MetadataConstraint>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MetadataConstraint {
    Exact(String),
    Any(Vec<String>),
    Pattern(MetadataPattern),
}

#[derive(Debug, Default, Deserialize)]
struct MetadataPattern {
    #[serde(default)]
    equals: Option<String>,
    #[serde(default)]
    any_of: Vec<String>,
    #[serde(default)]
    contains: Option<String>,
    #[serde(default)]
    regex: Option<String>,
}

impl MetadataConstraint {
    fn matches(&self, values: &[String]) -> bool {
        match self {
            Self::Exact(target) => values.iter().any(|v| v == target),
            Self::Any(targets) => values.iter().any(|v| targets.iter().any(|target| target == v)),
            Self::Pattern(pattern) => {
                if let Some(eq) = pattern.equals.as_ref()
                    && !values.iter().any(|v| v == eq)
                {
                    return false;
                }
                if !pattern.any_of.is_empty() && !values.iter().any(|v| pattern.any_of.iter().any(|target| target == v)) {
                    return false;
                }
                if let Some(needle) = pattern.contains.as_ref()
                    && !values.iter().any(|v| v.contains(needle))
                {
                    return false;
                }
                if let Some(regex_src) = pattern.regex.as_ref() {
                    let Ok(regex) = Regex::new(regex_src) else {
                        return false;
                    };
                    if !values.iter().any(|v| regex.is_match(v)) {
                        return false;
                    }
                }
                true
            }
        }
    }
}

fn repo_root_dir() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or(crate_dir)
}

fn resolve_repo_or_cwd_path(path: &str) -> PathBuf {
    let raw = PathBuf::from(path);
    if raw.is_absolute() {
        return raw;
    }

    let cwd_path = std::env::current_dir().map(|cwd| cwd.join(&raw)).unwrap_or_else(|_| raw.clone());
    if cwd_path.exists() {
        return cwd_path;
    }

    let repo_path = repo_root_dir().join(&raw);
    if repo_path.exists() {
        return repo_path;
    }

    let crate_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&raw);
    if crate_path.exists() {
        return crate_path;
    }

    cwd_path
}

fn registry_search_paths() -> Vec<PathBuf> {
    let repo_root = repo_root_dir();
    let mut out = Vec::with_capacity(2);
    if let Ok(cwd) = std::env::current_dir() {
        out.push(cwd.join("models/architecture.json"));
    }
    out.push(repo_root.join("models/architecture.json"));
    out
}

fn load_registry_from_disk() -> Result<ArchitectureRegistry, String> {
    let search_paths = registry_search_paths();
    for path in &search_paths {
        if !path.exists() {
            continue;
        }
        let content = std::fs::read_to_string(path).map_err(|e| format!("Failed to read architecture registry {:?}: {}", path, e))?;
        let registry: ArchitectureRegistry =
            serde_json::from_str(&content).map_err(|e| format!("Failed to parse architecture registry {:?}: {}", path, e))?;
        tracing::debug!("Loaded architecture registry from {:?}", path);
        return Ok(registry);
    }

    let searched = search_paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ");
    Err(format!(
        "Architecture registry not found. Expected models/architecture.json. Searched: {}",
        searched
    ))
}

fn registry() -> Result<&'static ArchitectureRegistry, String> {
    static REGISTRY: OnceLock<Result<ArchitectureRegistry, String>> = OnceLock::new();
    REGISTRY.get_or_init(load_registry_from_disk).as_ref().map_err(|e| e.clone())
}

fn metadata_values_to_strings(value: MetadataValue<'_>) -> Vec<String> {
    match value {
        MetadataValue::String(s) => vec![s.into_owned()],
        MetadataValue::Int(v) => vec![v.to_string()],
        MetadataValue::Float(v) => vec![v.to_string()],
        MetadataValue::Bool(v) => vec![v.to_string()],
        MetadataValue::Array(values) => values.into_iter().flat_map(metadata_values_to_strings).collect(),
    }
}

fn rule_matches(architecture: &str, metadata: &dyn ModelMetadata, rule: &ArchitectureRule) -> bool {
    if !architecture.eq_ignore_ascii_case(rule.architecture.as_str()) {
        return false;
    }

    for (key, constraint) in &rule.metadata_matches {
        let Some(value) = metadata.get(key) else {
            return false;
        };
        let values = metadata_values_to_strings(value);
        if values.is_empty() || !constraint.matches(&values) {
            return false;
        }
    }
    true
}

pub fn resolve_model_routing(architecture: &str, metadata: &dyn ModelMetadata) -> Result<ResolvedModelRouting, String> {
    let registry = registry()?;

    let Some(rule) = registry.entries.iter().find(|rule| rule_matches(architecture, metadata, rule)) else {
        return Err(format!(
            "Unsupported architecture '{}' for model routing. Add an entry in models/architecture.json.",
            architecture
        ));
    };

    let spec_path = resolve_repo_or_cwd_path(&rule.spec);
    if !spec_path.exists() {
        return Err(format!(
            "Model routing resolved spec {:?} for architecture '{}' but the file does not exist.",
            spec_path, architecture
        ));
    }

    let workflow_rel = rule
        .workflow
        .as_deref()
        .or(registry.default_workflow.as_deref())
        .unwrap_or(DEFAULT_WORKFLOW_REL_PATH);
    let workflow_path = resolve_repo_or_cwd_path(workflow_rel);
    if !workflow_path.exists() {
        return Err(format!(
            "Model routing resolved workflow {:?} for architecture '{}' but the file does not exist.",
            workflow_path, architecture
        ));
    }

    let matched_rule = rule.id.clone().unwrap_or_else(|| format!("{}=>{}", rule.architecture, rule.spec));
    tracing::debug!(
        architecture = architecture,
        matched_rule = matched_rule.as_str(),
        spec_path = ?spec_path,
        workflow_path = ?workflow_path,
        "Resolved model routing"
    );

    Ok(ResolvedModelRouting {
        architecture: architecture.to_string(),
        spec_path,
        workflow_path,
        matched_rule,
    })
}

pub fn resolve_model_routing_from_loaded_model(model: &dyn LoadedModel) -> Result<ResolvedModelRouting, String> {
    let architecture = model
        .architecture()
        .ok_or_else(|| "Architecture not found in model metadata".to_string())?;
    resolve_model_routing(architecture, model.metadata())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestMetadata {
        values: BTreeMap<String, MetadataValue<'static>>,
    }

    impl ModelMetadata for TestMetadata {
        fn get(&self, key: &str) -> Option<MetadataValue<'_>> {
            self.values.get(key).cloned()
        }
        fn parse_dtype(&self, _s: &str) -> Option<metallic_sdk::Dtype> {
            None
        }
        fn tokenizer_tokens(&self) -> Option<Vec<String>> {
            None
        }
        fn tokenizer_merges(&self) -> Option<Vec<String>> {
            None
        }
    }

    #[test]
    fn metadata_pattern_supports_contains_and_regex() {
        let constraint = MetadataConstraint::Pattern(MetadataPattern {
            equals: None,
            any_of: Vec::new(),
            contains: Some("qwen".to_string()),
            regex: Some("(?i)coder".to_string()),
        });
        let values = vec!["Qwen2.5-Coder-0.5B".to_string()];
        assert!(constraint.matches(&values));
    }

    #[test]
    fn rule_matches_requires_metadata_constraints() {
        let mut metadata_values = BTreeMap::new();
        metadata_values.insert("general.name".to_string(), MetadataValue::String("Qwen2.5-Coder-0.5B".into()));
        let metadata = TestMetadata { values: metadata_values };
        let rule = ArchitectureRule {
            id: Some("qwen-coder".to_string()),
            architecture: "qwen2".to_string(),
            spec: "models/qwen25-coder.json".to_string(),
            workflow: None,
            metadata_matches: BTreeMap::from([(
                "general.name".to_string(),
                MetadataConstraint::Pattern(MetadataPattern {
                    equals: None,
                    any_of: Vec::new(),
                    contains: Some("Coder".to_string()),
                    regex: None,
                }),
            )]),
        };
        assert!(rule_matches("qwen2", &metadata, &rule));
        assert!(!rule_matches("llama", &metadata, &rule));
    }
}
