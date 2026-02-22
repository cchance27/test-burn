#![cfg(test)]

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
