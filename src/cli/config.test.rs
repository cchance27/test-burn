use super::*;

#[test]
fn test_cli_config_default_prompt() {
    let config = CliConfig {
        gguf_path: "test.gguf".to_string(),
        prompts: Vec::new(),
        generation: GenerationConfig::default(),
        backend: None,
        verbose: 0,
        output_format: OutputFormat::Tui,
        sdpa_backend: None,
        compute_dtype: None,
        accum_dtype: None,
        foundry_env: Vec::new(),
        workflow: None,
        workflow_kwargs: Vec::new(),
        thinking: false,
        no_thinking: false,
    };

    assert_eq!(config.get_prompts(), Vec::<String>::new());
}

#[test]
fn test_cli_config_custom_prompt() {
    let config = CliConfig {
        gguf_path: "test.gguf".to_string(),
        prompts: vec!["Hello, world!".to_string()],
        generation: GenerationConfig::default(),
        backend: None,
        verbose: 0,
        output_format: OutputFormat::Tui,
        sdpa_backend: None,
        compute_dtype: None,
        accum_dtype: None,
        foundry_env: Vec::new(),
        workflow: None,
        workflow_kwargs: Vec::new(),
        thinking: false,
        no_thinking: false,
    };

    assert_eq!(config.get_prompts(), vec!["Hello, world!".to_string()]);
}

#[test]
fn test_cli_config_blank_prompt_tui_returns_empty() {
    let config = CliConfig {
        gguf_path: "test.gguf".to_string(),
        prompts: vec!["   ".to_string()],
        generation: GenerationConfig::default(),
        backend: None,
        verbose: 0,
        output_format: OutputFormat::Tui,
        sdpa_backend: None,
        compute_dtype: None,
        accum_dtype: None,
        foundry_env: Vec::new(),
        workflow: None,
        workflow_kwargs: Vec::new(),
        thinking: false,
        no_thinking: false,
    };

    assert_eq!(config.get_prompts(), Vec::<String>::new());
}

#[test]
fn test_cli_config_parses_workflow_kwargs() {
    let config = CliConfig {
        gguf_path: "test.gguf".to_string(),
        prompts: vec!["hello".to_string()],
        generation: GenerationConfig::default(),
        backend: None,
        verbose: 0,
        output_format: OutputFormat::Text,
        sdpa_backend: None,
        compute_dtype: None,
        accum_dtype: None,
        foundry_env: Vec::new(),
        workflow: None,
        workflow_kwargs: vec!["enable_thinking=0".to_string(), "tools=[]".to_string()],
        thinking: false,
        no_thinking: false,
    };

    let parsed = config.parsed_workflow_kwargs().expect("valid kwarg list");
    assert_eq!(
        parsed,
        vec![
            ("enable_thinking".to_string(), "0".to_string()),
            ("tools".to_string(), "[]".to_string())
        ]
    );
}

#[test]
fn test_cli_config_parses_foundry_env_overrides() {
    let config = CliConfig {
        gguf_path: "test.gguf".to_string(),
        prompts: vec!["hello".to_string()],
        generation: GenerationConfig::default(),
        backend: None,
        verbose: 0,
        output_format: OutputFormat::Text,
        sdpa_backend: None,
        compute_dtype: None,
        accum_dtype: None,
        foundry_env: vec![
            "METALLIC_ACCUM_DTYPE=f32".to_string(),
            "METALLIC_DEBUG_KERNEL_BINDINGS=1".to_string(),
        ],
        workflow: None,
        workflow_kwargs: Vec::new(),
        thinking: false,
        no_thinking: false,
    };

    let parsed = config.parsed_foundry_env_overrides().expect("valid --foundry-env list");
    assert_eq!(
        parsed,
        vec![
            ("METALLIC_ACCUM_DTYPE".to_string(), "f32".to_string()),
            ("METALLIC_DEBUG_KERNEL_BINDINGS".to_string(), "1".to_string())
        ]
    );
}
