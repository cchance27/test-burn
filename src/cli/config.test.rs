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
