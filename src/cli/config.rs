use clap::Parser;

/// Command-line interface configuration for the metallic CLI application
#[derive(Debug, Parser)]
#[command(name = "metallic_cli")]
#[command(about = "A high-performance Rust inference CLI with TUI", long_about = None)]
pub struct CliConfig {
    /// Path to the GGUF model file
    #[arg(value_name = "GGUF_PATH")]
    pub gguf_path: String,

    /// Optional prompt to process (default: Create a short javascript hello world app.)
    #[arg(value_name = "PROMPT")]
    pub prompt: Option<String>,

    /// Generation configuration options
    #[command(flatten)]
    pub generation: GenerationConfig,

    /// Global backend override for all kernels (defaults to environment-driven auto selection)
    #[arg(long, value_enum, value_name = "BACKEND")]
    pub backend: Option<GlobalBackendChoice>,

    /// Enable verbose output
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Output format (json, text, tui)
    #[arg(long, value_enum, default_value_t = OutputFormat::Tui)]
    pub output_format: OutputFormat,

    /// Override the SDPA backend (defaults to environment-driven auto selection)
    #[arg(long, value_enum, value_name = "BACKEND")]
    pub sdpa_backend: Option<SdpaBackendChoice>,

    /// Inference engine to use (context or foundry)
    #[arg(long, value_enum, default_value_t = Engine::Context)]
    pub engine: Engine,
}

/// Generation configuration options
#[derive(Debug, Parser)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 4096)]
    pub max_tokens: usize,

    /// Temperature for sampling (0.0-2.0)
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f64,

    /// Top-p sampling parameter (0.0-1.0)
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f64,

    /// Top-k sampling parameter (0-100)
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    /// Random seed for sampling (optional)
    #[arg(long)]
    pub seed: Option<u32>,
}

/// Output format options
#[derive(Debug, Clone, clap::ValueEnum)]
pub enum OutputFormat {
    /// Terminal UI mode (default)
    Tui,
    /// Text-only output mode
    Text,
    /// JSON output mode
    Json,
    /// No token output (metrics only)
    None,
}

/// Global backend override that applies to all kernels when supported.
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum GlobalBackendChoice {
    /// Allow the dispatcher to choose the backend automatically.
    Auto,
    /// Force the legacy Metal implementation.
    Legacy,
    /// Force the graph-backed implementation.
    Graph,
}

/// Available SDPA backend overrides exposed via the CLI.
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum SdpaBackendChoice {
    /// Allow the dispatcher to choose the backend automatically.
    Auto,
    /// Force the legacy Metal implementation.
    Legacy,
    /// Force the graph-backed implementation.
    Graph,
}

/// Inference engine backend choice.
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum Engine {
    /// Use the legacy Context<T> implementation
    Context,
    /// Use the new Foundry (DSL) implementation
    Foundry,
}

impl CliConfig {
    /// Get the prompt text, using default if not provided
    pub fn get_prompt(&self) -> String {
        self.prompt
            .clone()
            .unwrap_or_else(|| "Create a short javascript hello world app.".to_string())
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            seed: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_config_default_prompt() {
        let config = CliConfig {
            gguf_path: "test.gguf".to_string(),
            prompt: None,
            generation: GenerationConfig::default(),
            backend: None,
            verbose: 0,
            output_format: OutputFormat::Tui,
            sdpa_backend: None,
            engine: Engine::Context,
        };

        assert_eq!(config.get_prompt(), "Create a short javascript hello world app.");
    }

    #[test]
    fn test_cli_config_custom_prompt() {
        let config = CliConfig {
            gguf_path: "test.gguf".to_string(),
            prompt: Some("Hello, world!".to_string()),
            generation: GenerationConfig::default(),
            backend: None,
            verbose: 0,
            output_format: OutputFormat::Tui,
            sdpa_backend: None,
            engine: Engine::Context,
        };

        assert_eq!(config.get_prompt(), "Hello, world!");
    }
}
