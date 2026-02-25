use clap::Parser;

const DEFAULT_PROMPT: &str = "Create a short javascript hello world app.";

/// Command-line interface configuration for the metallic CLI application
#[derive(Debug, Parser)]
#[command(name = "metallic_cli")]
#[command(about = "A high-performance Rust inference CLI with TUI", long_about = None)]
pub struct CliConfig {
    /// Path to the GGUF model file
    #[arg(value_name = "GGUF_PATH")]
    pub gguf_path: String,

    /// Prompt(s) to process (repeat to run multiple turns; default: a short demo prompt)
    #[arg(value_name = "PROMPT", num_args = 0..)]
    pub prompts: Vec<String>,

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

    /// Compute dtype override for runtime math (`f16`, `bf16`, `f32`).
    #[arg(long, value_enum, value_name = "DTYPE")]
    pub compute_dtype: Option<RuntimeDtypeChoice>,

    /// Accumulation dtype override for runtime math (`f16`, `bf16`, `f32`).
    #[arg(long, value_enum, value_name = "DTYPE")]
    pub accum_dtype: Option<RuntimeDtypeChoice>,

    /// Foundry runtime env override (repeatable): --foundry-env KEY=VALUE
    #[arg(long = "foundry-env", value_name = "KEY=VALUE", action = clap::ArgAction::Append)]
    pub foundry_env: Vec<String>,

    /// Optional workflow JSON file path (Foundry engine only). If provided, may include model resources for multi-model workflows.
    #[arg(long, value_name = "WORKFLOW_JSON")]
    pub workflow: Option<String>,

    /// Workflow input override (repeatable): --kwarg key=value
    ///
    /// Applies to Foundry workflow inputs and is ignored by workflows/models that do not use the key.
    #[arg(long = "kwarg", value_name = "KEY=VALUE", action = clap::ArgAction::Append)]
    pub workflow_kwargs: Vec<String>,

    /// Convenience toggle for workflows/templates that support `enable_thinking`.
    ///
    /// Equivalent to passing `--kwarg enable_thinking=1`.
    #[arg(long, conflicts_with = "no_thinking")]
    pub thinking: bool,

    /// Convenience toggle for workflows/templates that support `enable_thinking`.
    ///
    /// Equivalent to passing `--kwarg enable_thinking=0`.
    #[arg(long = "no-thinking", conflicts_with = "thinking")]
    pub no_thinking: bool,
}

/// Generation configuration options
#[derive(Debug, Parser, Clone, Copy)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 4096)]
    pub max_tokens: usize,

    /// Temperature for sampling (0.0-2.0)
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f64,

    /// Top-p sampling parameter (0.0-1.0)
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f64,

    /// Min-p sampling parameter (0.0-1.0). 0.0 disables min-p.
    #[arg(long, default_value_t = 0.05)]
    pub min_p: f64,

    /// Top-k sampling parameter (0-100)
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    /// Repeat penalty (>= 1.0). 1.0 disables repeat penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f64,

    /// Number of recent tokens to consider for repeat penalty.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,

    /// Presence penalty (>= 0.0). 0.0 disables presence penalty.
    #[arg(long, default_value_t = 0.0)]
    pub presence_penalty: f64,

    /// Frequency penalty (>= 0.0). 0.0 disables frequency penalty.
    #[arg(long, default_value_t = 0.0)]
    pub frequency_penalty: f64,

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

/// Runtime dtype choices exposed via the CLI.
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum RuntimeDtypeChoice {
    F16,
    Bf16,
    F32,
}

impl RuntimeDtypeChoice {
    #[must_use]
    pub fn as_env_value(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        }
    }
}

impl CliConfig {
    /// Get all prompts, using a single default prompt if none were provided.
    pub fn get_prompts(&self) -> Vec<String> {
        let mut prompts: Vec<String> = self
            .prompts
            .iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if !prompts.is_empty() {
            return prompts;
        }

        // In TUI mode, an omitted prompt means "start empty and wait for user input".
        if matches!(self.output_format, OutputFormat::Tui) {
            return Vec::new();
        }

        prompts.push(DEFAULT_PROMPT.to_string());
        prompts
    }

    /// Parse repeatable `--kwarg key=value` flags into normalized pairs.
    pub fn parsed_workflow_kwargs(&self) -> Result<Vec<(String, String)>, String> {
        let mut out = Vec::with_capacity(self.workflow_kwargs.len());
        for raw in &self.workflow_kwargs {
            let Some((key_raw, value_raw)) = raw.split_once('=') else {
                return Err(format!("Invalid --kwarg '{raw}': expected key=value"));
            };
            let key = key_raw.trim();
            if key.is_empty() {
                return Err(format!("Invalid --kwarg '{raw}': key cannot be empty"));
            }
            out.push((key.to_string(), value_raw.trim().to_string()));
        }
        Ok(out)
    }

    /// Parse repeatable `--foundry-env key=value` flags into normalized pairs.
    pub fn parsed_foundry_env_overrides(&self) -> Result<Vec<(String, String)>, String> {
        let mut out = Vec::with_capacity(self.foundry_env.len());
        for raw in &self.foundry_env {
            let Some((key_raw, value_raw)) = raw.split_once('=') else {
                return Err(format!("Invalid --foundry-env '{raw}': expected key=value"));
            };
            let key = key_raw.trim();
            if key.is_empty() {
                return Err(format!("Invalid --foundry-env '{raw}': key cannot be empty"));
            }
            out.push((key.to_string(), value_raw.trim().to_string()));
        }
        Ok(out)
    }

    /// Returns explicit thinking override from convenience flags.
    pub fn thinking_override(&self) -> Option<bool> {
        if self.thinking {
            Some(true)
        } else if self.no_thinking {
            Some(false)
        } else {
            None
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.05,
            top_k: 40,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
        }
    }
}

#[cfg(test)]
#[path = "config.test.rs"]
mod tests;
