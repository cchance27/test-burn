use thiserror::Error;

/// CLI-specific errors
#[derive(Error, Debug)]
pub enum CliError {
    /// Error when parsing command line arguments
    #[error("Failed to parse command line arguments: {0}")]
    ArgParseError(#[from] clap::Error),

    /// Error when file path is invalid or doesn't exist
    #[error("File path error: {0}")]
    FilePathError(String),

    /// Error when required argument is missing
    #[error("Missing required argument: {0}")]
    MissingArgument(String),

    /// Configuration validation error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// General CLI error
    #[error("CLI error: {0}")]
    GeneralError(String),
}

impl CliError {
    /// Create a new file path error
    pub fn file_path_error(path: impl Into<String>) -> Self {
        Self::FilePathError(path.into())
    }

    /// Create a new missing argument error
    pub fn missing_argument(arg: impl Into<String>) -> Self {
        Self::MissingArgument(arg.into())
    }

    /// Create a new configuration error
    pub fn config_error(msg: impl Into<String>) -> Self {
        Self::ConfigError(msg.into())
    }

    /// Create a new general error
    pub fn general_error(msg: impl Into<String>) -> Self {
        Self::GeneralError(msg.into())
    }
}
