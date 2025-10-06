use crate::metallic::instrument::prelude::*;

use super::{EnvVarGuard, env_mutex};

use tracing::Level;

#[test]
fn app_config_parses_environment_and_initialises() {
    let _lock = env_mutex().lock().expect("env mutex poisoned");
    let _log_level = EnvVarGuard::set("METALLIC_LOG_LEVEL", "debug");
    let _jsonl_path = EnvVarGuard::set("METALLIC_METRICS_JSONL_PATH", "/tmp/metrics.jsonl");
    let _console = EnvVarGuard::set("METALLIC_METRICS_CONSOLE", "true");

    let config = AppConfig::from_env().expect("configuration should parse");
    assert_eq!(config.log_level, Level::DEBUG);
    assert_eq!(
        config.metrics_jsonl_path.as_deref(),
        Some(std::path::Path::new("/tmp/metrics.jsonl"))
    );
    assert!(config.enable_console_metrics);

    let initialised = AppConfig::initialise(config.clone()).expect("initialise should succeed once");
    assert_eq!(initialised.log_level, Level::DEBUG);
    assert_eq!(initialised.metrics_jsonl_path, config.metrics_jsonl_path);
    assert!(initialised.enable_console_metrics);

    match AppConfig::initialise(config) {
        Err(AppConfigError::AlreadyInitialised) => {}
        other => panic!("expected already initialised error, got {other:?}"),
    }
}

#[test]
fn app_config_rejects_invalid_log_level() {
    let _lock = env_mutex().lock().expect("env mutex poisoned");
    let _log_level = EnvVarGuard::set("METALLIC_LOG_LEVEL", "verbose");
    let _jsonl_path = EnvVarGuard::unset("METALLIC_METRICS_JSONL_PATH");
    let _console = EnvVarGuard::unset("METALLIC_METRICS_CONSOLE");

    match AppConfig::from_env() {
        Err(AppConfigError::InvalidLogLevel { value }) => assert_eq!(value, "verbose"),
        other => panic!("expected invalid log level error, got {other:?}"),
    }
}

#[test]
fn app_config_rejects_invalid_console_flag() {
    let _lock = env_mutex().lock().expect("env mutex poisoned");
    let _console = EnvVarGuard::set("METALLIC_METRICS_CONSOLE", "maybe");
    let _log_level = EnvVarGuard::unset("METALLIC_LOG_LEVEL");
    let _jsonl_path = EnvVarGuard::unset("METALLIC_METRICS_JSONL_PATH");

    match AppConfig::from_env() {
        Err(AppConfigError::InvalidBoolean { name, value }) => {
            assert_eq!(name, "METALLIC_METRICS_CONSOLE");
            assert_eq!(value, "maybe");
        }
        other => panic!("expected invalid boolean error, got {other:?}"),
    }
}
