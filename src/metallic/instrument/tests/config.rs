use crate::metallic::instrument::prelude::*;

#[test]
fn app_config_parses_environment_and_initialises() {
    let _lock = Environment::lock();
    let _log_level = EnvVarGuard::set(InstrumentEnvVar::LogLevel, "debug");
    let _jsonl_path = EnvVarGuard::set(InstrumentEnvVar::MetricsJsonlPath, "/tmp/metrics.jsonl");
    let _console = EnvVarGuard::set(InstrumentEnvVar::MetricsConsole, "true");

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
    let _lock = Environment::lock();
    let _log_level = EnvVarGuard::set(InstrumentEnvVar::LogLevel, "verbose");
    let _jsonl_path = EnvVarGuard::unset(InstrumentEnvVar::MetricsJsonlPath);
    let _console = EnvVarGuard::unset(InstrumentEnvVar::MetricsConsole);

    match AppConfig::from_env() {
        Err(AppConfigError::InvalidLogLevel { value }) => assert_eq!(value, "verbose"),
        other => panic!("expected invalid log level error, got {other:?}"),
    }
}

#[test]
fn app_config_rejects_invalid_console_flag() {
    let _lock = Environment::lock();
    let _console = EnvVarGuard::set(InstrumentEnvVar::MetricsConsole, "maybe");
    let _log_level = EnvVarGuard::unset(InstrumentEnvVar::LogLevel);
    let _jsonl_path = EnvVarGuard::unset(InstrumentEnvVar::MetricsJsonlPath);

    match AppConfig::from_env() {
        Err(AppConfigError::InvalidBoolean { name, value }) => {
            assert_eq!(name, InstrumentEnvVar::MetricsConsole.key());
            assert_eq!(value, "maybe");
        }
        other => panic!("expected invalid boolean error, got {other:?}"),
    }
}
