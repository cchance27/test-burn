use crate::metallic::instrument::prelude::*;
#[test]
fn app_config_parses_environment_and_initialises() {
    let _log_level = LOG_LEVEL_VAR.set_guard(Level::DEBUG).expect("log level should set");
    let _jsonl_path = METRICS_JSONL_PATH_VAR
        .set_guard("/tmp/metrics.jsonl")
        .expect("metrics path should set");
    let _console = METRICS_CONSOLE_VAR.set_guard(true).expect("console flag should set");

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
    let _log_level = EnvVarGuard::set(InstrumentEnvVar::LogLevel, "verbose");
    let _jsonl_path = METRICS_JSONL_PATH_VAR.unset_guard();
    let _console = METRICS_CONSOLE_VAR.unset_guard();

    match AppConfig::from_env() {
        Err(AppConfigError::InvalidLogLevel { value }) => assert_eq!(value, "verbose"),
        other => panic!("expected invalid log level error, got {other:?}"),
    }
}

#[test]
fn app_config_rejects_invalid_console_flag() {
    let _console = EnvVarGuard::set(InstrumentEnvVar::MetricsConsole, "maybe");
    let _log_level = LOG_LEVEL_VAR.unset_guard();
    let _jsonl_path = METRICS_JSONL_PATH_VAR.unset_guard();

    match AppConfig::from_env() {
        Err(AppConfigError::InvalidBoolean { name, value }) => {
            assert_eq!(name, InstrumentEnvVar::MetricsConsole.key());
            assert_eq!(value, "maybe");
        }
        other => panic!("expected invalid boolean error, got {other:?}"),
    }
}
