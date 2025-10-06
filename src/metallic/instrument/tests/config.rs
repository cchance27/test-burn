use crate::metallic::instrument::prelude::*;
#[test]
fn app_config_parses_environment_and_initialises() {
    let mut env_lock = Environment::lock();
    let _log_level = LOG_LEVEL_VAR
        .set_guard_with_lock(Level::DEBUG, &mut env_lock)
        .expect("log level should set");
    let _jsonl_path = METRICS_JSONL_PATH_VAR
        .set_guard_with_lock("/tmp/metrics.jsonl", &mut env_lock)
        .expect("metrics path should set");
    let _console = METRICS_CONSOLE_VAR
        .set_guard_with_lock(true, &mut env_lock)
        .expect("console flag should set");

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
    let mut env_lock = Environment::lock();
    let _log_level = EnvVarGuard::set_with_lock(InstrumentEnvVar::LogLevel, "verbose", &mut env_lock);
    let _jsonl_path = METRICS_JSONL_PATH_VAR.unset_guard_with_lock(&mut env_lock);
    let _console = METRICS_CONSOLE_VAR.unset_guard_with_lock(&mut env_lock);

    match AppConfig::from_env() {
        Err(AppConfigError::InvalidLogLevel { value }) => assert_eq!(value, "verbose"),
        other => panic!("expected invalid log level error, got {other:?}"),
    }
}

#[test]
fn app_config_rejects_invalid_console_flag() {
    let mut env_lock = Environment::lock();
    let _console = EnvVarGuard::set_with_lock(InstrumentEnvVar::MetricsConsole, "maybe", &mut env_lock);
    let _log_level = LOG_LEVEL_VAR.unset_guard_with_lock(&mut env_lock);
    let _jsonl_path = METRICS_JSONL_PATH_VAR.unset_guard_with_lock(&mut env_lock);

    match AppConfig::from_env() {
        Err(AppConfigError::InvalidBoolean { name, value }) => {
            assert_eq!(name, InstrumentEnvVar::MetricsConsole.key());
            assert_eq!(value, "maybe");
        }
        other => panic!("expected invalid boolean error, got {other:?}"),
    }
}
