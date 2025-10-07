use crate::prelude::*;

#[test]
fn app_config_parses_rejects() {
    reset_app_config_for_tests();
    let _log_level = LOG_LEVEL_VAR.set_guard(Level::DEBUG).expect("log level should set");
    let _jsonl_path = METRICS_JSONL_PATH_VAR
        .set_guard("/tmp/metrics.jsonl")
        .expect("metrics path should set");
    let _console = METRICS_CONSOLE_VAR.set_guard(true).expect("console flag should set");
    let _latency = EMIT_LATENCY_VAR.set_guard(false).expect("latency flag should set");

    let resolved = AppConfig::get_or_init_from_env().expect("configuration should initialise");
    assert_eq!(resolved.log_level, Level::DEBUG);
    assert_eq!(
        resolved.metrics_jsonl_path.as_deref(),
        Some(std::path::Path::new("/tmp/metrics.jsonl"))
    );
    assert!(resolved.enable_console_metrics);
    assert!(!resolved.emit_latency);
    assert!(std::ptr::eq(
        resolved,
        AppConfig::get_or_init_from_env().expect("config should already be initialised")
    ));

    reset_app_config_for_tests();

    let config = AppConfig::from_env().expect("configuration should parse");
    assert_eq!(config.log_level, Level::DEBUG);
    assert_eq!(
        config.metrics_jsonl_path.as_deref(),
        Some(std::path::Path::new("/tmp/metrics.jsonl"))
    );
    assert!(config.enable_console_metrics);
    assert!(!config.emit_latency);

    let initialised = AppConfig::initialise(config.clone()).expect("initialise should succeed once");
    assert_eq!(initialised.log_level, Level::DEBUG);
    assert_eq!(initialised.metrics_jsonl_path, config.metrics_jsonl_path);
    assert!(initialised.enable_console_metrics);
    assert!(!initialised.emit_latency);

    match AppConfig::initialise(config) {
        Err(AppConfigError::AlreadyInitialised) => {}
        other => panic!("expected already initialised error, got {other:?}"),
    }

    let _log_level = EnvVarGuard::set(InstrumentEnvVar::LogLevel, "verbose");
    let _jsonl_path = METRICS_JSONL_PATH_VAR.unset_guard();
    let _console = METRICS_CONSOLE_VAR.unset_guard();
    let _latency = EMIT_LATENCY_VAR.unset_guard();

    match AppConfig::from_env() {
        Err(AppConfigError::InvalidLogLevel { value }) => assert_eq!(value, "verbose"),
        other => panic!("expected invalid log level error, got {other:?}"),
    }

    let _console = EnvVarGuard::set(InstrumentEnvVar::MetricsConsole, "maybe");
    let _log_level = LOG_LEVEL_VAR.unset_guard();
    let _latency = EMIT_LATENCY_VAR.unset_guard();

    match AppConfig::from_env() {
        Err(AppConfigError::InvalidBoolean { name, value }) => {
            assert_eq!(name, InstrumentEnvVar::MetricsConsole.key());
            assert_eq!(value, "maybe");
        }
        other => panic!("expected invalid boolean error, got {other:?}"),
    }

    let _latency = EnvVarGuard::set(InstrumentEnvVar::EmitLatency, "definitely");
    let _console = METRICS_CONSOLE_VAR.unset_guard();

    match AppConfig::from_env() {
        Err(AppConfigError::InvalidBoolean { name, value }) => {
            assert_eq!(name, InstrumentEnvVar::EmitLatency.key());
            assert_eq!(value, "definitely");
        }
        other => panic!("expected invalid boolean error, got {other:?}"),
    }

    reset_app_config_for_tests();
}
