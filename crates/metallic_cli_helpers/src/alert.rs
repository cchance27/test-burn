use std::{
    env, fs::OpenOptions, io::Write, path::PathBuf, sync::{Mutex, OnceLock, mpsc::Sender}
};

use chrono::SecondsFormat;

use crate::app_event::{Alert, AppEvent};

const ERROR_LOG_ENV: &str = "TEST_BURN_ERROR_LOG";
const DEFAULT_ERROR_LOG: &str = "test-burn-error.log";

struct AlertLogger {
    file: Option<Mutex<std::fs::File>>,
}

impl AlertLogger {
    fn from_env() -> Self {
        let path = env::var(ERROR_LOG_ENV).ok().and_then(|value| resolve_path(value.trim()));

        let file = path.and_then(|path| match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => Some(Mutex::new(file)),
            Err(err) => {
                eprintln!("Failed to open error log at {path:?}: {err}");
                None
            }
        });

        Self { file }
    }

    fn log(&self, alert: &Alert) {
        let Some(file) = &self.file else {
            return;
        };

        if let Ok(mut file) = file.lock() {
            let line = format!(
                "{} [{}] {}\n",
                alert.timestamp.to_rfc3339_opts(SecondsFormat::Millis, true),
                alert.level.as_str(),
                alert.message
            );
            if let Err(err) = file.write_all(line.as_bytes()) {
                eprintln!("Failed to write to error log: {err}");
            }
        }
    }
}

fn resolve_path(value: &str) -> Option<PathBuf> {
    if value.is_empty() {
        return None;
    }

    let lowered = value.to_ascii_lowercase();
    if matches!(lowered.as_str(), "1" | "true" | "yes" | "on") {
        return Some(PathBuf::from(DEFAULT_ERROR_LOG));
    }

    Some(PathBuf::from(value))
}

static LOGGER: OnceLock<AlertLogger> = OnceLock::new();

fn logger() -> &'static AlertLogger {
    LOGGER.get_or_init(AlertLogger::from_env)
}

pub fn init_error_logging() {
    let _ = logger();
}

pub fn log_alert(alert: &Alert) {
    logger().log(alert);
}

pub fn emit(tx: &Sender<AppEvent>, alert: Alert) -> Result<(), std::sync::mpsc::SendError<AppEvent>> {
    log_alert(&alert);
    tx.send(AppEvent::Alert(alert))
}

pub fn emit_info(tx: &Sender<AppEvent>, message: impl Into<String>) {
    let alert = Alert::info(message);
    let _ = emit(tx, alert);
}

pub fn emit_warning(tx: &Sender<AppEvent>, message: impl Into<String>) {
    let alert = Alert::warning(message);
    let _ = emit(tx, alert);
}

pub fn emit_error(tx: &Sender<AppEvent>, message: impl Into<String>) {
    let alert = Alert::error(message);
    let _ = emit(tx, alert);
}

pub fn log_error(message: impl Into<String>) {
    let alert = Alert::error(message);
    log_alert(&alert);
}
