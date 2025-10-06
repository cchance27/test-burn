//! Concrete metric exporter implementations.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::mpsc::Sender;

use serde_json::to_string;

use crate::recorder::{EnrichedMetricEvent, MetricExporter};

/// Persist metrics as JSON lines to the provided file path.
pub struct JsonlExporter {
    writer: Mutex<BufWriter<File>>,
}

impl JsonlExporter {
    /// Create a new exporter writing to `path`, appending if the file already exists.
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }
}

impl MetricExporter for JsonlExporter {
    fn export(&self, event: &EnrichedMetricEvent) {
        if let Ok(serialised) = to_string(event)
            && let Ok(mut writer) = self.writer.lock()
            && let Err(error) = writeln!(writer, "{}", serialised)
        {
            tracing::error!(target: "instrument", ?error, "failed to write metric to jsonl");
        }
    }
}

/// Emit metrics to stdout for rapid prototyping and debugging.
pub struct ConsoleExporter;

impl Default for ConsoleExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsoleExporter {
    /// Construct a new console exporter.
    pub fn new() -> Self {
        Self
    }
}

impl MetricExporter for ConsoleExporter {
    fn export(&self, event: &EnrichedMetricEvent) {
        if let Ok(serialised) = to_string(event) {
            println!("METRIC: {}", serialised);
        }
    }
}

/// Send metrics through an in-process channel.
pub struct ChannelExporter {
    sender: Sender<EnrichedMetricEvent>,
}

impl ChannelExporter {
    /// Create a new exporter using the provided channel sender.
    pub fn new(sender: Sender<EnrichedMetricEvent>) -> Self {
        Self { sender }
    }
}

impl MetricExporter for ChannelExporter {
    fn export(&self, event: &EnrichedMetricEvent) {
        let _ = self.sender.send(event.clone());
    }
}
