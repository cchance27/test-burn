use crate::{tui::metrics::{HierarchicalMetric, RunningAverage}};
use metallic_cli_helpers::prelude::*;
use std::time::Duration;

/// Result type used throughout the TUI application
pub type AppResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Main application state
pub struct App {
    pub generated_text: String,
    pub iteration_latency: RunningAverage,
    pub prompt_token_count: usize,
    pub should_quit: bool,
    pub status: String,
    pub memory_rows: Vec<MemoryRow>,
    pub prompt_processing_time: Duration,
    pub latency_tree: Vec<HierarchicalMetric>,
    pub metrics_view: MetricsView,
    pub latency_collapse_depth: CollapseDepth,
    pub memory_collapse_depth: MemoryCollapseDepth,
    pub focus: FocusArea,
    pub text_scroll: u16,
    pub metrics_scroll: u16,
    pub text_follow_bottom: bool,
    pub request_follow_text: bool,
    pub text_area: ratatui::layout::Rect,
    pub metrics_area: ratatui::layout::Rect,
    pub alert_queue: std::collections::VecDeque<Alert>,
    pub active_alert: Option<Alert>,
    // Track prompt processing metrics separately for aggregation
    pub prompt_processing_total_last_ms: f64,
}

impl App {
    pub fn new() -> Self {
        Self {
            generated_text: String::new(),
            iteration_latency: RunningAverage::default(),
            prompt_token_count: 0,
            should_quit: false,
            status: "Initializing...".to_string(),
            memory_rows: Vec::new(),
            prompt_processing_time: Duration::default(),
            latency_tree: Vec::new(),
            metrics_view: MetricsView::Memory,
            latency_collapse_depth: CollapseDepth::new(),
            memory_collapse_depth: MemoryCollapseDepth::new(),
            focus: FocusArea::GeneratedText,
            text_scroll: 0,
            metrics_scroll: 0,
            text_follow_bottom: true,
            request_follow_text: false,
            text_area: ratatui::layout::Rect::new(0, 0, 0, 0),
            metrics_area: ratatui::layout::Rect::new(0, 0, 0, 0),
            alert_queue: std::collections::VecDeque::new(),
            active_alert: None,
            prompt_processing_total_last_ms: 0.0,
        }
    }

    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    pub fn toggle_collapse(&mut self) {
        match self.metrics_view {
            MetricsView::Latency => self.latency_collapse_depth.next(),
            MetricsView::Memory => self.memory_collapse_depth.next(),
        }
        self.reset_metrics_scroll();
    }

    pub fn reset_metrics_scroll(&mut self) {
        self.metrics_scroll = 0;
    }

    pub fn scroll_active(&mut self, delta: i32) {
        match self.focus {
            FocusArea::GeneratedText => {
                if delta != 0 {
                    self.text_follow_bottom = false;
                }
                Self::adjust_scroll(&mut self.text_scroll, delta);
                if self.text_scroll == 0 && delta <= 0 {
                    self.text_follow_bottom = true;
                }
            }
            FocusArea::Metrics => {
                Self::adjust_scroll(&mut self.metrics_scroll, delta);
            }
        }
    }

    fn adjust_scroll(value: &mut u16, delta: i32) {
        let current = i32::from(*value);
        let next = (current + delta).clamp(0, u16::MAX as i32);
        *value = next as u16;
    }

    pub fn scroll_active_to_start(&mut self) {
        match self.focus {
            FocusArea::GeneratedText => {
                self.text_scroll = 0;
                self.text_follow_bottom = false;
            }
            FocusArea::Metrics => self.metrics_scroll = 0,
        }
    }

    pub fn scroll_active_to_end(&mut self) {
        match self.focus {
            FocusArea::GeneratedText => {
                self.text_follow_bottom = true;
                self.request_follow_text = true;
            }
            FocusArea::Metrics => {
                self.metrics_scroll = u16::MAX;
            }
        }
    }

    pub fn focus_next(&mut self) {
        self.focus = match self.focus {
            FocusArea::GeneratedText => FocusArea::Metrics,
            FocusArea::Metrics => FocusArea::GeneratedText,
        };
    }

    pub fn push_alert(&mut self, alert: Alert) {
        if self.active_alert.is_some() {
            self.alert_queue.push_back(alert);
        } else {
            self.active_alert = Some(alert);
        }
    }

    pub fn active_alert(&self) -> Option<&Alert> {
        self.active_alert.as_ref()
    }

    pub fn has_active_alert(&self) -> bool {
        self.active_alert.is_some()
    }

    pub fn dismiss_active_alert(&mut self) {
        if let Some(next) = self.alert_queue.pop_front() {
            self.active_alert = Some(next);
        } else {
            self.active_alert = None;
        }
    }

    pub fn pending_alert_count(&self) -> usize {
        self.alert_queue.len()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MetricsView {
    Memory,
    Latency,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CollapseDepth {
    pub current: usize,
    pub max: Option<usize>,
}

impl CollapseDepth {
    pub fn new() -> Self {
        Self {
            current: usize::MAX, // Start fully uncollapsed (show all levels)
            max: None,           // Will be calculated dynamically based on data
        }
    }

    pub fn calculate_max_depth(&mut self, metrics: &[HierarchicalMetric]) {
        let max_found = find_max_depth(metrics, 0);
        self.max = Some(max_found);
        // If current is set to MAX (default), set it to the max found
        if self.current == usize::MAX {
            self.current = max_found;
        } else {
            // Ensure current doesn't exceed the max
            self.current = self.current.min(max_found);
        }
    }

    pub fn next(&mut self) {
        if let Some(max) = self.max {
            if max == 0 {
                // If max depth is 0, just toggle between 0 and 0
                self.current = 0;
            } else if self.current == 0 {
                // If at min depth (0), go back to max-1
                self.current = max - 1;
            } else if self.current == max {
                // If at max, go to max-1
                self.current = max - 1;
            } else {
                // Otherwise, decrease depth by 1
                self.current -= 1;
            }
        } else {
            // If max hasn't been calculated yet, go to 0
            self.current = 0;
        }
    }

    pub fn get_current_depth(&self) -> usize {
        self.current
    }

    pub fn label(&self) -> String {
        match self.max {
            Some(max) => {
                if self.current > max {
                    format!("Level {}", max)
                } else {
                    format!("Level {}", self.current)
                }
            }
            None => "Unknown".to_string(),
        }
    }
}

fn find_max_depth(metrics: &[HierarchicalMetric], current_depth: usize) -> usize {
    let mut max_depth = current_depth;
    for metric in metrics {
        let child_depth = find_max_depth(&metric.children, current_depth + 1);
        max_depth = max_depth.max(child_depth);
    }
    max_depth
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemoryCollapseDepth {
    pub current: u8,
    pub max: Option<u8>,
}

impl MemoryCollapseDepth {
    pub fn new() -> Self {
        Self {
            current: u8::MAX, // Start fully uncollapsed (show all levels)
            max: None,        // Will be calculated dynamically based on data
        }
    }

    pub fn calculate_max_depth(&mut self, rows: &[MemoryRow]) {
        let max_found = rows.iter().map(|row| row.level).max().unwrap_or(0);
        self.max = Some(max_found);
        // If current is set to MAX (default), set it to the max found
        if self.current == u8::MAX {
            self.current = max_found;
        } else {
            // Ensure current doesn't exceed the max
            self.current = self.current.min(max_found);
        }
    }

    pub fn next(&mut self) {
        if let Some(max) = self.max {
            if max == 0 {
                // If max depth is 0, just toggle between 0 and 0
                self.current = 0;
            } else if self.current == 0 {
                // If at min depth (0), go back to max-1
                self.current = max - 1;
            } else if self.current == max {
                // If at max, go to max-1
                self.current = max - 1;
            } else {
                // Otherwise, decrease depth by 1
                self.current -= 1;
            }
        } else {
            // If max hasn't been calculated yet, go to 0
            self.current = 0;
        }
    }

    pub fn get_current_depth(&self) -> u8 {
        self.current
    }

    pub fn label(&self) -> String {
        match self.max {
            Some(max) => {
                if self.current > max {
                    format!("Level {}", max)
                } else {
                    format!("Level {}", self.current)
                }
            }
            None => "Unknown".to_string(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FocusArea {
    GeneratedText,
    Metrics,
}