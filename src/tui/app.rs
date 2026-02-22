use std::time::Duration;

use metallic_cli_helpers::prelude::*;
use ratatui::layout::Position;
use rustc_hash::FxHashMap;

use crate::tui::{
    components::{AlertedState, StatusBar, StatusBarState}, metrics::{HierarchicalMetric, RunningAverage}
};

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
    pub generation_time: Duration,
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
    pub log_area: ratatui::layout::Rect,
    pub input_area: ratatui::layout::Rect,
    pub alert_queue: std::collections::VecDeque<Alert>,
    pub active_alert: Option<Alert>,
    // Track prompt processing metrics separately for aggregation
    pub prompt_processing_total_last_ms: f64,
    // Log box functionality
    pub log_messages: Vec<String>,
    pub log_scroll: u16,
    pub log_capacity: usize,
    pub log_visible: bool,
    pub request_scroll_to_log_end: bool,
    // Text selection functionality
    pub text_selection_start: Option<Position>,
    pub text_selection_end: Option<Position>,
    pub is_selecting: bool,
    // Interactive chat functionality
    pub input_buffer: String,
    pub is_processing: bool,

    // Stats metrics rows for the stats view
    pub tensor_preparation_stats: Vec<metallic_cli_helpers::app_event::StatsRow>,
    pub resource_cache_stats: rustc_hash::FxHashMap<String, Vec<metallic_cli_helpers::app_event::StatsRow>>,

    // Status bar
    pub status_bar: StatusBar,

    // Text display mode - whether to show as markdown or plain text
    pub text_display_mode: TextDisplayMode,
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
            generation_time: Duration::default(),
            latency_tree: Vec::new(),
            metrics_view: MetricsView::Memory,
            latency_collapse_depth: CollapseDepth::new(),
            memory_collapse_depth: MemoryCollapseDepth::new(),
            focus: FocusArea::Input,
            text_scroll: 0,
            metrics_scroll: 0,
            text_follow_bottom: true,
            request_follow_text: false,
            text_area: ratatui::layout::Rect::new(0, 0, 0, 0),
            metrics_area: ratatui::layout::Rect::new(0, 0, 0, 0),
            log_area: ratatui::layout::Rect::new(0, 0, 0, 0),
            input_area: ratatui::layout::Rect::new(0, 0, 0, 0),
            alert_queue: std::collections::VecDeque::new(),
            active_alert: None,
            prompt_processing_total_last_ms: 0.0,
            log_messages: Vec::new(),
            log_scroll: 0,
            log_capacity: 1000, // Keep only the last 1000 log messages
            log_visible: false, // Default to hidden
            request_scroll_to_log_end: false,
            text_selection_start: None,
            text_selection_end: None,
            is_selecting: false,
            input_buffer: String::new(),
            is_processing: false,
            tensor_preparation_stats: Vec::new(),
            resource_cache_stats: FxHashMap::default(),
            status_bar: StatusBar::new(StatusBarState::Normal),
            text_display_mode: TextDisplayMode::Plain,
        }
    }

    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    pub fn toggle_collapse(&mut self) {
        match self.metrics_view {
            MetricsView::Latency => self.latency_collapse_depth.next(),
            MetricsView::Memory => self.memory_collapse_depth.next(),
            MetricsView::Stats => {
                // Stats view doesn't use collapse depths, so just reset scroll
            }
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
                self.adjust_text_scroll(delta);
                if self.text_scroll == 0 && delta <= 0 {
                    self.text_follow_bottom = true;
                }
            }
            FocusArea::Metrics => {
                self.adjust_metrics_scroll(delta);
            }
            FocusArea::LogBox => {
                self.adjust_log_scroll(delta);
            }
            FocusArea::Input => {
                // Input box scrolling not implemented yet (single line mostly)
            }
        }
    }

    fn adjust_text_scroll(&mut self, delta: i32) {
        if delta == 0 {
            return;
        }

        // Calculate total visual lines based on text content and text area width
        let wrap_width = self.text_area.width.saturating_sub(2); // Subtract 2 for borders
        let mut total_visual_lines = 0u16;
        for line in self.generated_text.lines() {
            let visual_lines = self.count_visual_lines_for_content_line(line, wrap_width) as u16;
            total_visual_lines = total_visual_lines.saturating_add(visual_lines);
        }

        // Calculate visible lines in the text area (subtract 3: 2 for borders + 1 for title)
        let visible_lines = self.text_area.height.saturating_sub(3);

        // Calculate the maximum scroll position
        let max_scroll = total_visual_lines.saturating_sub(visible_lines);

        // Calculate the new scroll position, ensuring it's within bounds
        let current_scroll = self.text_scroll as i32;
        let new_scroll = (current_scroll + delta).clamp(0, max_scroll as i32);
        self.text_scroll = new_scroll as u16;
    }

    fn adjust_metrics_scroll(&mut self, delta: i32) {
        if delta == 0 {
            return;
        }

        // Calculate metrics content lines based on the current view and data
        let metrics_content_lines = match self.metrics_view {
            MetricsView::Memory => self.calculate_memory_metrics_lines(),
            MetricsView::Latency => self.calculate_latency_metrics_lines(),
            MetricsView::Stats => self.calculate_stats_metrics_lines(),
        } as u16;

        // Calculate visible lines in the metrics area (subtract 3: 2 for borders + 1 for title)
        let visible_lines = self.metrics_area.height.saturating_sub(3);

        // Calculate the maximum scroll position
        let max_scroll = metrics_content_lines.saturating_sub(visible_lines);

        // Calculate the new scroll position, ensuring it's within bounds
        let current_scroll = self.metrics_scroll as i32;
        let new_scroll = (current_scroll + delta).clamp(0, max_scroll as i32);
        self.metrics_scroll = new_scroll as u16;
    }

    fn calculate_memory_metrics_lines(&self) -> usize {
        if self.memory_rows.is_empty() {
            return 1; // "Collecting data..." line
        }

        // Calculate lines for help text and memory rows
        let help_lines = 2; // "[m] Memory [l] Latency [c] Collapse (...)" + blank line
        let memory_lines = self
            .memory_rows
            .iter()
            .filter(|row| row.level <= self.memory_collapse_depth.get_current_depth())
            .count();

        help_lines + memory_lines
    }

    fn calculate_latency_metrics_lines(&self) -> usize {
        if self.latency_tree.is_empty() {
            return 1; // "Collecting data..." line
        }

        // Calculate lines for help text and latency tree
        let help_lines = 2; // "[m] Memory [l] Latency [c] Collapse (...)" + blank line
        let latency_lines = Self::count_latency_tree_lines(&self.latency_tree, 0, self.latency_collapse_depth.get_current_depth());

        help_lines + latency_lines
    }

    fn count_latency_tree_lines(metrics: &[crate::tui::metrics::HierarchicalMetric], depth: usize, max_depth: usize) -> usize {
        if depth > max_depth || metrics.is_empty() {
            return 0;
        }

        let mut count = 0;
        for metric in metrics {
            count += 1; // The metric line itself
            // Add children if we haven't reached max depth
            if depth < max_depth {
                count += Self::count_latency_tree_lines(&metric.children, depth + 1, max_depth);
            }
        }
        count
    }

    fn calculate_stats_metrics_lines(&self) -> usize {
        // For now, we'll show a simple "Statistics" view
        // In the future, this can show tensor preparation metrics and other statistics
        let help_lines = 2; // Help text and blank line
        let stats_lines = 5; // Placeholder for stats content

        help_lines + stats_lines
    }

    fn adjust_log_scroll(&mut self, delta: i32) {
        if delta == 0 {
            return;
        }

        // Calculate visible lines in the log area (subtract 3: 2 for borders + 1 for title)
        let visible_lines = self.log_area.height.saturating_sub(3);
        let log_content_lines = self.log_messages.len() as u16;

        // Calculate the maximum scroll position
        let max_scroll = log_content_lines.saturating_sub(visible_lines);

        // Calculate the new scroll position, ensuring it's within bounds
        let current_scroll = self.log_scroll as i32;
        let new_scroll = (current_scroll + delta).clamp(0, max_scroll as i32);
        self.log_scroll = new_scroll as u16;
    }

    pub fn scroll_active_to_start(&mut self) {
        match self.focus {
            FocusArea::GeneratedText => {
                self.text_scroll = 0;
                self.text_follow_bottom = false;
            }
            FocusArea::Metrics => self.metrics_scroll = 0,
            FocusArea::LogBox => self.log_scroll = 0,
            FocusArea::Input => {} // Cursor move?
        }
    }

    pub fn scroll_text_to_end(&mut self) {
        self.text_follow_bottom = true;
        self.request_follow_text = true;
    }

    pub fn scroll_active_to_end(&mut self) {
        match self.focus {
            FocusArea::GeneratedText => {
                self.scroll_text_to_end();
            }
            FocusArea::Metrics => {
                self.metrics_scroll = u16::MAX;
            }
            FocusArea::LogBox => {
                // Calculate max scroll based on content
                let max_messages = self.log_messages.len();
                let visible_lines = self.log_area.height as usize;
                if max_messages > visible_lines {
                    self.log_scroll = (max_messages - visible_lines) as u16;
                } else {
                    self.log_scroll = 0;
                }
            }
            FocusArea::Input => {}
        }
    }

    pub fn focus_next(&mut self) {
        self.focus = match self.focus {
            FocusArea::GeneratedText => FocusArea::Metrics,
            FocusArea::Metrics => FocusArea::LogBox,
            FocusArea::LogBox => FocusArea::Input,
            FocusArea::Input => FocusArea::GeneratedText,
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

    pub fn toggle_log_visibility(&mut self) {
        let was_visible = self.log_visible;
        self.log_visible = !self.log_visible;

        // Set flag to scroll to bottom on next render when making the log visible
        if self.log_visible && !was_visible {
            self.request_scroll_to_log_end = true;
        }
    }

    pub fn pending_alert_count(&self) -> usize {
        self.alert_queue.len()
    }

    pub fn add_log_message(&mut self, message: &str) {
        // Add timestamp to the log message
        let timestamp = chrono::Utc::now().format("%H:%M:%S").to_string();
        let formatted_message = format!("[{}] {}", timestamp, message);

        self.log_messages.push(formatted_message);

        // Keep only the last N messages to prevent memory issues
        if self.log_messages.len() > self.log_capacity {
            let start = self.log_messages.len() - self.log_capacity;
            self.log_messages.drain(0..start);
        }

        // Auto-scroll to bottom if we were already at the bottom and log is visible
        if self.log_visible
            && (self.focus != FocusArea::LogBox
                || (self.log_area.height > 0
                    && self.log_scroll as usize >= self.log_messages.len().saturating_sub(self.log_area.height as usize)))
        {
            self.scroll_to_log_end();
        }
    }

    fn scroll_to_log_end(&mut self) {
        // Scroll to end of log messages
        // When log_area isn't fully calculated yet, use the message count to determine max scroll
        let max_scroll = if self.log_area.height > 0 && self.log_messages.len() > self.log_area.height as usize {
            (self.log_messages.len() - self.log_area.height as usize) as u16
        } else {
            0
        };

        self.log_scroll = max_scroll;
        // Clear the request flag if it was set
        self.request_scroll_to_log_end = false;
    }

    // Text selection methods
    pub fn start_text_selection(&mut self, pos: Position) {
        self.text_selection_start = Some(pos);
        self.text_selection_end = Some(pos);
        self.is_selecting = true;
    }

    pub fn update_text_selection(&mut self, pos: Position) {
        if self.is_selecting {
            self.text_selection_end = Some(pos);
        }
    }

    pub fn end_text_selection(&mut self) {
        self.is_selecting = false;
    }

    pub fn get_selected_text(&self, text_content: &str) -> String {
        if let (Some(start), Some(end)) = (self.text_selection_start, self.text_selection_end) {
            // Convert position to line and character indices for text selection
            let start_row = start.y as usize;
            let start_col = start.x as usize;
            let end_row = end.y as usize;
            let end_col = end.x as usize;

            let lines: Vec<&str> = text_content.lines().collect();

            // Ensure we're within bounds
            if lines.is_empty() {
                return String::new();
            }

            // Adjust row indices to not exceed available lines
            let max_row = lines.len().saturating_sub(1);
            let start_row = std::cmp::min(start_row, max_row);
            let end_row = std::cmp::min(end_row, max_row);

            // Sort the selection to ensure start <= end
            let (actual_start_row, actual_start_col, actual_end_row, actual_end_col) =
                if start_row > end_row || (start_row == end_row && start_col > end_col) {
                    (end_row, end_col, start_row, start_col)
                } else {
                    (start_row, start_col, end_row, end_col)
                };

            // Handle single line selection
            if actual_start_row == actual_end_row {
                let line = lines[actual_start_row];
                let line_chars: Vec<char> = line.chars().collect();

                if line_chars.is_empty() {
                    // If the line is empty, return empty string
                    return String::new();
                }

                // Adjust column indices to not exceed line length
                let actual_start_col = std::cmp::min(actual_start_col, line_chars.len());
                let actual_end_col = std::cmp::min(actual_end_col, line_chars.len());

                if actual_start_col < line_chars.len() {
                    let (start_idx, end_idx) = if actual_start_col <= actual_end_col {
                        (actual_start_col, actual_end_col)
                    } else {
                        (actual_end_col, actual_start_col)
                    };
                    return line_chars[start_idx..end_idx].iter().collect();
                } else {
                    // If start_col exceeds line length, return empty selection
                    return String::new();
                }
            } else {
                // Multi-line selection
                let mut result = String::new();
                for row in actual_start_row..=actual_end_row {
                    if row >= lines.len() {
                        continue; // Skip if we're beyond the available lines
                    }

                    let line = lines[row];
                    let line_chars: Vec<char> = line.chars().collect();

                    if row == actual_start_row {
                        // First line: from start_col to end of line
                        if actual_start_col < line_chars.len() {
                            let start_idx = std::cmp::min(actual_start_col, line_chars.len());
                            result.push_str(&line_chars[start_idx..].iter().collect::<String>());
                        }
                    } else if row == actual_end_row {
                        // Last line: from beginning to end_col
                        if actual_end_col <= line_chars.len() {
                            let end_idx = std::cmp::min(actual_end_col, line_chars.len());
                            result.push_str(&line_chars[..end_idx].iter().collect::<String>());
                        }
                    } else {
                        // Middle lines: entire line
                        result.push_str(line);
                    }

                    if row < actual_end_row {
                        result.push('\n');
                    }
                }
                return result;
            }
        }
        String::new()
    }

    // Helper functions for handling wrapped text
    pub fn get_content_position_from_visual(
        &self,
        visual_row: u16,
        visual_col: u16,
        scroll_offset: u16,
        text_content: &str,
        wrap_width: u16,
    ) -> (usize, usize) {
        // This function maps visual coordinates to content coordinates considering text wrapping and scroll
        // visual_row is the row relative to the viewport (including scrolling effect)
        // scroll_offset is how many lines the user has scrolled down
        // We need to find the actual content line and column
        let absolute_visual_row = visual_row.saturating_add(scroll_offset);

        let lines: Vec<&str> = text_content.lines().collect();
        let mut current_visual_row: u32 = 0; // Use u32 to prevent overflow issues

        for (idx, line) in lines.iter().enumerate() {
            // Simulate how ratatui wraps this line to count visual lines it would create
            let visual_line_count = self.count_visual_lines_for_content_line(line, wrap_width);

            // Check if the target visual row is within this content line's range
            if (current_visual_row as u16) <= absolute_visual_row
                && absolute_visual_row < (current_visual_row as u16 + visual_line_count as u16)
            {
                // The target is within this content line
                let row_in_content_line = absolute_visual_row.saturating_sub(current_visual_row as u16) as usize;

                // Calculate the content column position within this line, considering how ratatui would wrap
                let actual_content_col =
                    self.visual_position_to_content_position(line, row_in_content_line, visual_col as usize, wrap_width as usize);

                return (idx, actual_content_col);
            }

            current_visual_row += visual_line_count as u32;
        }

        // If we reach here, we're beyond the content, return last position
        if !lines.is_empty() {
            // Return the last line with the calculated column
            let last_line_idx = lines.len() - 1;
            let last_line = lines.last().unwrap();
            let last_line_chars: Vec<char> = last_line.chars().collect();
            let clamped_col = std::cmp::min(visual_col as usize, last_line_chars.len());
            (last_line_idx, clamped_col)
        } else {
            (0, 0) // No content, return (0,0)
        }
    }

    /// Count how many visual lines a content line would produce when wrapped
    pub fn count_visual_lines_for_content_line(&self, line: &str, wrap_width: u16) -> usize {
        if wrap_width == 0 || wrap_width as usize == 0 {
            return 1; // Minimum 1 visual line
        }

        // Use the same logic as create_visual_lines_for_content_line to match ratatui behavior
        let visual_lines = self.create_visual_lines_for_content_line(line, wrap_width as usize);
        visual_lines.len()
    }

    /// Convert visual position within a wrapped line to content position
    /// This method properly handles word-aware wrapping to match ratatui's behavior
    fn visual_position_to_content_position(
        &self,
        line: &str,
        visual_row_in_line: usize,
        visual_col_in_row: usize,
        wrap_width: usize,
    ) -> usize {
        if wrap_width == 0 {
            return 0;
        }

        // Create visual lines by simulating word-aware wrapping
        let visual_lines = self.create_visual_lines_for_content_line(line, wrap_width);

        // Get the target visual line
        if visual_row_in_line < visual_lines.len() {
            let target_visual_line = &visual_lines[visual_row_in_line];

            // Make sure visual_col_in_row doesn't exceed the length of this visual line
            let safe_visual_col = std::cmp::min(visual_col_in_row, target_visual_line.len());

            // The content position is the start of this visual line in the original content
            // plus the column within this visual line
            let mut content_pos = 0;
            (0..visual_row_in_line).for_each(|i| {
                content_pos += visual_lines[i].len();
            });

            // Add the column offset within the current visual line
            content_pos += safe_visual_col;

            // Ensure it doesn't exceed the total length of the original line
            std::cmp::min(content_pos, line.len())
        } else {
            // If visual row is beyond the available visual lines, return end of content
            line.len()
        }
    }

    /// Split a content line into visual lines according to word-aware wrapping
    fn create_visual_lines_for_content_line(&self, line: &str, wrap_width: usize) -> Vec<String> {
        if wrap_width == 0 || line.is_empty() {
            return vec![line.to_string()];
        }

        let mut visual_lines = Vec::new();
        let mut current_line = String::new();
        let words: Vec<&str> = line.split_inclusive(' ').collect(); // Split on spaces, keeping the space

        for word in words {
            // If adding this word would exceed the wrap width, start a new line
            if !current_line.is_empty() && current_line.len() + word.len() > wrap_width && !current_line.is_empty() {
                visual_lines.push(current_line);
                current_line = word.to_string();
            } else {
                // If the word itself is longer than wrap width, we need to break it
                if word.len() > wrap_width {
                    // Break long word into chunks
                    let word_chars: Vec<char> = word.chars().collect();
                    let mut word_pos = 0;

                    // If current line has content, finish it first
                    if !current_line.is_empty() {
                        visual_lines.push(current_line);
                        current_line = String::new();
                    }

                    // Add chunks of the long word to new lines
                    while word_pos < word_chars.len() {
                        let end_pos = std::cmp::min(word_pos + wrap_width, word_chars.len());
                        let chunk: String = word_chars[word_pos..end_pos].iter().collect();

                        if current_line.is_empty() {
                            current_line = chunk;
                        } else {
                            current_line.push_str(&chunk);
                        }

                        // If we've filled this line or reached the end of the word
                        if current_line.len() >= wrap_width || end_pos >= word_chars.len() {
                            visual_lines.push(current_line);
                            current_line = String::new();
                        }

                        word_pos = end_pos;
                    }
                } else {
                    // Add the word to the current line
                    current_line.push_str(word);
                }
            }

            // If current line has reached wrap width, start a new line
            if current_line.len() >= wrap_width {
                visual_lines.push(current_line);
                current_line = String::new();
            }
        }

        // Add the last line if it has content
        if !current_line.is_empty() {
            visual_lines.push(current_line);
        }

        // If no visual lines were created but the line isn't empty, it means the line is shorter than wrap width
        if visual_lines.is_empty() && !line.is_empty() {
            visual_lines.push(line.to_string());
        } else if visual_lines.is_empty() {
            visual_lines.push(String::new());
        }

        visual_lines
    }
}

impl App {
    pub fn set_profiling_active(&mut self, active: bool) {
        let state = if active {
            StatusBarState::Alerted(AlertedState::ProfilingEnabled)
        } else {
            StatusBarState::Normal
        };
        self.status_bar.set_state(state);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MetricsView {
    Memory,
    Latency,
    Stats,
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
    LogBox,
    Input,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TextDisplayMode {
    Plain,
    Markdown,
}
