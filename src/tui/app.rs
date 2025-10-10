use crate::tui::metrics::{HierarchicalMetric, RunningAverage};
use metallic_cli_helpers::prelude::*;
use ratatui::layout::Position;
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
    pub log_area: ratatui::layout::Rect,
    pub alert_queue: std::collections::VecDeque<Alert>,
    pub active_alert: Option<Alert>,
    // Track prompt processing metrics separately for aggregation
    pub prompt_processing_total_last_ms: f64,
    // Log box functionality
    pub log_messages: Vec<String>,
    pub log_scroll: u16,
    pub log_capacity: usize,
    // Text selection functionality
    pub text_selection_start: Option<Position>,
    pub text_selection_end: Option<Position>,
    pub is_selecting: bool,
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
            log_area: ratatui::layout::Rect::new(0, 0, 0, 0),
            alert_queue: std::collections::VecDeque::new(),
            active_alert: None,
            prompt_processing_total_last_ms: 0.0,
            log_messages: Vec::new(),
            log_scroll: 0,
            log_capacity: 1000, // Keep only the last 1000 log messages
            text_selection_start: None,
            text_selection_end: None,
            is_selecting: false,
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
            FocusArea::LogBox => {
                Self::adjust_scroll(&mut self.log_scroll, delta);
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
            FocusArea::LogBox => self.log_scroll = 0,
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
        }
    }

    pub fn focus_next(&mut self) {
        self.focus = match self.focus {
            FocusArea::GeneratedText => FocusArea::Metrics,
            FocusArea::Metrics => FocusArea::LogBox,
            FocusArea::LogBox => FocusArea::GeneratedText,
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
        
        // Auto-scroll to bottom if we were already at the bottom
        if self.focus != FocusArea::LogBox || (self.log_area.height > 0 && self.log_scroll as usize >= self.log_messages.len().saturating_sub(self.log_area.height as usize)) {
            self.scroll_to_log_end();
        }
    }

    fn scroll_to_log_end(&mut self) {
        if self.log_area.height > 0 && self.log_messages.len() > self.log_area.height as usize {
            self.log_scroll = (self.log_messages.len() - self.log_area.height as usize) as u16;
        } else {
            self.log_scroll = 0;
        }
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

    pub fn clear_text_selection(&mut self) {
        self.text_selection_start = None;
        self.text_selection_end = None;
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
    pub fn get_content_position_from_visual(&self, visual_row: u16, visual_col: u16, scroll_offset: u16, text_content: &str, wrap_width: u16) -> (usize, usize) {
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
            if (current_visual_row as u16) <= absolute_visual_row && 
               absolute_visual_row < (current_visual_row as u16 + visual_line_count as u16) {
                // The target is within this content line
                let row_in_content_line = absolute_visual_row.saturating_sub(current_visual_row as u16) as usize;
                
                // Calculate the content column position within this line, considering how ratatui would wrap
                let actual_content_col = self.visual_position_to_content_position(
                    line, 
                    row_in_content_line, 
                    visual_col as usize, 
                    wrap_width as usize
                );
                
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
    fn visual_position_to_content_position(&self, line: &str, visual_row_in_line: usize, visual_col_in_row: usize, wrap_width: usize) -> usize {
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
            for i in 0..visual_row_in_line {
                content_pos += visual_lines[i].len();
            }
            
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
    
    pub fn get_visual_position_from_content(&self, content_row: usize, content_col: usize, text_content: &str, wrap_width: u16) -> (u16, u16) {
        // This function maps content coordinates to visual coordinates
        // considering text wrapping
        let lines: Vec<&str> = text_content.lines().collect();
        let mut current_visual_row: u32 = 0u32; // Use u32 to prevent overflow issues
        
        for (idx, line) in lines.iter().enumerate() {
            if idx == content_row {
                let actual_content_col = std::cmp::min(content_col, line.len());
                
                // Create visual lines for this content line to find the visual position
                if wrap_width > 0 && wrap_width as usize > 0 {
                    let visual_lines = self.create_visual_lines_for_content_line(line, wrap_width as usize);
                    
                    // Find which visual line contains the requested content column
                    let mut content_pos_so_far = 0;
                    for (vis_idx, vis_line) in visual_lines.iter().enumerate() {
                        let next_content_pos = content_pos_so_far + vis_line.len();
                        
                        if actual_content_col < next_content_pos {
                            // The content column is in this visual line
                            let col_in_visual_line = actual_content_col - content_pos_so_far;
                            return ((current_visual_row as usize + vis_idx) as u16, col_in_visual_line as u16);
                        }
                        
                        content_pos_so_far = next_content_pos;
                    }
                    
                    // If we get here, it means the content_col is beyond the available text
                    // Return the end of the last visual line
                    if let Some(last_line) = visual_lines.last() {
                        let last_vis_idx = visual_lines.len() - 1;
                        return ((current_visual_row as usize + last_vis_idx) as u16, last_line.len() as u16);
                    } else {
                        return (current_visual_row as u16, 0);
                    }
                } else {
                    // No wrapping, return as is
                    return (current_visual_row as u16, actual_content_col as u16);
                }
            }
            
            // Calculate how many visual lines this content line takes using the same algorithm
            if wrap_width > 0 && wrap_width as usize > 0 {
                let visual_lines = self.create_visual_lines_for_content_line(line, wrap_width as usize);
                current_visual_row += visual_lines.len() as u32;
            } else {
                current_visual_row += 1; // No wrapping, one line per content line
            }
        }
        
        // If content_row is beyond the available lines, return appropriate values
        (current_visual_row.saturating_sub(1) as u16, 0)
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
    LogBox,
}
