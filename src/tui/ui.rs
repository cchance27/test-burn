use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Text,
    widgets::{Block, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
};

use crate::tui::{
    app::{App, FocusArea, MetricsView},
    metrics::HierarchicalMetric,
};

pub fn render(app: &mut App, frame: &mut Frame) {
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)]) // Body area + status bar
        .split(frame.area());

    let body_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
        .split(main_layout[0]);

    // Conditionally split the main text area to have generation text and log box
    let (text_area, log_area) = if app.log_visible {
        let areas = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(10)]) // Generation area + log box
            .split(body_layout[0]);
        (areas[0], Some(areas[1]))
    } else {
        (body_layout[0], None)
    };

    let text_block = Block::default()
        .title("Generated Text (q to quit)")
        .borders(Borders::ALL)
        .border_style(border_style(app.focus == FocusArea::GeneratedText));

    // Create text with selections highlighted
    let text_with_selection = create_text_with_selection(&app.generated_text, app, text_area);
    let text_area_widget = Paragraph::new(text_with_selection)
        .block(text_block)
        .wrap(Wrap { trim: false })
        .scroll((app.text_scroll, 0));

    let sidebar_block = Block::default().title("Metrics").borders(Borders::ALL);
    let sidebar_area = body_layout[1];
    let sidebar_inner = sidebar_block.inner(sidebar_area);

    let sidebar_sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(6), Constraint::Min(0)])
        .split(sidebar_inner);

    let prompt_section = Paragraph::new(format!(
        "Prompt Tokens: {}\nProcessing Time: {}",
        app.prompt_token_count,
        format_duration(app.prompt_processing_time)
    ))
    .block(Block::default().title("Prompt").borders(Borders::ALL));

    let metrics_block_title = match app.metrics_view {
        MetricsView::Memory => "Memory Usage",
        MetricsView::Latency => "Latency",
    };

    let collapse_label = match app.metrics_view {
        MetricsView::Memory => app.memory_collapse_depth.label(),
        MetricsView::Latency => app.latency_collapse_depth.label(),
    };
    let metrics_help = format!("[m] Memory [l] Latency [c] Collapse ({})", collapse_label);

    // Always calculate and update max depths before rendering to account for dynamic changes
    if matches!(app.metrics_view, MetricsView::Memory) && !app.memory_rows.is_empty() {
        app.memory_collapse_depth.calculate_max_depth(&app.memory_rows);
    } else if matches!(app.metrics_view, MetricsView::Latency) && !app.latency_tree.is_empty() {
        app.latency_collapse_depth.calculate_max_depth(&app.latency_tree);
    }

    let metrics_text = match app.metrics_view {
        MetricsView::Memory => render_memory_metrics(&app.memory_rows, app.memory_collapse_depth.get_current_depth()),
        MetricsView::Latency => render_hierarchical_latency_metrics(&app.latency_tree, app.latency_collapse_depth.get_current_depth()),
    };

    let metrics_block = Block::default()
        .title(metrics_block_title)
        .borders(Borders::ALL)
        .border_style(border_style(app.focus == FocusArea::Metrics));
    let metrics_section = Paragraph::new(format!("{}\n\n{}", metrics_help, metrics_text))
        .block(metrics_block)
        .wrap(Wrap { trim: false })
        .scroll((app.metrics_scroll, 0));

    // Log Box
    let log_block = Block::default()
        .title("Logs")
        .borders(Borders::ALL)
        .border_style(border_style(app.focus == FocusArea::LogBox));
    let log_text = app.log_messages.join("\n");
    let log_widget = Paragraph::new(log_text)
        .block(log_block)
        .wrap(Wrap { trim: false })
        .scroll((app.log_scroll, 0));

    frame.render_widget(text_area_widget, text_area); // Render text in the appropriate area
    if let Some(log_area) = log_area {
        frame.render_widget(log_widget, log_area);
    }
    frame.render_widget(sidebar_block, sidebar_area);
    frame.render_widget(prompt_section, sidebar_sections[0]);
    frame.render_widget(metrics_section, sidebar_sections[1]);

    // Update the status bar with current tokens per second and status
    let tokens_per_second = if app.iteration_latency.has_samples() {
        let average_ms = app.iteration_latency.average();
        if average_ms > 0.0 { Some(1000.0 / average_ms) } else { None }
    } else {
        None
    };

    app.status_bar.set_tokens_per_second(tokens_per_second);
    app.status_bar.set_status_text(&app.status);

    // Render the status bar using the component
    app.status_bar.render(frame, main_layout[1]);

    app.text_area = text_area; // Update app.text_area to refer to the actual text area
    app.metrics_area = sidebar_sections[1];
    app.log_area = log_area.unwrap_or_default();

    if let Some(alert) = app.active_alert() {
        render_alert_modal(frame, alert, app.pending_alert_count());
    }

    if app.request_follow_text {
        if app.text_area.height > 0 {
            let content_lines = app.generated_text.matches('\n').count() + 1;
            let visible = app.text_area.height as usize;
            let baseline = content_lines.saturating_sub(visible) as u16;
            app.text_scroll = baseline;
        }
        app.request_follow_text = false;
    }

    // Handle request to scroll log to end (when log is made visible)
    if app.request_scroll_to_log_end && app.log_visible {
        if app.log_area.height > 0 && app.log_messages.len() > app.log_area.height as usize {
            app.log_scroll = (app.log_messages.len() - app.log_area.height as usize) as u16;
        } else {
            app.log_scroll = 0;
        }
        app.request_scroll_to_log_end = false;
    }

    // Render scrollbars for widgets that need them

    // Text area scrollbar
    let wrap_width = text_area.width.saturating_sub(2); // Subtract 2 for borders
    let mut total_visual_lines = 0u16;
    for line in app.generated_text.lines() {
        let visual_lines = app.count_visual_lines_for_content_line(line, wrap_width) as u16;
        total_visual_lines = total_visual_lines.saturating_add(visual_lines);
    }

    let text_visible_lines = text_area.height.saturating_sub(2); // Subtract 2 for borders

    if total_visual_lines > text_visible_lines && text_visible_lines > 0 {
        let text_scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("↑"))
            .end_symbol(Some("↓"));

        // Calculate the maximum scroll position (total visual lines - visible lines)
        let max_scroll = (total_visual_lines - text_visible_lines) as usize;
        let current_scroll = (app.text_scroll as usize).min(max_scroll);

        let text_scrollbar_state = ScrollbarState::new(max_scroll).position(current_scroll);

        frame.render_stateful_widget(text_scrollbar, text_area, &mut text_scrollbar_state.clone());
    }

    // Metrics area scrollbar
    let metrics_text = match app.metrics_view {
        MetricsView::Memory => render_memory_metrics(&app.memory_rows, app.memory_collapse_depth.get_current_depth()),
        MetricsView::Latency => render_hierarchical_latency_metrics(&app.latency_tree, app.latency_collapse_depth.get_current_depth()),
    };
    let metrics_content = format!("{}\n\n{}", metrics_help, metrics_text);
    let metrics_content_lines = metrics_content.lines().count();
    let metrics_visible_lines = sidebar_sections[1].height.saturating_sub(2) as usize; // Subtract 2 for borders

    if metrics_content_lines > metrics_visible_lines && metrics_visible_lines > 0 {
        let metrics_scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("↑"))
            .end_symbol(Some("↓"));

        // Calculate the maximum scroll position (content lines - visible lines)
        let max_scroll = (metrics_content_lines - metrics_visible_lines).max(0);
        let current_scroll = (app.metrics_scroll as usize).min(max_scroll);

        let metrics_scrollbar_state = ScrollbarState::new(max_scroll).position(current_scroll);

        frame.render_stateful_widget(metrics_scrollbar, sidebar_sections[1], &mut metrics_scrollbar_state.clone());
    }

    // Log area scrollbar
    if app.log_visible
        && let Some(log_area) = log_area {
            let log_content_lines = app.log_messages.len();
            let log_visible_lines = log_area.height.saturating_sub(2) as usize; // Subtract 2 for borders

            if log_content_lines > log_visible_lines && log_visible_lines > 0 {
                let log_scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                    .begin_symbol(Some("↑"))
                    .end_symbol(Some("↓"));

                // Calculate the maximum scroll position (content lines - visible lines)
                let max_scroll = (log_content_lines - log_visible_lines).max(0);
                let current_scroll = (app.log_scroll as usize).min(max_scroll);

                let log_scrollbar_state = ScrollbarState::new(max_scroll).position(current_scroll);

                frame.render_stateful_widget(log_scrollbar, log_area, &mut log_scrollbar_state.clone());
            }
        }
}

pub fn render_memory_metrics(rows: &[metallic_cli_helpers::app_event::MemoryRow], collapse_depth: u8) -> String {
    if rows.is_empty() {
        return "Collecting data...".to_string();
    }

    rows.iter()
        .filter(|row| row.level <= collapse_depth)
        .map(|row| {
            let indent = "  ".repeat(row.level as usize);
            let mut line = format!(
                "{}{}: {}",
                indent,
                row.label,
                format_current_and_peak(row.current_total_mb, row.peak_total_mb)
            );

            // Consolidated pool/kv display: show used / reserved (percent)
            if row.label.trim() == "Tensor Pool" {
                let used = row.current_total_mb;
                let reserved = row.absolute_pool_mb;
                let pct = if reserved > 0.0 {
                    (used / reserved * 100.0).clamp(0.0, 9999.0)
                } else {
                    0.0
                };
                line = format!(
                    "{}{}: {} / {} ({:.2}%)",
                    indent,
                    row.label,
                    format_memory_amount(used),
                    format_memory_amount(reserved),
                    pct
                );
            }
            if row.label.trim() == "KV Pool" {
                let used = row.current_total_mb;
                let reserved = row.absolute_kv_mb;
                let pct = if reserved > 0.0 {
                    (used / reserved * 100.0).clamp(0.0, 9999.0)
                } else {
                    0.0
                };
                line = format!(
                    "{}{}: {} / {} ({:.2}%)",
                    indent,
                    row.label,
                    format_memory_amount(used),
                    format_memory_amount(reserved),
                    pct
                );
            }

            let mut deltas = Vec::new();
            if row.current_pool_mb > 0.0 || row.peak_pool_mb > 0.0 {
                deltas.push(format!("pool {}", format_current_and_peak(row.current_pool_mb, row.peak_pool_mb)));
            }
            if row.current_kv_mb > 0.0 || row.peak_kv_mb > 0.0 {
                deltas.push(format!("kv {}", format_current_and_peak(row.current_kv_mb, row.peak_kv_mb)));
            }
            if row.current_kv_cache_mb > 0.0 || row.peak_kv_cache_mb > 0.0 {
                deltas.push(format!(
                    "kv-cache {}",
                    format_current_and_peak(row.current_kv_cache_mb, row.peak_kv_cache_mb)
                ));
            }
            if !deltas.is_empty() {
                line.push_str(&format!(" | {}", deltas.join(", ")));
            }

            if row.show_absolute {
                let mut absolutes = Vec::new();
                if row.absolute_pool_mb > 0.0 {
                    absolutes.push(format!("pool {}", format_memory_amount(row.absolute_pool_mb)));
                }
                if row.absolute_kv_mb > 0.0 {
                    absolutes.push(format!("kv {}", format_memory_amount(row.absolute_kv_mb)));
                }
                if row.absolute_kv_cache_mb > 0.0 {
                    absolutes.push(format!("kv-cache {}", format_memory_amount(row.absolute_kv_cache_mb)));
                }
                if !absolutes.is_empty() {
                    line.push_str(&format!(" | abs {}", absolutes.join(", ")));
                }
            }

            line
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn render_hierarchical_latency_metrics(metrics: &[HierarchicalMetric], max_depth: usize) -> String {
    if metrics.is_empty() {
        return "Collecting data...".to_string();
    }

    let mut lines = Vec::new();
    for metric in metrics {
        render_metric(metric, 0, max_depth, &mut lines);
    }

    lines.join("\n")
}

fn render_metric(metric: &HierarchicalMetric, depth: usize, max_depth: usize, lines: &mut Vec<String>) {
    if depth > max_depth {
        return;
    }

    let level = u8::try_from(depth).unwrap_or(u8::MAX);
    // Use inclusive timing for display (parent + all descendants) to show total impact
    let (inclusive_last_ms, inclusive_average_ms) = metric.get_inclusive_timing();
    let (child_last_sum, child_average_sum) = metric.children.iter().fold((0.0, 0.0), |(last_acc, avg_acc), child| {
        let (child_last, child_avg) = child.get_inclusive_timing();
        (last_acc + child_last, avg_acc + child_avg)
    });
    let residual_last = (inclusive_last_ms - child_last_sum).max(0.0);
    let residual_avg = (inclusive_average_ms - child_average_sum).max(0.0);
    lines.push(format_latency_line(level, &metric.label, inclusive_last_ms, inclusive_average_ms));

    if depth == max_depth {
        return;
    }

    const RESIDUAL_THRESHOLD_MS: f64 = 0.05;
    if (residual_last > RESIDUAL_THRESHOLD_MS || residual_avg > RESIDUAL_THRESHOLD_MS) && !metric.children.is_empty() {
        let residual_level = level.saturating_add(1);
        lines.push(format_latency_line(residual_level, "Other", residual_last, residual_avg));
    }

    for child in &metric.children {
        render_metric(child, depth + 1, max_depth, lines);
    }
}

fn format_latency_line(level: u8, label: &str, last_ms: f64, avg_ms: f64) -> String {
    let indent = "  ".repeat(level as usize);
    format!("{}{} - {} ({} avg)", indent, label, format_time(last_ms), format_time(avg_ms))
}

fn format_time(ms: f64) -> String {
    if ms >= 1.0 {
        format!("{:.2}ms", ms)
    } else {
        // Convert to microseconds - ms * 1000 = microseconds
        let us = ms * 1000.0;
        format!("{:.0}µs", us)
    }
}

fn format_current_and_peak(current_mb: f64, peak_mb: f64) -> String {
    let current = format_memory_amount(current_mb);
    if nearly_equal(current_mb, peak_mb) {
        current
    } else {
        format!("{} (peak {})", current, format_memory_amount(peak_mb))
    }
}

fn format_memory_amount(mb: f64) -> String {
    if mb >= 1024.0 {
        format!("{:.2} GB", mb / 1024.0)
    } else if mb >= 1.0 {
        format!("{:.2} MB", mb)
    } else if mb > 0.0 {
        let kb = mb * 1024.0;
        if kb >= 10.0 {
            format!("{:.0} KB", kb)
        } else {
            format!("{:.2} KB", kb)
        }
    } else {
        "0 MB".to_string()
    }
}

fn nearly_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < 0.005
}

fn format_duration(duration: std::time::Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() >= 1 {
        format!("{:.0}ms", duration.as_secs_f64() * 1000.0)
    } else if duration.as_nanos() > 0 {
        "<1ms".to_string()
    } else {
        "0ms".to_string()
    }
}

fn border_style(active: bool) -> Style {
    if active {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    }
}

fn render_alert_modal(frame: &mut Frame, alert: &metallic_cli_helpers::app_event::Alert, pending: usize) {
    use chrono::SecondsFormat;

    let area = centered_rect(60, 40, frame.area());
    frame.render_widget(Clear, area);

    let timestamp = alert.timestamp.to_rfc3339_opts(SecondsFormat::Secs, true);
    let mut message = format!("{}\n\nTime: {}", alert.message, timestamp);
    if pending > 0 {
        message.push_str(&format!("\n\n{} more alert(s) pending", pending));
    }
    message.push_str("\n\nPress Enter, Space, or Esc to dismiss.");

    let block = Block::default()
        .title(alert.level.as_str().to_string())
        .borders(Borders::ALL)
        .border_style(Style::default().fg(alert_color(&alert.level)));

    let paragraph = Paragraph::new(message)
        .block(block)
        .wrap(Wrap { trim: false })
        .alignment(Alignment::Left)
        .style(Style::default().fg(Color::White).bg(Color::Black));

    frame.render_widget(paragraph, area);
}

fn alert_color(level: &metallic_cli_helpers::app_event::AlertLevel) -> Color {
    match level {
        metallic_cli_helpers::app_event::AlertLevel::Info => Color::LightBlue,
        metallic_cli_helpers::app_event::AlertLevel::Warning => Color::Yellow,
        metallic_cli_helpers::app_event::AlertLevel::Error => Color::Red,
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let remaining_y = 100u16.saturating_sub(percent_y);
    let top_margin = remaining_y / 2;
    let bottom_margin = remaining_y.saturating_sub(top_margin);
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(top_margin),
            Constraint::Percentage(percent_y.min(100)),
            Constraint::Percentage(bottom_margin),
        ])
        .split(area);

    let remaining_x = 100u16.saturating_sub(percent_x);
    let left_margin = remaining_x / 2;
    let right_margin = remaining_x.saturating_sub(left_margin);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(left_margin),
            Constraint::Percentage(percent_x.min(100)),
            Constraint::Percentage(right_margin),
        ])
        .split(vertical[1])[1]
}

impl App {
    pub fn update_generation_metrics(&mut self, iteration: Option<std::time::Duration>) {
        let Some(iteration) = iteration else {
            return;
        };

        if iteration.is_zero() {
            return;
        }

        self.iteration_latency.record(iteration.as_secs_f64() * 1000.0);
    }

    pub fn reset_generation_metrics(&mut self) {
        self.iteration_latency.reset();
    }

    pub fn reset_prompt_processing_metrics(&mut self) {
        self.prompt_processing_total_last_ms = 0.0;

        // Remove any existing Prompt Processing entry from the tree
        self.latency_tree.retain(|metric| metric.label != "Prompt Processing");
    }

    pub fn add_latency_metric(&mut self, row: metallic_cli_helpers::app_event::LatencyRow) {
        const PROMPT_PROCESSING_LABEL: &str = "Prompt Processing";

        let segments: Vec<&str> = row.label.split("::").collect();
        if segments.is_empty() {
            return;
        }

        // Check if this is a prompt processing related metric
        // Look for "Prompt Processing" anywhere in the path
        if segments.contains(&PROMPT_PROCESSING_LABEL) {
            // For prompt processing operations, just accumulate the time
            // but don't add individual sub-operations to maintain a single line display
            self.prompt_processing_total_last_ms += row.last_ms;

            // Find or create the "Prompt Processing" metric and update it
            let metric = match self.latency_tree.iter_mut().find(|metric| metric.label == PROMPT_PROCESSING_LABEL) {
                Some(metric) => metric,
                None => {
                    self.latency_tree
                        .push(HierarchicalMetric::new(PROMPT_PROCESSING_LABEL.to_string(), 0.0));
                    self.latency_tree.last_mut().expect("latency_tree cannot be empty after push")
                }
            };

            metric.last_ms = self.prompt_processing_total_last_ms;
            // Here, we record the accumulated time. This is not a perfect solution
            // as it records intermediate values, but it's an improvement.
            metric.running_average.record(self.prompt_processing_total_last_ms);
        } else {
            // For non-prompt processing metrics, add normally
            self.upsert_latency_path(&segments, row.last_ms);
        }

        // Recalculate max depth for latency metrics since the tree structure may have changed
        self.latency_collapse_depth.calculate_max_depth(&self.latency_tree);
    }

    fn upsert_latency_path(&mut self, path: &[&str], last_ms: f64) {
        if path.is_empty() {
            return;
        }

        let label = path[0];
        let entry = self.latency_tree.iter_mut().position(|metric| metric.label == label);

        let metric = if let Some(idx) = entry {
            &mut self.latency_tree[idx]
        } else {
            self.latency_tree.push(HierarchicalMetric::new(label.to_string(), 0.0));
            self.latency_tree
                .last_mut()
                .expect("latency_tree cannot be empty immediately after push")
        };

        if path.len() == 1 {
            metric.last_ms = last_ms;
            metric.running_average.record(last_ms);
            return;
        }

        metric.upsert_path(&path[1..], last_ms);
    }
}

fn create_text_with_selection(text: &str, app: &App, _area: Rect) -> ratatui::text::Text<'static> {
    use ratatui::text::{Line, Span};

    // If there's no selection, return the text as is
    if !app.is_selecting || app.text_selection_start.is_none() || app.text_selection_end.is_none() {
        return Text::from(text.to_string());
    }

    let lines: Vec<&str> = text.lines().collect();

    if let (Some(start_pos), Some(end_pos)) = (app.text_selection_start, app.text_selection_end) {
        // Selection coordinates are in content space (after accounting for scroll in mouse events)
        let (start_row, start_col) = (start_pos.y as usize, start_pos.x as usize);
        let (end_row, end_col) = (end_pos.y as usize, end_pos.x as usize);

        // Ensure we don't exceed the number of available lines
        let max_line_idx = lines.len().saturating_sub(1);
        let start_row = std::cmp::min(start_row, max_line_idx);
        let end_row = std::cmp::min(end_row, max_line_idx);

        // Sort start and end to ensure start <= end
        let (actual_start_row, actual_start_col, actual_end_row, actual_end_col) =
            if start_row > end_row || (start_row == end_row && start_col > end_col) {
                (end_row, end_col, start_row, start_col)
            } else {
                (start_row, start_col, end_row, end_col)
            };

        // Build the result lines for the text content
        let result_lines: Vec<Line> = lines
            .iter()
            .enumerate()
            .map(|(line_idx, line)| {
                let line_chars: Vec<char> = line.chars().collect();

                if line_idx >= actual_start_row && line_idx <= actual_end_row {
                    // This line is within the selection range
                    let start_char_idx = if line_idx == actual_start_row {
                        std::cmp::min(actual_start_col, line_chars.len())
                    } else {
                        0
                    };
                    let end_char_idx = if line_idx == actual_end_row {
                        std::cmp::min(actual_end_col, line_chars.len())
                    } else {
                        line_chars.len()
                    };

                    let mut spans = Vec::new();

                    // Case 1: Line is entirely within selection (not the start or end line of selection)
                    if line_idx != actual_start_row && line_idx != actual_end_row {
                        // The entire line is selected
                        let entire_line: String = line_chars.iter().collect();
                        spans.push(Span::styled(entire_line, Style::default().add_modifier(Modifier::REVERSED)));
                    }
                    // Case 2: Line is partially selected (start line, end line, or both are same line)
                    else {
                        // Add unselected text before the selection (if any) - only for start line
                        if line_idx == actual_start_row && start_char_idx > 0 && start_char_idx < line_chars.len() {
                            let before_text: String = line_chars[0..start_char_idx].iter().collect();
                            spans.push(Span::raw(before_text));
                        }

                        // Add selected text with REVERSED modifier
                        if start_char_idx < line_chars.len() && end_char_idx <= line_chars.len() && start_char_idx < end_char_idx {
                            let selected_text: String = line_chars[start_char_idx..end_char_idx].iter().collect();
                            spans.push(Span::styled(selected_text, Style::default().add_modifier(Modifier::REVERSED)));
                        }

                        // Add unselected text after the selection (if any) - only for end line
                        if line_idx == actual_end_row && end_char_idx < line_chars.len() {
                            let after_text: String = line_chars[end_char_idx..].iter().collect();
                            spans.push(Span::raw(after_text));
                        }
                    }

                    Line::from(spans)
                } else {
                    // This line is not in the selection range
                    Line::from(line.to_string())
                }
            })
            .collect();

        Text::from(result_lines)
    } else {
        // No active selection, return text as is
        Text::from(text.to_string())
    }
}
