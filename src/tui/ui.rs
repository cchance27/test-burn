use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};

use crate::tui::app::{App, FocusArea, MetricsView};

pub fn render(app: &mut App, frame: &mut Frame) {
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(frame.area());

    let body_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
        .split(main_layout[0]);

    let text_block = Block::default()
        .title("Generated Text (q to quit)")
        .borders(Borders::ALL)
        .border_style(border_style(app.focus == FocusArea::GeneratedText));
    let text_area_widget = Paragraph::new(app.generated_text.clone())
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

    let status_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(main_layout[1]);

    let status_text = Paragraph::new(app.status.clone()).style(Style::default().fg(Color::White).bg(Color::Blue));

    let throughput_text = Paragraph::new(app.throughput_display())
        .style(Style::default().fg(Color::White).bg(Color::Blue))
        .alignment(Alignment::Right);

    frame.render_widget(text_area_widget, body_layout[0]);
    frame.render_widget(sidebar_block, sidebar_area);
    frame.render_widget(prompt_section, sidebar_sections[0]);
    frame.render_widget(metrics_section, sidebar_sections[1]);
    frame.render_widget(status_text, status_layout[0]);
    frame.render_widget(throughput_text, status_layout[1]);

    app.text_area = body_layout[0];
    app.metrics_area = sidebar_sections[1];

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
}

fn render_memory_metrics(rows: &[metallic_cli_helpers::app_event::MemoryRow], collapse_depth: u8) -> String {
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

fn render_hierarchical_latency_metrics(metrics: &[crate::tui::app::HierarchicalMetric], max_depth: usize) -> String {
    if metrics.is_empty() {
        return "Collecting data...".to_string();
    }

    let mut lines = Vec::new();
    for metric in metrics {
        render_metric(metric, 0, max_depth, &mut lines);
    }

    lines.join("\n")
}

fn render_metric(metric: &crate::tui::app::HierarchicalMetric, depth: usize, max_depth: usize, lines: &mut Vec<String>) {
    if depth > max_depth {
        return;
    }

    let level = u8::try_from(depth).unwrap_or(u8::MAX);
    // Use inclusive timing for display (parent + all descendants) to show total impact
    let (inclusive_last_ms, inclusive_average_ms) = metric.get_inclusive_timing();
    lines.push(format_latency_line(level, &metric.label, inclusive_last_ms, inclusive_average_ms));

    if depth == max_depth {
        return;
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
    pub fn throughput_display(&self) -> String {
        if self.iteration_latency.has_samples() {
            let average_ms = self.iteration_latency.average();
            if average_ms > 0.0 {
                let tokens_per_second = 1000.0 / average_ms;
                format!("{:.1} tok/s ({} avg)", tokens_per_second, format_time(average_ms))
            } else {
                "-- tok/s (--)".to_string()
            }
        } else {
            "-- tok/s (--)".to_string()
        }
    }

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
        self.prompt_processing_total_average_ms = 0.0;
        self.prompt_processing_average_samples = 0;

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

            // Update the running average: (old_total + new_value) / (count + 1)
            let new_average_total =
                self.prompt_processing_total_average_ms * (self.prompt_processing_average_samples as f64) + row.average_ms;
            self.prompt_processing_average_samples += 1;
            self.prompt_processing_total_average_ms = new_average_total / (self.prompt_processing_average_samples as f64);

            // Update the Prompt Processing entry with the accumulated time
            self.update_prompt_processing_entry();

            // Don't add sub-items for prompt processing - this keeps it as a single line
        } else {
            // For non-prompt processing metrics, add normally
            self.upsert_latency_path(&segments, row.last_ms, row.average_ms);
        }

        // Recalculate max depth for latency metrics since the tree structure may have changed
        self.latency_collapse_depth.calculate_max_depth(&self.latency_tree);
    }

    fn update_prompt_processing_entry(&mut self) {
        const PROMPT_PROCESSING_LABEL: &str = "Prompt Processing";

        // Find if there's already a root-level "Prompt Processing" entry and update it
        // This is a simplified approach that creates "Prompt Processing" at the root level
        // with the aggregated time
        match self.latency_tree.iter_mut().find(|metric| metric.label == PROMPT_PROCESSING_LABEL) {
            Some(metric) => {
                // Update existing entry with accumulated time
                metric.last_ms = self.prompt_processing_total_last_ms;
                metric.average_ms = self.prompt_processing_total_average_ms;
            }
            None => {
                // Create new "Prompt Processing" entry at root level with accumulated time
                self.latency_tree.push(crate::tui::app::HierarchicalMetric::new(
                    PROMPT_PROCESSING_LABEL.to_string(),
                    self.prompt_processing_total_last_ms,
                    self.prompt_processing_total_average_ms,
                ));
            }
        }
    }

    fn upsert_latency_path(&mut self, path: &[&str], last_ms: f64, average_ms: f64) {
        if path.is_empty() {
            return;
        }

        let label = path[0];
        let entry = self.latency_tree.iter_mut().position(|metric| metric.label == label);

        let metric = if let Some(idx) = entry {
            &mut self.latency_tree[idx]
        } else {
            self.latency_tree
                .push(crate::tui::app::HierarchicalMetric::new(label.to_string(), 0.0, 0.0));
            self.latency_tree
                .last_mut()
                .expect("latency_tree cannot be empty immediately after push")
        };

        if path.len() == 1 {
            metric.last_ms = last_ms;
            metric.average_ms = average_ms;
            return;
        }

        metric.upsert_path(&path[1..], last_ms, average_ms);
    }
}
