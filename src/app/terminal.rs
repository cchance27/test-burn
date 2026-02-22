use std::io::stdout;

use anyhow::Result;
use crossterm::event::{self, MouseButton, MouseEvent};
use ratatui::{
    Terminal, backend::{Backend, CrosstermBackend}, layout::Position
};

use crate::tui::{App, app::FocusArea};

pub(super) fn setup_terminal() -> Result<Terminal<impl Backend>> {
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    crossterm::terminal::enable_raw_mode()?;
    crossterm::execute!(
        stdout(),
        crossterm::terminal::EnterAlternateScreen,
        crossterm::event::EnableMouseCapture
    )?;
    terminal.clear()?;
    Ok(terminal)
}

pub(super) fn restore_terminal() -> Result<()> {
    crossterm::execute!(
        stdout(),
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture
    )?;
    crossterm::terminal::disable_raw_mode()?;
    Ok(())
}

pub(super) fn handle_mouse_event(event: MouseEvent, app: &mut App) {
    let position = (event.column, event.row).into();

    match event.kind {
        event::MouseEventKind::Down(MouseButton::Left) => {
            // Check if the click is in one of the main focus areas
            if app.text_area.contains(position) {
                app.focus = FocusArea::GeneratedText;

                // Start text selection, converting screen coordinates to text content coordinates
                // Account for text wrapping and borders
                let relative_x = event.column.saturating_sub(app.text_area.x);
                let relative_y = event.row.saturating_sub(app.text_area.y);

                // Subtract 1 from relative_y to account for the border/title at the top of the text area
                let adjusted_y = relative_y.saturating_sub(1);

                // Convert visual coordinates to content coordinates considering text wrapping
                let wrap_width = if app.text_area.width > 2 {
                    app.text_area.width.saturating_sub(2) // width accounting for left/right borders
                } else {
                    1 // minimum width of 1 to avoid issues
                };
                let (content_row, content_col) = app.get_content_position_from_visual(
                    adjusted_y,      // visual row (relative to text area after removing border)
                    relative_x,      // visual column
                    app.text_scroll, // scroll offset
                    &app.generated_text,
                    wrap_width,
                );

                let relative_pos = Position::new(content_col as u16, content_row as u16);
                app.start_text_selection(relative_pos);
            } else if app.metrics_area.contains(position) {
                app.focus = FocusArea::Metrics;
            } else if app.log_visible && app.log_area.contains(position) {
                app.focus = FocusArea::LogBox;
            } else if app.input_area.contains(position) {
                app.focus = FocusArea::Input;
            }
        }
        event::MouseEventKind::Drag(MouseButton::Left) => {
            // Update text selection - allow dragging even if slightly outside the text area if we started inside
            if app.focus == FocusArea::GeneratedText && app.is_selecting {
                let relative_x = event.column.saturating_sub(app.text_area.x);
                let relative_y = event.row.saturating_sub(app.text_area.y);

                // Calculate if we're beyond the content area
                let lines: Vec<&str> = app.generated_text.lines().collect();
                if lines.is_empty() {
                    // If no content, just return early
                    return;
                }

                // Check if we're dragging beyond the bottom of content
                let wrap_width = if app.text_area.width > 2 {
                    app.text_area.width.saturating_sub(2) // width accounting for left/right borders
                } else {
                    1 // minimum width of 1 to avoid issues
                };

                // Calculate total visual lines for all content
                let mut total_visual_lines = 0u16;
                for line in &lines {
                    let visual_lines = app.count_visual_lines_for_content_line(line, wrap_width);
                    total_visual_lines += visual_lines as u16;
                }

                // If we're dragging beyond the content vertically, clamp to the end
                let adjusted_y = relative_y.saturating_sub(1);
                let absolute_visual_y = adjusted_y.saturating_add(app.text_scroll);

                if (absolute_visual_y as usize) >= total_visual_lines as usize {
                    // We're dragging beyond the content, set to the end position
                    if let Some(last_line) = lines.last() {
                        let last_line_chars: Vec<char> = last_line.chars().collect();
                        let last_line_idx = lines.len() - 1;
                        let relative_pos = Position::new(last_line_chars.len() as u16, last_line_idx as u16);
                        app.update_text_selection(relative_pos);
                    }
                } else {
                    // Normal case: convert visual coordinates to content coordinates considering text wrapping
                    let (content_row, content_col) = app.get_content_position_from_visual(
                        adjusted_y,      // visual row
                        relative_x,      // visual column
                        app.text_scroll, // scroll offset
                        &app.generated_text,
                        wrap_width,
                    );

                    let relative_pos = Position::new(content_col as u16, content_row as u16);
                    app.update_text_selection(relative_pos);
                }
            }
        }
        event::MouseEventKind::Up(MouseButton::Left) => {
            // End text selection and copy to clipboard if there's a selection
            if app.focus == FocusArea::GeneratedText && app.is_selecting {
                app.end_text_selection();

                // Copy selected text to clipboard if there's a selection
                let selected_text = app.get_selected_text(&app.generated_text);
                if !selected_text.trim().is_empty() {
                    copy_text_to_clipboard(&selected_text);
                }
            }
        }
        event::MouseEventKind::ScrollUp => {
            app.scroll_active(-1);
        }
        event::MouseEventKind::ScrollDown => {
            app.scroll_active(1);
        }
        _ => {}
    }
}

pub(super) fn copy_text_to_clipboard(text: &str) {
    // Attempt to copy text to clipboard using arboard - copy the text to avoid lifetime issues
    let text = text.to_string();
    std::thread::spawn(move || {
        if let Ok(mut clipboard) = arboard::Clipboard::new() {
            let _ = clipboard.set_text(text);
        }
    });
}
