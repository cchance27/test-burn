use ratatui::{
    prelude::*, widgets::{Block, Paragraph}
};

#[derive(Debug, Clone, PartialEq)]
pub enum StatusBarState {
    Normal,
    Alerted(AlertedState),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlertedState {
    ProfilingEnabled,
}

#[derive(Debug, Clone)]
pub struct StatusBar {
    state: StatusBarState,
    status_text: String,
    tokens_per_second: Option<f64>,
}

impl StatusBar {
    pub fn new(state: StatusBarState) -> Self {
        Self {
            state,
            status_text: String::new(),
            tokens_per_second: None,
        }
    }

    pub fn set_state(&mut self, state: StatusBarState) {
        self.state = state;
    }

    pub fn set_status_text(&mut self, text: impl Into<String>) {
        self.status_text = text.into();
    }

    pub fn set_tokens_per_second(&mut self, tps: Option<f64>) {
        self.tokens_per_second = tps;
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let bg_color = match &self.state {
            StatusBarState::Normal => Color::Blue,
            StatusBarState::Alerted(AlertedState::ProfilingEnabled) => Color::Red,
        };

        frame.render_widget(Block::default().style(Style::default().bg(bg_color)), area);

        let center_text = " PROFILING ACTIVE ";
        let center_width = if matches!(self.state, StatusBarState::Alerted(AlertedState::ProfilingEnabled)) {
            center_text.len() as u16
        } else {
            0
        };

        // Always display t/s, defaulting to 0.00.
        let tps_text = format!("{:.2} t/s", self.tokens_per_second.unwrap_or(0.0));

        // Create a three-column layout to center the profiling status.
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(0),               // Left (flexible)
                Constraint::Length(center_width), // Center (fixed)
                Constraint::Min(0),               // Right (flexible)
            ])
            .split(area);

        // Render left text (status)
        let left_paragraph = Paragraph::new(self.status_text.clone())
            .style(Style::default().fg(Color::White).bg(bg_color))
            .alignment(Alignment::Left);
        frame.render_widget(left_paragraph, chunks[0]);

        // Render center text (profiling active)
        if center_width > 0 {
            let center_paragraph = Paragraph::new(center_text)
                .style(Style::default().fg(Color::White).bg(Color::Black))
                .alignment(Alignment::Center);
            frame.render_widget(center_paragraph, chunks[1]);
        }

        // Render right text (tokens per second)
        let tps_paragraph = Paragraph::new(tps_text)
            .style(Style::default().fg(Color::White).bg(bg_color))
            .alignment(Alignment::Right);
        frame.render_widget(tps_paragraph, chunks[2]);
    }
}
