//! Welcome screen shown when no conversation is selected.

use gpui::{Entity, div, prelude::*};

use crate::{
    app::ChatApp, components::common::{Button, ButtonSize}, theme::{colors, spacing, typography}
};

/// Welcome screen component shown when no conversation is selected.
pub struct WelcomeScreen {
    app: Entity<ChatApp>,
}

impl WelcomeScreen {
    pub fn new(_cx: &mut gpui::Context<Self>, app: Entity<ChatApp>) -> Self {
        Self { app }
    }
}

impl gpui::Render for WelcomeScreen {
    fn render(&mut self, _window: &mut gpui::Window, _cx: &mut gpui::Context<Self>) -> impl IntoElement {
        let app = self.app.clone();

        div()
            .flex()
            .flex_col()
            .flex_1()
            .h_full()
            .items_center()
            .justify_center()
            .gap(spacing::lg())
            // Greeting
            .child(
                div()
                    .text_size(typography::xxl())
                    .text_color(colors::text_primary())
                    .child("👋 Welcome to Metallic"),
            )
            // Subtitle
            .child(
                div()
                    .text_size(typography::md())
                    .text_color(colors::text_secondary())
                    .child("Select a conversation or start a new chat"),
            )
            // New conversation button
            .child(Button::new("+ New Conversation").size(ButtonSize::Lg).on_click(move |_, cx| {
                app.update(cx, |this, cx| {
                    this.create_new_conversation(cx);
                });
            }))
    }
}
