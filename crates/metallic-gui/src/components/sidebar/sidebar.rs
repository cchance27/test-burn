//! Sidebar component showing conversation history.

use gpui::{Context, Entity, MouseButton, SharedString, Window, div, prelude::*, px};
use gpui_component::WindowExt;

use crate::{
    app::ChatApp, components::{chat::SettingsView, common::Button}, state::AppState, theme::{colors, radius, spacing, typography}
};

/// The sidebar component containing conversation list.
pub struct Sidebar {
    hovered_id: Option<u64>,
    state: Entity<AppState>,
    app: Entity<ChatApp>,
}

impl Sidebar {
    pub fn new(state: Entity<AppState>, app: Entity<ChatApp>) -> Self {
        Self {
            hovered_id: None,
            state,
            app,
        }
    }
}

impl Render for Sidebar {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);
        let conversations = state.conversations();
        let selected_id = state.selected_id();
        let hovered_id = self.hovered_id;
        let app = self.app.clone();

        div()
            .flex()
            .flex_col()
            .w(spacing::sidebar_width())
            .h_full()
            .bg(colors::bg_surface())
            .border_r_1()
            .border_color(colors::border())
            // Header with New Chat button - same height as chat header (56px)
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .h(px(56.0))
                    .px(spacing::lg())
                    .border_b_1()
                    .border_color(colors::border())
                    .child(div().text_size(typography::lg()).text_color(colors::text_primary()).child("Chats"))
                    .child(Button::new("+ New").id("new-chat-btn").on_click({
                        let app = app.clone();
                        move |_, cx| {
                            app.update(cx, |app, cx| {
                                app.create_new_conversation(cx);
                            });
                        }
                    })),
            )
            // Conversation list
            .child(
                div()
                    .id("sidebar-scroll")
                    .flex()
                    .flex_col()
                    .flex_1()
                    .overflow_y_scroll()
                    .py(spacing::sm())
                    .children(conversations.iter().enumerate().map(|(idx, convo)| {
                        let is_selected = selected_id == Some(convo.id);
                        let _is_hovered = hovered_id == Some(convo.id);
                        let title: SharedString = truncate_text(&convo.title, 25).into();
                        let preview: SharedString = truncate_text(convo.preview(), 35).into();
                        let time_str: SharedString = format_time(&convo.updated_at).into();
                        let convo_id = convo.id;
                        let app_select = app.clone();
                        let app_delete = app.clone();

                        div()
                            .id(("convo", idx))
                            .relative()
                            .group("convo-item")
                            .mx(spacing::sm())
                            .mb(spacing::xs())
                            .px(spacing::md())
                            .py(spacing::sm())
                            .rounded(radius::md())
                            .cursor_pointer()
                            .bg(if is_selected { colors::bg_selected() } else { colors::bg_surface() })
                            .hover(|style| if is_selected { style } else { style.bg(colors::bg_hover()) })
                            .on_mouse_down(MouseButton::Left, move |_, _, cx| {
                                app_select.update(cx, |app, cx| {
                                    app.select_conversation(convo_id, cx);
                                });
                            })
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap(spacing::xs())
                                    .w_full()
                                    // Title row with time and delete button
                                    .child(
                                        div()
                                            .flex()
                                            .justify_between()
                                            .items_center()
                                            .w_full()
                                            .child(
                                                div()
                                                    .flex_1()
                                                    .text_size(typography::base())
                                                    .text_color(colors::text_primary())
                                                    .overflow_hidden()
                                                    .whitespace_nowrap()
                                                    .child(title),
                                            )
                                            .child(
                                                // Time/delete container - fixed width to prevent layout shift
                                                div()
                                                    .relative()
                                                    .w(px(52.0))
                                                    .h(px(18.0))
                                                    .flex()
                                                    .items_center()
                                                    .justify_end()
                                                    // Time (visible by default, hidden on hover)
                                                    .child(
                                                        div()
                                                            .absolute()
                                                            .right_0()
                                                            .text_size(typography::xs())
                                                            .text_color(colors::text_muted())
                                                            .opacity(1.0)
                                                            .group_hover("convo-item", |style| style.opacity(0.0))
                                                            .child(time_str),
                                                    )
                                                    // Delete button (hidden by default, visible on hover)
                                                    .child(
                                                        div()
                                                            .id(("delete", idx))
                                                            .absolute()
                                                            .right_0()
                                                            .flex()
                                                            .items_center()
                                                            .justify_center()
                                                            .w(px(18.0))
                                                            .h(px(18.0))
                                                            .rounded(radius::sm())
                                                            .text_size(typography::xs())
                                                            .text_color(colors::text_muted())
                                                            .opacity(0.0)
                                                            .group_hover("convo-item", |style| style.opacity(1.0))
                                                            .hover(|style| style.bg(colors::danger()).text_color(colors::text_primary()))
                                                            .on_mouse_down(MouseButton::Left, {
                                                                let app = app_delete.clone();
                                                                move |_ev, _, cx| {
                                                                    app.update(cx, |app, cx| {
                                                                        app.delete_conversation(convo_id, cx);
                                                                    });
                                                                }
                                                            })
                                                            .child("✕"),
                                                    ),
                                            ),
                                    )
                                    // Preview row - single line, no wrap
                                    .child(
                                        div()
                                            .text_size(typography::sm())
                                            .text_color(colors::text_secondary())
                                            .overflow_hidden()
                                            .whitespace_nowrap()
                                            .child(preview),
                                    ),
                            )
                    })),
            )
            // Settings button at the bottom
            .child(
                div()
                    .px(spacing::md())
                    .py(spacing::sm())
                    .border_t_1()
                    .border_color(colors::border())
                    .child(
                        Button::new("⚙️ Settings")
                            .id("settings-btn")
                            .variant(crate::components::common::ButtonVariant::Ghost)
                            .on_click({
                                let state = self.state.clone();
                                move |window, cx| {
                                    let state = state.clone();
                                    let viewport = window.viewport_size();
                                    let max_width = (viewport.width - px(32.0)).max(px(320.0));
                                    let max_height = (viewport.height - px(32.0)).max(px(240.0));
                                    let dialog_width = px(1000.0).min(max_width);
                                    let dialog_height = px(600.0).min(max_height);
                                    let content_height = (dialog_height - px(96.0)).max(px(180.0));
                                    let view = cx.new(|cx| SettingsView::new(state, content_height, window, cx));
                                    window.open_dialog(cx, move |dialog, _, _| {
                                        dialog
                                            .title("Settings")
                                            .close_button(true)
                                            .w(dialog_width)
                                            .h(dialog_height)
                                            .child(view.clone())
                                    });
                                }
                            }),
                    ),
            )
    }
}

/// Format timestamp for display.
fn format_time(dt: &chrono::DateTime<chrono::Local>) -> String {
    let now = chrono::Local::now();
    let duration = now.signed_duration_since(*dt);

    if duration.num_minutes() < 1 {
        "Just now".to_string()
    } else if duration.num_hours() < 1 {
        format!("{}m ago", duration.num_minutes())
    } else if duration.num_hours() < 24 {
        format!("{}h ago", duration.num_hours())
    } else if duration.num_days() < 7 {
        format!("{}d ago", duration.num_days())
    } else {
        dt.format("%b %d").to_string()
    }
}

/// Truncate text with ellipsis.
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.chars().count() <= max_len {
        text.to_string()
    } else {
        format!("{}…", text.chars().take(max_len.saturating_sub(1)).collect::<String>())
    }
}
