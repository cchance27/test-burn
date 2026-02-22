//! Message bubble component for chat messages.

use std::rc::Rc;

use gpui::{App, ClipboardItem, MouseButton, SharedString, StatefulInteractiveElement, div, prelude::*, px};
use gpui_component::{WindowExt, notification::Notification, tooltip::Tooltip};

use crate::theme::{colors, radius, spacing, typography};

type ToggleCallback = Rc<dyn Fn(&mut App)>;

/// Message visual style variants.
#[derive(Clone, Copy, PartialEq)]
pub enum MessageStyle {
    /// Message from the user (right-aligned, blue).
    User,
    /// Message from the assistant (left-aligned, dark).
    Assistant,
    /// System notification (centered, muted).
    System,
}

impl From<crate::types::MessageRole> for MessageStyle {
    fn from(role: crate::types::MessageRole) -> Self {
        match role {
            crate::types::MessageRole::User => MessageStyle::User,
            crate::types::MessageRole::Assistant => MessageStyle::Assistant,
            crate::types::MessageRole::System => MessageStyle::System,
        }
    }
}

/// A message bubble component.
pub struct MessageBubble {
    message_id: u64,
    content: String,
    style: MessageStyle,
    timestamp: Option<SharedString>,
    label: Option<SharedString>,
    metadata: Option<crate::types::MessageMetadata>,
    expanded: bool,
    on_toggle: Option<ToggleCallback>,
}

impl MessageBubble {
    /// Create a new message bubble with the given content.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            message_id: 0,
            content: content.into(),
            style: MessageStyle::Assistant,
            timestamp: None,
            label: None,
            metadata: None,
            expanded: false,
            on_toggle: None,
        }
    }

    /// Set the message id (used for stable control ids).
    pub fn message_id(mut self, message_id: u64) -> Self {
        self.message_id = message_id;
        self
    }

    /// Set the message style (User, Assistant, System).
    pub fn style(mut self, style: MessageStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the timestamp display string.
    pub fn timestamp(mut self, timestamp: impl Into<SharedString>) -> Self {
        self.timestamp = Some(timestamp.into());
        self
    }

    /// Set a custom label (overrides default "You"/"Assistant").
    pub fn label(mut self, label: impl Into<SharedString>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set message metadata.
    pub fn metadata(mut self, metadata: Option<crate::types::MessageMetadata>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set whether the metadata panel is expanded.
    pub fn expanded(mut self, expanded: bool) -> Self {
        self.expanded = expanded;
        self
    }

    /// Set the toggle callback.
    pub fn on_toggle(mut self, on_toggle: impl Fn(&mut App) + 'static) -> Self {
        self.on_toggle = Some(Rc::new(on_toggle));
        self
    }
}

impl IntoElement for MessageBubble {
    type Element = gpui::Div;

    fn into_element(self) -> Self::Element {
        let is_user = self.style == MessageStyle::User;
        let is_system = self.style == MessageStyle::System;

        // Get colors based on style
        let (bg_color, default_label) = match self.style {
            MessageStyle::User => (colors::user_bubble(), "You"),
            MessageStyle::Assistant => (colors::assistant_bubble(), "Assistant"),
            MessageStyle::System => (colors::bg_elevated(), "System"),
        };

        let label = self.label.unwrap_or_else(|| default_label.into());
        let metadata = self.metadata.as_ref();

        div().flex().w_full().px(spacing::xl()).child(
            div().flex().flex_col().w_full().child(
                // Message container with alignment
                div()
                    .flex()
                    .w_full()
                    .when(is_user, |d| d.justify_end())
                    .when(is_system, |d| d.justify_center())
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .flex_shrink()
                            .max_w_3_4()
                            .gap(spacing::xs())
                            // Role label
                            .child(
                                div()
                                    .flex()
                                    .when(is_user, |d| d.justify_end())
                                    .when(is_system, |d| d.justify_center())
                                    .child(div().text_size(typography::xs()).text_color(colors::text_muted()).child(label)),
                            )
                            // Message bubble
                            .child(
                                div()
                                    .px(spacing::lg())
                                    .py(spacing::md())
                                    .rounded(radius::lg())
                                    .bg(bg_color)
                                    .text_size(typography::base())
                                    .text_color(colors::text_primary())
                                    .child(
                                        div()
                                            .flex()
                                            .flex_col()
                                            .gap(spacing::xs())
                                            .child(div().child(self.content.clone()))
                                            .when(self.style == MessageStyle::Assistant, |d| {
                                                d.child(
                                                    div()
                                                        .flex()
                                                        .justify_end()
                                                        .items_center()
                                                        .gap(spacing::xs())
                                                        .mr(px(-8.0))
                                                        .mb(px(-6.0))
                                                        .child(
                                                            div()
                                                                .id(("copy-msg-btn", self.message_id))
                                                                .group("copy-msg-btn")
                                                                .flex()
                                                                .items_center()
                                                                .justify_center()
                                                                .w(px(18.0))
                                                                .h(px(18.0))
                                                                .rounded(radius::sm())
                                                                .cursor_pointer()
                                                                .text_size(typography::md())
                                                                .hover(|s| s.bg(colors::bg_hover()))
                                                                .tooltip(move |window, cx| Tooltip::new("Copy message").build(window, cx))
                                                                .on_mouse_down(MouseButton::Left, {
                                                                    let content = self.content.clone();
                                                                    move |_, window, cx| {
                                                                        cx.write_to_clipboard(ClipboardItem::new_string(content.clone()));
                                                                        window.push_notification(
                                                                            Notification::new()
                                                                                .content(|_note, _window, _cx| {
                                                                                    div()
                                                                                        .flex()
                                                                                        .items_center()
                                                                                        .gap(spacing::sm())
                                                                                        .child(
                                                                                            div()
                                                                                                .text_size(typography::md())
                                                                                                .text_color(colors::text_primary())
                                                                                                .child("⧉"),
                                                                                        )
                                                                                        .child(
                                                                                            div()
                                                                                                .text_size(typography::sm())
                                                                                                .text_color(colors::text_primary())
                                                                                                .child("Copied to Clipboard"),
                                                                                        )
                                                                                        .into_any_element()
                                                                                })
                                                                                .w(px(220.0)),
                                                                            cx,
                                                                        );
                                                                    }
                                                                })
                                                                .child(
                                                                    div()
                                                                        .text_color(colors::text_muted())
                                                                        .group_hover("copy-msg-btn", |s| {
                                                                            s.text_color(colors::text_primary())
                                                                        })
                                                                        .child("⧉"),
                                                                ),
                                                        )
                                                        .when_some(metadata, |d, _meta| {
                                                            d.child(
                                                                div()
                                                                    .id(("meta-msg-btn", self.message_id))
                                                                    .group("meta-msg-btn")
                                                                    .flex()
                                                                    .items_center()
                                                                    .justify_center()
                                                                    .w(px(18.0))
                                                                    .h(px(18.0))
                                                                    .rounded(radius::sm())
                                                                    .cursor_pointer()
                                                                    .text_size(typography::md())
                                                                    .hover(|s| s.bg(colors::bg_hover()))
                                                                    .tooltip(move |window, cx| {
                                                                        Tooltip::new("Toggle metadata").build(window, cx)
                                                                    })
                                                                    .when_some(self.on_toggle.clone(), |d, on_toggle| {
                                                                        d.on_mouse_down(MouseButton::Left, move |_, _, cx| {
                                                                            on_toggle(cx);
                                                                        })
                                                                    })
                                                                    .child(
                                                                        div()
                                                                            .text_color(colors::text_muted())
                                                                            .group_hover("meta-msg-btn", |s| {
                                                                                s.text_color(colors::text_primary())
                                                                            })
                                                                            .child("ⓘ"),
                                                                    ),
                                                            )
                                                        }),
                                                )
                                            }),
                                    ),
                            )
                            // Expanded Metadata Panel
                            .when(self.expanded && metadata.is_some(), |d| {
                                let meta = metadata.unwrap();
                                let mut sorted_params: Vec<_> = meta.params.iter().collect();
                                sorted_params.sort_by_key(|(k, _)| *k);

                                d.child(
                                    div()
                                        .mt(spacing::xs())
                                        .p(spacing::sm())
                                        .rounded(radius::md())
                                        .bg(colors::bg_elevated())
                                        .border_1()
                                        .border_color(colors::border())
                                        .flex()
                                        .flex_col()
                                        .gap(spacing::xs())
                                        .child(
                                            div()
                                                .text_xs()
                                                .text_color(colors::text_secondary())
                                                .child(format!("Model: {}", meta.model_filename)),
                                        )
                                        .child(div().h_px().bg(colors::border()))
                                        .children(sorted_params.into_iter().map(|(k, v)| {
                                            let val_str = if let Some(f) = v.as_f64() {
                                                format!("{:.2}", f)
                                            } else {
                                                v.to_string()
                                            };
                                            div()
                                                .text_xs()
                                                .w_full()
                                                .flex()
                                                .flex_col()
                                                .child(div().text_color(colors::text_muted()).child(k.clone()))
                                                .child(div().w_full().overflow_hidden().text_color(colors::text_primary()).child(val_str))
                                        }))
                                        .child(div().h_px().bg(colors::border()))
                                        .child(
                                            div()
                                                .flex()
                                                .justify_between()
                                                .text_xs()
                                                .child(div().text_color(colors::text_muted()).child("Tokens"))
                                                .child(div().text_color(colors::text_primary()).child(meta.perf.tokens.to_string())),
                                        )
                                        .child(
                                            div()
                                                .flex()
                                                .justify_between()
                                                .text_xs()
                                                .child(div().text_color(colors::text_muted()).child("Throughput"))
                                                .child(
                                                    div()
                                                        .text_color(colors::text_primary())
                                                        .child(format!("{:.2} tok/s", meta.perf.decode_tok_per_sec)),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .flex()
                                                .justify_between()
                                                .text_xs()
                                                .child(div().text_color(colors::text_muted()).child("Wall Speed"))
                                                .child(
                                                    div()
                                                        .text_color(colors::text_primary())
                                                        .child(format!("{:.2} tok/s", meta.perf.wall_tok_per_sec)),
                                                ),
                                        )
                                        .child(div().h_px().bg(colors::border()))
                                        .child(
                                            div()
                                                .flex()
                                                .justify_between()
                                                .text_xs()
                                                .child(div().text_color(colors::text_muted()).child("Prefill Tokens"))
                                                .child(
                                                    div().text_color(colors::text_primary()).child(meta.perf.prefill_tokens.to_string()),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .flex()
                                                .justify_between()
                                                .text_xs()
                                                .child(div().text_color(colors::text_muted()).child("Prefill Speed"))
                                                .child(
                                                    div()
                                                        .text_color(colors::text_primary())
                                                        .child(format!("{:.2} tok/s", meta.perf.prefill_tok_per_sec)),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .flex()
                                                .justify_between()
                                                .text_xs()
                                                .child(div().text_color(colors::text_muted()).child("Prefill Latency"))
                                                .child(
                                                    div()
                                                        .text_color(colors::text_primary())
                                                        .child(format!("{}ms", meta.perf.prefill_ms)),
                                                ),
                                        )
                                        .when_some(meta.perf.setup_ms, |d, value| {
                                            d.child(
                                                div()
                                                    .flex()
                                                    .justify_between()
                                                    .text_xs()
                                                    .child(div().text_color(colors::text_muted()).child("Setup Latency"))
                                                    .child(div().text_color(colors::text_primary()).child(format!("{}ms", value))),
                                            )
                                        })
                                        .when_some(meta.perf.prompt_prep_ms, |d, value| {
                                            d.child(
                                                div()
                                                    .flex()
                                                    .justify_between()
                                                    .text_xs()
                                                    .child(div().text_color(colors::text_muted()).child("Prompt Prep"))
                                                    .child(div().text_color(colors::text_primary()).child(format!("{}ms", value))),
                                            )
                                        })
                                        .when_some(meta.perf.first_decode_ms, |d, value| {
                                            d.child(
                                                div()
                                                    .flex()
                                                    .justify_between()
                                                    .text_xs()
                                                    .child(div().text_color(colors::text_muted()).child("First Decode"))
                                                    .child(div().text_color(colors::text_primary()).child(format!("{}ms", value))),
                                            )
                                        })
                                        .when_some(meta.perf.decode_wait_ms, |d, value| {
                                            d.child(
                                                div()
                                                    .flex()
                                                    .justify_between()
                                                    .text_xs()
                                                    .child(div().text_color(colors::text_muted()).child("Decode Wait"))
                                                    .child(div().text_color(colors::text_primary()).child(format!("{}ms", value))),
                                            )
                                        })
                                        .child(
                                            div()
                                                .flex()
                                                .justify_between()
                                                .text_xs()
                                                .child(div().text_color(colors::text_muted()).child("Time to First Token"))
                                                .child(
                                                    div()
                                                        .text_color(colors::text_primary())
                                                        .child(format!("{}ms", meta.perf.first_token_ms)),
                                                ),
                                        ),
                                )
                            })
                            // Timestamp (if provided)
                            .when_some(self.timestamp.clone(), |d, ts| {
                                d.child(
                                    div()
                                        .flex()
                                        .when(is_user, |d| d.justify_end())
                                        .when(is_system, |d| d.justify_center())
                                        .child(div().text_size(typography::xs()).text_color(colors::text_muted()).child(ts)),
                                )
                            }),
                    ),
            ),
        )
    }
}

/// Convenience function to render a ChatMessage as a MessageBubble.
pub fn render_message(message: &crate::types::ChatMessage, expanded: bool, on_toggle: impl Fn(&mut App) + 'static) -> impl IntoElement {
    MessageBubble::new(message.content.clone())
        .message_id(message.id)
        .style(message.role.into())
        .timestamp(message.timestamp.format("%H:%M").to_string())
        .metadata(message.metadata.clone())
        .expanded(expanded)
        .on_toggle(on_toggle)
}
