//! Main chat view component.

use gpui::{Context, Entity, ScrollHandle, Window, div, prelude::*, px};
use gpui_component::{scroll::ScrollableElement as _, v_flex};

use super::{conversation_title::ConversationTitle, message::render_message, model_selector::ModelSelector, welcome_screen::WelcomeScreen};
use crate::{
    app::ChatApp, components::input::ChatInput, state::AppState, theme::{colors, spacing}
};

/// The main chat view showing messages and input.
pub struct ChatView {
    state: Entity<AppState>,
    input: Entity<ChatInput>,
    title: Entity<ConversationTitle>,
    model_loader: Entity<ModelSelector>,
    welcome: Entity<WelcomeScreen>,
    scroll_handle: ScrollHandle,
    should_scroll_to_bottom: bool,
    follow_streaming_output: bool,
    last_rendered_is_generating: bool,
    last_rendered_conversation_id: Option<u64>,
    last_rendered_message_count: usize,
    expanded_metadata_id: Option<u64>,
    _app: Entity<ChatApp>,
}

impl ChatView {
    pub fn new(cx: &mut Context<Self>, state: Entity<AppState>, app: Entity<ChatApp>) -> Self {
        let input = cx.new({
            let state = state.clone();
            let app = app.clone();
            move |cx| ChatInput::new(cx, state.clone(), app)
        });

        let title = cx.new({
            let state = state.clone();
            let app = app.clone();
            move |cx| ConversationTitle::new(cx, state.clone(), app)
        });

        let model_loader = cx.new({
            let state = state.clone();
            let app = app.clone();
            move |cx| ModelSelector::new(cx, state.clone(), app.clone())
        });

        let welcome = cx.new({
            let app = app.clone();
            move |cx| WelcomeScreen::new(cx, app)
        });

        Self {
            state,
            input,
            title,
            model_loader,
            welcome,
            scroll_handle: ScrollHandle::new(),
            should_scroll_to_bottom: true,
            follow_streaming_output: true,
            last_rendered_is_generating: false,
            last_rendered_conversation_id: None,
            last_rendered_message_count: 0,
            expanded_metadata_id: None,
            _app: app,
        }
    }

    #[allow(dead_code)]
    fn should_follow_output(&self) -> bool {
        self.distance_to_bottom_px() <= 24.0
    }

    fn distance_to_bottom_px(&self) -> f64 {
        let max = self.scroll_handle.max_offset().height;
        if max <= px(0.) {
            return 0.0;
        }
        let off = self.scroll_handle.offset().y;
        (off.to_f64() + max.to_f64()).abs()
    }
}

impl Render for ChatView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);
        let conversation = state.selected_conversation();
        let is_generating = state.is_generating();
        let selected_conversation_id = conversation.map(|c| c.id);
        let selected_message_count = conversation.map_or(0, |c| c.messages.len());

        if self.last_rendered_conversation_id != selected_conversation_id || self.last_rendered_message_count != selected_message_count {
            self.should_scroll_to_bottom = true;
            self.last_rendered_conversation_id = selected_conversation_id;
            self.last_rendered_message_count = selected_message_count;
        }

        if is_generating && !self.last_rendered_is_generating {
            // Engage sticky bottom-follow only if user was already near bottom.
            self.follow_streaming_output = self.should_follow_output();
        } else if !is_generating && self.last_rendered_is_generating {
            self.follow_streaming_output = false;
        }

        if is_generating && self.follow_streaming_output {
            // If user scrolls away meaningfully during generation, stop forcing follow.
            if self.distance_to_bottom_px() > 96.0 {
                self.follow_streaming_output = false;
            } else {
                self.should_scroll_to_bottom = true;
            }
        }

        if self.should_scroll_to_bottom {
            self.scroll_handle.scroll_to_bottom();
            self.should_scroll_to_bottom = false;
        }
        self.last_rendered_is_generating = is_generating;

        div()
            .flex()
            .flex_col()
            .flex_1()
            .h_full()
            .bg(colors::bg_base())
            // Header with title and model controls
            .child(
                div()
                    .flex_none()
                    .flex()
                    .items_center()
                    .justify_between()
                    .h(px(56.0))
                    .px(spacing::lg())
                    .border_b_1()
                    .border_color(colors::border())
                    .bg(colors::bg_surface())
                    // Left side: conversation title
                    .child(self.title.clone())
                    // Right side: model loader
                    .child(self.model_loader.clone()),
            )
            // Messages area
            .child(
                div()
                    .id("messages_container")
                    .flex()
                    .flex_col()
                    .flex_1()
                    .overflow_hidden()
                    .relative()
                    .child(
                        div()
                            .id("messages-scroll")
                            .flex_1()
                            .overflow_y_scroll()
                            .track_scroll(&self.scroll_handle)
                            .child(if let Some(convo) = conversation {
                                let message_count = convo.messages.len();
                                v_flex()
                                    .gap(spacing::md())
                                    .child(div().h(spacing::md()))
                                    .children(convo.messages.iter().enumerate().map(|(idx, msg)| {
                                        let msg_id = msg.id;
                                        let expanded = self.expanded_metadata_id == Some(msg_id);
                                        let is_last_message = idx + 1 == message_count;
                                        let this = cx.weak_entity();
                                        render_message(msg, expanded, move |cx| {
                                            if let Some(this) = this.upgrade() {
                                                this.update(cx, |this, _| {
                                                    if this.expanded_metadata_id == Some(msg_id) {
                                                        this.expanded_metadata_id = None;
                                                    } else {
                                                        this.expanded_metadata_id = Some(msg_id);
                                                        if is_last_message {
                                                            this.should_scroll_to_bottom = true;
                                                        }
                                                    }
                                                });
                                            }
                                        })
                                    }))
                                    .child(div().h(spacing::xxl()))
                                    .into_any_element()
                            } else {
                                // Welcome screen
                                self.welcome.clone().into_any_element()
                            }),
                    )
                    .vertical_scrollbar(&self.scroll_handle),
            )
            // Input area
            .child(self.input.clone())
    }
}
