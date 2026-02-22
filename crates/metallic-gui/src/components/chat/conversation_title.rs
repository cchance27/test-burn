//! Conversation title component with edit mode.

use gpui::{Context, Entity, MouseButton, Window, div, prelude::*, px};

use crate::{
    app::ChatApp, components::{
        common::{Button, ButtonSize, ButtonVariant}, input::TextInput
    }, state::AppState, theme::{colors, spacing, typography}
};

/// Displays the conversation title with edit capability.
pub struct ConversationTitle {
    conversation_id: Option<u64>,
    is_editing: bool,
    title_input: Option<Entity<TextInput>>,
    _edit_subscription: Option<gpui::Subscription>,
    state: Entity<AppState>,
    app: Entity<ChatApp>,
}

impl ConversationTitle {
    pub fn new(_cx: &mut Context<Self>, state: Entity<AppState>, app: Entity<ChatApp>) -> Self {
        Self {
            conversation_id: None,
            is_editing: false,
            title_input: None,
            _edit_subscription: None,
            state,
            app,
        }
    }

    fn save_title(&mut self, new_title: Option<String>, cx: &mut Context<Self>) {
        if !self.is_editing {
            return;
        }

        // Get title before clearing state
        let title = new_title.unwrap_or_else(|| {
            self.title_input
                .as_ref()
                .map(|input| input.read(cx).content().trim().to_string())
                .unwrap_or_default()
        });

        self.is_editing = false;

        // Capture for deferred task
        let conversation_id = self.conversation_id;
        let app = self.app.clone();
        let view = cx.weak_entity();

        cx.defer(move |cx| {
            // Call rename on app
            if let Some(id) = conversation_id
                && !title.is_empty()
            {
                app.update(cx, |app, cx| {
                    app.rename_conversation(id, title, cx);
                });
            }

            // Clear input
            if let Some(view) = view.upgrade() {
                view.update(cx, |this, cx| {
                    this.title_input = None;
                    this._edit_subscription = None;
                    cx.notify();
                });
            }
        });

        cx.notify();
    }

    fn cancel_editing(&mut self, cx: &mut Context<Self>) {
        if !self.is_editing {
            return;
        }
        self.is_editing = false;

        let view = cx.weak_entity();
        cx.defer(move |cx| {
            if let Some(view) = view.upgrade() {
                view.update(cx, |this, cx| {
                    this.title_input = None;
                    this._edit_subscription = None;
                    cx.notify();
                });
            }
        });

        cx.notify();
    }

    fn start_editing(&mut self, conversation_id: u64, title: String, window: &mut Window, cx: &mut Context<Self>) {
        self.is_editing = true;
        self.conversation_id = Some(conversation_id);

        let entity = cx.entity().clone();
        let e1 = entity.clone();
        let e2 = entity.clone();

        let title_input = cx.new(|cx| {
            TextInput::new(cx)
                .with_content(title)
                .on_submit(move |content, _window, cx| {
                    e1.update(cx, |this, cx| this.save_title(Some(content.to_string()), cx));
                })
                .on_escape(move |_, cx| {
                    e2.update(cx, |this, cx| this.cancel_editing(cx));
                })
        });

        let focus_handle = title_input.read(cx).focus_handle();
        window.focus(&focus_handle);

        // Save on blur
        let view = cx.weak_entity();
        let sub = cx.on_blur(&focus_handle, window, move |_, _, cx| {
            let view = view.clone();
            cx.defer(move |cx| {
                if let Some(view) = view.upgrade() {
                    view.update(cx, |this, cx| this.save_title(None, cx));
                }
            });
        });

        self.title_input = Some(title_input);
        self._edit_subscription = Some(sub);
        cx.notify();
    }
}

impl Render for ConversationTitle {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let selected = self
            .state
            .read(cx)
            .selected_conversation()
            .map(|conversation| (conversation.id, conversation.title.clone()));

        if self.is_editing && self.conversation_id != selected.as_ref().map(|(id, _)| *id) {
            self.is_editing = false;
            self.conversation_id = None;
            self.title_input = None;
            self._edit_subscription = None;
        }

        let has_title = selected.is_some();
        let title_text = selected
            .as_ref()
            .map(|(_, title)| title.clone())
            .unwrap_or_else(|| "Select a conversation".to_string());

        div().flex().items_center().gap(spacing::md()).child(if self.is_editing {
            // Edit mode
            let entity = cx.entity().clone();
            let e2 = entity.clone();

            div()
                .flex()
                .items_center()
                .gap(spacing::sm())
                .child(div().min_w(px(200.0)).child(self.title_input.as_ref().unwrap().clone()))
                .child({
                    let entity = entity.clone();
                    Button::new("Save")
                        .id("save-title-btn")
                        .size(ButtonSize::Sm)
                        .on_click(move |_, cx| {
                            entity.update(cx, |this, cx| {
                                this.save_title(None, cx);
                            });
                        })
                })
                .child({
                    Button::new("Cancel")
                        .id("cancel-title-btn")
                        .variant(ButtonVariant::Ghost)
                        .size(ButtonSize::Sm)
                        .on_click(move |_, cx| {
                            e2.update(cx, |this, cx| {
                                this.cancel_editing(cx);
                            });
                        })
                })
                .into_any_element()
        } else {
            // Display mode
            let entity = cx.entity().clone();

            div()
                .flex()
                .items_center()
                .gap(spacing::sm())
                .child(
                    div()
                        .text_size(typography::md())
                        .text_color(colors::text_primary())
                        .child(title_text),
                )
                .when(has_title, |d| {
                    let selected = selected.clone();
                    d.child(
                        div()
                            .id("edit-title-btn")
                            .px(spacing::xs())
                            .text_size(typography::xs())
                            .text_color(colors::text_muted())
                            .cursor_pointer()
                            .hover(|style| style.text_color(colors::accent()))
                            .on_mouse_down(MouseButton::Left, {
                                move |_, window, cx| {
                                    let selected = selected.clone();
                                    entity.update(cx, |this, cx| {
                                        if let Some((conversation_id, title)) = selected {
                                            this.start_editing(conversation_id, title, window, cx);
                                        }
                                    });
                                }
                            })
                            .child("Edit"),
                    )
                })
                .into_any_element()
        })
    }
}
