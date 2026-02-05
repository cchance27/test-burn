//! Root application component.

use gpui::{Context, Entity, Window, div, prelude::*};
use gpui_component::Root;

use crate::{
    components::{chat::ChatView, input::register_input_bindings, sidebar::Sidebar}, state::AppState, theme::{colors, spacing}
};

/// Root application UI component.
///
/// This is a thin UI coordinator that owns view entities and delegates
/// state management to `AppState`.
pub struct ChatApp {
    state: Entity<AppState>,
    sidebar: Entity<Sidebar>,
    chat_view: Entity<ChatView>,
}

impl ChatApp {
    pub fn new(cx: &mut Context<Self>) -> Self {
        // Register input bindings
        register_input_bindings(cx);

        // Create shared state
        let state = cx.new(AppState::new);

        let entity = cx.entity().clone();

        // Create sidebar with shared state and app entity for callbacks
        let sidebar = cx.new({
            let state = state.clone();
            let entity = entity.clone();
            move |_| Sidebar::new(state.clone(), entity.clone())
        });

        // Create chat view with shared state and submit handler
        let chat_view = cx.new({
            let state = state.clone();
            let entity = entity.clone();
            move |cx| ChatView::new(cx, state.clone(), entity.clone())
        });

        Self { state, sidebar, chat_view }
    }

    pub fn select_conversation(&mut self, id: u64, cx: &mut Context<Self>) {
        self.mutate_state(cx, |state, cx| {
            state.select_conversation(id, cx);
        });
    }

    pub fn create_new_conversation(&mut self, cx: &mut Context<Self>) {
        self.mutate_state(cx, |state, cx| {
            state.create_new_conversation(cx);
        });
    }

    pub fn delete_conversation(&mut self, id: u64, cx: &mut Context<Self>) {
        self.mutate_state(cx, |state, cx| {
            state.delete_conversation(id, cx);
        });
    }

    pub fn rename_conversation(&mut self, id: u64, new_title: String, cx: &mut Context<Self>) {
        self.mutate_state(cx, |state, cx| {
            state.rename_conversation(id, new_title, cx);
        });
    }

    pub fn send_message(&mut self, content: String, cx: &mut Context<Self>) {
        self.mutate_state(cx, |state, cx| {
            state.send_message(content, cx);
        });
    }

    pub fn stop_generation(&mut self, cx: &mut Context<Self>) -> bool {
        self.mutate_state(cx, |state, cx| state.stop_generation(cx))
    }

    pub fn cycle_selected_model(&mut self, cx: &mut Context<Self>) {
        self.mutate_state(cx, |state, cx| {
            state.cycle_selected_model(cx);
        });
    }

    pub fn select_model(&mut self, model: String, cx: &mut Context<Self>) -> bool {
        self.mutate_state(cx, |state, cx| state.select_model(&model, cx))
    }

    pub fn clear_selected_model(&mut self, cx: &mut Context<Self>) {
        self.mutate_state(cx, |state, cx| {
            state.clear_selected_model(cx);
        });
    }

    pub fn load_selected_model(&mut self, cx: &mut Context<Self>) -> bool {
        self.mutate_state(cx, |state, cx| state.load_selected_model(cx))
    }

    fn notify_views(&mut self, cx: &mut Context<Self>) {
        self.sidebar.update(cx, |_sidebar, cx| {
            cx.notify();
        });
        self.chat_view.update(cx, |_view, cx| {
            cx.notify();
        });
        cx.notify();
    }

    fn mutate_state<R>(&mut self, cx: &mut Context<Self>, f: impl FnOnce(&mut AppState, &mut Context<AppState>) -> R) -> R {
        let result = self.state.update(cx, f);
        self.notify_views(cx);
        result
    }
}

impl Render for ChatApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let dialog_layer = Root::render_dialog_layer(window, cx);

        div()
            .flex()
            .w_full()
            .h_full()
            .bg(colors::bg_base())
            // Sidebar
            .child(
                div()
                    .flex_none()
                    .w(spacing::sidebar_width())
                    .h_full()
                    .border_r_1()
                    .border_color(colors::border())
                    .child(self.sidebar.clone()),
            )
            // Main chat area
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .h_full()
                    .overflow_hidden()
                    .child(self.chat_view.clone()),
            )
            .children(dialog_layer)
    }
}
