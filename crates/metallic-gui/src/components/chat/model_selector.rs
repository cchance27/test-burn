//! Model selection and loading component.

use gpui::{Context, Entity, MouseButton, Subscription, Window, div, prelude::*, px};
use gpui_component::{
    Sizable, Size, select::{SearchableVec, Select, SelectEvent, SelectState}
};

use crate::{
    app::ChatApp, inference::ModelLoadStatus, state::AppState, theme::{colors, radius, spacing, typography}
};

type ModelItems = SearchableVec<String>;

/// Model selector dropdown and load button.
pub struct ModelSelector {
    state: Entity<AppState>,
    app: Entity<ChatApp>,
    select_state: Option<Entity<SelectState<ModelItems>>>,
    last_synced_items: Vec<String>,
    last_synced_selected: Option<String>,
    _subscriptions: Vec<Subscription>,
}

impl ModelSelector {
    pub fn new(_cx: &mut Context<Self>, state: Entity<AppState>, app: Entity<ChatApp>) -> Self {
        Self {
            state,
            app,
            select_state: None,
            last_synced_items: Vec::new(),
            last_synced_selected: None,
            _subscriptions: Vec::new(),
        }
    }

    fn ensure_select_state(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Entity<SelectState<ModelItems>> {
        if let Some(select_state) = self.select_state.clone() {
            return select_state;
        }

        // Get display names from ModelInfo
        let model_items: Vec<String> = self
            .state
            .read(cx)
            .available_models()
            .iter()
            .map(|m| m.display_name.clone())
            .collect();
        let select_state = cx.new(|cx| SelectState::new(SearchableVec::new(model_items), None, window, cx).searchable(true));

        let app = self.app.clone();
        let subscription = cx.subscribe(&select_state, move |_this, _, event: &SelectEvent<ModelItems>, cx| match event {
            SelectEvent::Confirm(Some(model)) => {
                let selected = model.clone();
                app.update(cx, |app, cx| {
                    app.select_model(selected, cx);
                });
            }
            SelectEvent::Confirm(None) => {
                app.update(cx, |app, cx| {
                    app.clear_selected_model(cx);
                });
            }
        });

        self._subscriptions.push(subscription);
        self.select_state = Some(select_state.clone());
        select_state
    }

    fn sync_select_state(&mut self, select_state: &Entity<SelectState<ModelItems>>, window: &mut Window, cx: &mut Context<Self>) {
        let (current_items, selected_model) = {
            let state = self.state.read(cx);
            (
                state.available_models().iter().map(|m| m.display_name.clone()).collect::<Vec<_>>(),
                state.selected_model_name().map(str::to_string),
            )
        };

        let items_changed = current_items != self.last_synced_items;
        let selected_changed = selected_model != self.last_synced_selected;
        if !(items_changed || selected_changed) {
            return;
        }

        self.last_synced_items = current_items.clone();
        self.last_synced_selected = selected_model.clone();

        select_state.update(cx, |select, cx| {
            if items_changed {
                select.set_items(SearchableVec::new(current_items), window, cx);
            }

            if let Some(model) = selected_model.as_ref() {
                select.set_selected_value(model, window, cx);
            } else {
                select.set_selected_index(None, window, cx);
            }
        });
    }
}

impl Render for ModelSelector {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (load_status, has_selection, context_readout) = {
            let state = self.state.read(cx);
            (
                state.model_load_status().clone(),
                state.selected_model().is_some(),
                state.selected_model_context_readout(),
            )
        };
        let is_loaded = load_status == ModelLoadStatus::Loaded;
        let is_loading = load_status == ModelLoadStatus::Loading;

        let select_state = self.ensure_select_state(window, cx);
        self.sync_select_state(&select_state, window, cx);

        // Determine button text and style
        let (button_text, button_enabled) = match load_status {
            ModelLoadStatus::Idle => ("Load Model", has_selection),
            ModelLoadStatus::Loading => ("Loading...", false),
            ModelLoadStatus::Loaded => ("Loaded ✓", true),
            ModelLoadStatus::Error(ref _msg) => ("Load Model", has_selection),
        };

        div()
            .flex()
            .items_center()
            .gap(spacing::sm())
            .child(
                div()
                    .w(px(78.0))
                    .text_right()
                    .text_size(typography::xs())
                    .text_color(colors::text_secondary())
                    .child(context_readout),
            )
            // Model selector dropdown (disabled during loading)
            .child(
                div()
                    .id("model-selector")
                    .w(px(350.0))
                    .when(is_loading, |this| this.opacity(0.6))
                    .child(
                        Select::new(&select_state)
                            .placeholder("Select a model...")
                            .search_placeholder("Search models...")
                            .appearance(true)
                            .bg(colors::bg_elevated())
                            .with_size(Size::Small)
                            .disabled(is_loading),
                    ),
            )
            // Load button
            .child(
                div()
                    .when(button_enabled && !is_loading, |this| {
                        this.on_mouse_down(MouseButton::Left, {
                            let app = self.app.clone();
                            move |_, _, cx| {
                                app.update(cx, |app, cx| {
                                    app.load_selected_model(cx);
                                });
                            }
                        })
                    })
                    .child(
                        div()
                            .id("load-model-btn")
                            .px(spacing::md())
                            .py(spacing::xs())
                            .bg(if is_loaded {
                                colors::bg_selected()
                            } else if is_loading {
                                colors::bg_surface()
                            } else if button_enabled {
                                colors::accent()
                            } else {
                                colors::bg_surface()
                            })
                            .rounded(radius::sm())
                            .text_size(typography::sm())
                            .text_color(if button_enabled || is_loading {
                                colors::text_primary()
                            } else {
                                colors::text_muted()
                            })
                            .when(button_enabled && !is_loading, |this| {
                                this.cursor_pointer().hover(move |style| {
                                    if is_loaded {
                                        style.bg(colors::bg_hover())
                                    } else {
                                        style.bg(colors::accent_hover())
                                    }
                                })
                            })
                            .child(button_text),
                    ),
            )
    }
}
