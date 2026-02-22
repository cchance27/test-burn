//! Settings view with tabbed interface.

use gpui::{App, Context, Entity, Pixels, SharedString, Subscription, Window, div, prelude::*, px};
use gpui_component::{
    h_flex, setting::{SettingField, SettingGroup, SettingItem, SettingPage, Settings}, slider::{Slider, SliderEvent, SliderState}, v_flex
};

use crate::{
    components::{common::Button, input::TextInput}, state::AppState
};

const SETTINGS_MODEL_TARGET_KEY: &str = "settings_model_target";
const DEFAULT_MODEL_NAME: &str = "default";

struct SliderFieldState {
    slider: Entity<SliderState>,
    _subscription: Subscription,
    model_name: String,
    input_name: String,
}

pub struct SettingsView {
    state: Entity<AppState>,
    content_height: Pixels,
    system_prompt_input: Entity<TextInput>,
}

impl SettingsView {
    fn selected_settings_model_name(state: &AppState) -> Option<String> {
        if let Some(saved) = state.get_setting(SETTINGS_MODEL_TARGET_KEY)
            && state.available_models().iter().any(|model| model.display_name == *saved)
        {
            return Some(saved.clone());
        }

        if let Some(active) = state.selected_model_name() {
            return Some(active.to_string());
        }

        state.available_models().first().map(|model| model.display_name.clone())
    }

    fn dynamic_slider_field(
        state_entity: Entity<AppState>,
        input_name: String,
        min: f32,
        max: f32,
        step: f32,
    ) -> SettingField<SharedString> {
        SettingField::render(move |options, window: &mut Window, cx: &mut App| {
            let model_name = {
                let state = state_entity.read(cx);
                SettingsView::selected_settings_model_name(state).unwrap_or_else(|| DEFAULT_MODEL_NAME.to_string())
            };
            let current_value = {
                let state = state_entity.read(cx);
                let settings = state.get_model_settings(&model_name);
                settings
                    .values
                    .get(&input_name)
                    .and_then(|v| v.as_f64().map(|f| f as f32))
                    .unwrap_or(0.0) // Fallback should ideally come from workflow default but for now 0.0
            };

            let key = SharedString::from(format!(
                "settings-{}-{}-{}-{}-{}",
                input_name, options.page_ix, options.group_ix, options.item_ix, model_name
            ));

            let slider_field_state = window.use_keyed_state(key, cx, {
                let state_entity = state_entity.clone();
                let model_name = model_name.clone();
                let input_name = input_name.clone();
                move |window, cx| {
                    let slider = cx.new(|_| SliderState::new().min(min).max(max).step(step).default_value(current_value));

                    let subscription = cx.subscribe_in(&slider, window, {
                        let state_entity = state_entity.clone();
                        let input_name_inner = input_name.clone();
                        move |_, _, event: &SliderEvent, _, cx| {
                            let SliderEvent::Change(value) = event;
                            let value = value.start();
                            state_entity.update(cx, |state, cx| {
                                let model_name =
                                    SettingsView::selected_settings_model_name(state).unwrap_or_else(|| DEFAULT_MODEL_NAME.to_string());
                                let mut settings = state.get_model_settings(&model_name);
                                settings.values.insert(input_name_inner.clone(), serde_json::Value::from(value));
                                state.update_model_settings(&model_name, &settings);
                                cx.notify();
                            });
                        }
                    });

                    SliderFieldState {
                        slider,
                        _subscription: subscription,
                        model_name,
                        input_name,
                    }
                }
            });

            slider_field_state.update(cx, |field_state, cx| {
                let slider_value = field_state.slider.read(cx).value().start();
                if field_state.model_name != model_name
                    || field_state.input_name != input_name
                    || (slider_value - current_value).abs() > 0.0001
                {
                    field_state.model_name = model_name.clone();
                    field_state.slider.update(cx, |slider, cx| {
                        slider.set_value(current_value, window, cx);
                    });
                }
            });

            let slider = slider_field_state.read(cx).slider.clone();

            // Format display value: hide decimals if step is integer and value has no fractional part
            let display_value = if step >= 1.0 && current_value.fract() == 0.0 {
                format!("{}", current_value as i64)
            } else {
                format!("{:.2}", current_value)
            };

            h_flex()
                .w(px(240.0)) // Approximately w_60 if 1 unit = 4px
                .items_center()
                .gap_2()
                .child(div().text_sm().min_w(px(72.0)).text_right().child(display_value))
                .child(div().flex_1().min_w_0().child(Slider::new(&slider)))
        })
    }

    pub fn new(state: Entity<AppState>, content_height: Pixels, _window: &mut Window, cx: &mut Context<Self>) -> Self {
        let system_prompt = state.read(cx).get_setting("system_prompt").cloned().unwrap_or_default();
        let state_entity = state.clone();
        let system_prompt_input = cx.new(move |cx| {
            TextInput::new(cx)
                .with_placeholder("Enter system prompt...")
                .with_content(system_prompt)
                .on_change(move |content, _window, cx| {
                    state_entity.update(cx, |state, cx| {
                        state.update_setting("system_prompt".to_string(), content.to_string());
                        cx.notify();
                    });
                })
        });

        Self {
            state,
            content_height,
            system_prompt_input,
        }
    }
}

impl Render for SettingsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state_entity = self.state.clone();

        v_flex()
            .w_full()
            .h(self.content_height)
            .overflow_hidden()
            .child(Settings::new("app-settings").pages(vec![
                SettingPage::new("General").default_open(true).group(
                    SettingGroup::new()
                        .title("Preferences")
                        .item(
                            SettingItem::new(
                                "Remember last used model",
                                SettingField::switch(
                                    {
                                        let state = state_entity.clone();
                                        move |cx: &App| state.read(cx).get_setting("remember_model") == Some(&"true".to_string())
                                    },
                                    {
                                        let state = state_entity.clone();
                                        move |val: bool, cx: &mut App| {
                                            state.update(cx, |state, cx| {
                                                state.update_setting("remember_model".to_string(), val.to_string());
                                                cx.notify();
                                            });
                                        }
                                    },
                                )
                                .cursor_pointer(),
                            )
                            .description("Always select the model you used in your last session."),
                        )
                        .item(
                            SettingItem::new(
                                "Auto-load model",
                                SettingField::switch(
                                    {
                                        let state = state_entity.clone();
                                        move |cx: &App| state.read(cx).get_setting("auto_load_model") == Some(&"true".to_string())
                                    },
                                    {
                                        let state = state_entity.clone();
                                        move |val: bool, cx: &mut App| {
                                            state.update(cx, |state, cx| {
                                                state.update_setting("auto_load_model".to_string(), val.to_string());
                                                cx.notify();
                                            });
                                        }
                                    },
                                )
                                .cursor_pointer(),
                            )
                            .description("Automatically load the weights into memory on startup."),
                        )
                        .item(
                            SettingItem::new(
                                "System Prompt",
                                SettingField::render({
                                    let input = self.system_prompt_input.clone();
                                    move |_, _, _| {
                                h_flex()
                                    .w_full()
                                    .min_w(px(400.0))
                                    .child(input.clone())
                                    }
                                }),
                            )
                            .description("Initial instructions for the AI model."),
                        ),
                ),
                SettingPage::new("Model")
                    .group({
                        let state = state_entity.read(cx);
                        let models = state.available_models();
                        let options: Vec<(SharedString, SharedString)> = models
                            .iter()
                            .map(|m| (m.display_name.clone().into(), m.display_name.clone().into()))
                            .collect();

                        SettingGroup::new().title("Active Model").item(
                            SettingItem::new(
                                "Model Selection",
                                SettingField::dropdown(
                                    options,
                                    {
                                        let state = state_entity.clone();
                                        move |cx: &App| {
                                            let state = state.read(cx);
                                            SettingsView::selected_settings_model_name(state).unwrap_or_default().into()
                                        }
                                    },
                                    {
                                        let state = state_entity.clone();
                                        move |val: SharedString, cx: &mut App| {
                                            state.update(cx, |state, cx| {
                                                state.update_setting(SETTINGS_MODEL_TARGET_KEY.to_string(), val.to_string());
                                                cx.notify();
                                            });
                                        }
                                    },
                                ),
                            )
                            .description("Choose which model configuration to edit."),
                        )
                    })
                    .group({
                        let state = state_entity.read(cx);
                        let mut group = SettingGroup::new().title("Generation Parameters");

                        if let Some(workflow) = state.loaded_model().map(|m| &m.workflow) {
                            for input in &workflow.inputs {
                                if input.hidden {
                                    continue;
                                }
                                if let (Some(min), Some(max)) = (input.min, input.max) {
                                    let step = input.step.unwrap_or(0.1);
                                    let label = input.label.as_deref().unwrap_or(&input.name);
                                    let desc = input.description.clone().unwrap_or_default();
                                    let input_name = input.name.clone();

                                    if input_name == "seed" {
                                        let is_random = {
                                            let state = state_entity.read(cx);
                                            let model_name = SettingsView::selected_settings_model_name(state)
                                                .unwrap_or_else(|| DEFAULT_MODEL_NAME.to_string());
                                            let settings = state.get_model_settings(&model_name);
                                            settings.values.get("seed_random").and_then(|v| v.as_bool()).unwrap_or(false)
                                        };

                                        group = group.item(
                                            SettingItem::new(
                                                "Randomize Seed",
                                                SettingField::switch(
                                                    {
                                                        let state_entity = state_entity.clone();
                                                        move |cx| {
                                                            let state = state_entity.read(cx);
                                                            let model_name = SettingsView::selected_settings_model_name(state)
                                                                .unwrap_or_else(|| DEFAULT_MODEL_NAME.to_string());
                                                            let settings = state.get_model_settings(&model_name);
                                                            settings.values.get("seed_random").and_then(|v| v.as_bool()).unwrap_or(false)
                                                        }
                                                    },
                                                    {
                                                        let state_entity = state_entity.clone();
                                                        move |val, cx| {
                                                            state_entity.update(cx, |state, cx| {
                                                                let model_name = SettingsView::selected_settings_model_name(state)
                                                                    .unwrap_or_else(|| DEFAULT_MODEL_NAME.to_string());
                                                                let mut settings = state.get_model_settings(&model_name);
                                                                settings
                                                                    .values
                                                                    .insert("seed_random".to_string(), serde_json::Value::Bool(val));
                                                                state.update_model_settings(&model_name, &settings);
                                                                cx.notify();
                                                            });
                                                        }
                                                    },
                                                ),
                                            )
                                            .description("Use a different random seed for every generation."),
                                        );

                                        if is_random {
                                            continue;
                                        }
                                    }

                                    group = group.item(
                                        SettingItem::new(
                                            label.to_string(),
                                            SettingsView::dynamic_slider_field(
                                                state_entity.clone(),
                                                input_name,
                                                min as f32,
                                                max as f32,
                                                step as f32,
                                            ),
                                        )
                                        .description(desc),
                                    );
                                } else if let Some(_options) = &input.options {
                                    // Dropdown implementation can be added here
                                }
                            }
                        }

                        group.item(SettingItem::render({
                            let state = state_entity.clone();
                            move |_options, _window, _cx| {
                                h_flex().w_full().justify_center().child(
                                    Button::new("Reset to defaults")
                                        .id("reset-model-params")
                                        .variant(crate::components::common::ButtonVariant::Secondary)
                                        .on_click({
                                            let state = state.clone();
                                            move |_, cx| {
                                                state.update(cx, |state, cx| {
                                                    let model_name = SettingsView::selected_settings_model_name(state)
                                                        .unwrap_or_else(|| DEFAULT_MODEL_NAME.to_string());
                                                    let mut settings = state.get_model_settings(&model_name);

                                                    // Reset based on workflow defaults
                                                    if let Some(workflow) = state.loaded_model().map(|m| &m.workflow) {
                                                        for input in &workflow.inputs {
                                                            if let Some(default) = &input.default {
                                                                let val: serde_json::Value = default.clone();
                                                                settings.values.insert(input.name.clone(), val);
                                                            }
                                                        }
                                                    }

                                                    state.update_model_settings(&model_name, &settings);
                                                    cx.notify();
                                                });
                                            }
                                        }),
                                )
                            }
                        }))
                    }),
            ]))
    }
}
