//! Unified Button and IconButton components.

use std::rc::Rc;

use gpui::{App, ElementId, SharedString, Window, div, prelude::*, px};

use crate::theme::{colors, radius, spacing, typography};

/// Button visual style variants.
#[derive(Clone, Copy, Default, PartialEq)]
pub enum ButtonVariant {
    /// Primary action button with accent color.
    #[default]
    Primary,
    /// Secondary button with elevated background.
    Secondary,
    /// Ghost button with no background, text-only.
    Ghost,
    /// Danger button for destructive actions.
    Danger,
}

/// Button size presets.
#[derive(Clone, Copy, Default, PartialEq)]
pub enum ButtonSize {
    /// Small: less padding, smaller text.
    Sm,
    /// Medium: default size.
    #[default]
    Md,
    /// Large: more padding, larger text.
    Lg,
}

/// A unified button component with configurable variant and size.
pub struct Button {
    id: gpui::ElementId,
    label: String,
    variant: ButtonVariant,
    size: ButtonSize,
    disabled: bool,
    on_click: Option<Rc<dyn Fn(&mut Window, &mut App)>>,
}

impl Button {
    /// Create a new button with the given label.
    pub fn new(label: impl Into<String>) -> Self {
        let label = label.into();
        Self {
            id: ElementId::Name(SharedString::new(label.clone())), // Stable ID from label
            label,
            variant: ButtonVariant::Primary,
            size: ButtonSize::Md,
            disabled: false,
            on_click: None,
        }
    }

    /// Set the button's element ID.
    pub fn id(mut self, id: &'static str) -> Self {
        self.id = ElementId::Name(SharedString::new(id));
        self
    }

    /// Set the button variant.
    pub fn variant(mut self, variant: ButtonVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Set the button size.
    pub fn size(mut self, size: ButtonSize) -> Self {
        self.size = size;
        self
    }

    /// Set whether the button is disabled.
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Set the click handler.
    pub fn on_click(mut self, handler: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        self.on_click = Some(Rc::new(handler));
        self
    }
}

impl IntoElement for Button {
    type Element = gpui::Stateful<gpui::Div>;

    fn into_element(self) -> Self::Element {
        let (px_h, px_v, text_size) = match self.size {
            ButtonSize::Sm => (spacing::sm(), px(2.0), typography::sm()),
            ButtonSize::Md => (spacing::md(), spacing::xs(), typography::sm()),
            ButtonSize::Lg => (spacing::xl(), spacing::md(), typography::base()),
        };

        let (bg, hover_bg, text_color, border) = match self.variant {
            ButtonVariant::Primary => (colors::accent(), colors::accent_hover(), colors::text_primary(), None),
            ButtonVariant::Secondary => (
                colors::bg_elevated(),
                colors::bg_hover(),
                colors::text_secondary(),
                Some(colors::border()),
            ),
            ButtonVariant::Ghost => (colors::transparent(), colors::transparent(), colors::text_muted(), None),
            ButtonVariant::Danger => (colors::transparent(), colors::danger(), colors::text_muted(), None),
        };

        let hover_text = match self.variant {
            ButtonVariant::Ghost => Some(colors::text_primary()),
            ButtonVariant::Danger => Some(colors::text_primary()),
            _ => None,
        };

        let on_click = self.on_click;
        let disabled = self.disabled;
        let _variant = self.variant;

        let mut element = div()
            .id(self.id)
            .px(px_h)
            .py(px_v)
            .bg(bg)
            .rounded(radius::sm())
            .text_size(text_size)
            .text_color(text_color)
            .child(self.label);

        if let Some(border_color) = border {
            element = element.border_1().border_color(border_color);
        }

        if !disabled {
            element = element.cursor_pointer();
            element = element.hover(move |style| {
                let style = style.bg(hover_bg);
                if let Some(txt) = hover_text { style.text_color(txt) } else { style }
            });

            if let Some(handler) = on_click {
                element = element.on_click(move |_, window, cx| {
                    handler(window, cx);
                });
            }
        } else {
            element = element.opacity(0.5);
        }

        element
    }
}

/// An icon-only button for actions like send or delete.
pub struct IconButton {
    id: gpui::ElementId,
    icon: String,
    variant: ButtonVariant,
    size: f32,
    on_click: Option<Rc<dyn Fn(&mut Window, &mut App)>>,
}

impl IconButton {
    /// Create a new icon button with the given icon character/emoji.
    pub fn new(icon: impl Into<String>) -> Self {
        let icon = icon.into();
        Self {
            id: ElementId::Name(SharedString::new(icon.clone())),
            icon,
            variant: ButtonVariant::Primary,
            size: 32.0,
            on_click: None,
        }
    }

    /// Set the button's element ID.
    pub fn id(mut self, id: &'static str) -> Self {
        self.id = ElementId::Name(SharedString::new(id));
        self
    }

    /// Set the button variant.
    pub fn variant(mut self, variant: ButtonVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Set the button size in pixels.
    pub fn size_px(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    /// Set the click handler.
    pub fn on_click(mut self, handler: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        self.on_click = Some(Rc::new(handler));
        self
    }
}

impl IntoElement for IconButton {
    type Element = gpui::Stateful<gpui::Div>;

    fn into_element(self) -> Self::Element {
        let (bg, hover_bg, text_color) = match self.variant {
            ButtonVariant::Primary => (colors::accent(), colors::accent_hover(), colors::text_primary()),
            ButtonVariant::Secondary => (colors::bg_elevated(), colors::bg_hover(), colors::text_secondary()),
            ButtonVariant::Ghost => (colors::transparent(), colors::transparent(), colors::text_muted()),
            ButtonVariant::Danger => (colors::transparent(), colors::danger(), colors::text_muted()),
        };

        let hover_text = match self.variant {
            ButtonVariant::Ghost => Some(colors::text_primary()),
            ButtonVariant::Danger => Some(colors::text_primary()),
            _ => None,
        };

        let on_click = self.on_click;
        let size = px(self.size);

        let mut element = div()
            .id(self.id)
            .flex()
            .items_center()
            .justify_center()
            .w(size)
            .h(size)
            .bg(bg)
            .rounded(radius::sm())
            .text_size(typography::md())
            .text_color(text_color)
            .cursor_pointer()
            .hover(move |style| {
                let style = style.bg(hover_bg);
                if let Some(txt) = hover_text { style.text_color(txt) } else { style }
            })
            .child(self.icon);

        if let Some(handler) = on_click {
            element = element.on_click(move |_, window, cx| {
                handler(window, cx);
            });
        }

        element
    }
}
