//! Theme constants and color palette for the chat interface.

use gpui::{Hsla, Rgba, hsla, rgb};

/// Dark theme color palette.
pub mod colors {
    use super::*;

    // Backgrounds
    pub fn bg_base() -> Rgba {
        rgb(0x0f0f0f)
    }

    pub fn bg_surface() -> Rgba {
        rgb(0x1a1a1a)
    }

    pub fn bg_elevated() -> Rgba {
        rgb(0x252525)
    }

    pub fn bg_hover() -> Rgba {
        rgb(0x2a2a2a)
    }

    pub fn bg_selected() -> Rgba {
        rgb(0x333333)
    }

    // User message bubble
    pub fn user_bubble() -> Rgba {
        rgb(0x2563eb) // Blue
    }

    pub fn user_bubble_hover() -> Rgba {
        rgb(0x3b82f6)
    }

    // Assistant message bubble
    pub fn assistant_bubble() -> Rgba {
        rgb(0x262626)
    }

    // Text colors
    pub fn text_primary() -> Rgba {
        rgb(0xf5f5f5)
    }

    pub fn text_secondary() -> Rgba {
        rgb(0xa0a0a0)
    }

    pub fn text_muted() -> Rgba {
        rgb(0x666666)
    }

    // Accent colors
    pub fn accent() -> Rgba {
        rgb(0x8b5cf6) // Purple
    }

    pub fn accent_hover() -> Rgba {
        rgb(0xa78bfa)
    }

    pub fn accent_gradient_start() -> Hsla {
        hsla(262.0 / 360.0, 0.83, 0.58, 1.0) // Purple
    }

    pub fn accent_gradient_end() -> Hsla {
        hsla(217.0 / 360.0, 0.91, 0.60, 1.0) // Blue
    }

    // Borders
    pub fn border() -> Rgba {
        rgb(0x333333)
    }

    pub fn border_focused() -> Rgba {
        rgb(0x8b5cf6)
    }

    // Scrollbar
    pub fn scrollbar_track() -> Rgba {
        rgb(0x1a1a1a)
    }

    pub fn scrollbar_thumb() -> Rgba {
        rgb(0x404040)
    }

    // Danger/destructive action
    pub fn danger_base() -> Rgba {
        rgb(0x7f1d1d) // Dark red
    }

    pub fn danger() -> Rgba {
        rgb(0xef4444) // Red
    }

    pub fn danger_hover() -> Rgba {
        rgb(0xf87171)
    }

    /// Fully transparent color.
    pub fn transparent() -> Rgba {
        Rgba {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        }
    }
}

/// Spacing constants (in pixels).
pub mod spacing {
    use gpui::Pixels;

    pub fn xs() -> Pixels {
        gpui::px(4.0)
    }

    pub fn sm() -> Pixels {
        gpui::px(8.0)
    }

    pub fn md() -> Pixels {
        gpui::px(12.0)
    }

    pub fn lg() -> Pixels {
        gpui::px(16.0)
    }

    pub fn xl() -> Pixels {
        gpui::px(24.0)
    }

    pub fn xxl() -> Pixels {
        gpui::px(32.0)
    }

    pub fn sidebar_width() -> Pixels {
        gpui::px(280.0)
    }

    pub fn input_height() -> Pixels {
        gpui::px(120.0)
    }

    pub fn message_max_width() -> Pixels {
        gpui::px(700.0)
    }
}

/// Border radius constants.
pub mod radius {
    use gpui::Pixels;

    pub fn sm() -> Pixels {
        gpui::px(4.0)
    }

    pub fn md() -> Pixels {
        gpui::px(8.0)
    }

    pub fn lg() -> Pixels {
        gpui::px(12.0)
    }

    pub fn xl() -> Pixels {
        gpui::px(16.0)
    }

    pub fn full() -> Pixels {
        gpui::px(9999.0)
    }
}

/// Typography sizes.
pub mod typography {
    use gpui::Pixels;

    pub fn xs() -> Pixels {
        gpui::px(11.0)
    }

    pub fn sm() -> Pixels {
        gpui::px(13.0)
    }

    pub fn base() -> Pixels {
        gpui::px(14.0)
    }

    pub fn md() -> Pixels {
        gpui::px(16.0)
    }

    pub fn lg() -> Pixels {
        gpui::px(18.0)
    }

    pub fn xl() -> Pixels {
        gpui::px(24.0)
    }

    pub fn xxl() -> Pixels {
        gpui::px(32.0)
    }
}
