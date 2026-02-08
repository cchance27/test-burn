//! Metallic GUI - AI Chat Interface
//!
//! A modern, GPU-accelerated chat interface built with GPUI.

use gpui::{App, Application, Bounds, KeyBinding, Menu, MenuItem, WindowBounds, WindowOptions, prelude::*, px, size};
use gpui_component::{Root, Theme, ThemeMode};
use metallic_gui::{ChatApp, Quit, theme::colors};

fn main() {
    // Initialize tracing for RUST_LOG support
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    Application::new().run(|cx: &mut App| {
        // Initialize gpui-component systems (theme, root, select, menus, etc.)
        gpui_component::init(cx);
        Theme::change(ThemeMode::Dark, None, cx);

        // Harmonize gpui-component controls (Select/List/Menu) with Metallic dark palette.
        let theme = Theme::global_mut(cx);
        theme.background = colors::bg_surface().into();
        theme.popover = colors::bg_surface().into();
        theme.list = colors::bg_surface().into();
        theme.list_head = colors::bg_surface().into();
        theme.list_hover = colors::bg_hover().into();
        theme.input = colors::border().into();
        theme.border = colors::border().into();
        theme.foreground = colors::text_primary().into();
        theme.muted_foreground = colors::text_secondary().into();
        theme.accent = colors::bg_selected().into();
        theme.accent_foreground = colors::text_primary().into();
        theme.ring = colors::border_focused().into();

        // Register global keybindings
        cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);

        // Define a native app menu so macOS treats this as a foreground app with
        // a standard menubar (app menu + File menu).
        cx.set_menus(vec![Menu {
            name: "Metallic".into(),
            items: vec![MenuItem::action("Quit", Quit)],
        }]);

        // Handle global actions
        cx.on_action(|_: &Quit, cx| cx.quit());
        // Ensure closing the last window (traffic light / cmd-w / window close action)
        // actually terminates the process instead of leaving the event loop running.
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        // Keep the app activated when launched from terminal, so the menubar
        // shows this app as active.
        cx.activate(true);

        // Create window with reasonable default size and minimum size
        let bounds = Bounds::centered(None, size(px(1200.0), px(800.0)), cx);

        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_min_size: Some(size(px(800.0), px(600.0))),
                ..Default::default()
            },
            |window, cx| {
                // Ensure the window is key/frontmost on macOS so the app appears active
                // in the system menubar when launched from terminal.
                window.activate_window();
                cx.activate(true);
                let app = cx.new(ChatApp::new);
                cx.new(|cx| Root::new(app, window, cx))
            },
        )
        .expect("Failed to open window");
    });
}
