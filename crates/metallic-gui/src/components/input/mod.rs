//! Reusable text input component and chat-specific wrapper.

use std::{ops::Range, rc::Rc};

use gpui::{
    App, AppContext, Bounds, Context, ElementId, ElementInputHandler, Entity, EntityInputHandler, FocusHandle, Focusable, GlobalElementId, LayoutId, MouseButton, Pixels, Point, ShapedLine, SharedString, Style, TextRun, UTF16Selection, Window, div, prelude::*, px
};
use rustc_hash::FxHashMap;
use unicode_segmentation::*;

use crate::{
    app::ChatApp, components::common::{Button, ButtonVariant, IconButton}, state::AppState, theme::{colors, radius, spacing}
};

// Actions for text input
gpui::actions!(
    text_input,
    [
        Backspace, Delete, Left, Right, SelectAll, Home, End, Paste, Copy, Cut, Submit, Escape
    ]
);

/// A generic text input view.
pub struct TextInput {
    focus_handle: FocusHandle,
    content: String,
    placeholder: SharedString,
    selected_range: Range<usize>,
    selection_reversed: bool,
    marked_range: Option<Range<usize>>,
    last_layout: Option<ShapedLine>,
    last_bounds: Option<Bounds<Pixels>>,
    on_submit: Option<Rc<dyn Fn(&str, &mut Window, &mut App)>>,
    on_change: Option<Rc<dyn Fn(&str, &mut Window, &mut App)>>,
    on_escape: Option<Rc<dyn Fn(&mut Window, &mut App)>>,
    /// Track if mouse is dragging for text selection
    is_selecting: bool,
    /// Anchor point for mouse selection (where mouse down started)
    selection_anchor: usize,
    /// Horizontal scroll offset for long text
    scroll_offset: Pixels,
    /// Whether this input should ignore edits/submit.
    disabled: bool,
}

impl TextInput {
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            focus_handle: cx.focus_handle(),
            content: String::new(),
            placeholder: "Type...".into(),
            selected_range: 0..0,
            selection_reversed: false,
            marked_range: None,
            last_layout: None,
            last_bounds: None,
            on_submit: None,
            on_change: None,
            on_escape: None,
            is_selecting: false,
            selection_anchor: 0,
            scroll_offset: px(0.),
            disabled: false,
        }
    }

    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    fn handle_mouse_down(&mut self, event: &gpui::MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(bounds) = self.last_bounds
            && let Some(layout) = &self.last_layout
        {
            let local_x = event.position.x - bounds.left() + self.scroll_offset;
            let offset = layout.index_for_x(local_x).unwrap_or(self.content.len());

            // Start selection from this point
            self.is_selecting = true;
            self.selection_anchor = offset;
            self.selected_range = offset..offset;
            self.selection_reversed = false;
            cx.notify();
        }
        window.focus(&self.focus_handle);
    }

    fn handle_mouse_up(&mut self, _event: &gpui::MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
        self.is_selecting = false;
        cx.notify();
    }

    fn handle_mouse_move(&mut self, event: &gpui::MouseMoveEvent, _window: &mut Window, cx: &mut Context<Self>) {
        if !self.is_selecting {
            return;
        }

        if let Some(bounds) = self.last_bounds
            && let Some(layout) = &self.last_layout
        {
            let local_x = event.position.x - bounds.left() + self.scroll_offset;
            let offset = layout.index_for_x(local_x).unwrap_or(self.content.len());

            // Update selection from anchor to current position
            if offset < self.selection_anchor {
                self.selected_range = offset..self.selection_anchor;
                self.selection_reversed = true;
            } else {
                self.selected_range = self.selection_anchor..offset;
                self.selection_reversed = false;
            }
            cx.notify();
        }
    }

    pub fn with_placeholder(mut self, placeholder: impl Into<SharedString>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = content.into();
        self.selected_range = 0..self.content.len();
        self
    }

    pub fn on_submit(mut self, callback: impl Fn(&str, &mut Window, &mut App) + 'static) -> Self {
        self.on_submit = Some(Rc::new(callback));
        self
    }

    pub fn on_change(mut self, callback: impl Fn(&str, &mut Window, &mut App) + 'static) -> Self {
        self.on_change = Some(Rc::new(callback));
        self
    }

    pub fn on_escape(mut self, callback: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        self.on_escape = Some(Rc::new(callback));
        self
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn disabled(&self) -> bool {
        self.disabled
    }

    pub fn set_disabled(&mut self, disabled: bool, cx: &mut Context<Self>) {
        if self.disabled != disabled {
            self.disabled = disabled;
            cx.notify();
        }
    }

    pub fn set_content(&mut self, content: String, cx: &mut Context<Self>) {
        self.content = content;
        self.selected_range = 0..0;
        cx.notify();
    }

    pub fn clear(&mut self) {
        self.content.clear();
        self.selected_range = 0..0;
        self.marked_range = None;
        self.scroll_offset = px(0.);
        self.selection_anchor = 0;
        self.selection_reversed = false;
    }

    fn do_submit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        if let Some(on_submit) = self.on_submit.clone() {
            on_submit(&self.content, window, cx);
        }
        cx.notify();
    }

    fn submit(&mut self, _: &Submit, window: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        self.do_submit(window, cx);
    }

    fn escape(&mut self, _: &Escape, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(on_escape) = self.on_escape.clone() {
            on_escape(window, cx);
        }
    }

    fn backspace(&mut self, _: &Backspace, window: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        if self.selected_range.is_empty() {
            self.select_to(self.previous_boundary(self.cursor_offset()), cx);
        }
        self.replace_text_in_range(None, "", window, cx);
    }

    fn delete(&mut self, _: &Delete, window: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        if self.selected_range.is_empty() {
            self.select_to(self.next_boundary(self.cursor_offset()), cx);
        }
        self.replace_text_in_range(None, "", window, cx);
    }

    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        if self.selected_range.is_empty() {
            self.move_to(self.previous_boundary(self.cursor_offset()), cx);
        } else {
            self.move_to(self.selected_range.start, cx);
        }
    }

    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        if self.selected_range.is_empty() {
            self.move_to(self.next_boundary(self.selected_range.end), cx);
        } else {
            self.move_to(self.selected_range.end, cx);
        }
    }

    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        self.move_to(0, cx);
        self.select_to(self.content.len(), cx);
    }

    fn home(&mut self, _: &Home, _: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        self.move_to(0, cx);
    }

    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        self.move_to(self.content.len(), cx);
    }

    fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
            self.replace_text_in_range(None, &text.replace("\n", " "), window, cx);
        }
    }

    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(gpui::ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
        }
    }

    fn cut(&mut self, _: &Cut, window: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(gpui::ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
            self.replace_text_in_range(None, "", window, cx);
        }
    }

    fn move_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.selected_range = offset..offset;
        cx.notify();
    }

    fn cursor_offset(&self) -> usize {
        if self.selection_reversed {
            self.selected_range.start
        } else {
            self.selected_range.end
        }
    }

    fn select_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        if self.selection_reversed {
            self.selected_range.start = offset;
        } else {
            self.selected_range.end = offset;
        }
        if self.selected_range.end < self.selected_range.start {
            self.selection_reversed = !self.selection_reversed;
            self.selected_range = self.selected_range.end..self.selected_range.start;
        }
        cx.notify();
    }

    fn previous_boundary(&self, offset: usize) -> usize {
        self.content
            .grapheme_indices(true)
            .rev()
            .find_map(|(idx, _)| (idx < offset).then_some(idx))
            .unwrap_or(0)
    }

    fn next_boundary(&self, offset: usize) -> usize {
        self.content
            .grapheme_indices(true)
            .find_map(|(idx, _)| (idx > offset).then_some(idx))
            .unwrap_or(self.content.len())
    }

    fn offset_from_utf16(&self, offset: usize) -> usize {
        let mut utf8_offset = 0;
        let mut utf16_count = 0;
        for ch in self.content.chars() {
            if utf16_count >= offset {
                break;
            }
            utf16_count += ch.len_utf16();
            utf8_offset += ch.len_utf8();
        }
        utf8_offset
    }

    fn offset_to_utf16(&self, offset: usize) -> usize {
        let mut utf16_offset = 0;
        let mut utf8_count = 0;
        for ch in self.content.chars() {
            if utf8_count >= offset {
                break;
            }
            utf8_count += ch.len_utf8();
            utf16_offset += ch.len_utf16();
        }
        utf16_offset
    }

    fn range_to_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.offset_to_utf16(range.start)..self.offset_to_utf16(range.end)
    }

    fn range_from_utf16(&self, range_utf16: &Range<usize>) -> Range<usize> {
        self.offset_from_utf16(range_utf16.start)..self.offset_from_utf16(range_utf16.end)
    }

    #[allow(dead_code)]
    fn offset_for_x(&self, x: Pixels) -> usize {
        if let Some(layout) = &self.last_layout {
            layout.index_for_x(x).unwrap_or(self.content.len())
        } else {
            0
        }
    }
}

impl EntityInputHandler for TextInput {
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        actual_range: &mut Option<Range<usize>>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<String> {
        let range = self.range_from_utf16(&range_utf16);
        actual_range.replace(self.range_to_utf16(&range));
        Some(self.content[range].to_string())
    }

    fn selected_text_range(
        &mut self,
        _ignore_disabled_input: bool,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        Some(UTF16Selection {
            range: self.range_to_utf16(&self.selected_range),
            reversed: self.selection_reversed,
        })
    }

    fn marked_text_range(&self, _window: &mut Window, _cx: &mut Context<Self>) -> Option<Range<usize>> {
        self.marked_range.as_ref().map(|range| self.range_to_utf16(range))
    }

    fn unmark_text(&mut self, _window: &mut Window, _cx: &mut Context<Self>) {
        self.marked_range = None;
    }

    fn replace_text_in_range(&mut self, range_utf16: Option<Range<usize>>, new_text: &str, window: &mut Window, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        let range = range_utf16
            .as_ref()
            .map(|range_utf16| self.range_from_utf16(range_utf16))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());

        // Clamp range to valid bounds to prevent panics
        let content_len = self.content.len();
        let start = range.start.min(content_len);
        let end = range.end.min(content_len).max(start);

        self.content = self.content[0..start].to_owned() + new_text + &self.content[end..];
        self.selected_range = start + new_text.len()..start + new_text.len();
        self.marked_range.take();

        if let Some(on_change) = self.on_change.clone() {
            on_change(&self.content, window, cx);
        }

        cx.notify();
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range_utf16: Option<Range<usize>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.disabled {
            return;
        }
        let range = range_utf16
            .as_ref()
            .map(|range_utf16| self.range_from_utf16(range_utf16))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());

        self.content = self.content[0..range.start].to_owned() + new_text + &self.content[range.end..];
        if !new_text.is_empty() {
            self.marked_range = Some(range.start..range.start + new_text.len());
        } else {
            self.marked_range = None;
        }
        self.selected_range = new_selected_range_utf16
            .as_ref()
            .map(|range_utf16| self.range_from_utf16(range_utf16))
            .map(|new_range| new_range.start + range.start..new_range.end + range.end)
            .unwrap_or_else(|| range.start + new_text.len()..range.start + new_text.len());

        if let Some(on_change) = self.on_change.clone() {
            on_change(&self.content, window, cx);
        }

        cx.notify();
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        bounds: Bounds<Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let last_layout = self.last_layout.as_ref()?;
        let range = self.range_from_utf16(&range_utf16);
        Some(Bounds::from_corners(
            Point::new(bounds.left() + last_layout.x_for_index(range.start), bounds.top()),
            Point::new(bounds.left() + last_layout.x_for_index(range.end), bounds.bottom()),
        ))
    }

    fn character_index_for_point(&mut self, point: Point<Pixels>, _window: &mut Window, _cx: &mut Context<Self>) -> Option<usize> {
        let line_point = self.last_bounds?.localize(&point)?;
        let last_layout = self.last_layout.as_ref()?;
        let utf8_index = last_layout.index_for_x(point.x - line_point.x + self.scroll_offset)?;
        Some(self.offset_to_utf16(utf8_index))
    }
}

impl Focusable for TextInput {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

/// Custom element for rendering text input
struct TextInputElement {
    input: Entity<TextInput>,
}

impl gpui::IntoElement for TextInputElement {
    type Element = Self;
    fn into_element(self) -> Self::Element {
        self
    }
}

impl gpui::Element for TextInputElement {
    type RequestLayoutState = ();
    type PrepaintState = Option<ShapedLine>;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.size.width = gpui::relative(1.).into();
        style.size.height = window.line_height().into();
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let input = self.input.read(cx);
        // Layout/hit-testing must always reflect actual editable content.
        let content: SharedString = input.content.clone().into();

        let style = window.text_style();
        let text_color = colors::text_primary();

        let run = TextRun {
            len: content.len(),
            font: style.font(),
            color: text_color.into(),
            background_color: None,
            underline: None,
            strikethrough: None,
        };

        let font_size = style.font_size.to_pixels(window.rem_size());
        let line = window.text_system().shape_line(content, font_size, &[run], None);

        self.input.update(cx, |input, _| {
            input.last_layout = Some(line.clone());
            input.last_bounds = Some(bounds);
        });

        Some(line)
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let input_view = self.input.clone();
        let focus_handle = input_view.read(cx).focus_handle.clone();
        window.handle_input(&focus_handle, ElementInputHandler::new(bounds, input_view.clone()), cx);

        if let Some(line) = prepaint.take() {
            let (selected_range, cursor_offset, scroll_offset, is_placeholder, placeholder_text) = {
                let input = input_view.read(cx);
                (
                    input.selected_range.clone(),
                    input.cursor_offset(),
                    input.scroll_offset,
                    input.content.is_empty(),
                    input.placeholder.clone(),
                )
            };

            // Calculate vertical centering for cursor/highlight
            let line_height = window.line_height();
            let box_height = bounds.size.height;
            let vertical_offset = (box_height - line_height) / 2.0;
            let text_top = bounds.top() + vertical_offset;
            let text_bottom = text_top + line_height;

            // Calculate cursor position and update scroll offset if needed
            let cursor_x = line.x_for_index(cursor_offset);
            let visible_width = bounds.size.width;
            let padding = px(8.0); // Small padding before edge

            // Update scroll offset to keep cursor visible
            let new_scroll_offset = {
                let cursor_in_view = cursor_x - scroll_offset;
                if cursor_in_view < px(0.) {
                    // Cursor is to the left of visible area
                    cursor_x
                } else if cursor_in_view > visible_width - padding {
                    // Cursor is to the right of visible area
                    cursor_x - visible_width + padding
                } else {
                    scroll_offset
                }
            };

            // Clamp scroll offset
            let text_width = line.width;
            let max_scroll = (text_width - visible_width + padding).max(px(0.));
            let final_scroll_offset = new_scroll_offset.max(px(0.)).min(max_scroll);

            // Update scroll offset in the input if changed
            if final_scroll_offset != scroll_offset {
                input_view.update(cx, |input, _| {
                    input.scroll_offset = final_scroll_offset;
                });
            }

            // Apply clipping to bounds
            window.with_content_mask(Some(gpui::ContentMask { bounds }), |window| {
                // Paint selection highlight
                if !selected_range.is_empty() {
                    let start_x = line.x_for_index(selected_range.start) - final_scroll_offset;
                    let end_x = line.x_for_index(selected_range.end) - final_scroll_offset;
                    window.paint_quad(gpui::fill(
                        Bounds::from_corners(
                            Point::new(bounds.left() + start_x, text_top),
                            Point::new(bounds.left() + end_x, text_bottom),
                        ),
                        gpui::rgba(0x4a9eff66), // Semi-transparent blue
                    ));
                }

                // Paint text with scroll offset
                if is_placeholder {
                    let style = window.text_style();
                    let run = TextRun {
                        len: placeholder_text.len(),
                        font: style.font(),
                        color: colors::text_muted().into(),
                        background_color: None,
                        underline: None,
                        strikethrough: None,
                    };
                    let font_size = style.font_size.to_pixels(window.rem_size());
                    let placeholder_line = window.text_system().shape_line(placeholder_text, font_size, &[run], None);
                    placeholder_line
                        .paint(Point::new(bounds.left() + px(6.), text_top), line_height, window, cx)
                        .ok();
                } else {
                    line.paint(Point::new(bounds.left() - final_scroll_offset, text_top), line_height, window, cx)
                        .ok();
                }

                // Paint cursor
                if focus_handle.is_focused(window) && selected_range.is_empty() {
                    let cursor_draw_x = cursor_x - final_scroll_offset;
                    let cursor_inset = px(3.);
                    let cursor_top = text_top + cursor_inset;
                    let cursor_height = (line_height - cursor_inset * 2.).max(px(1.));
                    window.paint_quad(gpui::fill(
                        Bounds::new(
                            Point::new(bounds.left() + cursor_draw_x, cursor_top),
                            gpui::size(px(2.), cursor_height),
                        ),
                        colors::text_primary(),
                    ));
                }
            });
        }
    }
}

impl Render for TextInput {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().clone();

        div()
            .flex_1()
            .h(px(32.0))
            .items_center()
            .px(spacing::sm())
            .bg(colors::bg_elevated())
            .rounded(radius::sm())
            .border_1()
            .border_color(colors::border())
            .cursor_text()
            .key_context("TextInput")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::left))
            .on_action(cx.listener(Self::right))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::home))
            .on_action(cx.listener(Self::end))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::copy))
            .on_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::submit))
            .on_action(cx.listener(Self::escape))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::handle_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::handle_mouse_up))
            .on_mouse_move(cx.listener(Self::handle_mouse_move))
            .hover(|style| style.border_color(colors::accent()))
            .child(TextInputElement { input: entity })
    }
}

/// The chat input area for composing messages.
pub struct ChatInput {
    input: Entity<TextInput>,
    state: Entity<AppState>,
    app: Entity<ChatApp>,
    should_focus_on_render: bool,
    active_draft_scope: DraftScope,
    drafts_by_conversation: FxHashMap<u64, String>,
    draft_without_selection: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DraftScope {
    Conversation(u64),
    Unscoped,
}

impl ChatInput {
    pub fn new(cx: &mut Context<Self>, state: Entity<AppState>, app: Entity<ChatApp>) -> Self {
        let entity = cx.entity().clone();
        let input = cx.new(|cx| {
            TextInput::new(cx)
                .with_placeholder("Type a message...")
                .on_change({
                    let entity = entity.clone();
                    move |content, _window, cx| {
                        entity.update(cx, |this, _| {
                            this.record_current_draft(content.to_string());
                        });
                    }
                })
                .on_submit(move |content, window, cx| {
                    entity.update(cx, |this, cx| {
                        this.do_submit(content.to_string(), window, cx);
                    });
                })
        });

        Self {
            input,
            state,
            app,
            should_focus_on_render: true,
            active_draft_scope: DraftScope::Unscoped,
            drafts_by_conversation: FxHashMap::default(),
            draft_without_selection: String::new(),
        }
    }

    fn selected_scope(&self, cx: &App) -> DraftScope {
        self.state
            .read(cx)
            .selected_id()
            .map(DraftScope::Conversation)
            .unwrap_or(DraftScope::Unscoped)
    }

    fn record_draft_for_scope(&mut self, scope: DraftScope, content: String) {
        match scope {
            DraftScope::Conversation(id) => {
                if content.is_empty() {
                    self.drafts_by_conversation.remove(&id);
                } else {
                    self.drafts_by_conversation.insert(id, content);
                }
            }
            DraftScope::Unscoped => {
                self.draft_without_selection = content;
            }
        }
    }

    fn draft_for_scope(&self, scope: DraftScope) -> String {
        match scope {
            DraftScope::Conversation(id) => self.drafts_by_conversation.get(&id).cloned().unwrap_or_default(),
            DraftScope::Unscoped => self.draft_without_selection.clone(),
        }
    }

    fn record_current_draft(&mut self, content: String) {
        self.record_draft_for_scope(self.active_draft_scope, content);
    }

    fn sync_input_with_selected_scope(&mut self, cx: &mut Context<Self>) {
        let next_scope = self.selected_scope(cx);
        if next_scope == self.active_draft_scope {
            return;
        }

        let current_content = self.input.read(cx).content().to_string();
        self.record_draft_for_scope(self.active_draft_scope, current_content);

        let next_content = self.draft_for_scope(next_scope);
        self.input.update(cx, |input, cx| {
            input.set_content(next_content, cx);
        });

        self.active_draft_scope = next_scope;
    }

    fn do_submit(&mut self, content: String, _: &mut Window, cx: &mut Context<Self>) {
        if self.state.read(cx).is_generating() {
            return;
        }
        if content.trim().is_empty() {
            return;
        }

        self.record_current_draft(String::new());

        let input = self.input.clone();
        cx.defer(move |cx| {
            input.update(cx, |input, _| input.clear());
        });

        let app = self.app.clone();
        app.update(cx, |app, cx| {
            app.send_message(content, cx);
        });

        cx.notify();
    }
}

impl Render for ChatInput {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.sync_input_with_selected_scope(cx);

        let is_generating = self.state.read(cx).is_generating();

        let input_disabled = self.input.read(cx).disabled();
        if input_disabled != is_generating {
            self.input.update(cx, |input, cx| {
                input.set_disabled(is_generating, cx);
            });
        }

        if self.should_focus_on_render && !is_generating {
            let focus_handle = self.input.read(cx).focus_handle();
            window.focus(&focus_handle);
            self.should_focus_on_render = false;
        }

        let input = self.input.clone();
        let app = self.app.clone();

        div()
            .flex_none()
            .flex()
            .w_full()
            .p(spacing::sm())
            .bg(colors::bg_surface())
            .border_t_1()
            .border_color(colors::border())
            .child(
                div()
                    .flex()
                    .w_full()
                    .gap(spacing::sm())
                    .items_center()
                    .child(self.input.clone())
                    .child(if is_generating {
                        Button::new("Stop")
                            .id("stop-btn")
                            .variant(ButtonVariant::Danger)
                            .on_click({
                                let app = app.clone();
                                move |_, cx| {
                                    app.update(cx, |app, cx| {
                                        app.stop_generation(cx);
                                    });
                                }
                            })
                            .into_any_element()
                    } else {
                        // Send button
                        IconButton::new("→")
                            .id("send-btn")
                            .on_click({
                                move |window, cx| {
                                    input.update(cx, |this, cx| {
                                        this.do_submit(window, cx);
                                    });
                                }
                            })
                            .into_any_element()
                    }),
            )
    }
}

pub fn register_input_bindings(cx: &mut App) {
    cx.bind_keys([
        gpui::KeyBinding::new("backspace", Backspace, Some("TextInput")),
        gpui::KeyBinding::new("delete", Delete, Some("TextInput")),
        gpui::KeyBinding::new("left", Left, Some("TextInput")),
        gpui::KeyBinding::new("right", Right, Some("TextInput")),
        gpui::KeyBinding::new("cmd-a", SelectAll, Some("TextInput")),
        gpui::KeyBinding::new("cmd-v", Paste, Some("TextInput")),
        gpui::KeyBinding::new("cmd-c", Copy, Some("TextInput")),
        gpui::KeyBinding::new("cmd-x", Cut, Some("TextInput")),
        gpui::KeyBinding::new("home", Home, Some("TextInput")),
        gpui::KeyBinding::new("end", End, Some("TextInput")),
        gpui::KeyBinding::new("enter", Submit, Some("TextInput")),
        gpui::KeyBinding::new("escape", Escape, Some("TextInput")),
    ]);
}
