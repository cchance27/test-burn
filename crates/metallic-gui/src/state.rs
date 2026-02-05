//! Application state and business logic.
//!
//! This module contains the core state management separated from UI rendering.

use std::{
    collections::HashMap, sync::{
        Arc, atomic::{AtomicBool, Ordering}
    }
};

use gpui::{AsyncApp, Context, WeakEntity};

use crate::{
    db::{Database, ModelSettings}, inference::{LoadedModelState, ModelInfo, ModelLoadStatus, load_model, run_inference_streaming, scan_models_directory}, types::{ChatMessage, Conversation, MessageRole}
};

/// Core application state, managing conversations and messages.
///
/// This is separate from UI components to allow for backend integration.
pub struct AppState {
    conversations: Vec<Conversation>,
    selected_id: Option<u64>,
    next_message_id: u64,
    next_conversation_id: u64,
    /// Available models discovered from models/ directory.
    available_models: Vec<ModelInfo>,
    /// Index of selected model in available_models.
    selected_model_idx: Option<usize>,
    /// Current model loading status.
    model_load_status: ModelLoadStatus,
    /// The loaded model state (Foundry, CompiledModel, Workflow).
    loaded_model: Option<LoadedModelState>,
    /// Whether a generation task is currently running.
    inference_in_flight: bool,
    /// Cancellation flag for the active generation task, when present.
    generation_cancel: Option<Arc<AtomicBool>>,
    /// SQLite database for persistence.
    db: Arc<Database>,
    /// Global app settings.
    settings: HashMap<String, String>,
    /// Approximate active-session context usage (prompt + generated) for the selected conversation.
    selected_model_context_used: usize,
    /// Whether we have a valid context usage sample for the current conversation/model session.
    selected_model_context_has_value: bool,
    /// Whether context usage is currently pending (generation in flight).
    selected_model_context_pending: bool,
    /// Cached model context readout shown in the header.
    selected_model_context_readout: String,
}

impl AppState {
    pub fn new(cx: &mut Context<Self>) -> Self {
        // Initialize database in application data directory or current dir for now
        let db_path = "metallic.db";
        let db = Arc::new(Database::new(db_path).expect("Failed to initialize database"));

        // Scan for models at startup
        let models = scan_models_directory();

        // Load settings from database
        let mut settings = HashMap::new();
        if let Ok(Some(last_model)) = db.get_app_setting("last_model")
            && !last_model.is_empty()
        {
            settings.insert("last_model".to_string(), last_model);
        }
        if let Ok(Some(remember_model)) = db.get_app_setting("remember_model") {
            settings.insert("remember_model".to_string(), remember_model);
        } else {
            settings.insert("remember_model".to_string(), "true".to_string());
        }
        if let Ok(Some(auto_load)) = db.get_app_setting("auto_load_model") {
            settings.insert("auto_load_model".to_string(), auto_load);
        } else {
            settings.insert("auto_load_model".to_string(), "false".to_string());
        }
        if let Ok(Some(system_prompt)) = db.get_app_setting("system_prompt") {
            settings.insert("system_prompt".to_string(), system_prompt);
        } else {
            settings.insert(
                "system_prompt".to_string(),
                "You are a helpful assistant, named alloy, you should always give short concise answers.".to_string(),
            );
        }

        let mut selected_idx = None;
        if settings.get("remember_model") == Some(&"true".to_string())
            && let Some(last_model_name) = settings.get("last_model")
        {
            selected_idx = models.iter().position(|m| &m.display_name == last_model_name);
        }

        // Default to first if none persisted or found
        if selected_idx.is_none() && !models.is_empty() {
            selected_idx = Some(0);
        }

        let state = Self {
            conversations: Vec::new(),
            selected_id: None,
            next_message_id: 1,
            next_conversation_id: 1,
            available_models: models,
            selected_model_idx: selected_idx,
            model_load_status: ModelLoadStatus::Idle,
            loaded_model: None,
            inference_in_flight: false,
            generation_cancel: None,
            db,
            settings,
            selected_model_context_used: 0,
            selected_model_context_has_value: false,
            selected_model_context_pending: false,
            selected_model_context_readout: "----/-----".to_string(),
        };

        // Auto-load if enabled
        if state.settings.get("auto_load_model") == Some(&"true".to_string()) && state.selected_model_idx.is_some() {
            let this = cx.weak_entity();
            cx.defer(move |cx| {
                if let Some(this) = this.upgrade() {
                    this.update(cx, |this, cx| {
                        this.load_selected_model(cx);
                    });
                }
            });
        }

        state
    }

    // --- Model Management ---

    pub fn available_models(&self) -> &[ModelInfo] {
        &self.available_models
    }

    pub fn selected_model_idx(&self) -> Option<usize> {
        self.selected_model_idx
    }

    pub fn set_selected_model_idx(&mut self, idx: Option<usize>, cx: &mut Context<Self>) {
        if self.selected_model_idx != idx {
            self.selected_model_idx = idx;

            // Persist selection only when remember-model is enabled.
            if let Some(idx) = idx
                && let Some(model) = self.available_models.get(idx)
                && self.settings.get("remember_model") == Some(&"true".to_string())
            {
                let _ = self.db.set_app_setting("last_model", &model.display_name);
                self.settings.insert("last_model".to_string(), model.display_name.clone());
            }

            // Unload current model when selection changes
            self.loaded_model = None;
            self.model_load_status = ModelLoadStatus::Idle;
            self.inference_in_flight = false;
            self.generation_cancel = None;
            self.reset_selected_model_context_usage();
            cx.notify();
        }
    }

    pub fn is_generating(&self) -> bool {
        self.inference_in_flight
    }

    pub fn stop_generation(&mut self, cx: &mut Context<Self>) -> bool {
        let Some(cancel) = self.generation_cancel.as_ref() else {
            return false;
        };
        cancel.store(true, Ordering::Release);
        tracing::info!("Generation stop requested");
        cx.notify();
        true
    }

    pub fn selected_model(&self) -> Option<&ModelInfo> {
        self.selected_model_idx.and_then(|idx| self.available_models.get(idx))
    }

    pub fn selected_model_name(&self) -> Option<&str> {
        self.selected_model().map(|model| model.display_name.as_str())
    }

    pub fn select_model(&mut self, model_name: &str, cx: &mut Context<Self>) -> bool {
        let selected_idx = self.available_models.iter().position(|model| model.display_name == model_name);
        if selected_idx.is_none() {
            return false;
        }
        self.set_selected_model_idx(selected_idx, cx);
        true
    }

    pub fn clear_selected_model(&mut self, cx: &mut Context<Self>) {
        self.set_selected_model_idx(None, cx);
    }

    pub fn cycle_selected_model(&mut self, cx: &mut Context<Self>) {
        let model_count = self.available_models.len();
        if model_count == 0 {
            return;
        }

        let next_idx = match self.selected_model_idx {
            Some(idx) => Some((idx + 1) % model_count),
            None => Some(0),
        };
        self.set_selected_model_idx(next_idx, cx);
    }

    pub fn model_load_status(&self) -> &ModelLoadStatus {
        &self.model_load_status
    }

    pub fn loaded_model(&self) -> Option<&LoadedModelState> {
        self.loaded_model.as_ref()
    }

    fn refresh_selected_model_context_readout(&mut self) {
        let Some(loaded_model) = self.loaded_model.as_ref() else {
            self.selected_model_context_readout = "----/-----".to_string();
            return;
        };

        let max_context = loaded_model.model.architecture().max_seq_len();
        if max_context == 0 {
            self.selected_model_context_readout = "----/-----".to_string();
            return;
        }
        if !self.selected_model_context_has_value {
            self.selected_model_context_readout = if self.selected_model_context_pending {
                "----/-----*".to_string()
            } else {
                "----/-----".to_string()
            };
            return;
        }

        let used = self.selected_model_context_used.min(max_context);
        self.selected_model_context_readout = if self.selected_model_context_pending {
            format!("{:>4}/{:>5}*", used, max_context)
        } else {
            format!("{:>4}/{:>5}", used, max_context)
        };
    }

    fn reset_selected_model_context_usage(&mut self) {
        self.selected_model_context_used = 0;
        self.selected_model_context_has_value = false;
        self.selected_model_context_pending = false;
        self.refresh_selected_model_context_readout();
    }

    fn mark_selected_model_context_pending(&mut self, pending: bool) {
        self.selected_model_context_pending = pending;
        self.refresh_selected_model_context_readout();
    }

    fn set_selected_model_context_usage(&mut self, used_tokens: usize) {
        self.selected_model_context_used = used_tokens;
        self.selected_model_context_has_value = true;
        self.selected_model_context_pending = false;
        self.refresh_selected_model_context_readout();
    }

    pub fn selected_model_context_readout(&self) -> String {
        self.selected_model_context_readout.clone()
    }

    /// Load the currently selected model.
    pub fn load_selected_model(&mut self, cx: &mut Context<Self>) -> bool {
        let Some(idx) = self.selected_model_idx else {
            return false;
        };
        let Some(model_info) = self.available_models.get(idx) else {
            return false;
        };

        if self.model_load_status == ModelLoadStatus::Loading {
            return false;
        }

        let model_path = model_info.path.clone();
        let system_prompt = self.get_setting("system_prompt").cloned();

        // Update status to loading
        self.loaded_model = None;
        self.model_load_status = ModelLoadStatus::Loading;
        self.inference_in_flight = false;
        self.generation_cancel = None;
        self.reset_selected_model_context_usage();
        cx.notify();

        tracing::info!("Starting background model load for {:?}", model_path);

        cx.spawn(|this: WeakEntity<AppState>, cx_ref: &mut AsyncApp| {
            let mut cx_owned = cx_ref.clone();
            async move {
                tracing::debug!("Background model load task started");
                let load_task = smol::unblock(move || {
                    std::panic::catch_unwind(|| load_model(&model_path, system_prompt)).unwrap_or_else(|panic_payload| {
                        let panic_msg = panic_payload
                            .downcast_ref::<&str>()
                            .map(|msg| (*msg).to_string())
                            .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "unknown panic payload".to_string());
                        Err(format!("Model load panicked: {panic_msg}"))
                    })
                });
                // smol::unblock work is not cancellable once scheduled, so avoid a fake timeout
                // state transition that can leave the heavy load still running in the background.
                let result = load_task.await;
                tracing::debug!("Background model load task finished");

                if let Err(err) = this.update(&mut cx_owned, |this: &mut AppState, cx: &mut Context<AppState>| {
                    match result {
                        Ok(loaded_state) => {
                            tracing::info!("Model loaded successfully");
                            this.loaded_model = Some(loaded_state);
                            this.model_load_status = ModelLoadStatus::Loaded;
                            this.reset_selected_model_context_usage();
                        }
                        Err(err) => {
                            tracing::error!("Model load failed: {}", err);
                            this.model_load_status = ModelLoadStatus::Error(err);
                            this.reset_selected_model_context_usage();
                        }
                    }
                    cx.notify();
                }) {
                    tracing::error!("Failed to publish model-load status update: {}", err);
                }
            }
        })
        .detach();

        true
    }

    // --- Conversation Management ---

    pub fn conversations(&self) -> &[Conversation] {
        &self.conversations
    }

    pub fn selected_id(&self) -> Option<u64> {
        self.selected_id
    }

    fn resolved_selected_id(&self) -> Option<u64> {
        self.selected_id.or_else(|| self.conversations.first().map(|c| c.id))
    }

    fn reset_session_for_conversation_boundary(&mut self) {
        if let Some(model_state) = &mut self.loaded_model {
            model_state.model.rewind_session();
            if let Ok(mut runner) = model_state.runner.lock() {
                runner.reset();
            }
            tracing::info!("Rewound model session and reset workflow state for conversation boundary");
        }
        self.reset_selected_model_context_usage();
    }

    pub fn select_conversation(&mut self, id: u64, cx: &mut Context<Self>) {
        if self.selected_id != Some(id) {
            self.selected_id = Some(id);
            // Clear KV cache and reset workflow state when switching conversations to prevent cross-talk.
            self.reset_session_for_conversation_boundary();
            cx.notify();
        }
    }

    pub fn selected_conversation(&self) -> Option<&Conversation> {
        let selected_id = self.resolved_selected_id()?;
        self.conversation(selected_id)
    }

    pub fn create_new_conversation(&mut self, cx: &mut Context<Self>) -> u64 {
        self.create_conversation(cx)
    }

    pub fn rename_conversation(&mut self, id: u64, new_title: String, cx: &mut Context<Self>) {
        let trimmed_title = new_title.trim();
        if trimmed_title.is_empty() {
            return;
        }
        if let Some(conversation) = self.conversation_mut(id) {
            conversation.title = trimmed_title.to_string();
            conversation.updated_at = chrono::Local::now();
            cx.notify();
        }
    }

    pub fn create_conversation(&mut self, cx: &mut Context<Self>) -> u64 {
        let id = self.next_conversation_id;
        self.next_conversation_id += 1;

        let now = chrono::Local::now();
        let conversation = Conversation {
            id,
            title: format!("New Conversation #{}", id),
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
        };

        self.conversations.insert(0, conversation);
        let selection_changed = self.selected_id != Some(id);
        self.selected_id = Some(id);
        if selection_changed {
            // New conversation must start from a clean session.
            self.reset_session_for_conversation_boundary();
        }
        cx.notify();
        id
    }

    pub fn delete_conversation(&mut self, id: u64, cx: &mut Context<Self>) {
        self.conversations.retain(|c| c.id != id);
        let previous_selected = self.selected_id;
        if self.selected_id == Some(id) {
            self.selected_id = self.conversations.first().map(|c| c.id);
        }
        if self.selected_id != previous_selected {
            self.reset_session_for_conversation_boundary();
        }
        cx.notify();
    }

    // --- Message Management ---

    pub fn conversation(&self, id: u64) -> Option<&Conversation> {
        self.conversations.iter().find(|c| c.id == id)
    }

    pub fn conversation_mut(&mut self, id: u64) -> Option<&mut Conversation> {
        self.conversations.iter_mut().find(|c| c.id == id)
    }

    pub fn send_message(&mut self, content: String, cx: &mut Context<Self>) {
        if content.trim().is_empty() {
            return;
        }

        let selected_id = self.ensure_selected_conversation(cx).unwrap_or_default();
        let system_prompt = self.effective_system_prompt();
        self.ensure_conversation_system_prompt(selected_id, &system_prompt);

        // Add user message
        self.append_message(selected_id, MessageRole::User, content);
        cx.notify();

        // Trigger AI response (now async/background)
        self.add_ai_response(cx);
    }

    fn append_message(&mut self, conversation_id: u64, role: MessageRole, content: String) {
        let id = self.next_message_id;
        self.next_message_id += 1;
        if let Some(convo) = self.conversation_mut(conversation_id) {
            convo.messages.push(ChatMessage::new(id, role, content));
        }
    }

    fn effective_system_prompt(&self) -> String {
        self.get_setting("system_prompt")
            .cloned()
            .unwrap_or_else(|| "You are a helpful assistant.".to_string())
    }

    fn ensure_conversation_system_prompt(&mut self, conversation_id: u64, system_prompt: &str) {
        enum Action {
            None,
            Update,
            Insert,
        }

        let action = self
            .conversation(conversation_id)
            .map(|convo| match convo.messages.first() {
                Some(msg) if matches!(msg.role, MessageRole::System) => {
                    if msg.content == system_prompt {
                        Action::None
                    } else {
                        Action::Update
                    }
                }
                _ => Action::Insert,
            })
            .unwrap_or(Action::None);

        match action {
            Action::None => {}
            Action::Update => {
                if let Some(convo) = self.conversation_mut(conversation_id)
                    && let Some(first_msg) = convo.messages.first_mut()
                {
                    first_msg.content.clear();
                    first_msg.content.push_str(system_prompt);
                    first_msg.timestamp = chrono::Local::now();
                }
            }
            Action::Insert => {
                let id = self.next_message_id;
                self.next_message_id += 1;
                if let Some(convo) = self.conversation_mut(conversation_id) {
                    convo
                        .messages
                        .insert(0, ChatMessage::new(id, MessageRole::System, system_prompt.to_string()));
                }
            }
        }
    }

    fn add_ai_response(&mut self, cx: &mut Context<AppState>) {
        let Some(selected_id) = self.resolved_selected_id() else {
            return;
        };
        if self.inference_in_flight {
            tracing::warn!("Inference already in flight, rejecting overlapping generation request");
            self.append_message(
                selected_id,
                MessageRole::Assistant,
                "⚠️ A response is already generating. Please wait for it to finish.".to_string(),
            );
            self.mark_selected_model_context_pending(false);
            cx.notify();
            return;
        }

        // Keep a real system message at the head of the conversation so UI and inference payload match.
        let effective_system_prompt = self.effective_system_prompt();
        self.ensure_conversation_system_prompt(selected_id, &effective_system_prompt);

        // Get the full conversation messages after system-message synchronization.
        let messages: Vec<ChatMessage> = self.conversation(selected_id).map(|c| c.messages.clone()).unwrap_or_default();

        if messages.is_empty() {
            return;
        }

        // Check if model is loaded and clone it (it's wrapped in Arc<Mutex>)
        let Some(loaded_model) = self.loaded_model.clone() else {
            tracing::warn!("No model loaded, cannot generate response");
            self.append_message(
                selected_id,
                MessageRole::Assistant,
                "⚠️ Please load a model first to generate responses.".to_string(),
            );
            self.mark_selected_model_context_pending(false);
            cx.notify();
            return;
        };

        let assistant_message_id = self.next_message_id;
        self.next_message_id += 1;

        // Add an empty assistant message to be filled by streaming
        self.append_message_with_id(selected_id, MessageRole::Assistant, String::new(), assistant_message_id);
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.inference_in_flight = true;
        self.generation_cancel = Some(cancel_flag.clone());
        self.mark_selected_model_context_pending(true);
        cx.notify();

        tracing::info!("Starting background streaming inference for conversation {}", selected_id);

        // Prepare all inputs for the workflow.
        // Runtime-owned keys must never be overridden by persisted settings.
        let mut inputs = rustc_hash::FxHashMap::default();
        // Merge model-specific settings (includes global system_prompt fallback)
        let model_name = self.selected_model_name().unwrap_or("default");
        let settings = self.get_model_settings(model_name);
        for (key, val) in settings.values {
            if matches!(
                key.as_str(),
                "messages" | "system_prompt" | "run_generation" | "sample_cpu_fallback" | "conversation_id"
            ) {
                tracing::debug!("Skipping runtime-managed workflow input from settings: {}", key);
                continue;
            }
            inputs.insert(key, metallic_foundry::workflow::Value::from_json(val));
        }
        inputs.insert("run_generation".to_string(), metallic_foundry::workflow::Value::U32(1));
        inputs.insert("messages".to_string(), crate::inference::messages_to_value(&messages));
        inputs.insert(
            "conversation_id".to_string(),
            metallic_foundry::workflow::Value::Text(std::sync::Arc::<str>::from(format!("conversation-{selected_id}"))),
        );
        inputs.insert(
            "system_prompt".to_string(),
            metallic_foundry::workflow::Value::Text(std::sync::Arc::from(effective_system_prompt.as_str())),
        );
        let force_cpu_sampler = std::env::var("METALLIC_GUI_USE_CPU_SAMPLER")
            .ok()
            .map(|v| {
                let trimmed = v.trim();
                !matches!(trimmed.to_ascii_lowercase().as_str(), "" | "0" | "false" | "no" | "off")
            })
            .unwrap_or(false);
        if force_cpu_sampler {
            inputs.insert("sample_cpu_fallback".to_string(), metallic_foundry::workflow::Value::Bool(true));
        }
        let input_message_roles = inputs
            .get("messages")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|entry| entry.as_map())
                    .filter_map(|map| map.get("role"))
                    .filter_map(|role| role.as_text())
                    .collect::<Vec<_>>()
                    .join(">")
            })
            .unwrap_or_else(|| "<missing-or-non-array>".to_string());
        let param_keys = [
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "repeat_penalty",
            "repeat_last_n",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "seed_random",
            "max_tokens",
            "eos_token",
            "sample_cpu_fallback",
        ];
        let mut param_snapshot = Vec::new();
        for key in param_keys {
            if let Some(v) = inputs.get(key) {
                param_snapshot.push(format!("{key}={}", v.to_json()));
            }
        }
        let message_roles = messages
            .iter()
            .map(|m| crate::inference::role_to_string(m.role))
            .collect::<Vec<_>>()
            .join(">");
        tracing::debug!(
            "Using system prompt for inference (len={} chars); model='{}'; request_messages={}; roles={}; params: {}",
            effective_system_prompt.chars().count(),
            model_name,
            messages.len(),
            format!("{message_roles}; input_roles={input_message_roles}"),
            param_snapshot.join(", ")
        );

        cx.spawn(move |this: WeakEntity<AppState>, cx_ref: &mut AsyncApp| {
            let mut cx_owned = cx_ref.clone();
            let loaded_model = loaded_model.clone();
            let inputs = inputs; // Capture prepared inputs
            let cancel_flag = cancel_flag.clone();
            let model_filename = loaded_model
                .path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| loaded_model.model.name().to_string());
            let mut params = HashMap::new();
            for (k, v) in &inputs {
                if k != "messages" {
                    params.insert(k.clone(), v.to_json());
                }
            }

            async move {
                let (tx, rx) = smol::channel::bounded(2048);
                let (done_tx, done_rx) = smol::channel::bounded(1);
                let loaded_model_task = loaded_model.clone();
                let cancel_flag_task = cancel_flag.clone();

                // Run inference in a background thread pool and report completion status.
                smol::spawn(async move {
                    let result = smol::unblock(move || {
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            run_inference_streaming(&loaded_model_task, inputs, tx, cancel_flag_task.as_ref())
                        })) {
                            Ok(Ok(perf)) => Ok(perf),
                            Ok(Err(err)) => Err(err.to_string()),
                            Err(panic_payload) => {
                                let panic_msg = panic_payload
                                    .downcast_ref::<&str>()
                                    .map(|msg| (*msg).to_string())
                                    .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                                    .unwrap_or_else(|| "unknown panic payload".to_string());
                                Err(format!("Inference panicked: {panic_msg}"))
                            }
                        }
                    })
                    .await;
                    let _ = done_tx.send(result).await;
                })
                .detach();

                // Stream tokens back to UI
                let generation_started_at = std::time::Instant::now();
                let mut full_tokens = Vec::new();
                let mut token_stream_closed = false;
                let mut pending_tokens = 0usize;
                let mut last_flush = std::time::Instant::now();

                let flush_ui = |full_tokens: &Vec<u32>,
                                metadata: Option<crate::types::MessageMetadata>,
                                this: &WeakEntity<AppState>,
                                cx_owned: &mut AsyncApp|
                 -> Result<(), String> {
                    let current_text = if let Ok(tokenizer) = loaded_model.model.tokenizer() {
                        tokenizer.decode(full_tokens).unwrap_or_else(|_| " (decode error) ".to_string())
                    } else {
                        String::from(" (tokenizer error) ")
                    };

                    this.update(cx_owned, |this: &mut AppState, cx: &mut Context<AppState>| {
                        if let Some(convo) = this.conversation_mut(selected_id)
                            && let Some(msg) = convo.messages.iter_mut().find(|m| m.id == assistant_message_id)
                        {
                            msg.content = current_text;
                            if let Some(meta) = metadata {
                                msg.metadata = Some(meta);
                            }
                        }
                        cx.notify();
                    })
                    .map_err(|err| err.to_string())
                };

                let inference_result = loop {
                    enum StreamEvent {
                        Token(Result<u32, smol::channel::RecvError>),
                        Done(Result<Result<crate::types::InferencePerf, String>, smol::channel::RecvError>),
                    }

                    let event = if token_stream_closed {
                        StreamEvent::Done(done_rx.recv().await)
                    } else {
                        smol::future::or(async { StreamEvent::Token(rx.recv().await) }, async {
                            StreamEvent::Done(done_rx.recv().await)
                        })
                        .await
                    };

                    match event {
                        StreamEvent::Token(Ok(token_id)) => {
                            full_tokens.push(token_id);
                            pending_tokens += 1;

                            let should_flush = pending_tokens >= 8 || last_flush.elapsed() >= std::time::Duration::from_millis(10);
                            if should_flush {
                                if let Err(err) = flush_ui(&full_tokens, None, &this, &mut cx_owned) {
                                    tracing::error!("Failed to publish streaming token update: {}", err);
                                    break Err(format!("Failed to publish streaming token update: {err}"));
                                }
                                pending_tokens = 0;
                                last_flush = std::time::Instant::now();
                                // Yield after a UI flush so render work runs without throttling every token.
                                smol::future::yield_now().await;
                            }
                        }
                        StreamEvent::Token(Err(_)) => {
                            token_stream_closed = true;
                        }
                        StreamEvent::Done(Ok(result)) => {
                            if result.is_ok() {
                                while let Ok(token_id) = rx.try_recv() {
                                    full_tokens.push(token_id);
                                    pending_tokens += 1;
                                }
                            }
                            break result;
                        }
                        StreamEvent::Done(Err(_)) => {
                            if full_tokens.is_empty() {
                                break Err("Inference task ended without reporting completion".to_string());
                            }
                            tracing::warn!(
                                "Inference channels closed after {} tokens without explicit completion status",
                                full_tokens.len()
                            );
                            // Synthesize a minimal/empty perf if we don't have one but have tokens
                            break Ok(crate::types::InferencePerf {
                                tokens: full_tokens.len(),
                                wall_ms: generation_started_at.elapsed().as_millis(),
                                wall_tok_per_sec: 0.0,
                                decode_tok_per_sec: 0.0,
                                first_token_ms: 0,
                                prefill_ms: 0,
                                setup_ms: None,
                                prompt_prep_ms: None,
                                first_decode_ms: None,
                                decode_wait_ms: None,
                                prefill_tokens: 0,
                                prefill_tok_per_sec: 0.0,
                                context_tokens: None,
                            });
                        }
                    }
                };

                if pending_tokens > 0 {
                    let final_metadata = if let Ok(perf) = &inference_result {
                        Some(crate::types::MessageMetadata {
                            model_filename: model_filename.clone(),
                            params: params.clone(),
                            perf: perf.clone(),
                        })
                    } else {
                        None
                    };
                    if let Err(err) = flush_ui(&full_tokens, final_metadata, &this, &mut cx_owned) {
                        tracing::error!("Failed to publish final streaming token update: {}", err);
                    }
                } else if let Ok(perf) = &inference_result {
                    // Even if no pending tokens, we need to attach the metadata if we have a successful result
                    let final_metadata = crate::types::MessageMetadata {
                        model_filename: model_filename.clone(),
                        params: params.clone(),
                        perf: perf.clone(),
                    };
                    let _ = flush_ui(&full_tokens, Some(final_metadata), &this, &mut cx_owned);
                }

                let elapsed = generation_started_at.elapsed();
                let elapsed_secs = elapsed.as_secs_f64();
                let toks_per_sec = if elapsed_secs > 0.0 {
                    full_tokens.len() as f64 / elapsed_secs
                } else {
                    0.0
                };
                tracing::debug!(
                    "Streaming generation complete: conversation_id={}, tokens={}, elapsed_ms={}, tok_per_sec={:.2}",
                    selected_id,
                    full_tokens.len(),
                    elapsed.as_millis(),
                    toks_per_sec
                );

                let perf_for_usage = inference_result.as_ref().ok().cloned();
                if let Err(err_msg) = &inference_result {
                    tracing::error!("Streaming inference failed: {}", err_msg);
                    let fallback = format!("⚠️ Inference failed: {err_msg}");
                    let _ = this.update(&mut cx_owned, |this: &mut AppState, cx: &mut Context<AppState>| {
                        if let Some(convo) = this.conversation_mut(selected_id)
                            && let Some(msg) = convo.messages.iter_mut().find(|m| m.id == assistant_message_id)
                            && msg.content.is_empty()
                        {
                            msg.content = fallback;
                        }
                        cx.notify();
                    });
                } else if full_tokens.is_empty() {
                    tracing::warn!("Streaming inference completed with zero tokens");
                    let _ = this.update(&mut cx_owned, |this: &mut AppState, cx: &mut Context<AppState>| {
                        if let Some(convo) = this.conversation_mut(selected_id)
                            && let Some(msg) = convo.messages.iter_mut().find(|m| m.id == assistant_message_id)
                            && msg.content.is_empty()
                        {
                            msg.content = "⚠️ Model returned no tokens.".to_string();
                        }
                        cx.notify();
                    });
                }

                let canceled = cancel_flag.load(Ordering::Acquire);
                if canceled {
                    tracing::info!("Generation canceled; rewinding model session and resetting workflow state");
                    loaded_model.model.rewind_session();
                    if let Ok(mut runner) = loaded_model.runner.lock() {
                        runner.reset();
                    }
                }

                let _ = this.update(&mut cx_owned, |this: &mut AppState, cx: &mut Context<AppState>| {
                    this.inference_in_flight = false;
                    this.generation_cancel = None;
                    if canceled {
                        this.reset_selected_model_context_usage();
                    } else if let Some(perf) = perf_for_usage.as_ref()
                        && let Some(context_tokens) = perf.context_tokens
                    {
                        this.set_selected_model_context_usage(context_tokens);
                    } else {
                        this.reset_selected_model_context_usage();
                    }
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn append_message_with_id(&mut self, conversation_id: u64, role: MessageRole, content: String, id: u64) {
        if let Some(convo) = self.conversation_mut(conversation_id) {
            convo.messages.push(ChatMessage::new(id, role, content));
        }
    }

    fn ensure_selected_conversation(&mut self, cx: &mut Context<Self>) -> Option<u64> {
        if self.resolved_selected_id().is_none() {
            Some(self.create_conversation(cx))
        } else {
            self.resolved_selected_id()
        }
    }

    // --- Settings Persistence Helpers ---

    pub fn get_setting(&self, key: &str) -> Option<&String> {
        self.settings.get(key)
    }

    pub fn update_setting(&mut self, key: String, value: String) {
        let _ = self.db.set_app_setting(&key, &value);
        if key == "remember_model" {
            if value == "true" {
                if let Some(model) = self.selected_model() {
                    let _ = self.db.set_app_setting("last_model", &model.display_name);
                    self.settings.insert("last_model".to_string(), model.display_name.clone());
                }
            } else {
                self.settings.remove("last_model");
                let _ = self.db.set_app_setting("last_model", "");
            }
        }

        self.settings.insert(key, value);
        self.refresh_selected_model_context_readout();
    }

    pub fn get_model_settings(&self, model_id: &str) -> ModelSettings {
        let mut settings = self.db.get_model_settings(model_id).unwrap_or_default();

        if let Some(loaded_model) = self.loaded_model() {
            for input in &loaded_model.workflow.inputs {
                if let Some(default) = &input.default {
                    if input.name == "system_prompt" {
                        let global_val = self.get_setting("system_prompt").map(|s| serde_json::Value::String(s.clone()));

                        if let Some(g) = &global_val {
                            tracing::debug!("Using global system prompt override: {:?}", g);
                        } else {
                            tracing::debug!("Using workflow default system prompt: {:?}", default);
                        }

                        if let Some(g) = global_val {
                            // Global system prompt is authoritative when set.
                            settings.values.insert(input.name.clone(), g);
                        } else {
                            settings.values.entry(input.name.clone()).or_insert_with(|| default.clone());
                        }
                    } else {
                        settings.values.entry(input.name.clone()).or_insert_with(|| default.clone());
                    }
                }
            }
        }

        settings
    }

    pub fn update_model_settings(&self, model_id: &str, settings: &crate::db::ModelSettings) {
        let _ = self.db.set_model_settings(model_id, settings);
    }
}
