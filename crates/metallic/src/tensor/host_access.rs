use std::{
    ffi::c_void, ops::Deref, sync::{Arc, Mutex, OnceLock, Weak}
};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLBuffer;
use rustc_hash::FxHashMap;

pub type RetainedBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

// CPU fill threshold in MB
pub const DEFAULT_CPU_FILL_THRESHOLD_MB: usize = 1;

#[derive(Clone)]
pub struct ThreadSafeBuffer {
    inner: RetainedBuffer,
}

impl ThreadSafeBuffer {
    pub fn new(inner: RetainedBuffer) -> Self {
        Self { inner }
    }

    pub fn clone_inner(&self) -> RetainedBuffer {
        self.inner.clone()
    }
}

unsafe impl Send for ThreadSafeBuffer {}
unsafe impl Sync for ThreadSafeBuffer {}

impl Deref for ThreadSafeBuffer {
    type Target = RetainedBuffer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Clone)]
pub struct HostAccessState {
    pub staging: Option<ThreadSafeBuffer>,
    pub staging_valid: bool,
    pub host_dirty: bool,
    pub base_offset: usize,
    pub region_len: usize,
}

impl HostAccessState {
    pub fn new(base_offset: usize, region_len: usize) -> Self {
        Self {
            staging: None,
            staging_valid: false,
            host_dirty: false,
            base_offset,
            region_len,
        }
    }

    pub fn region_end(&self) -> usize {
        self.base_offset.saturating_add(self.region_len)
    }
}

pub type HostAccessRegistry = FxHashMap<usize, Vec<Weak<Mutex<HostAccessState>>>>;

pub fn host_access_registry() -> &'static Mutex<HostAccessRegistry> {
    static REGISTRY: OnceLock<Mutex<HostAccessRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(FxHashMap::default()))
}

pub fn buffer_registry_key(buffer: &RetainedBuffer) -> usize {
    Retained::as_ptr(buffer).cast::<c_void>() as usize
}

pub fn shared_host_access_state(buffer: &RetainedBuffer, offset: usize, len_bytes: usize) -> Arc<Mutex<HostAccessState>> {
    let mut registry = host_access_registry().lock().expect("host access registry mutex poisoned");
    let entry = registry.entry(buffer_registry_key(buffer)).or_default();

    let req_start = offset;
    let req_end = offset.saturating_add(len_bytes);
    let mut idx = 0;
    while idx < entry.len() {
        if let Some(state_arc) = entry[idx].upgrade() {
            let mut selected = false;
            {
                let mut state = state_arc.lock().expect("host access state mutex poisoned");
                let state_start = state.base_offset;
                let state_end = state.region_end();
                if req_start >= state_start && req_end <= state_end {
                    selected = true;
                } else if req_end > state_start && req_start < state_end {
                    let new_start = req_start.min(state_start);
                    let new_end = req_end.max(state_end);
                    if new_start != state_start || new_end != state_end {
                        state.base_offset = new_start;
                        state.region_len = new_end - new_start;
                        state.staging = None;
                        state.staging_valid = false;
                        state.host_dirty = false;
                    }
                    selected = true;
                }
            }

            if selected {
                return state_arc;
            }

            idx += 1;
        } else {
            entry.remove(idx);
        }
    }

    let state_arc = Arc::new(Mutex::new(HostAccessState::new(offset, len_bytes)));
    entry.push(Arc::downgrade(&state_arc));
    state_arc
}
