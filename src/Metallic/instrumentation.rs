use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

/// Handle to a shared latency collector used to instrument fine-grained timing inside
/// the Metal execution context. The collector is populated by `Context` while the
/// inference loops execute and later inspected by higher-level orchestration code.
pub type LatencyCollectorHandle = Rc<RefCell<StepLatencyCollector>>;

/// Enumeration of the latency events that can be emitted from the low-level kernels.
#[derive(Clone, Copy, Debug)]
pub enum LatencyEvent {
    ForwardStep,
    Block { index: usize },
}

/// Per-token latency collector that stores the most recent measurements for the
/// surrounding forward step as well as each transformer block.
#[derive(Debug)]
pub struct StepLatencyCollector {
    forward_step: Option<Duration>,
    block_durations: Vec<Option<Duration>>,
}

impl StepLatencyCollector {
    /// Create a collector that can store measurements for `block_count` transformer blocks.
    pub fn new(block_count: usize) -> Self {
        Self {
            forward_step: None,
            block_durations: vec![None; block_count],
        }
    }

    /// Update the collector with a new latency measurement for the given event.
    pub fn record(&mut self, event: LatencyEvent, duration: Duration) {
        match event {
            LatencyEvent::ForwardStep => {
                self.forward_step = Some(duration);
            }
            LatencyEvent::Block { index } => {
                if let Some(slot) = self.block_durations.get_mut(index) {
                    *slot = Some(duration);
                }
            }
        }
    }

    /// Read the most recent measurements without consuming the collector.
    pub fn snapshot(&self) -> StepLatencySnapshot {
        StepLatencySnapshot {
            forward_step: self.forward_step.unwrap_or_default(),
            blocks: self.block_durations.iter().map(|entry| entry.unwrap_or_default()).collect(),
        }
    }
}

/// Copy of the most recent timings recorded for a single iteration.
#[derive(Clone, Debug, Default)]
pub struct StepLatencySnapshot {
    pub forward_step: Duration,
    pub blocks: Vec<Duration>,
}

impl StepLatencySnapshot {
    /// Convenience to create an empty snapshot with space for `block_count` blocks.
    pub fn empty(block_count: usize) -> Self {
        Self {
            forward_step: Duration::default(),
            blocks: vec![Duration::default(); block_count],
        }
    }
}

/// Helper to create a new collector handle for the desired number of transformer blocks.
pub fn new_collector(block_count: usize) -> LatencyCollectorHandle {
    Rc::new(RefCell::new(StepLatencyCollector::new(block_count)))
}
