use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

/// Handle to a shared latency collector used to instrument fine-grained timing inside
/// the Metal execution context. The collector is populated by `Context` while the
/// inference loops execute and later inspected by higher-level orchestration code.
pub type LatencyCollectorHandle = Rc<RefCell<StepLatencyCollector>>;

/// Enumeration of the latency events that can be emitted from the low-level kernels.
#[derive(Clone, Debug)]
pub enum LatencyEvent<'a> {
    ForwardStep,
    Block { index: usize },
    BlockPhase { index: usize, label: Cow<'a, str> },
}

impl<'a> LatencyEvent<'a> {
    pub fn block_phase<T>(index: usize, label: T) -> Self
    where
        T: Into<Cow<'a, str>>,
    {
        LatencyEvent::BlockPhase {
            index,
            label: label.into(),
        }
    }
}

/// Per-token latency collector that stores the most recent measurements for the
/// surrounding forward step as well as each transformer block.
#[derive(Debug)]
pub struct StepLatencyCollector {
    forward_step: Option<Duration>,
    block_durations: Vec<BlockLatencyEntry>,
}

impl StepLatencyCollector {
    /// Create a collector that can store measurements for `block_count` transformer blocks.
    pub fn new(block_count: usize) -> Self {
        Self {
            forward_step: None,
            block_durations: vec![BlockLatencyEntry::default(); block_count],
        }
    }

    /// Update the collector with a new latency measurement for the given event.
    pub fn record(&mut self, event: LatencyEvent<'_>, duration: Duration) {
        match event {
            LatencyEvent::ForwardStep => {
                self.forward_step = Some(duration);
            }
            LatencyEvent::Block { index } => {
                if let Some(slot) = self.block_durations.get_mut(index) {
                    slot.total = Some(duration);
                }
            }
            LatencyEvent::BlockPhase { index, label } => {
                if let Some(slot) = self.block_durations.get_mut(index) {
                    let label_owned = label.into_owned();
                    if let Some(existing) = slot.phases.iter_mut().find(|phase| phase.label == label_owned) {
                        existing.duration = duration;
                    } else {
                        slot.phases.push(BlockPhaseEntry {
                            label: label_owned,
                            duration,
                        });
                    }
                }
            }
        }
    }

    /// Read the most recent measurements without consuming the collector.
    pub fn snapshot(&self) -> StepLatencySnapshot {
        StepLatencySnapshot {
            forward_step: self.forward_step.unwrap_or_default(),
            blocks: self.block_durations.iter().map(BlockLatencyEntry::snapshot).collect(),
        }
    }
}

/// Copy of the most recent timings recorded for a single iteration.
#[derive(Clone, Debug, Default)]
pub struct StepLatencySnapshot {
    pub forward_step: Duration,
    pub blocks: Vec<BlockLatencySnapshot>,
}

impl StepLatencySnapshot {
    /// Convenience to create an empty snapshot with space for `block_count` blocks.
    pub fn empty(block_count: usize) -> Self {
        Self {
            forward_step: Duration::default(),
            blocks: vec![BlockLatencySnapshot::default(); block_count],
        }
    }
}

#[derive(Clone, Debug, Default)]
struct BlockLatencyEntry {
    total: Option<Duration>,
    phases: Vec<BlockPhaseEntry>,
}

impl BlockLatencyEntry {
    fn snapshot(&self) -> BlockLatencySnapshot {
        BlockLatencySnapshot {
            total: self.total.unwrap_or_default(),
            phases: self
                .phases
                .iter()
                .map(|phase| BlockPhaseSnapshot {
                    label: phase.label.clone(),
                    duration: phase.duration,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
struct BlockPhaseEntry {
    label: String,
    duration: Duration,
}

#[derive(Clone, Debug, Default)]
pub struct BlockLatencySnapshot {
    pub total: Duration,
    pub phases: Vec<BlockPhaseSnapshot>,
}

#[derive(Clone, Debug)]
pub struct BlockPhaseSnapshot {
    pub label: String,
    pub duration: Duration,
}

/// Helper to create a new collector handle for the desired number of transformer blocks.
pub fn new_collector(block_count: usize) -> LatencyCollectorHandle {
    Rc::new(RefCell::new(StepLatencyCollector::new(block_count)))
}
