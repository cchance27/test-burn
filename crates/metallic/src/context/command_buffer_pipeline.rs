use std::{collections::VecDeque, time::Instant};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLCommandQueue;

use super::utils::GpuProfilerLabel;
use crate::{error::MetalError, operation::CommandBuffer};

const DEFAULT_MAX_INFLIGHT: usize = 3;

pub struct CommandBufferPipeline {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    inflight: VecDeque<PipelineEntry>,
    max_inflight: usize,
}

struct PipelineEntry {
    command_buffer: CommandBuffer,
    label: Option<GpuProfilerLabel>,
}

pub struct PipelineCompletion {
    pub command_buffer: CommandBuffer,
    pub label: Option<GpuProfilerLabel>,
    pub wait_duration: std::time::Duration,
}

impl CommandBufferPipeline {
    pub fn new(queue: Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Self {
        Self {
            queue,
            inflight: VecDeque::with_capacity(DEFAULT_MAX_INFLIGHT),
            max_inflight: DEFAULT_MAX_INFLIGHT,
        }
    }

    pub fn submit(&mut self, command_buffer: CommandBuffer, label: Option<GpuProfilerLabel>) {
        command_buffer.commit();
        self.inflight.push_back(PipelineEntry { command_buffer, label });
    }

    pub fn acquire(&mut self) -> Result<(CommandBuffer, Vec<PipelineCompletion>), MetalError> {
        let completed = self.reserve_slot();
        let command_buffer = CommandBuffer::new(&self.queue)?;
        Ok((command_buffer, completed))
    }

    pub fn reserve_slot(&mut self) -> Vec<PipelineCompletion> {
        let mut completed = Vec::new();
        while self.inflight.len() >= self.max_inflight {
            if let Some(done) = self.wait_oldest() {
                completed.push(done);
            }
        }
        completed
    }

    pub fn flush_all(&mut self) -> Vec<PipelineCompletion> {
        let mut completed = Vec::new();
        while let Some(done) = self.wait_oldest() {
            completed.push(done);
        }
        completed
    }

    fn wait_oldest(&mut self) -> Option<PipelineCompletion> {
        self.inflight.pop_front().map(|entry| {
            let wait_start = Instant::now();
            entry.command_buffer.wait();
            PipelineCompletion {
                wait_duration: wait_start.elapsed(),
                label: entry.label,
                command_buffer: entry.command_buffer,
            }
        })
    }
}
