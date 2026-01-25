use super::{BlitCommandEncoder, ComputeCommandEncoder, MetalCommandBuffer, MetalError};

/// A stateful manager for a single Metal command buffer that automatically
/// handles encoder transitions and resource barriers.
///
/// Designed to be ICB-compatible by separating command recording from buffer lifecycle.
pub struct CommandStream {
    command_buffer: MetalCommandBuffer,
    current_encoder: StreamEncoder,
}

pub enum StreamEncoder {
    Compute(ComputeCommandEncoder),
    Blit(BlitCommandEncoder),
    None,
}

impl CommandStream {
    pub fn new(command_buffer: MetalCommandBuffer) -> Self {
        Self {
            command_buffer,
            current_encoder: StreamEncoder::None,
        }
    }

    /// Access the underlying command buffer.
    pub fn command_buffer(&self) -> &MetalCommandBuffer {
        &self.command_buffer
    }

    /// Ensure a compute encoder is active, ending any other active encoders.
    pub fn compute_encoder(&mut self) -> Result<ComputeCommandEncoder, MetalError> {
        match &self.current_encoder {
            StreamEncoder::Compute(enc) => Ok(enc.clone()),
            _ => {
                self.end_encoding();
                let enc = self.command_buffer.compute_command_encoder()?;
                self.current_encoder = StreamEncoder::Compute(enc.clone());
                Ok(enc)
            }
        }
    }

    /// Ensure a blit encoder is active, ending any other active encoders.
    pub fn blit_encoder(&mut self) -> Result<BlitCommandEncoder, MetalError> {
        match &self.current_encoder {
            StreamEncoder::Blit(enc) => Ok(enc.clone()),
            _ => {
                self.end_encoding();
                let enc = self.command_buffer.blit_command_encoder()?;
                self.current_encoder = StreamEncoder::Blit(enc.clone());
                Ok(enc)
            }
        }
    }

    /// Explicitly end the current encoder if one exists.
    pub fn end_encoding(&mut self) {
        match std::mem::replace(&mut self.current_encoder, StreamEncoder::None) {
            StreamEncoder::Compute(enc) => enc.end_encoding(),
            StreamEncoder::Blit(enc) => enc.end_encoding(),
            StreamEncoder::None => {}
        }
    }

    /// Commit the command buffer, ending any active encoders first.
    pub fn commit(mut self) {
        self.end_encoding();
        self.command_buffer.commit();
    }

    /// Commit and wait for the command buffer to complete.
    pub fn commit_and_wait(mut self) {
        self.end_encoding();
        self.command_buffer.commit();
        self.command_buffer.wait_until_completed();
    }
}

impl Drop for CommandStream {
    fn drop(&mut self) {
        self.end_encoding();
    }
}
