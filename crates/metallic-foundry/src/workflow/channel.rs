use crate::{
    Foundry, error::MetalError, metals::channel::{ChannelU32Init, ChannelU32Push, ChannelU32PushScalar}, tensor::Dtype, types::{MetalBuffer, MetalResourceOptions, TensorArg}
};

#[derive(Debug, Clone)]
pub struct ChannelU32 {
    pub header: TensorArg, // u32[4]
    pub data: TensorArg,   // u32[capacity]
    pub capacity: u32,
}

impl ChannelU32 {
    pub fn allocate(foundry: &mut Foundry, capacity: u32) -> Result<Self, MetalError> {
        if capacity == 0 {
            return Err(MetalError::InvalidOperation("ChannelU32 capacity must be > 0".into()));
        }
        let header_buf = foundry
            .device
            .new_buffer(4 * 4, MetalResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed("Failed to allocate ChannelU32 header buffer".into()))?;
        let data_bytes = capacity as usize * 4;
        let data_buf = foundry
            .device
            .new_buffer(data_bytes.max(4), MetalResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed("Failed to allocate ChannelU32 data buffer".into()))?;

        let header = TensorArg::from_buffer(header_buf, Dtype::U32, vec![4], vec![1]);
        let data = TensorArg::from_buffer(data_buf, Dtype::U32, vec![capacity as usize], vec![1]);

        // Initialize header + clear data (deterministic tests).
        let init = ChannelU32Init::new(&header, &data, capacity);
        foundry.run(&init)?;

        Ok(Self { header, data, capacity })
    }

    pub fn push_value_buffer(&self, foundry: &mut Foundry, value_buf: &TensorArg) -> Result<(), MetalError> {
        let push = ChannelU32Push::new(&self.header, &self.data, value_buf);
        foundry.run(&push)
    }

    pub fn push_scalar(&self, foundry: &mut Foundry, value: u32) -> Result<(), MetalError> {
        let push = ChannelU32PushScalar::new(&self.header, &self.data, value);
        foundry.run(&push)
    }

    pub fn header_buffer(&self) -> Result<MetalBuffer, MetalError> {
        self.header
            .buffer
            .clone()
            .ok_or_else(|| MetalError::InvalidOperation("ChannelU32 header tensor missing buffer".into()))
    }

    pub fn data_buffer(&self) -> Result<MetalBuffer, MetalError> {
        self.data
            .buffer
            .clone()
            .ok_or_else(|| MetalError::InvalidOperation("ChannelU32 data tensor missing buffer".into()))
    }
}

pub struct ChannelU32Reader {
    chan: ChannelU32,
    next_read_idx: u32,
}

impl ChannelU32Reader {
    pub fn new(chan: ChannelU32) -> Self {
        Self { chan, next_read_idx: 0 }
    }

    pub fn channel(&self) -> &ChannelU32 {
        &self.chan
    }

    pub fn try_next(&mut self) -> Result<Option<u32>, MetalError> {
        let header = self.chan.header_buffer()?;
        // Header layout: [write_idx, read_idx, capacity, flags]
        let raw: Vec<u32> = header.read_to_vec(4);
        let write_idx = raw.first().copied().unwrap_or(0);
        let cap = raw.get(2).copied().unwrap_or(self.chan.capacity);
        if cap == 0 {
            return Ok(None);
        }

        // Drop-oldest semantics if producer overwrote unread values.
        let min_valid = write_idx.saturating_sub(cap);
        if self.next_read_idx < min_valid {
            self.next_read_idx = min_valid;
        }

        if self.next_read_idx >= write_idx {
            return Ok(None);
        }

        let slot = (self.next_read_idx % cap) as usize;
        let data = self.chan.data_buffer()?;
        let value = unsafe {
            let ptr = data.contents() as *const u32;
            std::ptr::read_volatile(ptr.add(slot))
        };
        self.next_read_idx = self.next_read_idx.wrapping_add(1);
        Ok(Some(value))
    }

    pub fn drain_into(&mut self, out: &mut Vec<u32>) -> Result<(), MetalError> {
        while let Some(v) = self.try_next()? {
            out.push(v);
        }
        Ok(())
    }
}

impl Iterator for ChannelU32Reader {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        self.try_next().ok().flatten()
    }
}
