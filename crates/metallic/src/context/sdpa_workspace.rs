use super::main::Context;
use crate::{
    MetalError, Tensor, kernels::scaled_dot_product_attention::{ScaledDotProductAttentionDispatchOp, cache::SdpaKey}, tensor::TensorElement
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SdpaWorkspaceKey {
    buffer: usize,
    offset: usize,
}

impl SdpaWorkspaceKey {
    fn from_tensor<T: TensorElement>(tensor: &Tensor<T>) -> Self {
        let buffer = objc2::rc::Retained::as_ptr(&tensor.buf) as *const _ as usize;
        Self {
            buffer,
            offset: tensor.offset,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SdpaWorkspaceState {
    descriptor: SdpaKey,
    last_seq_q: usize,
    last_seq_k: usize,
}

impl SdpaWorkspaceState {
    fn new(descriptor: SdpaKey) -> Self {
        Self {
            descriptor,
            last_seq_q: 0,
            last_seq_k: 0,
        }
    }

    #[inline]
    fn reset(&mut self, descriptor: SdpaKey) {
        self.descriptor = descriptor;
        self.last_seq_q = 0;
        self.last_seq_k = 0;
    }
}

impl<T: TensorElement> Context<T> {
    #[inline]
    pub(crate) fn sdpa_workspace_key_for(&self, tensor: &Tensor<T>) -> SdpaWorkspaceKey {
        SdpaWorkspaceKey::from_tensor(tensor)
    }

    pub(crate) fn sdpa_seq_delta(&mut self, key: SdpaWorkspaceKey, descriptor: SdpaKey, seq_q: usize, seq_k: usize) -> usize {
        let entry = self
            .sdpa_workspaces
            .entry(key)
            .or_insert_with(|| SdpaWorkspaceState::new(descriptor.clone()));

        if entry.descriptor != descriptor {
            entry.reset(descriptor);
        }

        let delta = seq_k.saturating_sub(entry.last_seq_k);
        entry.last_seq_q = seq_q;
        entry.last_seq_k = seq_k;

        if delta == 0 { seq_k } else { delta }
    }

    #[inline]
    pub fn scaled_dot_product_attention(
        &mut self,
        q: &Tensor<T>,
        k: &Tensor<T>,
        v: &Tensor<T>,
        causal: bool,
    ) -> Result<Tensor<T>, MetalError> {
        self.scaled_dot_product_attention_with_offset(q, k, v, causal, 0)
    }

    #[inline]
    pub fn scaled_dot_product_attention_with_offset(
        &mut self,
        q: &Tensor<T>,
        k: &Tensor<T>,
        v: &Tensor<T>,
        causal: bool,
        query_offset: usize,
    ) -> Result<Tensor<T>, MetalError> {
        // Use the kernel system for SDPA
        self.call::<ScaledDotProductAttentionDispatchOp>((q, k, v, causal, query_offset as u32))
    }
}
