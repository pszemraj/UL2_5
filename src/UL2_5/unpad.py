"""Unpadding utilities for Flash Attention varlen kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class UnpadOutput:
    """Container for unpadded tensor with varlen metadata.

    :param Tensor hidden_states: Flattened non-padding tokens, shape (total_tokens,)
        for IDs or (total_tokens, hidden_dim) for embeddings/hidden states.
    :param Tensor indices: Position of each token in original flattened batch (total_tokens,).
    :param Tensor cu_seqlens: Cumulative sequence lengths starting with 0 (batch_size + 1,),
        uses int32 dtype as required by Flash Attention CUDA kernels.
    :param int max_seqlen: Maximum sequence length in the batch.
    """

    hidden_states: Tensor
    indices: Tensor
    cu_seqlens: Tensor
    max_seqlen: int


def unpad_input(inputs: Tensor, attention_mask: Tensor) -> UnpadOutput:
    """Remove padding from batched inputs for Flash Attention varlen kernels.

    :param Tensor inputs: Padded input tensor, shape (batch, seqlen) or (batch, seqlen, hidden_dim).
    :param Tensor attention_mask: Binary mask where 1 = valid token, 0 = padding (batch, seqlen).
    :return UnpadOutput: Container with flattened non-padding tokens and varlen metadata.

    Example::

        >>> inputs = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        >>> mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        >>> out = unpad_input(inputs, mask)
        >>> out.hidden_states  # tensor([1, 2, 3, 4, 5])
        >>> out.cu_seqlens     # tensor([0, 2, 5], dtype=torch.int32)
        >>> out.max_seqlen     # 3
    """
    device = inputs.device

    # Compute sequence lengths from attention mask
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)

    # Find indices of non-padding positions
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    # Max sequence length in batch
    max_seqlen = int(seqlens.max().item())

    # Cumulative sequence lengths (prepend 0) - int32 for Flash Attention
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

    # Extract non-padding tokens
    if inputs.dim() == 2:
        # Token IDs: (batch, seqlen) -> (total_tokens,)
        unpadded = inputs.flatten()[indices]
    else:
        # Hidden states: (batch, seqlen, hidden) -> (total_tokens, hidden)
        batch, seqlen = inputs.shape[:2]
        rest = inputs.shape[2:]
        unpadded = inputs.view(batch * seqlen, *rest)[indices]

    return UnpadOutput(
        hidden_states=unpadded,
        indices=indices,
        cu_seqlens=cu_seqlens.to(device),
        max_seqlen=max_seqlen,
    )


def pad_input(
    inputs: Tensor,
    indices: Tensor,
    batch_size: int,
    seqlen: int,
    pad_value: int | float = 0,
) -> Tensor:
    """Re-pad unpadded inputs to original batch shape.

    :param Tensor inputs: Unpadded tensor, shape (total_tokens,) or (total_tokens, hidden_dim).
    :param Tensor indices: Original positions from unpad_input output.
    :param int batch_size: Original batch size.
    :param int seqlen: Original sequence length.
    :param pad_value: Value for padding positions, defaults to 0.
    :type pad_value: int | float
    :return Tensor: Padded tensor, shape (batch, seqlen) or (batch, seqlen, hidden_dim).

    Example::

        >>> unpadded = torch.tensor([1, 2, 3, 4, 5])
        >>> indices = torch.tensor([0, 1, 4, 5, 6])
        >>> padded = pad_input(unpadded, indices, batch_size=2, seqlen=4)
        >>> padded  # tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    """
    if inputs.dim() == 1:
        # Token IDs
        output = torch.full(
            (batch_size * seqlen,),
            pad_value,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        output[indices] = inputs
        return output.view(batch_size, seqlen)
    else:
        # Hidden states
        rest = inputs.shape[1:]
        output = torch.full(
            (batch_size * seqlen, *rest),
            pad_value,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        output[indices] = inputs
        return output.view(batch_size, seqlen, *rest)
