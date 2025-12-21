"""Sentinel token processing for T5-style models."""

from __future__ import annotations

import torch
from torch import Tensor


def create_sentinel_ids(mask: Tensor, sentinel_start: int) -> Tensor:
    """Convert boolean mask to sentinel token IDs."""
    device = mask.device

    shifted = torch.cat([torch.zeros(1, dtype=torch.bool, device=device), mask[:-1]])
    span_starts = mask & ~shifted

    cumsum = torch.cumsum(span_starts.int(), dim=0)
    sentinel_ids = torch.where(
        span_starts, sentinel_start - cumsum + 1, torch.zeros_like(cumsum)
    )

    continuations = mask & ~span_starts
    sentinel_ids = torch.where(
        continuations, torch.full_like(sentinel_ids, -1), sentinel_ids
    )

    return sentinel_ids


def apply_sentinel_mask(
    input_ids: Tensor,
    sentinel_ids: Tensor,
    prefix_ids: Tensor | None = None,
    eos_id: int | None = None,
) -> Tensor:
    """Apply sentinel mask to input_ids."""
    device = input_ids.device

    result = torch.where(sentinel_ids > 0, sentinel_ids, input_ids)
    result = result[sentinel_ids != -1]

    if prefix_ids is not None and prefix_ids.numel() > 0:
        result = torch.cat([prefix_ids.to(device), result])

    if eos_id is not None:
        result = torch.cat(
            [result, torch.tensor([eos_id], dtype=result.dtype, device=device)]
        )

    return result
