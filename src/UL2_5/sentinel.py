"""Sentinel token processing for T5-style models."""

from __future__ import annotations

import torch
from torch import Tensor


def create_sentinel_ids(
    mask: Tensor,
    sentinel_start: int,
    max_sentinels: int = 100,
) -> Tensor:
    """
    Convert boolean mask to sentinel token IDs.

    Args:
        mask: Boolean mask tensor [seq_len] where True = masked position
        sentinel_start: Highest sentinel token ID (e.g., 32099 for <extra_id_0>)
        max_sentinels: Maximum number of sentinel tokens available (default: 100)

    Returns:
        Tensor of sentinel IDs where:
        - Span starts get sentinel IDs (sentinel_start, sentinel_start-1, ...)
        - Span continuations get -1 (to be filtered later)
        - Non-masked positions get 0
    """
    device = mask.device

    # Handle empty mask
    if mask.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    shifted = torch.cat([torch.zeros(1, dtype=torch.bool, device=device), mask[:-1]])
    span_starts = mask & ~shifted

    cumsum = torch.cumsum(span_starts.int(), dim=0)

    # Clamp to max sentinels (no-op when under limit, avoids GPU sync)
    cumsum = torch.clamp(cumsum, max=max_sentinels)

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

    # Ensure sentinel_ids is on same device as input_ids for torch.where
    if sentinel_ids.device != device:
        sentinel_ids = sentinel_ids.to(device)

    result = torch.where(sentinel_ids > 0, sentinel_ids, input_ids)
    result = result[sentinel_ids != -1]

    if prefix_ids is not None and prefix_ids.numel() > 0:
        result = torch.cat([prefix_ids.to(device), result])

    if eos_id is not None:
        result = torch.cat(
            [result, torch.tensor([eos_id], dtype=result.dtype, device=device)]
        )

    return result
