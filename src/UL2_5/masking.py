"""Masking functions for UL2.5 denoising tasks."""

from __future__ import annotations

import random
import warnings
from typing import Any

import torch
from torch import Tensor

# Minimum sequence lengths for meaningful masking
MIN_SEQ_LEN_SPAN = 3  # Need at least 3 tokens for span corruption
MIN_SEQ_LEN_PREFIX = 2  # Need at least prefix + suffix
MIN_SEQ_LEN_INFILL = 3  # Need context + hole

# Warning state for GPU boundary snapping
_BOUNDARY_SNAP_GPU_WARNING_SHOWN = False


def _random_segmentation(n_items: int, n_segments: int, device: torch.device) -> Tensor:
    """Partition n_items into n_segments non-empty segments."""
    if n_segments <= 0 or n_items <= 0:
        return torch.ones(1, dtype=torch.long, device=device)
    if n_segments >= n_items:
        return torch.ones(n_items, dtype=torch.long, device=device)

    dividers = torch.randperm(n_items - 1, device=device)[: n_segments - 1]
    dividers = torch.sort(dividers).values

    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), dividers + 1])
    ends = torch.cat(
        [dividers + 1, torch.tensor([n_items], dtype=torch.long, device=device)]
    )

    return ends - starts


def span_corruption_mask(
    seq_len: int,
    r: float,
    mu: float,
    max_spans: int,
    device: torch.device,
) -> Tensor:
    """Generate T5-style span corruption mask."""
    # Handle edge cases
    if seq_len < MIN_SEQ_LEN_SPAN:
        return torch.zeros(seq_len, dtype=torch.bool, device=device)

    num_noise = max(1, min(int(round(seq_len * r)), seq_len - 1))
    num_spans = max(1, min(max_spans, int(round(num_noise / mu))))
    num_keep = seq_len - num_noise

    # Compute segmentation on CPU (small tensors, avoids GPU sync on .tolist())
    cpu_device = torch.device("cpu")
    noise_lens = _random_segmentation(num_noise, num_spans, cpu_device)
    keep_lens = _random_segmentation(num_keep, num_spans, cpu_device)

    # Interleave segments with random start to avoid edge bias
    n = min(len(noise_lens), len(keep_lens))
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    pos = 0

    # Convert to lists (fast on CPU tensors)
    noise_list = noise_lens.tolist()
    keep_list = keep_lens.tolist()

    # Randomize starting pattern (CPU-only, no sync)
    start_with_noise = random.random() < 0.5

    for i in range(n):
        if start_with_noise:
            # Noise segment first
            noise_len = noise_list[i]
            end = min(pos + noise_len, seq_len)
            mask[pos:end] = True
            pos = end
            # Then keep segment
            pos += keep_list[i]
        else:
            # Keep segment first (original order)
            pos += keep_list[i]
            # Then noise segment
            noise_len = noise_list[i]
            end = min(pos + noise_len, seq_len)
            mask[pos:end] = True
            pos = end
        if pos >= seq_len:
            break

    return mask


def middle_heavy_span_mask(
    seq_len: int,
    noise_density: float,
    mean_span_length: float,
    device: torch.device,
) -> Tensor:
    """
    Position-biased SPAN corruption preferring middle positions.

    Unlike the previous implementation that sampled individual tokens,
    this version samples span START positions with Gaussian weighting,
    then extends spans. This creates contiguous masked regions for
    better retrieval training.

    The function ensures the total masked tokens stays close to the
    target noise_density * seq_len (within a small tolerance).

    Args:
        seq_len: Sequence length
        noise_density: Target fraction of tokens to mask (r)
        mean_span_length: Average span length (mu)
        device: Torch device

    Returns:
        Boolean tensor [seq_len] where True = corrupted/masked
    """
    # Handle edge cases
    if seq_len < MIN_SEQ_LEN_SPAN:
        return torch.zeros(seq_len, dtype=torch.bool, device=device)

    num_noise = max(1, min(int(round(seq_len * noise_density)), seq_len - 1))
    num_spans = max(1, int(round(num_noise / mean_span_length)))

    # Compute weights on CPU (avoids GPU sync on multinomial + .tolist())
    cpu_device = torch.device("cpu")

    # Gaussian weights for span START positions (prefer middle)
    positions = torch.arange(seq_len, dtype=torch.float32, device=cpu_device)
    center = seq_len / 2
    sigma = seq_len / 4
    weights = torch.exp(-0.5 * ((positions - center) / sigma) ** 2)

    # Don't start spans too close to end (leave room for span extension)
    cutoff = max(1, int(0.9 * seq_len))
    weights[cutoff:] = 0

    # Ensure weights sum to 1
    if weights.sum() < 1e-8:
        weights = torch.ones(seq_len, dtype=torch.float32, device=cpu_device)
        weights[cutoff:] = 0
    weights = weights / weights.sum()

    # Sample span start positions (without replacement)
    max_possible_spans = min(num_spans, seq_len // 2, cutoff)
    if max_possible_spans < 1:
        max_possible_spans = 1

    starts = torch.multinomial(weights, max_possible_spans, replacement=False)
    starts = torch.sort(starts).values

    # Generate span lengths using Poisson distribution (on CPU)
    avg_span_len = max(1.0, num_noise / max_possible_spans)
    span_lens = torch.poisson(
        torch.full((len(starts),), avg_span_len, device=cpu_device)
    ).long()
    span_lens = torch.clamp(span_lens, min=1, max=max(1, seq_len // max_possible_spans))

    # Create mask by extending spans from start positions
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    starts_list = starts.tolist()
    lens_list = span_lens.tolist()

    # Track occupied positions and remaining budget
    occupied = set()
    tokens_masked = 0

    for start, length in zip(starts_list, lens_list):
        # Check remaining budget - stop if we've reached target
        remaining_budget = num_noise - tokens_masked
        if remaining_budget <= 0:
            break

        # Cap span length to remaining budget
        length = min(length, remaining_budget)

        # Extend span, respecting boundaries and avoiding overlaps
        end = min(start + length, seq_len)
        # Check if any position in range is already occupied
        span_positions = set(range(start, end))
        if not span_positions & occupied:
            actual_len = end - start
            # Only add if it doesn't exceed budget
            if tokens_masked + actual_len <= num_noise:
                mask[start:end] = True
                occupied.update(span_positions)
                tokens_masked += actual_len
            elif remaining_budget > 0:
                # Truncate span to fit budget
                truncated_end = start + remaining_budget
                if truncated_end > start:
                    truncated_positions = set(range(start, truncated_end))
                    if not truncated_positions & occupied:
                        mask[start:truncated_end] = True
                        occupied.update(truncated_positions)
                        tokens_masked += truncated_end - start

    # If we haven't reached target noise, fill in more from high-weight positions
    # Use locally tracked tokens_masked (no GPU sync)
    if tokens_masked < num_noise:
        remaining = num_noise - tokens_masked
        # Get unmasked positions weighted by original Gaussian
        # Need to move mask to CPU for indexing (single sync, acceptable for fallback)
        mask_cpu = mask.cpu()
        unmasked_weights = weights.clone()
        unmasked_weights[mask_cpu] = 0
        if unmasked_weights.sum() > 1e-8:
            unmasked_weights = unmasked_weights / unmasked_weights.sum()
            extra_count = min(int(remaining), seq_len - tokens_masked)
            if extra_count > 0:
                extra_indices = torch.multinomial(
                    unmasked_weights, extra_count, replacement=False
                )
                mask[extra_indices] = True

    return mask


def prefix_lm_mask(seq_len: int, mode: str, device: torch.device) -> tuple[Tensor, int]:
    """Generate prefix LM mask with various split strategies."""
    # Handle edge cases
    if seq_len < MIN_SEQ_LEN_PREFIX:
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if seq_len > 1:
            mask[1:] = True
        return mask, max(0, seq_len - 1)

    if mode == "random":
        low, high = int(0.2 * seq_len), int(0.8 * seq_len)
        split = random.randint(low, high) if high > low else low
    elif mode == "short":
        frac = 0.05 + 0.10 * random.random()
        split = int((1 - frac) * seq_len)
    elif mode == "long":
        frac = 0.05 + 0.15 * random.random()
        split = int(frac * seq_len)
    else:
        split = int(0.75 * seq_len)

    split = max(1, min(split, seq_len - 1))
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[split:] = True
    return mask, split


def infilling_mask(
    seq_len: int, hole_frac: float, device: torch.device
) -> tuple[Tensor, int, int]:
    """Generate infilling mask (mask middle portion)."""
    # Handle edge cases
    if seq_len < MIN_SEQ_LEN_INFILL:
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if seq_len > 1:
            mask[1:] = True
        return mask, 1 if seq_len > 1 else 0, seq_len

    hole_size = max(1, int(hole_frac * seq_len))
    min_start = int(0.1 * seq_len)
    max_start = max(min_start, int(0.9 * seq_len) - hole_size)

    if max_start <= min_start:
        hole_start = seq_len // 3
    else:
        hole_start = random.randint(min_start, max_start)

    hole_end = min(hole_start + hole_size, seq_len)

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[hole_start:hole_end] = True
    return mask, hole_start, hole_end


def snap_mask_to_word_boundaries(
    mask: Tensor,
    input_ids: Tensor,
    tokenizer: Any,
    warn_on_skip: bool = True,
) -> Tensor:
    """
    Snap mask boundaries to word/token boundaries.

    This adjusts span starts to align with word-initial tokens (those starting
    with space, Ġ for GPT-style, or ▁ for SentencePiece). This makes the
    corruption more semantically meaningful.

    Note: This only runs on CPU tensors to avoid GPU sync overhead.

    Args:
        mask: Boolean mask tensor [seq_len]
        input_ids: Token IDs tensor [seq_len]
        tokenizer: Tokenizer with convert_ids_to_tokens method
        warn_on_skip: Whether to warn when skipping GPU tensors (default True)

    Returns:
        Adjusted mask tensor [seq_len]
    """
    global _BOUNDARY_SNAP_GPU_WARNING_SHOWN

    if mask.device.type != "cpu":
        if warn_on_skip and not _BOUNDARY_SNAP_GPU_WARNING_SHOWN:
            warnings.warn(
                "snap_mask_to_word_boundaries skipped for GPU tensor. "
                "Pass CPU tensors or set enable_boundary_snapping=False in config.",
                stacklevel=2,
            )
            _BOUNDARY_SNAP_GPU_WARNING_SHOWN = True
        return mask

    seq_len = int(mask.shape[0])
    if seq_len == 0:
        return mask

    if isinstance(input_ids, Tensor):
        if input_ids.device.type != "cpu":
            return mask
        ids = input_ids.tolist()
    else:
        ids = list(input_ids)

    if len(ids) != seq_len:
        return mask

    tokens = None
    try:
        tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        pass

    if not tokens or len(tokens) != seq_len:
        try:
            tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in ids]
        except Exception:
            return mask

    if not tokens or len(tokens) != seq_len:
        return mask

    # Detect word-start tokens
    is_word_start = [False] * seq_len
    is_word_start[0] = True  # First token is always a boundary

    for i in range(1, seq_len):
        token = tokens[i]
        if token is None:
            continue
        if (
            token.startswith("▁")
            or token.startswith("Ġ")
            or token.startswith(" ")
            or token.startswith("<")  # Special tokens
        ):
            is_word_start[i] = True

    def _find_spans(flags: list[bool]) -> list[tuple[int, int]]:
        spans = []
        in_span = False
        start = 0
        for i, val in enumerate(flags):
            if val and not in_span:
                start = i
                in_span = True
            elif not val and in_span:
                spans.append((start, i))
                in_span = False
        if in_span:
            spans.append((start, len(flags)))
        return spans

    mask_list = mask.tolist()
    target_count = sum(mask_list)
    if target_count == 0:
        return mask

    new_mask = mask_list[:]
    spans = _find_spans(new_mask)
    max_shift = 3

    for start, _ in spans:
        if start >= seq_len or is_word_start[start]:
            continue

        new_start = None
        for offset in range(1, max_shift + 1):
            idx = start + offset
            if idx >= seq_len:
                break
            if is_word_start[idx]:
                new_start = idx
                break

        if new_start is None:
            continue

        for i in range(start, new_start):
            new_mask[i] = False

    missing = target_count - sum(new_mask)
    if missing > 0:
        spans = _find_spans(new_mask)
        for i, (_, end) in enumerate(spans):
            if missing <= 0:
                break
            next_start = spans[i + 1][0] if i + 1 < len(spans) else seq_len
            available = max(0, next_start - end)
            if available == 0:
                continue
            extend_by = min(available, missing)
            for j in range(end, end + extend_by):
                new_mask[j] = True
            missing -= extend_by

    if missing > 0:
        for i, is_start in enumerate(is_word_start):
            if missing <= 0:
                break
            if is_start and not new_mask[i]:
                new_mask[i] = True
                missing -= 1

    if missing > 0:
        for i in range(seq_len):
            if missing <= 0:
                break
            if not new_mask[i]:
                new_mask[i] = True
                missing -= 1

    return torch.tensor(new_mask, dtype=torch.bool, device=mask.device)
