"""
UL2.5 Data Collator - PyTorch Native, GPU-Ready
================================================

Transformers-style implementation optimized for GPU training.

Features:
- Pure PyTorch tensors (GPU-compatible)
- Follows HuggingFace DataCollatorMixin pattern
- Vectorized operations where possible
- Compatible with Trainer and DataLoader
- Pydantic-based configuration validation
- Fixed middle-heavy span corruption (creates actual spans)
- Span boundary snapping to word boundaries
- Length-adaptive task selection
- Curriculum learning support

Usage:
    from ul2_5_torch import UL25DataCollator, UL25Config

    collator = UL25DataCollator(
        tokenizer=tokenizer,
        config=UL25Config.recommended(),
        max_length=512,
        max_labels_length=128,
    )

    # With DataLoader
    dataloader = DataLoader(dataset, collate_fn=collator, batch_size=32)

    # With Trainer
    trainer = Trainer(..., data_collator=collator)
"""

from __future__ import annotations

import warnings
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object


# =============================================================================
# CONFIGURATION (Pydantic-based)
# =============================================================================


class Task(IntEnum):
    """Task types as integers for efficient torch operations."""

    SPAN = 0  # Standard T5-style span corruption
    SPAN_MIDDLE = 1  # Position-biased toward middle (creates actual spans)
    PREFIX_RANDOM = 2  # Random split (20-80%)
    PREFIX_SHORT = 3  # Long prefix, short target
    PREFIX_LONG = 4  # Short prefix, long target
    INFILLING = 5  # Middle-out masking


if PYDANTIC_AVAILABLE:

    class DenoiserSpec(BaseModel):
        """Single denoiser specification with validation."""

        task: Task
        mu: float = Field(default=3.0, gt=0, description="Mean span length")
        r: float = Field(default=0.15, ge=0.01, le=0.99, description="Noise density")
        max_spans: int = Field(default=512, gt=0, description="Maximum number of spans")
        prefix: str = Field(default="", description="Task prefix token")
        variable_r: bool = Field(default=False, description="Sample r from bounds")
        r_bounds: Tuple[float, float] = Field(
            default=(0.05, 0.50), description="Bounds for variable r"
        )

        model_config = {"frozen": False, "extra": "forbid"}

        @field_validator("r_bounds")
        @classmethod
        def validate_r_bounds(cls, v: Tuple[float, float]) -> Tuple[float, float]:
            if v[0] >= v[1]:
                raise ValueError("r_bounds[0] must be less than r_bounds[1]")
            if v[0] < 0.01 or v[1] > 0.99:
                raise ValueError("r_bounds must be within [0.01, 0.99]")
            return v

        def __repr__(self):
            return f"DenoiserSpec({self.task.name}, r={self.r}, μ={self.mu}, prefix='{self.prefix}')"

    class UL25Config(BaseModel):
        """UL2.5 mixture configuration with validation."""

        denoisers: List[DenoiserSpec] = Field(default_factory=list)
        weights: List[float] = Field(default_factory=list)
        curriculum_start: Optional[List[float]] = Field(
            default=None, description="Starting weights for curriculum"
        )
        curriculum_end: Optional[List[float]] = Field(
            default=None, description="Ending weights for curriculum"
        )
        enable_length_adaptive: bool = Field(
            default=True, description="Enable length-adaptive task selection"
        )
        enable_boundary_snapping: bool = Field(
            default=True, description="Enable span boundary snapping"
        )

        model_config = {"frozen": False, "extra": "forbid"}

        @model_validator(mode="after")
        def validate_config(self) -> "UL25Config":
            # Normalize weights
            if self.weights:
                total = sum(self.weights)
                if total > 0:
                    self.weights = [w / total for w in self.weights]
            if self.curriculum_start:
                total = sum(self.curriculum_start)
                if total > 0:
                    self.curriculum_start = [w / total for w in self.curriculum_start]
            if self.curriculum_end:
                total = sum(self.curriculum_end)
                if total > 0:
                    self.curriculum_end = [w / total for w in self.curriculum_end]

            # Validate lengths match
            n = len(self.denoisers)
            if self.weights and len(self.weights) != n:
                raise ValueError(
                    f"weights length ({len(self.weights)}) must match denoisers ({n})"
                )
            if self.curriculum_start and len(self.curriculum_start) != n:
                raise ValueError(
                    f"curriculum_start length must match denoisers ({n})"
                )
            if self.curriculum_end and len(self.curriculum_end) != n:
                raise ValueError(
                    f"curriculum_end length must match denoisers ({n})"
                )

            return self

        def get_weights(self, progress: float = 0.0) -> List[float]:
            """Get interpolated weights based on training progress."""
            if self.curriculum_start is None or self.curriculum_end is None:
                return self.weights

            progress = max(0.0, min(1.0, progress))
            return [
                (1 - progress) * s + progress * e
                for s, e in zip(self.curriculum_start, self.curriculum_end)
            ]

        @classmethod
        def recommended(cls) -> "UL25Config":
            """Recommended mixture based on feasibility analysis."""
            return cls(
                denoisers=[
                    DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                    DenoiserSpec(task=Task.SPAN, mu=8.0, r=0.25, prefix="[R]"),
                    DenoiserSpec(task=Task.SPAN_MIDDLE, mu=12.0, r=0.20, prefix="[X]"),
                    DenoiserSpec(task=Task.PREFIX_RANDOM, prefix="[S]"),
                    DenoiserSpec(task=Task.PREFIX_SHORT, prefix="[S]"),
                    DenoiserSpec(task=Task.PREFIX_LONG, prefix="[S]"),
                    DenoiserSpec(task=Task.INFILLING, r=0.30, prefix="[I]"),
                ],
                weights=[0.10, 0.10, 0.10, 0.20, 0.15, 0.15, 0.20],
            )

        @classmethod
        def recommended_with_curriculum(cls) -> "UL25Config":
            """
            Recommended config with curriculum learning.

            Early: More span denoising (easier)
            Late: More prefix LM (matches inference)
            """
            return cls(
                denoisers=[
                    DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                    DenoiserSpec(task=Task.SPAN, mu=8.0, r=0.25, prefix="[R]"),
                    DenoiserSpec(task=Task.SPAN_MIDDLE, mu=12.0, r=0.20, prefix="[X]"),
                    DenoiserSpec(task=Task.PREFIX_RANDOM, prefix="[S]"),
                    DenoiserSpec(task=Task.PREFIX_SHORT, prefix="[S]"),
                    DenoiserSpec(task=Task.PREFIX_LONG, prefix="[S]"),
                    DenoiserSpec(task=Task.INFILLING, r=0.30, prefix="[I]"),
                ],
                weights=[0.10, 0.10, 0.10, 0.20, 0.15, 0.15, 0.20],
                curriculum_start=[0.25, 0.25, 0.10, 0.15, 0.10, 0.10, 0.05],
                curriculum_end=[0.05, 0.05, 0.10, 0.25, 0.20, 0.20, 0.15],
            )

        @classmethod
        def span_heavy(cls) -> "UL25Config":
            """Original UL2-style with more span denoising."""
            return cls(
                denoisers=[
                    DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                    DenoiserSpec(task=Task.SPAN, mu=8.0, r=0.15, prefix="[R]"),
                    DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.50, prefix="[X]"),
                    DenoiserSpec(task=Task.SPAN, mu=64.0, r=0.50, prefix="[X]"),
                    DenoiserSpec(task=Task.PREFIX_RANDOM, prefix="[S]"),
                ],
                weights=[0.20, 0.20, 0.15, 0.15, 0.30],
            )

        @classmethod
        def minimal(cls) -> "UL25Config":
            """Minimal config for testing."""
            return cls(
                denoisers=[
                    DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15),
                    DenoiserSpec(task=Task.PREFIX_RANDOM),
                ],
                weights=[0.5, 0.5],
            )

else:
    # Fallback dataclass-based config when pydantic is not available
    from dataclasses import dataclass, field

    @dataclass
    class DenoiserSpec:
        """Single denoiser specification."""

        task: Task
        mu: float = 3.0
        r: float = 0.15
        max_spans: int = 512
        prefix: str = ""
        variable_r: bool = False
        r_bounds: Tuple[float, float] = (0.05, 0.50)

        def __repr__(self):
            return f"DenoiserSpec({self.task.name}, r={self.r}, μ={self.mu}, prefix='{self.prefix}')"

    @dataclass
    class UL25Config:
        """UL2.5 mixture configuration."""

        denoisers: List[DenoiserSpec] = field(default_factory=list)
        weights: List[float] = field(default_factory=list)
        curriculum_start: Optional[List[float]] = None
        curriculum_end: Optional[List[float]] = None
        enable_length_adaptive: bool = True
        enable_boundary_snapping: bool = True

        def __post_init__(self):
            if self.weights:
                total = sum(self.weights)
                self.weights = [w / total for w in self.weights]
            if self.curriculum_start:
                total = sum(self.curriculum_start)
                self.curriculum_start = [w / total for w in self.curriculum_start]
            if self.curriculum_end:
                total = sum(self.curriculum_end)
                self.curriculum_end = [w / total for w in self.curriculum_end]

        def get_weights(self, progress: float = 0.0) -> List[float]:
            """Get interpolated weights based on training progress."""
            if self.curriculum_start is None or self.curriculum_end is None:
                return self.weights
            progress = max(0.0, min(1.0, progress))
            return [
                (1 - progress) * s + progress * e
                for s, e in zip(self.curriculum_start, self.curriculum_end)
            ]

        @classmethod
        def recommended(cls) -> "UL25Config":
            return cls(
                denoisers=[
                    DenoiserSpec(Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                    DenoiserSpec(Task.SPAN, mu=8.0, r=0.25, prefix="[R]"),
                    DenoiserSpec(Task.SPAN_MIDDLE, mu=12.0, r=0.20, prefix="[X]"),
                    DenoiserSpec(Task.PREFIX_RANDOM, prefix="[S]"),
                    DenoiserSpec(Task.PREFIX_SHORT, prefix="[S]"),
                    DenoiserSpec(Task.PREFIX_LONG, prefix="[S]"),
                    DenoiserSpec(Task.INFILLING, r=0.30, prefix="[I]"),
                ],
                weights=[0.10, 0.10, 0.10, 0.20, 0.15, 0.15, 0.20],
            )

        @classmethod
        def recommended_with_curriculum(cls) -> "UL25Config":
            return cls(
                denoisers=[
                    DenoiserSpec(Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                    DenoiserSpec(Task.SPAN, mu=8.0, r=0.25, prefix="[R]"),
                    DenoiserSpec(Task.SPAN_MIDDLE, mu=12.0, r=0.20, prefix="[X]"),
                    DenoiserSpec(Task.PREFIX_RANDOM, prefix="[S]"),
                    DenoiserSpec(Task.PREFIX_SHORT, prefix="[S]"),
                    DenoiserSpec(Task.PREFIX_LONG, prefix="[S]"),
                    DenoiserSpec(Task.INFILLING, r=0.30, prefix="[I]"),
                ],
                weights=[0.10, 0.10, 0.10, 0.20, 0.15, 0.15, 0.20],
                curriculum_start=[0.25, 0.25, 0.10, 0.15, 0.10, 0.10, 0.05],
                curriculum_end=[0.05, 0.05, 0.10, 0.25, 0.20, 0.20, 0.15],
            )

        @classmethod
        def span_heavy(cls) -> "UL25Config":
            return cls(
                denoisers=[
                    DenoiserSpec(Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                    DenoiserSpec(Task.SPAN, mu=8.0, r=0.15, prefix="[R]"),
                    DenoiserSpec(Task.SPAN, mu=3.0, r=0.50, prefix="[X]"),
                    DenoiserSpec(Task.SPAN, mu=64.0, r=0.50, prefix="[X]"),
                    DenoiserSpec(Task.PREFIX_RANDOM, prefix="[S]"),
                ],
                weights=[0.20, 0.20, 0.15, 0.15, 0.30],
            )

        @classmethod
        def minimal(cls) -> "UL25Config":
            return cls(
                denoisers=[
                    DenoiserSpec(Task.SPAN, mu=3.0, r=0.15),
                    DenoiserSpec(Task.PREFIX_RANDOM),
                ],
                weights=[0.5, 0.5],
            )


# =============================================================================
# MASKING FUNCTIONS (Vectorized PyTorch)
# =============================================================================


def _random_segmentation(
    num_items: int,
    num_segments: int,
    device: torch.device,
) -> Tensor:
    """Partition num_items into num_segments non-empty segments."""
    if num_segments <= 0 or num_items <= 0:
        return torch.ones(1, dtype=torch.long, device=device)
    if num_segments >= num_items:
        return torch.ones(num_items, dtype=torch.long, device=device)

    # Sample divider positions
    dividers = torch.randperm(num_items - 1, device=device)[: num_segments - 1]
    dividers = torch.sort(dividers).values

    # Compute segment lengths from divider positions
    starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            dividers + 1,
        ]
    )
    ends = torch.cat(
        [
            dividers + 1,
            torch.tensor([num_items], dtype=torch.long, device=device),
        ]
    )

    return ends - starts


def span_corruption_mask(
    seq_len: int,
    noise_density: float,
    mean_span_length: float,
    max_spans: int,
    device: torch.device,
) -> Tensor:
    """
    Generate T5-style span corruption mask.

    Returns:
        Boolean tensor [seq_len] where True = corrupted/masked
    """
    num_noise = max(1, min(int(round(seq_len * noise_density)), seq_len - 1))
    num_spans = max(1, min(max_spans, int(round(num_noise / mean_span_length))))
    num_nonnoise = seq_len - num_noise

    noise_lengths = _random_segmentation(num_noise, num_spans, device)
    nonnoise_lengths = _random_segmentation(num_nonnoise, num_spans, device)

    # Interleave segments with random start to avoid edge bias
    n = min(len(noise_lengths), len(nonnoise_lengths))
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    pos = 0

    # Convert to lists once to avoid repeated .item() calls
    noise_list = noise_lengths.tolist()
    nonnoise_list = nonnoise_lengths.tolist()

    # Randomize starting pattern to eliminate edge effect at seq end
    start_with_noise = torch.rand(1, device=device).item() < 0.5

    for i in range(n):
        if start_with_noise:
            # Noise segment first
            noise_len = noise_list[i]
            end = min(pos + noise_len, seq_len)
            mask[pos:end] = True
            pos = end
            # Then nonnoise segment
            pos += nonnoise_list[i]
        else:
            # Nonnoise segment first (original order)
            pos += nonnoise_list[i]
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
    num_noise = max(1, min(int(round(seq_len * noise_density)), seq_len - 1))
    num_spans = max(1, int(round(num_noise / mean_span_length)))

    # Gaussian weights for span START positions (prefer middle)
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    center = seq_len / 2
    sigma = seq_len / 4
    weights = torch.exp(-0.5 * ((positions - center) / sigma) ** 2)

    # Don't start spans too close to end (leave room for span extension)
    cutoff = max(1, int(0.9 * seq_len))
    weights[cutoff:] = 0

    # Ensure weights sum to 1
    if weights.sum() < 1e-8:
        weights = torch.ones(seq_len, dtype=torch.float32, device=device)
        weights[cutoff:] = 0
    weights = weights / weights.sum()

    # Sample span start positions (without replacement)
    max_possible_spans = min(num_spans, seq_len // 2, cutoff)
    if max_possible_spans < 1:
        max_possible_spans = 1

    starts = torch.multinomial(weights, max_possible_spans, replacement=False)
    starts = torch.sort(starts).values

    # Generate span lengths using Poisson distribution
    avg_span_len = max(1.0, num_noise / max_possible_spans)
    span_lens = torch.poisson(
        torch.full((len(starts),), avg_span_len, device=device)
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
    current_noise = mask.sum().item()
    if current_noise < num_noise:
        remaining = num_noise - current_noise
        # Get unmasked positions weighted by original Gaussian
        unmasked_weights = weights.clone()
        unmasked_weights[mask] = 0
        if unmasked_weights.sum() > 1e-8:
            unmasked_weights = unmasked_weights / unmasked_weights.sum()
            extra_count = min(int(remaining), int((~mask).sum().item()))
            if extra_count > 0:
                extra_indices = torch.multinomial(
                    unmasked_weights, extra_count, replacement=False
                )
                mask[extra_indices] = True

    return mask


def prefix_lm_mask(
    seq_len: int,
    mode: str,  # "random", "short", "long"
    device: torch.device,
) -> Tuple[Tensor, int]:
    """
    Generate prefix LM mask.

    Returns:
        (mask, split_index) where mask[split:] = True
    """
    if mode == "random":
        min_s, max_s = int(0.2 * seq_len), int(0.8 * seq_len)
        split = torch.randint(min_s, max_s + 1, (1,), device=device).item()
    elif mode == "short":
        # Short target: 5-15% of sequence
        frac = 0.05 + 0.10 * torch.rand(1, device=device).item()
        split = int((1 - frac) * seq_len)
    elif mode == "long":
        # Long target: prefix is 5-20%
        frac = 0.05 + 0.15 * torch.rand(1, device=device).item()
        split = int(frac * seq_len)
    else:
        split = int(0.75 * seq_len)

    split = max(1, min(split, seq_len - 1))

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[split:] = True
    return mask, split


def infilling_mask(
    seq_len: int,
    hole_frac: float,
    device: torch.device,
) -> Tuple[Tensor, int, int]:
    """
    Generate infilling mask (mask middle portion).

    Returns:
        (mask, hole_start, hole_end)
    """
    hole_size = max(1, int(hole_frac * seq_len))

    min_start = int(0.1 * seq_len)
    max_start = max(min_start, int(0.9 * seq_len) - hole_size)

    if max_start <= min_start:
        hole_start = seq_len // 3
    else:
        hole_start = torch.randint(min_start, max_start + 1, (1,), device=device).item()

    hole_end = min(hole_start + hole_size, seq_len)

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[hole_start:hole_end] = True
    return mask, hole_start, hole_end


# =============================================================================
# SPAN BOUNDARY SNAPPING
# =============================================================================


def snap_mask_to_word_boundaries(
    mask: Tensor,
    input_ids: Tensor,
    tokenizer: Any,
) -> Tensor:
    """
    Snap mask boundaries to word/token boundaries.

    This adjusts span starts to align with word-initial tokens (those starting
    with space, Ġ for GPT-style, or ▁ for SentencePiece). This makes the
    corruption more semantically meaningful.

    Args:
        mask: Boolean mask tensor [seq_len]
        input_ids: Token IDs tensor [seq_len]
        tokenizer: Tokenizer with convert_ids_to_tokens method

    Returns:
        Adjusted mask tensor [seq_len]
    """
    device = mask.device
    seq_len = mask.shape[0]

    # Detect word-start tokens
    is_word_start = torch.zeros(seq_len, dtype=torch.bool, device=device)
    is_word_start[0] = True  # First token is always a boundary

    for i in range(1, seq_len):
        try:
            token = tokenizer.convert_ids_to_tokens(input_ids[i].item())
            if token is None:
                continue
            # SentencePiece uses ▁, BPE uses Ġ or leading space
            if (
                token.startswith("▁")
                or token.startswith("Ġ")
                or token.startswith(" ")
                or token.startswith("<")  # Special tokens
            ):
                is_word_start[i] = True
        except Exception:
            continue

    # Find span starts in current mask (False -> True transitions)
    shifted = torch.cat([torch.zeros(1, dtype=torch.bool, device=device), mask[:-1]])
    span_starts = mask & ~shifted

    new_mask = mask.clone()
    start_indices = span_starts.nonzero().squeeze(-1)

    if start_indices.dim() == 0:
        start_indices = start_indices.unsqueeze(0)

    for idx in start_indices.tolist():
        if is_word_start[idx]:
            continue  # Already aligned

        # Find next word boundary >= idx
        candidates = is_word_start[idx:].nonzero()
        if len(candidates) > 0:
            offset = candidates[0].item()
            new_start = idx + offset

            # Don't snap too far (max 3 tokens)
            if new_start > idx and new_start - idx <= 3:
                # Clear positions between old start and new start
                new_mask[idx:new_start] = False

    return new_mask


# =============================================================================
# SENTINEL PROCESSING (Vectorized)
# =============================================================================


def create_sentinel_ids(
    mask: Tensor,
    sentinel_start: int,
) -> Tensor:
    """
    Convert boolean mask to sentinel token IDs.

    Returns tensor where:
        - Span starts have sentinel IDs (sentinel_start, sentinel_start-1, ...)
        - Continuation positions have -1 (to be removed)
        - Unmasked positions have 0
    """
    device = mask.device

    # Find span starts: transition from False->True
    shifted = torch.cat([torch.zeros(1, dtype=torch.bool, device=device), mask[:-1]])
    span_starts = mask & ~shifted

    # Assign decreasing sentinel IDs to span starts
    cumsum = torch.cumsum(span_starts.int(), dim=0)

    # sentinel_start for first span, sentinel_start-1 for second, etc.
    sentinel_ids = torch.where(
        span_starts,
        sentinel_start - cumsum + 1,
        torch.zeros_like(cumsum),
    )

    # Mark continuation positions with -1
    continuations = mask & ~span_starts
    sentinel_ids = torch.where(
        continuations,
        torch.full_like(sentinel_ids, -1),
        sentinel_ids,
    )

    return sentinel_ids


def apply_sentinel_mask(
    input_ids: Tensor,
    sentinel_ids: Tensor,
    prefix_ids: Optional[Tensor] = None,
    eos_id: Optional[int] = None,
) -> Tensor:
    """
    Apply sentinel mask: replace spans with sentinels, remove continuations.

    Args:
        input_ids: Original token IDs [seq_len]
        sentinel_ids: From create_sentinel_ids [seq_len]
        prefix_ids: Optional prefix tokens to prepend
        eos_id: Optional EOS token to append

    Returns:
        Filtered token IDs with sentinels
    """
    device = input_ids.device

    # Replace masked positions with sentinels
    result = torch.where(sentinel_ids > 0, sentinel_ids, input_ids)

    # Filter out -1 positions (continuations)
    keep_mask = sentinel_ids != -1
    result = result[keep_mask]

    # Prepend prefix
    if prefix_ids is not None and prefix_ids.numel() > 0:
        result = torch.cat([prefix_ids.to(device), result])

    # Append EOS
    if eos_id is not None:
        result = torch.cat(
            [result, torch.tensor([eos_id], dtype=result.dtype, device=device)]
        )

    return result


# =============================================================================
# MAIN COLLATOR
# =============================================================================


class UL25DataCollator:
    """
    PyTorch-native UL2.5 Data Collator.

    Follows HuggingFace DataCollator conventions for use with
    Trainer and DataLoader.

    Args:
        tokenizer: HuggingFace tokenizer with extra_id tokens
        config: UL25Config specifying denoiser mixture
        max_length: Maximum encoder input length
        max_labels_length: Maximum decoder target length
        pad_to_multiple_of: Pad sequences to multiple of this value
        return_tensors: Return type ("pt" for PyTorch)

    Example:
        >>> collator = UL25DataCollator(tokenizer, UL25Config.recommended())
        >>> batch = collator([{"input_ids": torch.randint(0, 1000, (128,))}])
    """

    def __init__(
        self,
        tokenizer: Any,
        config: Optional[UL25Config] = None,
        max_length: int = 512,
        max_labels_length: int = 128,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.config = config or UL25Config.recommended()
        self.max_length = max_length
        self.max_labels_length = max_labels_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

        # Token IDs
        self.sentinel_start = self._get_sentinel_start()
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id or 0

        # Pre-encode prefixes (store as CPU tensors, move to device during forward)
        self._prefix_cache: Dict[str, Tensor] = {}
        for spec in self.config.denoisers:
            if spec.prefix and spec.prefix not in self._prefix_cache:
                ids = tokenizer.encode(spec.prefix, add_special_tokens=False)
                self._prefix_cache[spec.prefix] = torch.tensor(ids, dtype=torch.long)

        # Sampling weights as tensor
        self._weights = torch.tensor(self.config.weights, dtype=torch.float32)

        # Training progress for curriculum
        self._progress = 0.0

    def _get_sentinel_start(self) -> int:
        """Get highest extra_id token ID.

        Detection order:
        1. Direct conversion of <extra_id_0> (most reliable for T5 tokenizers)
        2. Search special tokens for extra_id pattern (fallback)
        3. Hardcoded 32099 default
        """
        # Method 1: Direct token conversion (most reliable)
        try:
            token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
            if token_id != self.tokenizer.unk_token_id:
                return token_id
        except Exception:
            pass

        # Method 2: Search special tokens for extra_id pattern
        extra_ids = []
        for i, tok in enumerate(self.tokenizer.all_special_tokens):
            if "extra_id" in tok.lower():
                extra_ids.append(self.tokenizer.all_special_ids[i])
        if extra_ids:
            return max(extra_ids)

        # Method 3: Hardcoded default with warning
        warnings.warn(
            "No extra_id tokens found. Using 32099 as default sentinel start."
        )
        return 32099

    @property
    def progress(self) -> float:
        """Current training progress (0-1) for curriculum."""
        return self._progress

    @progress.setter
    def progress(self, value: float):
        """Set training progress for curriculum learning."""
        self._progress = max(0.0, min(1.0, value))

    def _get_curriculum_weights(self) -> Tensor:
        """Get denoiser sampling weights (supports curriculum learning)."""
        # Always use get_weights to properly handle curriculum_start at progress 0
        weights = self.config.get_weights(self._progress)
        return torch.tensor(weights, dtype=torch.float32)

    def _get_length_adaptive_weights(self, seq_len: int) -> Tensor:
        """
        Adjust task weights based on sequence length.

        For longer sequences, boost tasks that exercise long-context abilities:
        - Prefix LM variants
        - Middle-heavy span corruption
        - Infilling
        """
        if not getattr(self.config, "enable_length_adaptive", True):
            return self._get_curriculum_weights()

        base_weights = self._get_curriculum_weights()

        if seq_len < 1024:
            return base_weights

        # Identify task indices by type
        prefix_indices = []
        middle_indices = []
        infill_indices = []

        for i, d in enumerate(self.config.denoisers):
            if d.task in (Task.PREFIX_RANDOM, Task.PREFIX_SHORT, Task.PREFIX_LONG):
                prefix_indices.append(i)
            elif d.task == Task.SPAN_MIDDLE:
                middle_indices.append(i)
            elif d.task == Task.INFILLING:
                infill_indices.append(i)

        # Boost long-context-relevant tasks (scale with length)
        adjusted = base_weights.clone()
        boost = min(0.3, (seq_len - 1024) / 8192)

        for idx in prefix_indices + middle_indices + infill_indices:
            adjusted[idx] *= 1 + boost

        # Renormalize
        adjusted = adjusted / adjusted.sum()
        return adjusted

    def _get_prefix_ids(self, prefix: str, device: torch.device) -> Optional[Tensor]:
        """Get prefix token IDs on correct device (caches per-device)."""
        if not prefix:
            return None
        # Check for device-specific cached tensor
        cache_key = (prefix, str(device))
        if cache_key in self._prefix_cache:
            return self._prefix_cache[cache_key]
        # Get base tensor and cache device-specific version
        base_tensor = self._prefix_cache.get(prefix)
        if base_tensor is not None:
            device_tensor = base_tensor.to(device)
            self._prefix_cache[cache_key] = device_tensor
            return device_tensor
        return None

    def _sample_r(self, spec: DenoiserSpec) -> float:
        """Sample corruption rate."""
        if spec.variable_r:
            lo, hi = spec.r_bounds
            return lo + (hi - lo) * torch.rand(1).item()
        return spec.r

    def _generate_mask(
        self,
        seq_len: int,
        spec: DenoiserSpec,
        device: torch.device,
    ) -> Tensor:
        """Generate corruption mask based on task type."""
        r = self._sample_r(spec)

        if spec.task == Task.SPAN:
            return span_corruption_mask(seq_len, r, spec.mu, spec.max_spans, device)
        elif spec.task == Task.SPAN_MIDDLE:
            # Use the new middle-heavy SPAN mask (creates actual spans)
            return middle_heavy_span_mask(seq_len, r, spec.mu, device)
        elif spec.task == Task.PREFIX_RANDOM:
            return prefix_lm_mask(seq_len, "random", device)[0]
        elif spec.task == Task.PREFIX_SHORT:
            return prefix_lm_mask(seq_len, "short", device)[0]
        elif spec.task == Task.PREFIX_LONG:
            return prefix_lm_mask(seq_len, "long", device)[0]
        elif spec.task == Task.INFILLING:
            return infilling_mask(seq_len, r, device)[0]
        else:
            return span_corruption_mask(seq_len, r, spec.mu, spec.max_spans, device)

    def _process_single(
        self,
        input_ids: Tensor,
        denoiser_idx: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Process single sequence with UL2.5 denoising."""
        device = input_ids.device
        seq_len = input_ids.shape[0]

        # Sample denoiser (with length-adaptive weights if enabled)
        if denoiser_idx is None:
            weights = self._get_length_adaptive_weights(seq_len)
            denoiser_idx = torch.multinomial(weights, 1).item()

        spec = self.config.denoisers[denoiser_idx]
        prefix_ids = self._get_prefix_ids(spec.prefix, device)

        # Generate mask
        mask = self._generate_mask(seq_len, spec, device)

        # Apply boundary snapping if enabled and tokenizer supports it
        if getattr(self.config, "enable_boundary_snapping", True):
            if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                try:
                    mask = snap_mask_to_word_boundaries(mask, input_ids, self.tokenizer)
                except Exception:
                    pass  # Fall back to unsnapped mask

        # Create sentinel IDs for encoder (masked spans) and decoder (unmasked spans)
        enc_sentinels = create_sentinel_ids(mask, self.sentinel_start)
        dec_sentinels = create_sentinel_ids(~mask, self.sentinel_start)

        # Apply masks
        encoder_ids = apply_sentinel_mask(
            input_ids, enc_sentinels, prefix_ids, self.eos_id
        )
        decoder_ids = apply_sentinel_mask(input_ids, dec_sentinels, None, self.eos_id)

        return {
            "encoder_ids": encoder_ids,
            "decoder_ids": decoder_ids,
        }

    def _pad_to_multiple(self, length: int) -> int:
        """Round up to multiple if specified."""
        if self.pad_to_multiple_of is None:
            return length
        return (
            (length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
        ) * self.pad_to_multiple_of

    def __call__(
        self,
        examples: List[Dict[str, Any]],
    ) -> Dict[str, Tensor]:
        """
        Collate batch of examples.

        Args:
            examples: List of dicts with "input_ids" (Tensor or list)

        Returns:
            Dict with "input_ids", "attention_mask", "labels"
        """
        # Determine device from first example
        first_ids = examples[0]["input_ids"]
        if isinstance(first_ids, Tensor):
            device = first_ids.device
        else:
            device = torch.device("cpu")

        batch_size = len(examples)

        # Get sequence lengths to determine weights
        seq_lens = []
        for ex in examples:
            ids = ex["input_ids"]
            if isinstance(ids, Tensor):
                seq_lens.append(ids.shape[-1] if ids.dim() > 0 else 1)
            else:
                seq_lens.append(len(ids))

        # Batch sample denoiser indices (use max seq_len for weight calculation)
        max_seq_len = max(seq_lens)
        weights = self._get_length_adaptive_weights(max_seq_len)
        denoiser_indices = (
            torch.multinomial(weights.expand(batch_size, -1), 1)
            .squeeze(-1)
            .tolist()
        )

        # Process each example
        processed = []
        for i, ex in enumerate(examples):
            input_ids = ex["input_ids"]

            # Convert to tensor if needed
            if not isinstance(input_ids, Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)

            # Flatten if 2D
            if input_ids.dim() == 2:
                input_ids = input_ids.squeeze(0)

            input_ids = input_ids.to(device)
            processed.append(self._process_single(input_ids, denoiser_indices[i]))

        # Compute padded lengths
        max_enc = min(
            self.max_length, max(p["encoder_ids"].shape[0] for p in processed)
        )
        max_dec = min(
            self.max_labels_length, max(p["decoder_ids"].shape[0] for p in processed)
        )

        max_enc = self._pad_to_multiple(max_enc)
        max_dec = self._pad_to_multiple(max_dec)

        batch_size = len(processed)

        # Allocate output tensors
        input_ids_batch = torch.full(
            (batch_size, max_enc), self.pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            (batch_size, max_enc), dtype=torch.long, device=device
        )
        labels = torch.full(
            (batch_size, max_dec), -100, dtype=torch.long, device=device
        )

        # Fill tensors
        for i, p in enumerate(processed):
            enc = p["encoder_ids"]
            dec = p["decoder_ids"]

            enc_len = min(enc.shape[0], max_enc)
            dec_len = min(dec.shape[0], max_dec)

            input_ids_batch[i, :enc_len] = enc[:enc_len]
            attention_mask[i, :enc_len] = 1
            labels[i, :dec_len] = dec[:dec_len]

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# =============================================================================
# TESTING
# =============================================================================


def _run_tests():
    """Run basic functionality tests."""
    print("=" * 60)
    print("UL2.5 PyTorch Collator - Tests")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Pydantic available: {PYDANTIC_AVAILABLE}")

    # Test masking functions
    print("\n[TEST] Masking functions")
    seq_len = 100

    # Span mask
    mask = span_corruption_mask(seq_len, 0.15, 3.0, 512, device)
    density = mask.float().mean().item()
    print(
        f"  span_corruption_mask: {mask.sum().item()}/{seq_len} masked (density={density:.3f})"
    )
    assert 0.10 < density < 0.25, "Density out of range"

    # Middle-heavy SPAN mask (new implementation)
    mask = middle_heavy_span_mask(seq_len, 0.20, 12.0, device)
    density = mask.float().mean().item()
    middle = mask[25:75].float().mean().item()
    edges = (mask[:25].float().mean().item() + mask[75:].float().mean().item()) / 2

    # Count contiguous spans
    shifted = torch.cat([torch.zeros(1, dtype=torch.bool, device=device), mask[:-1]])
    num_spans = (mask & ~shifted).sum().item()

    print(f"  middle_heavy_span_mask: {mask.sum().item()}/{seq_len} masked")
    print(f"    density={density:.3f}, middle={middle:.3f}, edges={edges:.3f}")
    print(f"    num_spans={num_spans} (should be coherent spans, not scattered)")

    # Verify it creates actual spans (not scattered tokens)
    assert num_spans < mask.sum().item(), "Should create spans, not individual tokens"

    # Prefix LM
    for mode in ["random", "short", "long"]:
        mask, split = prefix_lm_mask(seq_len, mode, device)
        print(f"  prefix_lm_mask ({mode}): split at {split}")

    # Infilling
    mask, start, end = infilling_mask(seq_len, 0.3, device)
    print(f"  infilling_mask: hole [{start}, {end})")

    # Test sentinel creation
    print("\n[TEST] Sentinel creation")
    mask = torch.tensor(
        [False, False, True, True, False, True, True, True, False, False], device=device
    )
    sids = create_sentinel_ids(mask, 32099)
    print(f"  Mask:      {mask.int().tolist()}")
    print(f"  Sentinels: {sids.tolist()}")
    assert sids[2] == 32099, "First span should get sentinel_start"
    assert sids[3] == -1, "Continuation should be -1"
    assert sids[5] == 32098, "Second span decrements"

    # Test configuration
    print("\n[TEST] Configuration")
    config = UL25Config.recommended()
    print(f"  recommended: {len(config.denoisers)} denoisers")

    config_curriculum = UL25Config.recommended_with_curriculum()
    print(f"  with curriculum: start={config_curriculum.curriculum_start is not None}")

    # Test curriculum weights
    w0 = config_curriculum.get_weights(0.0)
    w1 = config_curriculum.get_weights(1.0)
    print(f"  weights at progress=0: {[f'{w:.2f}' for w in w0[:3]]}...")
    print(f"  weights at progress=1: {[f'{w:.2f}' for w in w1[:3]]}...")

    # Test full collator with mock tokenizer
    print("\n[TEST] Full collator")

    class MockTokenizer:
        eos_token_id = 1
        pad_token_id = 0
        all_special_tokens = [f"<extra_id_{i}>" for i in range(100)]
        all_special_ids = list(range(32000, 32100))
        unk_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text[:10]]

        def convert_ids_to_tokens(self, token_id):
            if token_id < 100:
                return f"▁tok{token_id}"
            return f"tok{token_id}"

    tokenizer = MockTokenizer()
    config = UL25Config.recommended()
    collator = UL25DataCollator(
        tokenizer=tokenizer,
        config=config,
        max_length=128,
        max_labels_length=64,
    )

    # Test batch
    examples = [
        {"input_ids": torch.randint(100, 1000, (64,), device=device)},
        {"input_ids": torch.randint(100, 1000, (48,), device=device)},
        {"input_ids": torch.randint(100, 1000, (72,), device=device)},
    ]

    batch = collator(examples)

    print(f"  Batch size: {len(examples)}")
    print(
        f"  input_ids: {batch['input_ids'].shape}, device={batch['input_ids'].device}"
    )
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")

    # Verify shapes and dtypes
    assert batch["input_ids"].shape[0] == 3
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long

    # Test curriculum
    print("\n[TEST] Curriculum learning")
    collator.progress = 0.0
    print(f"  progress=0.0: set successfully")
    collator.progress = 0.5
    print(f"  progress=0.5: set successfully")
    collator.progress = 1.0
    print(f"  progress=1.0: set successfully")

    # Test length-adaptive weights
    print("\n[TEST] Length-adaptive weights")
    short_weights = collator._get_length_adaptive_weights(512)
    long_weights = collator._get_length_adaptive_weights(4096)
    print(f"  short seq (512): {[f'{w:.3f}' for w in short_weights.tolist()[:3]]}...")
    print(f"  long seq (4096): {[f'{w:.3f}' for w in long_weights.tolist()[:3]]}...")

    # Benchmark
    print("\n[BENCHMARK] Throughput")
    import time

    n_batches = 100
    batch_size = 32
    seq_len = 256

    examples = [
        {"input_ids": torch.randint(100, 1000, (seq_len,), device=device)}
        for _ in range(batch_size)
    ]

    # Warmup
    _ = collator(examples)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_batches):
        _ = collator(examples)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    batches_per_sec = n_batches / elapsed
    samples_per_sec = n_batches * batch_size / elapsed
    tokens_per_sec = samples_per_sec * seq_len

    print(f"  {batches_per_sec:.1f} batches/sec")
    print(f"  {samples_per_sec:.0f} samples/sec")
    print(f"  {tokens_per_sec:,.0f} tokens/sec")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
