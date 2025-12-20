"""
UL2.5 Data Collator - HuggingFace Transformers Integration
==========================================================

Production-ready collator compatible with:
- transformers.Trainer
- torch.utils.data.DataLoader
- Accelerate distributed training

Features:
- Pure PyTorch, GPU-ready
- Follows DataCollatorMixin pattern
- Efficient batch processing
- Curriculum learning support
- Multiple preset configurations

Installation:
    pip install torch transformers

Usage with Trainer:
    from ul2_5_hf import UL25DataCollator, UL25Config
    from transformers import Trainer, T5ForConditionalGeneration

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    # Add special tokens
    tokenizer.add_special_tokens({"additional_special_tokens": ["[R]", "[S]", "[X]", "[I]"]})
    model.resize_token_embeddings(len(tokenizer))

    collator = UL25DataCollator(
        tokenizer=tokenizer,
        config=UL25Config.recommended(),
    )

    trainer = Trainer(
        model=model,
        data_collator=collator,
        train_dataset=dataset,
    )

Usage with DataLoader:
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collator,
        num_workers=4,
    )
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from torch import Tensor

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any

try:
    from transformers import PreTrainedTokenizerBase
    from transformers.data.data_collator import DataCollatorMixin

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    PreTrainedTokenizerBase = Any
    DataCollatorMixin = object


# =============================================================================
# CONFIGURATION
# =============================================================================


class Task(IntEnum):
    """Denoising task types."""

    SPAN = 0  # Standard T5 span corruption
    SPAN_MIDDLE = 1  # Position-biased (middle-heavy)
    PREFIX_RANDOM = 2  # Random prefix/suffix split
    PREFIX_SHORT = 3  # Long prefix, short target (QA-like)
    PREFIX_LONG = 4  # Short prefix, long target (generation)
    INFILLING = 5  # Middle-out (bidirectional context)


@dataclass
class DenoiserSpec:
    """Single denoiser configuration."""

    task: Task
    mu: float = 3.0  # Mean span length
    r: float = 0.15  # Noise density
    max_spans: int = 512  # Max corruption spans
    prefix: str = ""  # Task prefix token
    variable_r: bool = False  # Sample r from bounds
    r_bounds: Tuple[float, float] = (0.05, 0.50)

    def __repr__(self):
        return f"DenoiserSpec({self.task.name}, r={self.r}, μ={self.mu}, prefix='{self.prefix}')"


@dataclass
class UL25Config:
    """
    UL2.5 mixture configuration.

    Attributes:
        denoisers: List of DenoiserSpec defining the task mixture
        weights: Sampling weights (normalized automatically)
        curriculum_start: Optional starting weights for curriculum
        curriculum_end: Optional ending weights for curriculum
    """

    denoisers: List[DenoiserSpec] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    curriculum_start: Optional[List[float]] = None
    curriculum_end: Optional[List[float]] = None

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
        """
        Recommended configuration based on feasibility analysis.

        Mixture:
        - 30% span denoising (standard + middle-heavy)
        - 50% prefix LM variants
        - 20% infilling
        """
        return cls(
            denoisers=[
                DenoiserSpec(Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                DenoiserSpec(Task.SPAN, mu=8.0, r=0.15, prefix="[R]"),
                DenoiserSpec(Task.SPAN_MIDDLE, mu=16.0, r=0.20, prefix="[X]"),
                DenoiserSpec(Task.PREFIX_RANDOM, prefix="[S]"),
                DenoiserSpec(Task.PREFIX_SHORT, prefix="[S]"),
                DenoiserSpec(Task.PREFIX_LONG, prefix="[S]"),
                DenoiserSpec(Task.INFILLING, r=0.30, prefix="[I]"),
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
                DenoiserSpec(Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                DenoiserSpec(Task.SPAN, mu=8.0, r=0.15, prefix="[R]"),
                DenoiserSpec(Task.SPAN_MIDDLE, mu=16.0, r=0.20, prefix="[X]"),
                DenoiserSpec(Task.PREFIX_RANDOM, prefix="[S]"),
                DenoiserSpec(Task.PREFIX_SHORT, prefix="[S]"),
                DenoiserSpec(Task.PREFIX_LONG, prefix="[S]"),
                DenoiserSpec(Task.INFILLING, r=0.30, prefix="[I]"),
            ],
            weights=[0.10, 0.10, 0.10, 0.20, 0.15, 0.15, 0.20],
            curriculum_start=[0.25, 0.25, 0.10, 0.15, 0.10, 0.10, 0.05],  # Span-heavy
            curriculum_end=[0.05, 0.05, 0.10, 0.25, 0.20, 0.20, 0.15],  # Prefix-heavy
        )

    @classmethod
    def ul2_original(cls) -> "UL25Config":
        """Original UL2 7-denoiser mixture."""
        return cls(
            denoisers=[
                DenoiserSpec(Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                DenoiserSpec(Task.SPAN, mu=8.0, r=0.15, prefix="[R]"),
                DenoiserSpec(Task.PREFIX_RANDOM, prefix="[S]"),
                DenoiserSpec(Task.SPAN, mu=3.0, r=0.50, prefix="[X]"),
                DenoiserSpec(Task.SPAN, mu=8.0, r=0.15, prefix="[X]"),
                DenoiserSpec(Task.SPAN, mu=64.0, r=0.15, prefix="[X]"),
                DenoiserSpec(Task.SPAN, mu=64.0, r=0.50, prefix="[X]"),
            ],
            weights=[0.165, 0.165, 0.34, 0.0825, 0.0825, 0.0825, 0.0825],
        )

    @classmethod
    def t5_standard(cls) -> "UL25Config":
        """Standard T5 span corruption."""
        return cls(
            denoisers=[DenoiserSpec(Task.SPAN, mu=3.0, r=0.15)],
            weights=[1.0],
        )

    @classmethod
    def minimal(cls) -> "UL25Config":
        """Minimal config for testing."""
        return cls(
            denoisers=[
                DenoiserSpec(Task.SPAN, mu=3.0, r=0.15),
                DenoiserSpec(Task.PREFIX_RANDOM),
            ],
            weights=[0.5, 0.5],
        )


# =============================================================================
# MASKING FUNCTIONS
# =============================================================================

if TORCH_AVAILABLE:

    def _random_segmentation(
        n_items: int, n_segments: int, device: torch.device
    ) -> Tensor:
        """Partition n_items into n_segments non-empty segments."""
        if n_segments <= 0 or n_items <= 0:
            return torch.ones(1, dtype=torch.long, device=device)
        if n_segments >= n_items:
            return torch.ones(n_items, dtype=torch.long, device=device)

        dividers = torch.randperm(n_items - 1, device=device)[: n_segments - 1]
        dividers = torch.sort(dividers).values

        starts = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), dividers + 1]
        )
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
        num_noise = max(1, min(int(round(seq_len * r)), seq_len - 1))
        num_spans = max(1, min(max_spans, int(round(num_noise / mu))))
        num_keep = seq_len - num_noise

        noise_lens = _random_segmentation(num_noise, num_spans, device)
        keep_lens = _random_segmentation(num_keep, num_spans, device)

        # Interleave segments with random start to avoid edge bias
        n = min(len(noise_lens), len(keep_lens))
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        pos = 0

        # Convert to lists once to avoid repeated .item() calls
        noise_list = noise_lens.tolist()
        keep_list = keep_lens.tolist()

        # Randomize starting pattern to eliminate edge effect at seq end
        start_with_noise = torch.rand(1, device=device).item() < 0.5

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

    def middle_heavy_mask(seq_len: int, r: float, device: torch.device) -> Tensor:
        """Position-biased mask preferring middle positions."""
        num_noise = max(1, min(int(round(seq_len * r)), seq_len - 1))

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        center, sigma = seq_len / 2, seq_len / 4
        weights = torch.exp(-0.5 * ((positions - center) / sigma) ** 2)
        weights = weights / weights.sum()

        indices = torch.multinomial(weights, num_noise, replacement=False)
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[indices] = True
        return mask

    def prefix_lm_mask(
        seq_len: int, mode: str, device: torch.device
    ) -> Tuple[Tensor, int]:
        """Generate prefix LM mask with various split strategies."""
        if mode == "random":
            split = torch.randint(
                int(0.2 * seq_len), int(0.8 * seq_len) + 1, (1,), device=device
            ).item()
        elif mode == "short":
            frac = 0.05 + 0.10 * torch.rand(1, device=device).item()
            split = int((1 - frac) * seq_len)
        elif mode == "long":
            frac = 0.05 + 0.15 * torch.rand(1, device=device).item()
            split = int(frac * seq_len)
        else:
            split = int(0.75 * seq_len)

        split = max(1, min(split, seq_len - 1))
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[split:] = True
        return mask, split

    def infilling_mask(
        seq_len: int, hole_frac: float, device: torch.device
    ) -> Tuple[Tensor, int, int]:
        """Generate infilling mask (mask middle portion)."""
        hole_size = max(1, int(hole_frac * seq_len))
        min_start = int(0.1 * seq_len)
        max_start = max(min_start, int(0.9 * seq_len) - hole_size)

        if max_start <= min_start:
            hole_start = seq_len // 3
        else:
            hole_start = torch.randint(
                min_start, max_start + 1, (1,), device=device
            ).item()

        hole_end = min(hole_start + hole_size, seq_len)

        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[hole_start:hole_end] = True
        return mask, hole_start, hole_end

    def create_sentinel_ids(mask: Tensor, sentinel_start: int) -> Tensor:
        """Convert boolean mask to sentinel token IDs."""
        device = mask.device

        shifted = torch.cat(
            [torch.zeros(1, dtype=torch.bool, device=device), mask[:-1]]
        )
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
        prefix_ids: Optional[Tensor] = None,
        eos_id: Optional[int] = None,
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


# =============================================================================
# MAIN COLLATOR
# =============================================================================


class UL25DataCollator(DataCollatorMixin if HF_AVAILABLE else object):
    """
    UL2.5 Data Collator for encoder-decoder models.

    Compatible with HuggingFace Trainer and PyTorch DataLoader.

    Args:
        tokenizer: HuggingFace tokenizer with extra_id sentinel tokens
        config: UL25Config specifying the denoiser mixture
        max_length: Maximum encoder sequence length
        max_labels_length: Maximum decoder sequence length
        pad_to_multiple_of: Pad to multiple of this value (for tensor cores)
        return_tensors: Output format ("pt" for PyTorch)

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
        >>> tokenizer.add_special_tokens({"additional_special_tokens": ["[R]", "[S]", "[X]", "[I]"]})
        >>> collator = UL25DataCollator(tokenizer, UL25Config.recommended())
        >>> batch = collator([{"input_ids": [1, 2, 3, 4, 5]}])
    """

    return_tensors: str = "pt"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[UL25Config] = None,
        max_length: int = 512,
        max_labels_length: int = 128,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.tokenizer = tokenizer
        self.config = config or UL25Config.recommended()
        self.max_length = max_length
        self.max_labels_length = max_labels_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

        # Token IDs
        self.sentinel_start = self._get_sentinel_start()
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

        # Cache encoded prefixes (store as CPU tensors, move to device during forward)
        self._prefix_cache: Dict[str, Tensor] = {}
        for spec in self.config.denoisers:
            if spec.prefix and spec.prefix not in self._prefix_cache:
                ids = tokenizer.encode(spec.prefix, add_special_tokens=False)
                self._prefix_cache[spec.prefix] = torch.tensor(ids, dtype=torch.long)

        # Cache sampling weights tensor
        self._weights = torch.tensor(self.config.weights, dtype=torch.float32)

        # Training progress for curriculum
        self._progress = 0.0

    def _get_sentinel_start(self) -> int:
        """Get highest extra_id token ID.

        Detection order:
        1. Direct conversion of <extra_id_0> (most reliable for T5 tokenizers)
        2. Search special tokens for extra_id pattern (fallback)
        3. Hardcoded 32099 default with warning
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
        if self._progress == 0.0:
            return self._weights
        return torch.tensor(
            self.config.get_weights(self._progress), dtype=torch.float32
        )

    def _get_prefix(self, prefix: str, device: torch.device) -> Optional[Tensor]:
        """Get prefix tensor on correct device (caches per-device)."""
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
        self, seq_len: int, spec: DenoiserSpec, device: torch.device
    ) -> Tensor:
        """Generate corruption mask based on task type."""
        r = self._sample_r(spec)

        if spec.task == Task.SPAN:
            return span_corruption_mask(seq_len, r, spec.mu, spec.max_spans, device)
        elif spec.task == Task.SPAN_MIDDLE:
            return middle_heavy_mask(seq_len, r, device)
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
        self, input_ids: Tensor, denoiser_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """Process single sequence with UL2.5 denoising."""
        device = input_ids.device
        seq_len = input_ids.shape[0]

        # Use provided index or sample denoiser
        if denoiser_idx is None:
            weights = self._get_curriculum_weights()
            denoiser_idx = torch.multinomial(weights, 1).item()

        spec = self.config.denoisers[denoiser_idx]
        prefix_ids = self._get_prefix(spec.prefix, device)

        # Generate mask
        mask = self._generate_mask(seq_len, spec, device)

        # Create sentinels
        enc_sentinels = create_sentinel_ids(mask, self.sentinel_start)
        dec_sentinels = create_sentinel_ids(~mask, self.sentinel_start)

        # Apply masks
        encoder_ids = apply_sentinel_mask(
            input_ids, enc_sentinels, prefix_ids, self.eos_id
        )
        decoder_ids = apply_sentinel_mask(input_ids, dec_sentinels, None, self.eos_id)

        return {"encoder_ids": encoder_ids, "decoder_ids": decoder_ids}

    def _pad_length(self, length: int) -> int:
        """Round up to pad_to_multiple_of if specified."""
        if self.pad_to_multiple_of is None:
            return length
        return (
            (length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
        ) * self.pad_to_multiple_of

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        Process batch of examples (HuggingFace DataCollatorMixin interface).

        Args:
            examples: List of dicts with "input_ids"

        Returns:
            Dict with "input_ids", "attention_mask", "labels"
        """
        # Determine device
        first = examples[0]["input_ids"]
        device = first.device if isinstance(first, Tensor) else torch.device("cpu")

        # Batch sample all denoiser indices at once (single multinomial call)
        batch_size = len(examples)
        weights = self._get_curriculum_weights()
        denoiser_indices = (
            torch.multinomial(weights.expand(batch_size, -1), 1).squeeze(-1).tolist()
        )

        # Process examples
        processed = []
        for i, ex in enumerate(examples):
            ids = ex["input_ids"]
            if not isinstance(ids, Tensor):
                ids = torch.tensor(ids, dtype=torch.long, device=device)
            if ids.dim() == 2:
                ids = ids.squeeze(0)
            ids = ids.to(device)
            processed.append(self._process_single(ids, denoiser_indices[i]))

        # Compute padded lengths
        max_enc = min(
            self.max_length, max(p["encoder_ids"].shape[0] for p in processed)
        )
        max_dec = min(
            self.max_labels_length, max(p["decoder_ids"].shape[0] for p in processed)
        )
        max_enc = self._pad_length(max_enc)
        max_dec = self._pad_length(max_dec)

        batch_size = len(processed)

        # Allocate tensors
        input_ids = torch.full(
            (batch_size, max_enc), self.pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            (batch_size, max_enc), dtype=torch.long, device=device
        )
        labels = torch.full(
            (batch_size, max_dec), -100, dtype=torch.long, device=device
        )

        # Fill
        for i, p in enumerate(processed):
            enc, dec = p["encoder_ids"], p["decoder_ids"]
            enc_len, dec_len = min(enc.shape[0], max_enc), min(dec.shape[0], max_dec)

            input_ids[i, :enc_len] = enc[:enc_len]
            attention_mask[i, :enc_len] = 1
            labels[i, :dec_len] = dec[:dec_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """Collate batch of examples."""
        return self.torch_call(examples)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_ul25_collator(
    tokenizer: PreTrainedTokenizerBase,
    preset: str = "recommended",
    max_length: int = 512,
    max_labels_length: int = 128,
    **kwargs,
) -> UL25DataCollator:
    """
    Create UL2.5 collator with preset configuration.

    Args:
        tokenizer: HuggingFace tokenizer
        preset: One of "recommended", "ul2", "t5", "minimal"
        max_length: Max encoder length
        max_labels_length: Max decoder length
        **kwargs: Additional args for UL25DataCollator

    Returns:
        Configured UL25DataCollator
    """
    presets = {
        "recommended": UL25Config.recommended,
        "curriculum": UL25Config.recommended_with_curriculum,
        "ul2": UL25Config.ul2_original,
        "t5": UL25Config.t5_standard,
        "minimal": UL25Config.minimal,
    }

    if preset not in presets:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}"
        )

    config = presets[preset]()

    return UL25DataCollator(
        tokenizer=tokenizer,
        config=config,
        max_length=max_length,
        max_labels_length=max_labels_length,
        **kwargs,
    )


# =============================================================================
# TESTING
# =============================================================================


def _test():
    """Run tests if torch is available."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping tests.")
        return

    print("=" * 60)
    print("UL2.5 HuggingFace Collator - Tests")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Mock tokenizer
    class MockTokenizer:
        eos_token_id = 1
        pad_token_id = 0
        all_special_tokens = [f"<extra_id_{i}>" for i in range(100)]
        all_special_ids = list(range(32000, 32100))

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text[:10]]

    tokenizer = MockTokenizer()

    # Test all configs
    print("[TEST] Configuration presets")
    for name in ["recommended", "curriculum", "ul2", "t5", "minimal"]:
        config = getattr(
            UL25Config, name if name != "curriculum" else "recommended_with_curriculum"
        )()
        print(f"  {name}: {len(config.denoisers)} denoisers")

    # Test collator
    print("\n[TEST] Collator")
    collator = UL25DataCollator(tokenizer, UL25Config.recommended())

    examples = [
        {"input_ids": torch.randint(100, 1000, (64,), device=device)},
        {"input_ids": torch.randint(100, 1000, (48,), device=device)},
        {"input_ids": torch.randint(100, 1000, (72,), device=device)},
    ]

    batch = collator(examples)
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")

    # Test curriculum
    print("\n[TEST] Curriculum")
    collator.progress = 0.0
    w0 = collator.config.get_weights(0.0)
    collator.progress = 1.0
    w1 = collator.config.get_weights(1.0)
    print(f"  progress=0.0: weights={[f'{w:.2f}' for w in w0]}")
    print(f"  progress=1.0: weights={[f'{w:.2f}' for w in w1]}")

    # Benchmark
    print("\n[BENCHMARK]")
    import time

    n_batches, batch_size, seq_len = 50, 32, 256
    examples = [
        {"input_ids": torch.randint(100, 1000, (seq_len,), device=device)}
        for _ in range(batch_size)
    ]

    # Warmup
    for _ in range(3):
        _ = collator(examples)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_batches):
        _ = collator(examples)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"  {n_batches * batch_size / elapsed:.0f} samples/sec")
    print(f"  {n_batches * batch_size * seq_len / elapsed:,.0f} tokens/sec")

    print("\n✓ All tests passed")
    print("=" * 60)


if __name__ == "__main__":
    _test()
