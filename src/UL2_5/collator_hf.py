"""HuggingFace Transformers-compatible UL2.5 Data Collator."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

from .config import DenoiserSpec, Task, UL25Config
from .masking import (
    infilling_mask,
    middle_heavy_span_mask,
    prefix_lm_mask,
    snap_mask_to_word_boundaries,
    span_corruption_mask,
)
from .sentinel import apply_sentinel_mask, create_sentinel_ids


class UL25DataCollator(DataCollatorMixin):
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
        return_task_info: If True, include task_indices in output for debugging

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
        config: UL25Config | None = None,
        max_length: int = 512,
        max_labels_length: int = 128,
        pad_to_multiple_of: int | None = None,
        return_tensors: str = "pt",
        return_task_info: bool = False,
    ):
        self.tokenizer = tokenizer
        self.config = config or UL25Config.recommended()
        self.max_length = max_length
        self.max_labels_length = max_labels_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.return_task_info = return_task_info

        # Token IDs
        self.sentinel_start = self._get_sentinel_start()
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

        # Cache encoded prefixes (store as CPU tensors, move to device during forward)
        self._prefix_cache: dict[str, Tensor] = {}
        for spec in self.config.denoisers:
            if spec.prefix and spec.prefix not in self._prefix_cache:
                ids = self._encode_prefix(spec.prefix)
                self._prefix_cache[spec.prefix] = torch.tensor(ids, dtype=torch.long)

        # Cache sampling weights tensor
        self._weights = torch.tensor(self.config.weights, dtype=torch.float32)

        # Weight caching for curriculum (invalidated on progress change)
        self._weight_cache: dict[tuple[float, str], Tensor] = {}

        # Training progress for curriculum
        self._progress = 0.0

    def _encode_prefix(self, prefix: str) -> list[int]:
        """
        Encode prefix token, preferring special token lookup.

        Checks if prefix is a registered special token first (single ID),
        falls back to encode() with a warning if multi-token encoding occurs.
        """
        if not prefix:
            return []

        # Try special token lookup first (preferred for [R], [S], [X], [I])
        try:
            token_id = self.tokenizer.convert_tokens_to_ids(prefix)
            unk_id = getattr(self.tokenizer, "unk_token_id", None)
            if token_id != unk_id and token_id is not None:
                return [token_id]
        except Exception:
            pass

        # Fallback to encode (may produce multiple tokens)
        ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        if len(ids) > 1:
            warnings.warn(
                f"Prefix '{prefix}' encoded to {len(ids)} tokens. "
                f"Consider adding it as a special token: "
                f"tokenizer.add_special_tokens({{'additional_special_tokens': ['{prefix}']}})",
                stacklevel=3,
            )
        return ids

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
            "No extra_id tokens found. Using 32099 as default sentinel start.",
            stacklevel=2,
        )
        return 32099

    @property
    def progress(self) -> float:
        """Current training progress (0-1) for curriculum."""
        return self._progress

    @progress.setter
    def progress(self, value: float):
        """Set training progress for curriculum learning."""
        new_progress = max(0.0, min(1.0, value))
        if new_progress != self._progress:
            self._weight_cache.clear()  # Invalidate cache on progress change
        self._progress = new_progress

    def _get_curriculum_weights(self, device: torch.device | None = None) -> Tensor:
        """Get denoiser sampling weights (cached per progress/device)."""
        device_key = str(device) if device else "cpu"
        cache_key = (round(self._progress, 4), device_key)

        if cache_key not in self._weight_cache:
            weights = self.config.get_weights(self._progress)
            tensor = torch.tensor(weights, dtype=torch.float32)
            if device is not None and device.type != "cpu":
                tensor = tensor.to(device)
            self._weight_cache[cache_key] = tensor

        return self._weight_cache[cache_key]

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

    def _get_length_adaptive_weights_batch(self, seq_lens: list[int]) -> Tensor:
        """Get per-example length-adaptive weights for a batch."""
        return torch.stack(
            [self._get_length_adaptive_weights(seq_len) for seq_len in seq_lens],
            dim=0,
        )

    def _get_prefix(self, prefix: str, device: torch.device) -> Tensor | None:
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
        self, input_ids: Tensor, denoiser_idx: int | None = None
    ) -> dict[str, Tensor]:
        """Process single sequence with UL2.5 denoising."""
        device = input_ids.device
        seq_len = input_ids.shape[0]

        # Use provided index or sample denoiser (with length-adaptive weights)
        if denoiser_idx is None:
            weights = self._get_length_adaptive_weights(seq_len)
            denoiser_idx = torch.multinomial(weights, 1).item()

        spec = self.config.denoisers[denoiser_idx]
        prefix_ids = self._get_prefix(spec.prefix, device)

        # Generate mask
        mask = self._generate_mask(seq_len, spec, device)

        # Apply boundary snapping if enabled and tokenizer supports it
        if getattr(self.config, "enable_boundary_snapping", True) and spec.task in (
            Task.SPAN,
            Task.SPAN_MIDDLE,
        ):
            if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                try:
                    mask = snap_mask_to_word_boundaries(mask, input_ids, self.tokenizer)
                except Exception:
                    pass  # Fall back to unsnapped mask

        # Create sentinels
        enc_sentinels = create_sentinel_ids(mask, self.sentinel_start)
        dec_sentinels = create_sentinel_ids(~mask, self.sentinel_start)

        # Apply masks
        encoder_ids = apply_sentinel_mask(
            input_ids, enc_sentinels, prefix_ids, self.eos_id
        )
        decoder_ids = apply_sentinel_mask(input_ids, dec_sentinels, None, self.eos_id)

        return {"encoder_ids": encoder_ids, "decoder_ids": decoder_ids}

    def _pad_length(self, length: int, cap: int | None = None) -> int:
        """Round up to pad_to_multiple_of if specified, capped at cap."""
        if self.pad_to_multiple_of is None:
            return length
        padded = (
            (length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
        ) * self.pad_to_multiple_of
        if cap is not None:
            padded = min(padded, cap)
        return padded

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        """
        Process batch of examples (HuggingFace DataCollatorMixin interface).

        Args:
            examples: List of dicts with "input_ids"

        Returns:
            Dict with "input_ids", "attention_mask", "labels", "decoder_input_ids"
        """
        # Determine device
        first = examples[0]["input_ids"]
        device = first.device if isinstance(first, Tensor) else torch.device("cpu")

        batch_size = len(examples)

        # Get sequence lengths to determine weights
        seq_lens = []
        for ex in examples:
            ids = ex["input_ids"]
            if isinstance(ids, Tensor):
                seq_lens.append(ids.shape[-1] if ids.dim() > 0 else 1)
            else:
                seq_lens.append(len(ids))

        # Sample denoiser indices per example (length-adaptive weights per sequence)
        weights = self._get_length_adaptive_weights_batch(seq_lens)
        denoiser_indices = torch.multinomial(weights, 1).squeeze(-1).tolist()

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
        max_enc = self._pad_length(max_enc, self.max_length)
        max_dec = self._pad_length(max_dec, self.max_labels_length)

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
        decoder_input_ids = torch.full(
            (batch_size, max_dec), self.pad_id, dtype=torch.long, device=device
        )

        # Fill
        for i, p in enumerate(processed):
            enc, dec = p["encoder_ids"], p["decoder_ids"]
            enc_len, dec_len = min(enc.shape[0], max_enc), min(dec.shape[0], max_dec)

            input_ids[i, :enc_len] = enc[:enc_len]
            attention_mask[i, :enc_len] = 1
            labels[i, :dec_len] = dec[:dec_len]

            # Right-shift for decoder_input_ids (T5-style teacher forcing)
            # Position 0 is already pad_id; fill positions 1:dec_len with labels[0:dec_len-1]
            if dec_len > 1:
                decoder_input_ids[i, 1:dec_len] = dec[: dec_len - 1]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }

        if self.return_task_info:
            result["task_indices"] = torch.tensor(
                denoiser_indices, dtype=torch.long, device=device
            )

        return result

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        """Collate batch of examples."""
        return self.torch_call(examples)


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
        preset: One of "recommended", "curriculum", "ul2", "t5", "minimal",
                "span_heavy", "flan_ul2"
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
        "span_heavy": UL25Config.span_heavy,
        "flan_ul2": UL25Config.flan_ul2_finetune,
        "all_features": UL25Config.all_features,
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
