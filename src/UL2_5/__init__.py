"""
UL2.5 Data Collator - GPU-optimized denoising for encoder-decoder models.

This package provides data collators implementing the UL2 mixture-of-denoisers
training paradigm for T5, FLAN, and other encoder-decoder models.

Usage:
    from UL2_5 import UL25DataCollator, UL25Config

    # With HuggingFace Trainer
    collator = UL25DataCollator(tokenizer, UL25Config.recommended())

    # Pure PyTorch (no transformers dependency for collator)
    from UL2_5.collator_torch import UL25DataCollator as TorchCollator
"""

# Core configuration (always available)
from .config import DenoiserSpec, Task, UL25Config

# Masking functions (for advanced usage)
from .masking import (
    infilling_mask,
    middle_heavy_span_mask,
    prefix_lm_mask,
    snap_mask_to_word_boundaries,
    span_corruption_mask,
)

# Sentinel processing
from .sentinel import apply_sentinel_mask, create_sentinel_ids

# Unpadding utilities (for Flash Attention varlen kernels)
from .unpad import UnpadOutput, pad_input, unpad_input

# Default to HF collator (most common use case)
try:
    from .collator_hf import UL25DataCollator, create_ul25_collator
except ImportError:
    # Fall back to torch-only collator if transformers not installed
    from .collator_torch import UL25DataCollator

    create_ul25_collator = None

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0")

__all__ = [
    # Config
    "Task",
    "DenoiserSpec",
    "UL25Config",
    # Collator
    "UL25DataCollator",
    "create_ul25_collator",
    # Masking
    "span_corruption_mask",
    "middle_heavy_span_mask",
    "prefix_lm_mask",
    "infilling_mask",
    "snap_mask_to_word_boundaries",
    # Sentinel
    "create_sentinel_ids",
    "apply_sentinel_mask",
    # Unpadding
    "unpad_input",
    "pad_input",
    "UnpadOutput",
]
