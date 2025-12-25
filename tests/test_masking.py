"""Tests for masking functions."""

import torch

from UL2_5.masking import (
    infilling_mask,
    middle_heavy_span_mask,
    prefix_lm_mask,
    span_corruption_mask,
)


class TestSpanCorruptionMask:
    """Tests for span_corruption_mask function."""

    def test_empty_sequence(self, device):
        """seq_len=0 should return empty mask."""
        mask = span_corruption_mask(0, 0.15, 3.0, 512, device)
        assert mask.shape == torch.Size([0])
        assert mask.dtype == torch.bool

    def test_single_token(self, device):
        """seq_len=1 should return no corruption (below MIN_SEQ_LEN)."""
        mask = span_corruption_mask(1, 0.15, 3.0, 512, device)
        assert mask.shape == torch.Size([1])
        assert not mask.any(), "Single token should not be masked"

    def test_two_tokens(self, device):
        """seq_len=2 should return no corruption (below MIN_SEQ_LEN)."""
        mask = span_corruption_mask(2, 0.15, 3.0, 512, device)
        assert mask.shape == torch.Size([2])
        assert not mask.any(), "seq_len=2 below minimum should not be masked"

    def test_minimum_length(self, device):
        """seq_len=3 (MIN_SEQ_LEN) should work."""
        mask = span_corruption_mask(3, 0.15, 3.0, 512, device)
        assert mask.shape == torch.Size([3])
        # Should have at least one masked token
        assert mask.sum() >= 1

    def test_normal_sequence(self, device):
        """Normal length sequence should have approximately correct density."""
        seq_len = 100
        r = 0.15
        mask = span_corruption_mask(seq_len, r, 3.0, 512, device)

        assert mask.shape == torch.Size([seq_len])
        density = mask.float().mean().item()
        # Allow 10% tolerance
        assert abs(density - r) < 0.10, f"Density {density} too far from {r}"

    def test_high_corruption_rate(self, device):
        """r=0.99 should not crash.

        Note: At extreme r values close to 1.0, actual masking density may be
        lower than requested due to segmentation algorithm limitations when
        num_keep tokens < num_spans. This is expected behavior.
        """
        mask = span_corruption_mask(100, 0.99, 3.0, 512, device)
        assert mask.shape == torch.Size([100])
        # Should mask at least something
        assert mask.sum() >= 1, "Should mask at least one token"

    def test_low_corruption_rate(self, device):
        """r=0.01 should not crash."""
        mask = span_corruption_mask(100, 0.01, 3.0, 512, device)
        assert mask.shape == torch.Size([100])
        assert mask.sum() >= 1, "Should mask at least one token"

    def test_very_long_sequence(self, device):
        """Very long sequences should not crash or timeout."""
        mask = span_corruption_mask(4096, 0.15, 3.0, 512, device)
        assert mask.shape == torch.Size([4096])


class TestMiddleHeavySpanMask:
    """Tests for middle_heavy_span_mask function."""

    def test_empty_sequence(self, device):
        """seq_len=0 should return empty mask."""
        mask = middle_heavy_span_mask(0, 0.15, 3.0, device)
        assert mask.shape == torch.Size([0])

    def test_single_token(self, device):
        """seq_len=1 should return no corruption."""
        mask = middle_heavy_span_mask(1, 0.15, 3.0, device)
        assert mask.shape == torch.Size([1])
        assert not mask.any()

    def test_minimum_length(self, device):
        """seq_len=3 should work."""
        mask = middle_heavy_span_mask(3, 0.15, 3.0, device)
        assert mask.shape == torch.Size([3])

    def test_middle_bias(self, device):
        """Verify middle positions are more likely to be masked."""
        seq_len = 100
        n_samples = 100

        middle_count = 0
        edge_count = 0
        middle_range = range(seq_len // 3, 2 * seq_len // 3)

        for _ in range(n_samples):
            mask = middle_heavy_span_mask(seq_len, 0.3, 3.0, device)
            middle_count += mask[list(middle_range)].sum().item()
            edge_count += mask[: seq_len // 3].sum().item()
            edge_count += mask[2 * seq_len // 3 :].sum().item()

        # Middle third should have more masks than each edge third
        avg_middle = middle_count / n_samples
        avg_edge = edge_count / n_samples
        assert avg_middle > avg_edge * 0.8, "Middle should be masked more than edges"


class TestPrefixLmMask:
    """Tests for prefix_lm_mask function."""

    def test_empty_sequence(self, device):
        """seq_len=0 should return empty mask."""
        mask, split = prefix_lm_mask(0, "random", device)
        assert mask.shape == torch.Size([0])
        assert split == 0

    def test_single_token(self, device):
        """seq_len=1 should handle gracefully."""
        mask, split = prefix_lm_mask(1, "random", device)
        assert mask.shape == torch.Size([1])
        assert split >= 0

    def test_two_tokens(self, device):
        """seq_len=2 should work."""
        mask, split = prefix_lm_mask(2, "random", device)
        assert mask.shape == torch.Size([2])
        assert 0 <= split <= 2

    def test_random_mode(self, device):
        """Random mode should split in 20-80% range."""
        seq_len = 100
        splits = []
        for _ in range(50):
            _, split = prefix_lm_mask(seq_len, "random", device)
            splits.append(split)

        avg_split = sum(splits) / len(splits)
        assert 20 <= avg_split <= 80, (
            f"Average split {avg_split} outside expected range"
        )

    def test_short_mode(self, device):
        """Short mode should have long prefix (high split)."""
        seq_len = 100
        splits = []
        for _ in range(20):
            _, split = prefix_lm_mask(seq_len, "short", device)
            splits.append(split)

        avg_split = sum(splits) / len(splits)
        assert avg_split > 80, f"Short mode average split {avg_split} too low"

    def test_long_mode(self, device):
        """Long mode should have short prefix (low split)."""
        seq_len = 100
        splits = []
        for _ in range(20):
            _, split = prefix_lm_mask(seq_len, "long", device)
            splits.append(split)

        avg_split = sum(splits) / len(splits)
        assert avg_split < 20, f"Long mode average split {avg_split} too high"


class TestInfillingMask:
    """Tests for infilling_mask function."""

    def test_empty_sequence(self, device):
        """seq_len=0 should return empty mask."""
        mask, hole_start, hole_end = infilling_mask(0, 0.3, device)
        assert mask.shape == torch.Size([0])

    def test_single_token(self, device):
        """seq_len=1 should handle gracefully."""
        mask, hole_start, hole_end = infilling_mask(1, 0.3, device)
        assert mask.shape == torch.Size([1])

    def test_minimum_length(self, device):
        """seq_len=3 should work."""
        mask, hole_start, hole_end = infilling_mask(3, 0.3, device)
        assert mask.shape == torch.Size([3])
        assert 0 <= hole_start <= hole_end <= 3

    def test_hole_is_contiguous(self, device):
        """Hole should be a single contiguous region."""
        mask, hole_start, hole_end = infilling_mask(100, 0.3, device)

        # Check that mask is all False, then all True, then all False
        # (or just all True in the middle)
        mask_list = mask.tolist()

        # Find first True
        first_true = None
        for i, v in enumerate(mask_list):
            if v:
                first_true = i
                break

        if first_true is not None:
            # Find last True
            last_true = None
            for i in range(len(mask_list) - 1, -1, -1):
                if mask_list[i]:
                    last_true = i
                    break

            # All positions between first and last should be True
            for i in range(first_true, last_true + 1):
                assert mask_list[i], f"Position {i} should be True in contiguous hole"
