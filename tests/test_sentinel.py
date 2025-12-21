"""Tests for sentinel token processing."""

import pytest
import torch

from UL2_5.sentinel import apply_sentinel_mask, create_sentinel_ids


class TestCreateSentinelIds:
    """Tests for create_sentinel_ids function."""

    def test_empty_mask(self, device):
        """Empty mask should return empty tensor."""
        mask = torch.zeros(0, dtype=torch.bool, device=device)
        sentinel_ids = create_sentinel_ids(mask, 32099)
        assert sentinel_ids.shape == torch.Size([0])

    def test_no_masked_positions(self, device):
        """All-False mask should return all zeros."""
        mask = torch.zeros(10, dtype=torch.bool, device=device)
        sentinel_ids = create_sentinel_ids(mask, 32099)
        assert (sentinel_ids == 0).all()

    def test_single_span(self, device):
        """Single span should get one sentinel."""
        mask = torch.tensor([False, True, True, False, False], dtype=torch.bool)
        sentinel_ids = create_sentinel_ids(mask, 32099)

        # First masked position should be sentinel_start (32099)
        assert sentinel_ids[1] == 32099
        # Continuation should be -1
        assert sentinel_ids[2] == -1
        # Non-masked should be 0
        assert sentinel_ids[0] == 0
        assert sentinel_ids[3] == 0
        assert sentinel_ids[4] == 0

    def test_multiple_spans(self, device):
        """Multiple spans should get decreasing sentinel IDs."""
        mask = torch.tensor(
            [True, True, False, True, False, True, True, True],
            dtype=torch.bool,
        )
        sentinel_ids = create_sentinel_ids(mask, 32099)

        # First span starts at 0 -> sentinel 32099
        assert sentinel_ids[0] == 32099
        assert sentinel_ids[1] == -1  # continuation

        # Second span starts at 3 -> sentinel 32098
        assert sentinel_ids[3] == 32098

        # Third span starts at 5 -> sentinel 32097
        assert sentinel_ids[5] == 32097
        assert sentinel_ids[6] == -1  # continuation
        assert sentinel_ids[7] == -1  # continuation

    def test_overflow_protection(self, device):
        """More spans than sentinels should be clamped with warning."""
        # Create mask with many single-token spans (150 spans)
        mask_list = []
        for _ in range(150):
            mask_list.extend([True, False])
        mask = torch.tensor(mask_list[:300], dtype=torch.bool, device=device)

        with pytest.warns(UserWarning, match="Clamping"):
            sentinel_ids = create_sentinel_ids(mask, 32099, max_sentinels=100)

        # Sentinel IDs should not go below 32099 - 100 + 1 = 32000
        span_starts = (sentinel_ids > 0)
        if span_starts.any():
            min_sentinel = sentinel_ids[span_starts].min().item()
            assert min_sentinel >= 32000, f"Sentinel {min_sentinel} below minimum"

    def test_max_sentinels_parameter(self, device):
        """max_sentinels parameter should limit sentinel range."""
        # Create 10 spans
        mask = torch.tensor([True, False] * 10, dtype=torch.bool, device=device)

        sentinel_ids = create_sentinel_ids(mask, 32099, max_sentinels=5)

        # Should only use 5 sentinels (32099, 32098, 32097, 32096, 32095)
        span_starts = sentinel_ids[sentinel_ids > 0]
        if len(span_starts) > 0:
            assert span_starts.min() >= 32095


class TestApplySentinelMask:
    """Tests for apply_sentinel_mask function."""

    def test_basic_application(self, device):
        """Basic sentinel application."""
        input_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long, device=device)
        sentinel_ids = torch.tensor([0, 32099, -1, 0, 0], dtype=torch.long, device=device)

        result = apply_sentinel_mask(input_ids, sentinel_ids)

        # Should be: [10, 32099, 40, 50] (20 replaced with 32099, 30 filtered out)
        expected = torch.tensor([10, 32099, 40, 50], dtype=torch.long, device=device)
        assert torch.equal(result, expected)

    def test_with_prefix(self, device):
        """Prefix should be prepended."""
        input_ids = torch.tensor([10, 20, 30], dtype=torch.long, device=device)
        sentinel_ids = torch.tensor([32099, -1, 0], dtype=torch.long, device=device)
        prefix_ids = torch.tensor([100, 101], dtype=torch.long, device=device)

        result = apply_sentinel_mask(input_ids, sentinel_ids, prefix_ids=prefix_ids)

        # Should be: [100, 101, 32099, 30]
        assert result[0] == 100
        assert result[1] == 101
        assert result[2] == 32099
        assert result[3] == 30

    def test_with_eos(self, device):
        """EOS should be appended."""
        input_ids = torch.tensor([10, 20, 30], dtype=torch.long, device=device)
        sentinel_ids = torch.tensor([0, 32099, 0], dtype=torch.long, device=device)

        result = apply_sentinel_mask(input_ids, sentinel_ids, eos_id=1)

        # Should end with EOS
        assert result[-1] == 1

    def test_all_masked(self, device):
        """All positions masked should result in just sentinels."""
        input_ids = torch.tensor([10, 20, 30], dtype=torch.long, device=device)
        sentinel_ids = torch.tensor([32099, -1, -1], dtype=torch.long, device=device)

        result = apply_sentinel_mask(input_ids, sentinel_ids)

        # Should be just [32099] (continuations filtered)
        assert len(result) == 1
        assert result[0] == 32099
