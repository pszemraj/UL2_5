"""Tests for unpadding utilities."""

import pytest
import torch

from UL2_5.unpad import UnpadOutput, pad_input, unpad_input


class TestUnpadInput:
    """Tests for unpad_input function."""

    def test_basic_2d(self) -> None:
        """Verify basic unpadding of 2D token IDs produces correct output."""
        inputs = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(inputs, mask)

        assert isinstance(out, UnpadOutput)
        assert out.hidden_states.tolist() == [1, 2, 3, 4, 5]
        assert out.cu_seqlens.tolist() == [0, 2, 5]
        assert out.max_seqlen == 3
        assert out.cu_seqlens.dtype == torch.int32

    def test_3d_hidden_states(self) -> None:
        """Verify unpadding of 3D hidden states preserves hidden dimension."""
        batch, seqlen, hidden = 2, 4, 8
        inputs = torch.randn(batch, seqlen, hidden)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(inputs, mask)

        assert out.hidden_states.shape == (5, hidden)
        assert out.max_seqlen == 3

    def test_all_valid(self) -> None:
        """Verify handling when all positions are valid (no padding)."""
        inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mask = torch.ones(2, 3, dtype=torch.long)

        out = unpad_input(inputs, mask)

        assert out.hidden_states.shape[0] == 6
        assert out.cu_seqlens.tolist() == [0, 3, 6]
        assert out.max_seqlen == 3

    def test_varying_lengths(self) -> None:
        """Verify handling of sequences with varying lengths."""
        inputs = torch.tensor([[1, 0, 0, 0], [2, 3, 0, 0], [4, 5, 6, 7]])
        mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]])

        out = unpad_input(inputs, mask)

        assert out.hidden_states.tolist() == [1, 2, 3, 4, 5, 6, 7]
        assert out.cu_seqlens.tolist() == [0, 1, 3, 7]
        assert out.max_seqlen == 4

    def test_single_sequence(self) -> None:
        """Verify handling of single sequence batch."""
        inputs = torch.tensor([[1, 2, 3, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0, 0]])

        out = unpad_input(inputs, mask)

        assert out.cu_seqlens.tolist() == [0, 3]
        assert out.max_seqlen == 3

    def test_indices_correct(self) -> None:
        """Verify indices correctly map to original flattened positions."""
        inputs = torch.tensor([[10, 20, 0, 0], [30, 40, 50, 0]])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(inputs, mask)

        # Row 0: positions 0, 1 (tokens 10, 20)
        # Row 1: positions 4, 5, 6 (tokens 30, 40, 50)
        assert out.indices.tolist() == [0, 1, 4, 5, 6]

    def test_non_contiguous_3d(self) -> None:
        """Verify unpadding handles non-contiguous tensors from transpose/permute."""
        batch, seqlen, hidden = 2, 4, 8
        # Create non-contiguous tensor via transpose
        inputs = torch.randn(batch, hidden, seqlen).transpose(
            1, 2
        )  # (batch, seqlen, hidden)
        assert not inputs.is_contiguous(), "Test requires non-contiguous input"
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(inputs, mask)

        assert out.hidden_states.shape == (5, hidden)
        assert out.max_seqlen == 3
        # Verify values match expected positions
        assert torch.allclose(out.hidden_states[0], inputs[0, 0])
        assert torch.allclose(out.hidden_states[1], inputs[0, 1])
        assert torch.allclose(out.hidden_states[2], inputs[1, 0])


class TestPadInput:
    """Tests for pad_input function."""

    def test_roundtrip_2d(self) -> None:
        """Verify unpad then pad recovers original for valid positions."""
        original = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(original, mask)
        recovered = pad_input(out.hidden_states, out.indices, 2, 4, pad_value=0)

        assert torch.equal(recovered, original)

    def test_roundtrip_3d(self) -> None:
        """Verify roundtrip for 3D hidden states preserves valid positions."""
        batch, seqlen, hidden = 2, 4, 8
        original = torch.randn(batch, seqlen, hidden)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(original, mask)
        recovered = pad_input(out.hidden_states, out.indices, batch, seqlen)

        # Valid positions should match
        assert torch.allclose(recovered[0, :2], original[0, :2])
        assert torch.allclose(recovered[1, :3], original[1, :3])
        # Padding positions should be zero
        assert torch.all(recovered[0, 2:] == 0)
        assert torch.all(recovered[1, 3:] == 0)

    def test_custom_pad_value(self) -> None:
        """Verify custom padding value is applied correctly."""
        unpadded = torch.tensor([1, 2, 3])
        indices = torch.tensor([0, 1, 4])

        recovered = pad_input(unpadded, indices, 2, 3, pad_value=-1)

        expected = torch.tensor([[1, 2, -1], [-1, 3, -1]])
        assert torch.equal(recovered, expected)

    def test_single_element(self) -> None:
        """Verify handling of single element unpadded tensor."""
        unpadded = torch.tensor([42])
        indices = torch.tensor([0])

        recovered = pad_input(unpadded, indices, 1, 3, pad_value=0)

        expected = torch.tensor([[42, 0, 0]])
        assert torch.equal(recovered, expected)


class TestUnpadDevice:
    """Device placement tests for unpadding functions."""

    def test_cpu_tensors(self) -> None:
        """Verify unpadding preserves CPU device placement."""
        inputs = torch.tensor([[1, 2, 0], [3, 4, 5]], device="cpu")
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], device="cpu")

        out = unpad_input(inputs, mask)

        assert out.hidden_states.device.type == "cpu"
        assert out.indices.device.type == "cpu"
        assert out.cu_seqlens.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensors(self) -> None:
        """Verify unpadding preserves CUDA device placement."""
        inputs = torch.tensor([[1, 2, 0], [3, 4, 5]], device="cuda")
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], device="cuda")

        out = unpad_input(inputs, mask)

        assert out.hidden_states.device.type == "cuda"
        assert out.indices.device.type == "cuda"
        assert out.cu_seqlens.device.type == "cuda"


class TestUnpadEdgeCases:
    """Edge case tests for encoder-decoder training dynamics."""

    def test_empty_sequence_in_batch(self) -> None:
        """Verify handling when one sequence is all padding."""
        inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mask = torch.tensor([[1, 1, 1], [0, 0, 0]])  # Second seq is all padding

        out = unpad_input(inputs, mask)

        # Only first sequence's tokens
        assert out.hidden_states.tolist() == [1, 2, 3]
        # cu_seqlens has repeat for empty sequence
        assert out.cu_seqlens.tolist() == [0, 3, 3]
        assert out.max_seqlen == 3

    def test_4d_multi_head_tensors(self) -> None:
        """Verify handling of 4D tensors (batch, seqlen, heads, dim)."""
        batch, seqlen, heads, dim = 2, 4, 8, 64
        inputs = torch.randn(batch, seqlen, heads, dim)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(inputs, mask)

        assert out.hidden_states.shape == (5, heads, dim)
        # Verify values
        assert torch.allclose(out.hidden_states[0], inputs[0, 0])
        assert torch.allclose(out.hidden_states[4], inputs[1, 2])

    def test_4d_roundtrip(self) -> None:
        """Verify roundtrip for 4D tensors preserves values."""
        batch, seqlen, heads, dim = 2, 4, 8, 64
        original = torch.randn(batch, seqlen, heads, dim)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(original, mask)
        recovered = pad_input(out.hidden_states, out.indices, batch, seqlen)

        assert recovered.shape == original.shape
        assert torch.allclose(recovered[0, :2], original[0, :2])
        assert torch.allclose(recovered[1, :3], original[1, :3])

    def test_boolean_mask(self) -> None:
        """Verify handling of boolean attention masks."""
        inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mask = torch.tensor([[True, True, False], [True, True, True]])

        out = unpad_input(inputs, mask)

        assert out.hidden_states.tolist() == [1, 2, 4, 5, 6]

    def test_non_contiguous_2d(self) -> None:
        """Verify unpadding handles non-contiguous 2D tensors."""
        # Create non-contiguous via transpose
        inputs = torch.arange(8).reshape(4, 2).T  # (2, 4) non-contiguous
        assert not inputs.is_contiguous()
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(inputs, mask)

        assert out.hidden_states.shape == (5,)
        # Values should match the non-contiguous layout
        assert out.hidden_states[0] == inputs[0, 0]
        assert out.hidden_states[1] == inputs[0, 1]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency(self) -> None:
        """Verify all outputs are on same device as inputs, even with cross-device mask."""
        inputs = torch.tensor([[1, 2, 3]], device="cuda")
        mask = torch.tensor([[1, 1, 0]], device="cpu")

        out = unpad_input(inputs, mask)

        # All outputs should be on inputs' device
        assert out.hidden_states.device.type == "cuda"
        assert out.indices.device.type == "cuda"
        assert out.cu_seqlens.device.type == "cuda"

    def test_bfloat16_dtype(self) -> None:
        """Verify bfloat16 is preserved (common in modern training)."""
        inputs = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(inputs, mask)

        assert out.hidden_states.dtype == torch.bfloat16


class TestUnpadDtype:
    """Dtype handling tests for unpadding functions."""

    def test_cu_seqlens_int32(self) -> None:
        """Verify cu_seqlens is always int32 for Flash Attention compatibility."""
        inputs = torch.tensor([[1, 2, 0], [3, 4, 5]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        out = unpad_input(inputs, mask)

        assert out.cu_seqlens.dtype == torch.int32

    def test_preserves_input_dtype(self) -> None:
        """Verify hidden states preserve input dtype (e.g., float16)."""
        inputs = torch.randn(2, 4, 8, dtype=torch.float16)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(inputs, mask)

        assert out.hidden_states.dtype == torch.float16

    def test_indices_long(self) -> None:
        """Verify indices tensor uses long dtype for indexing."""
        inputs = torch.tensor([[1, 2, 0], [3, 4, 5]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        out = unpad_input(inputs, mask)

        assert out.indices.dtype == torch.long
