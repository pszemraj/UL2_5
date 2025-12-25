"""Tests for unpadding utilities."""

import pytest
import torch

from UL2_5.unpad import UnpadOutput, pad_input, unpad_input


class TestUnpadInput:
    """Tests for unpad_input function."""

    def test_basic_2d(self):
        """Basic unpadding of 2D token IDs."""
        inputs = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(inputs, mask)

        assert isinstance(out, UnpadOutput)
        assert out.hidden_states.tolist() == [1, 2, 3, 4, 5]
        assert out.cu_seqlens.tolist() == [0, 2, 5]
        assert out.max_seqlen == 3
        assert out.cu_seqlens.dtype == torch.int32

    def test_3d_hidden_states(self):
        """Unpadding of 3D hidden states."""
        batch, seqlen, hidden = 2, 4, 8
        inputs = torch.randn(batch, seqlen, hidden)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(inputs, mask)

        assert out.hidden_states.shape == (5, hidden)
        assert out.max_seqlen == 3

    def test_all_valid(self):
        """All positions valid (no padding)."""
        inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mask = torch.ones(2, 3, dtype=torch.long)

        out = unpad_input(inputs, mask)

        assert out.hidden_states.shape[0] == 6
        assert out.cu_seqlens.tolist() == [0, 3, 6]
        assert out.max_seqlen == 3

    def test_varying_lengths(self):
        """Sequences of varying lengths."""
        inputs = torch.tensor([[1, 0, 0, 0], [2, 3, 0, 0], [4, 5, 6, 7]])
        mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]])

        out = unpad_input(inputs, mask)

        assert out.hidden_states.tolist() == [1, 2, 3, 4, 5, 6, 7]
        assert out.cu_seqlens.tolist() == [0, 1, 3, 7]
        assert out.max_seqlen == 4

    def test_single_sequence(self):
        """Single sequence batch."""
        inputs = torch.tensor([[1, 2, 3, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0, 0]])

        out = unpad_input(inputs, mask)

        assert out.cu_seqlens.tolist() == [0, 3]
        assert out.max_seqlen == 3

    def test_indices_correct(self):
        """Indices correctly map to original positions."""
        inputs = torch.tensor([[10, 20, 0, 0], [30, 40, 50, 0]])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(inputs, mask)

        # Verify indices point to correct flattened positions
        # Row 0: positions 0, 1 (tokens 10, 20)
        # Row 1: positions 4, 5, 6 (tokens 30, 40, 50)
        assert out.indices.tolist() == [0, 1, 4, 5, 6]


class TestPadInput:
    """Tests for pad_input function."""

    def test_roundtrip_2d(self):
        """Unpad then pad should recover original (for valid positions)."""
        original = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(original, mask)
        recovered = pad_input(out.hidden_states, out.indices, 2, 4, pad_value=0)

        assert torch.equal(recovered, original)

    def test_roundtrip_3d(self):
        """Roundtrip for 3D hidden states."""
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

    def test_custom_pad_value(self):
        """Custom padding value."""
        unpadded = torch.tensor([1, 2, 3])
        indices = torch.tensor([0, 1, 4])

        recovered = pad_input(unpadded, indices, 2, 3, pad_value=-1)

        expected = torch.tensor([[1, 2, -1], [-1, 3, -1]])
        assert torch.equal(recovered, expected)

    def test_single_element(self):
        """Single element unpadded tensor."""
        unpadded = torch.tensor([42])
        indices = torch.tensor([0])

        recovered = pad_input(unpadded, indices, 1, 3, pad_value=0)

        expected = torch.tensor([[42, 0, 0]])
        assert torch.equal(recovered, expected)


class TestUnpadDevice:
    """Device placement tests."""

    def test_cpu_tensors(self):
        """Unpadding should work on CPU tensors."""
        inputs = torch.tensor([[1, 2, 0], [3, 4, 5]], device="cpu")
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], device="cpu")

        out = unpad_input(inputs, mask)

        assert out.hidden_states.device.type == "cpu"
        assert out.indices.device.type == "cpu"
        assert out.cu_seqlens.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensors(self):
        """Unpadding should work on CUDA tensors."""
        inputs = torch.tensor([[1, 2, 0], [3, 4, 5]], device="cuda")
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], device="cuda")

        out = unpad_input(inputs, mask)

        assert out.hidden_states.device.type == "cuda"
        assert out.indices.device.type == "cuda"
        assert out.cu_seqlens.device.type == "cuda"


class TestUnpadDtype:
    """Dtype preservation tests."""

    def test_cu_seqlens_int32(self):
        """cu_seqlens should always be int32 (Flash Attention requirement)."""
        inputs = torch.tensor([[1, 2, 0], [3, 4, 5]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        out = unpad_input(inputs, mask)

        assert out.cu_seqlens.dtype == torch.int32

    def test_preserves_input_dtype(self):
        """Hidden states should preserve input dtype."""
        inputs = torch.randn(2, 4, 8, dtype=torch.float16)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        out = unpad_input(inputs, mask)

        assert out.hidden_states.dtype == torch.float16

    def test_indices_long(self):
        """Indices should be long dtype."""
        inputs = torch.tensor([[1, 2, 0], [3, 4, 5]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        out = unpad_input(inputs, mask)

        assert out.indices.dtype == torch.long
