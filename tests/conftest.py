"""Shared fixtures for UL2.5 tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def cuda_device():
    """Return CUDA device if available, skip test otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class MockTokenizer:
    """Mock tokenizer for testing without HuggingFace dependency."""

    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = 2

    # Simulate extra_id tokens like T5
    all_special_tokens = ["<pad>", "<eos>", "<unk>"] + [
        f"<extra_id_{i}>" for i in range(100)
    ]
    all_special_ids = [0, 1, 2] + list(range(32099, 31999, -1))

    def encode(self, text, add_special_tokens=False):
        """Encode text to token IDs (simple character-based)."""
        return [ord(c) % 1000 + 100 for c in text[:50]]

    def convert_tokens_to_ids(self, token):
        """Convert token string to ID."""
        if token == "[R]":
            return 32100
        elif token == "[S]":
            return 32101
        elif token == "[X]":
            return 32102
        elif token == "[I]":
            return 32103
        elif token.startswith("<extra_id_"):
            try:
                idx = int(token.split("_")[-1].rstrip(">"))
                return 32099 - idx
            except ValueError:
                return self.unk_token_id
        return self.unk_token_id

    def convert_ids_to_tokens(self, ids):
        """Convert IDs to token strings."""
        if isinstance(ids, int):
            if 32000 <= ids <= 32099:
                return f"<extra_id_{32099 - ids}>"
            return f"▁tok_{ids}"
        return [self.convert_ids_to_tokens(i) for i in ids]


@pytest.fixture
def mock_tokenizer():
    """Return mock tokenizer for testing."""
    return MockTokenizer()


@pytest.fixture
def real_tokenizer():
    """Return real T5 tokenizer for integration tests."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[R]", "[S]", "[X]", "[I]"]}
    )
    return tokenizer
