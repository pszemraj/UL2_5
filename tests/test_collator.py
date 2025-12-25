"""Tests for data collator."""

import pytest
import torch

from UL2_5 import UL25Config, UL25DataCollator


class TestUL25DataCollator:
    """Tests for UL25DataCollator class."""

    def test_basic_collation(self, mock_tokenizer):
        """Basic collation should work."""
        collator = UL25DataCollator(
            mock_tokenizer,
            UL25Config.minimal(),
            max_length=128,
            max_labels_length=64,
        )

        examples = [
            {"input_ids": torch.randint(100, 1000, (50,))},
            {"input_ids": torch.randint(100, 1000, (60,))},
        ]

        batch = collator(examples)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "decoder_input_ids" in batch

        assert batch["input_ids"].shape[0] == 2
        assert batch["attention_mask"].shape[0] == 2
        assert batch["labels"].shape[0] == 2
        assert batch["decoder_input_ids"].shape[0] == 2

    def test_decoder_input_ids_right_shift(self, mock_tokenizer):
        """decoder_input_ids should be right-shifted labels."""
        collator = UL25DataCollator(
            mock_tokenizer,
            UL25Config.t5_standard(),
            max_length=128,
            max_labels_length=64,
        )

        examples = [{"input_ids": torch.randint(100, 1000, (50,))}]
        batch = collator(examples)

        decoder_ids = batch["decoder_input_ids"][0]
        labels = batch["labels"][0]

        # First position should be pad_id
        assert decoder_ids[0].item() == mock_tokenizer.pad_token_id

        # Valid labels (not -100) should be shifted
        valid_labels = labels[labels != -100]
        if len(valid_labels) > 1:
            n = len(valid_labels)
            assert torch.equal(decoder_ids[1:n], valid_labels[:-1]), (
                "decoder_input_ids should be right-shifted labels"
            )

    def test_task_info(self, mock_tokenizer):
        """return_task_info should include task_indices."""
        collator = UL25DataCollator(
            mock_tokenizer,
            UL25Config.recommended(),
            max_length=128,
            max_labels_length=64,
            return_task_info=True,
        )

        examples = [
            {"input_ids": torch.randint(100, 1000, (50,))},
            {"input_ids": torch.randint(100, 1000, (60,))},
        ]

        batch = collator(examples)

        assert "task_indices" in batch
        assert batch["task_indices"].shape == torch.Size([2])
        assert batch["task_indices"].dtype == torch.long

    def test_max_length_respected(self, mock_tokenizer):
        """Output should not exceed max_length."""
        collator = UL25DataCollator(
            mock_tokenizer,
            UL25Config.minimal(),
            max_length=64,
            max_labels_length=32,
        )

        # Input longer than max_length
        examples = [{"input_ids": torch.randint(100, 1000, (200,))}]
        batch = collator(examples)

        assert batch["input_ids"].shape[1] <= 64
        assert batch["labels"].shape[1] <= 32

    def test_pad_to_multiple_with_cap(self, mock_tokenizer):
        """pad_to_multiple_of should not exceed max_length."""
        collator = UL25DataCollator(
            mock_tokenizer,
            UL25Config.minimal(),
            max_length=100,
            max_labels_length=50,
            pad_to_multiple_of=64,  # Would round to 128 without capping
        )

        examples = [{"input_ids": torch.randint(100, 1000, (90,))}]
        batch = collator(examples)

        assert batch["input_ids"].shape[1] <= 100

    def test_progress_property(self, mock_tokenizer):
        """progress property should work."""
        collator = UL25DataCollator(
            mock_tokenizer,
            UL25Config.recommended_with_curriculum(),
        )

        assert collator.progress == 0.0

        collator.progress = 0.5
        assert collator.progress == 0.5

        # Should clamp to [0, 1]
        collator.progress = 1.5
        assert collator.progress == 1.0

        collator.progress = -0.5
        assert collator.progress == 0.0

    def test_list_input_ids(self, mock_tokenizer):
        """Should handle list input_ids (not just tensors)."""
        collator = UL25DataCollator(
            mock_tokenizer,
            UL25Config.minimal(),
            max_length=128,
            max_labels_length=64,
        )

        examples = [
            {"input_ids": [100, 200, 300, 400, 500]},
            {"input_ids": [100, 200, 300]},
        ]

        batch = collator(examples)
        assert batch["input_ids"].shape[0] == 2

    def test_empty_batch(self, mock_tokenizer):
        """Empty batch should raise or handle gracefully."""
        collator = UL25DataCollator(
            mock_tokenizer,
            UL25Config.minimal(),
        )

        with pytest.raises(Exception):
            collator([])


class TestCollatorWithRealTokenizer:
    """Integration tests with real HuggingFace tokenizer."""

    @pytest.mark.slow
    def test_with_real_tokenizer(self, real_tokenizer):
        """Full integration test with real tokenizer."""
        collator = UL25DataCollator(
            real_tokenizer,
            UL25Config.recommended(),
            max_length=128,
            max_labels_length=64,
        )

        # Encode some text
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "Machine learning is transforming industries."

        examples = [
            {"input_ids": real_tokenizer.encode(text1)},
            {"input_ids": real_tokenizer.encode(text2)},
        ]

        batch = collator(examples)

        assert batch["input_ids"].shape[0] == 2
        assert batch["decoder_input_ids"].shape[0] == 2
        assert batch["labels"].shape[0] == 2

        # Check that outputs are valid token IDs (not negative except labels padding)
        assert (batch["input_ids"] >= 0).all()
        assert (batch["attention_mask"] >= 0).all()
        assert (batch["decoder_input_ids"] >= 0).all()


class TestUnpadCollator:
    """Tests for unpadding feature in collator."""

    def test_encoder_unpad(self, mock_tokenizer):
        """enable_unpad_encoder should add encoder unpad outputs."""
        config = UL25Config.minimal()
        config.enable_unpad_encoder = True

        collator = UL25DataCollator(
            mock_tokenizer,
            config,
            max_length=128,
            max_labels_length=64,
        )
        examples = [
            {"input_ids": torch.randint(100, 1000, (50,))},
            {"input_ids": torch.randint(100, 1000, (30,))},
        ]

        batch = collator(examples)

        # Standard outputs still present
        assert "input_ids" in batch
        assert "attention_mask" in batch

        # Unpad outputs present
        assert "input_ids_unpad" in batch
        assert "encoder_indices" in batch
        assert "encoder_cu_seqlens" in batch
        assert "encoder_max_seqlen" in batch

        # Verify shapes
        total_tokens = batch["attention_mask"].sum().item()
        assert batch["input_ids_unpad"].shape[0] == total_tokens
        assert batch["encoder_cu_seqlens"].shape[0] == 3  # batch_size + 1

    def test_decoder_unpad(self, mock_tokenizer):
        """enable_unpad_decoder should add decoder unpad outputs."""
        config = UL25Config.minimal()
        config.enable_unpad_decoder = True

        collator = UL25DataCollator(
            mock_tokenizer,
            config,
            max_length=128,
            max_labels_length=64,
        )
        examples = [
            {"input_ids": torch.randint(100, 1000, (50,))},
            {"input_ids": torch.randint(100, 1000, (30,))},
        ]

        batch = collator(examples)

        # Decoder unpad outputs present
        assert "decoder_input_ids_unpad" in batch
        assert "labels_unpad" in batch
        assert "decoder_indices" in batch
        assert "decoder_cu_seqlens" in batch
        assert "decoder_max_seqlen" in batch

    def test_both_unpad(self, mock_tokenizer):
        """Both encoder and decoder unpadding."""
        config = UL25Config.minimal()
        config.enable_unpad_encoder = True
        config.enable_unpad_decoder = True

        collator = UL25DataCollator(mock_tokenizer, config)
        examples = [{"input_ids": torch.randint(100, 1000, (50,))}]

        batch = collator(examples)

        assert "input_ids_unpad" in batch
        assert "decoder_input_ids_unpad" in batch

    def test_unpad_disabled_by_default(self, mock_tokenizer):
        """Unpadding should be disabled by default."""
        collator = UL25DataCollator(mock_tokenizer, UL25Config.minimal())
        examples = [{"input_ids": torch.randint(100, 1000, (50,))}]

        batch = collator(examples)

        assert "input_ids_unpad" not in batch
        assert "decoder_input_ids_unpad" not in batch

    def test_flash_attention_preset(self, mock_tokenizer):
        """flash_attention preset should have unpadding enabled."""
        config = UL25Config.flash_attention()

        assert config.enable_unpad_encoder is True
        assert config.enable_unpad_decoder is True

        collator = UL25DataCollator(mock_tokenizer, config)
        examples = [{"input_ids": torch.randint(100, 1000, (50,))}]

        batch = collator(examples)

        assert "input_ids_unpad" in batch
        assert "decoder_input_ids_unpad" in batch

    def test_cu_seqlens_dtype(self, mock_tokenizer):
        """cu_seqlens must be int32 for Flash Attention kernels."""
        config = UL25Config.minimal()
        config.enable_unpad_encoder = True
        config.enable_unpad_decoder = True

        collator = UL25DataCollator(mock_tokenizer, config)
        examples = [{"input_ids": torch.randint(100, 1000, (50,))}]

        batch = collator(examples)

        assert batch["encoder_cu_seqlens"].dtype == torch.int32
        assert batch["decoder_cu_seqlens"].dtype == torch.int32

    def test_unpad_roundtrip(self, mock_tokenizer):
        """Unpadded tensors should reconstruct original when re-padded."""
        from UL2_5 import pad_input

        config = UL25Config.minimal()
        config.enable_unpad_encoder = True

        collator = UL25DataCollator(mock_tokenizer, config)
        examples = [
            {"input_ids": torch.randint(100, 1000, (50,))},
            {"input_ids": torch.randint(100, 1000, (30,))},
        ]

        batch = collator(examples)

        # Re-pad and compare
        batch_size = batch["input_ids"].shape[0]
        seqlen = batch["input_ids"].shape[1]
        recovered = pad_input(
            batch["input_ids_unpad"],
            batch["encoder_indices"],
            batch_size,
            seqlen,
            pad_value=mock_tokenizer.pad_token_id,
        )

        assert torch.equal(recovered, batch["input_ids"])
