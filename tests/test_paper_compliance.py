"""Tests for UL2 paper compliance."""

import pytest

from UL2_5 import Task, UL25Config


class TestUL2PaperCompliance:
    """Tests verifying UL2 paper semantics."""

    def test_ul2_original_weights_match_paper(self):
        """Verify ul2_original() weights match Table 2 from paper."""
        config = UL25Config.ul2_original()

        # Paper specifies: R=33%, S=34%, X=33%
        r_weight = sum(
            w for d, w in zip(config.denoisers, config.weights) if d.prefix == "[R]"
        )
        s_weight = sum(
            w for d, w in zip(config.denoisers, config.weights) if d.prefix == "[S]"
        )
        x_weight = sum(
            w for d, w in zip(config.denoisers, config.weights) if d.prefix == "[X]"
        )

        assert abs(r_weight - 0.33) < 0.01, f"R-denoiser weight {r_weight} should be ~33%"
        assert abs(s_weight - 0.34) < 0.01, f"S-denoiser weight {s_weight} should be ~34%"
        assert abs(x_weight - 0.33) < 0.01, f"X-denoiser weight {x_weight} should be ~33%"

    def test_ul2_original_denoiser_count(self):
        """UL2 paper specifies 7 denoisers."""
        config = UL25Config.ul2_original()
        assert len(config.denoisers) == 7

    def test_x_denoiser_semantics(self):
        """Verify [X] is only used for extreme settings (r>=50% OR mu>=12)."""
        config = UL25Config.recommended()

        for d in config.denoisers:
            if d.prefix == "[X]":
                is_extreme = d.r >= 0.5 or d.mu >= 12
                assert is_extreme, (
                    f"[X] denoiser should have r>=50% or mu>=12, "
                    f"got r={d.r}, mu={d.mu}"
                )

    def test_r_denoiser_semantics(self):
        """Verify [R] is only used for regular settings (r<50% AND mu<12)."""
        config = UL25Config.recommended()

        for d in config.denoisers:
            if d.prefix == "[R]":
                is_regular = d.r < 0.5 and d.mu < 12
                assert is_regular, (
                    f"[R] denoiser should have r<50% and mu<12, "
                    f"got r={d.r}, mu={d.mu}"
                )

    def test_s_denoiser_is_prefix_lm(self):
        """Verify [S] is only used for prefix LM tasks."""
        config = UL25Config.recommended()

        prefix_tasks = {Task.PREFIX_RANDOM, Task.PREFIX_SHORT, Task.PREFIX_LONG}

        for d in config.denoisers:
            if d.prefix == "[S]":
                assert d.task in prefix_tasks, (
                    f"[S] denoiser should be prefix LM task, got {d.task}"
                )

    def test_ul2_original_r_denoisers(self):
        """R-denoisers should have r=15%, mu in {3, 8}."""
        config = UL25Config.ul2_original()

        r_denoisers = [d for d in config.denoisers if d.prefix == "[R]"]
        assert len(r_denoisers) == 2

        mus = {d.mu for d in r_denoisers}
        assert mus == {3.0, 8.0}, f"R-denoiser mus should be {{3, 8}}, got {mus}"

        for d in r_denoisers:
            assert d.r == 0.15, f"R-denoiser r should be 0.15, got {d.r}"

    def test_ul2_original_has_extreme_spans(self):
        """X-denoisers should include extreme settings."""
        config = UL25Config.ul2_original()

        x_denoisers = [d for d in config.denoisers if d.prefix == "[X]"]
        assert len(x_denoisers) == 4

        # Should have at least one with r>=50%
        has_high_r = any(d.r >= 0.5 for d in x_denoisers)
        assert has_high_r, "X-denoisers should include high corruption rate"

        # Should have at least one with mu>=12
        has_long_mu = any(d.mu >= 12 for d in x_denoisers)
        assert has_long_mu, "X-denoisers should include long mean span"


class TestUL25Extensions:
    """Tests for UL2.5 extensions (not in original paper)."""

    def test_infilling_task_exists(self):
        """recommended() should include infilling task."""
        config = UL25Config.recommended()

        infill_denoisers = [d for d in config.denoisers if d.task == Task.INFILLING]
        assert len(infill_denoisers) >= 1, "Should have at least one infilling denoiser"

    def test_infilling_uses_i_prefix(self):
        """Infilling tasks should use [I] prefix."""
        config = UL25Config.recommended()

        for d in config.denoisers:
            if d.task == Task.INFILLING:
                assert d.prefix == "[I]", f"Infilling should use [I] prefix, got '{d.prefix}'"

    def test_middle_heavy_exists(self):
        """recommended() should include middle-heavy span task."""
        config = UL25Config.recommended()

        middle_denoisers = [d for d in config.denoisers if d.task == Task.SPAN_MIDDLE]
        assert len(middle_denoisers) >= 1, "Should have at least one middle-heavy denoiser"

    def test_flan_ul2_no_prefixes(self):
        """flan_ul2_finetune() should have no mode tokens."""
        config = UL25Config.flan_ul2_finetune()

        for d in config.denoisers:
            assert d.prefix == "", (
                f"Flan-UL2 config should have no prefixes, "
                f"found '{d.prefix}' on {d.task}"
            )
