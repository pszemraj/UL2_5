"""Tests for configuration classes."""

import pytest

from UL2_5 import DenoiserSpec, Task, UL25Config


class TestDenoiserSpec:
    """Tests for DenoiserSpec class."""

    def test_default_values(self):
        """Default values should be valid."""
        spec = DenoiserSpec(task=Task.SPAN)
        assert spec.mu == 3.0
        assert spec.r == 0.15
        assert spec.max_spans == 512
        assert spec.prefix == ""
        assert spec.variable_r is False

    def test_r_bounds_validation(self):
        """r_bounds should be validated."""
        # Valid bounds
        spec = DenoiserSpec(task=Task.SPAN, r_bounds=(0.1, 0.5))
        assert spec.r_bounds == (0.1, 0.5)

        # Invalid: lower >= upper
        with pytest.raises(ValueError, match="r_bounds"):
            DenoiserSpec(task=Task.SPAN, r_bounds=(0.5, 0.5))

        with pytest.raises(ValueError, match="r_bounds"):
            DenoiserSpec(task=Task.SPAN, r_bounds=(0.6, 0.5))

    def test_r_validation(self):
        """r should be in valid range."""
        # Too low
        with pytest.raises(ValueError):
            DenoiserSpec(task=Task.SPAN, r=0.001)

        # Too high
        with pytest.raises(ValueError):
            DenoiserSpec(task=Task.SPAN, r=0.999)

    def test_mu_validation(self):
        """mu should be positive."""
        with pytest.raises(ValueError):
            DenoiserSpec(task=Task.SPAN, mu=0)

        with pytest.raises(ValueError):
            DenoiserSpec(task=Task.SPAN, mu=-1)


class TestUL25Config:
    """Tests for UL25Config class."""

    def test_weights_required(self):
        """Weights must be provided with denoisers."""
        with pytest.raises(ValueError, match="weights must be provided"):
            UL25Config(
                denoisers=[DenoiserSpec(task=Task.SPAN)],
                weights=[],  # Empty weights
            )

    def test_weights_normalized(self):
        """Weights should be normalized to sum to 1."""
        config = UL25Config(
            denoisers=[
                DenoiserSpec(task=Task.SPAN),
                DenoiserSpec(task=Task.PREFIX_RANDOM),
            ],
            weights=[1, 3],  # Will be normalized to [0.25, 0.75]
        )
        assert abs(sum(config.weights) - 1.0) < 1e-6
        assert abs(config.weights[0] - 0.25) < 1e-6
        assert abs(config.weights[1] - 0.75) < 1e-6

    def test_curriculum_both_required(self):
        """curriculum_start and curriculum_end must both be set or neither."""
        with pytest.raises(ValueError, match="must both be set"):
            UL25Config(
                denoisers=[DenoiserSpec(task=Task.SPAN)],
                weights=[1.0],
                curriculum_start=[1.0],
                # curriculum_end missing
            )

    def test_get_weights_no_curriculum(self):
        """get_weights without curriculum should return static weights."""
        config = UL25Config(
            denoisers=[DenoiserSpec(task=Task.SPAN)],
            weights=[1.0],
        )
        assert config.get_weights(0.0) == [1.0]
        assert config.get_weights(0.5) == [1.0]
        assert config.get_weights(1.0) == [1.0]

    def test_get_weights_with_curriculum(self):
        """get_weights with curriculum should interpolate."""
        config = UL25Config(
            denoisers=[
                DenoiserSpec(task=Task.SPAN),
                DenoiserSpec(task=Task.PREFIX_RANDOM),
            ],
            weights=[0.5, 0.5],
            curriculum_start=[0.8, 0.2],
            curriculum_end=[0.2, 0.8],
        )

        # At progress=0, should be curriculum_start
        weights_0 = config.get_weights(0.0)
        assert abs(weights_0[0] - 0.8) < 1e-6
        assert abs(weights_0[1] - 0.2) < 1e-6

        # At progress=1, should be curriculum_end
        weights_1 = config.get_weights(1.0)
        assert abs(weights_1[0] - 0.2) < 1e-6
        assert abs(weights_1[1] - 0.8) < 1e-6

        # At progress=0.5, should be midpoint
        weights_05 = config.get_weights(0.5)
        assert abs(weights_05[0] - 0.5) < 1e-6
        assert abs(weights_05[1] - 0.5) < 1e-6


class TestConfigPresets:
    """Tests for configuration presets."""

    def test_recommended(self):
        """recommended() should be valid."""
        config = UL25Config.recommended()
        assert len(config.denoisers) > 0
        assert abs(sum(config.weights) - 1.0) < 1e-6

    def test_recommended_with_curriculum(self):
        """recommended_with_curriculum() should have curriculum weights."""
        config = UL25Config.recommended_with_curriculum()
        assert config.curriculum_start is not None
        assert config.curriculum_end is not None
        assert abs(sum(config.curriculum_start) - 1.0) < 1e-6
        assert abs(sum(config.curriculum_end) - 1.0) < 1e-6

    def test_ul2_original(self):
        """ul2_original() should match paper specs."""
        config = UL25Config.ul2_original()
        assert len(config.denoisers) == 7

        # Check prefix distribution
        r_count = sum(1 for d in config.denoisers if d.prefix == "[R]")
        s_count = sum(1 for d in config.denoisers if d.prefix == "[S]")
        x_count = sum(1 for d in config.denoisers if d.prefix == "[X]")

        assert r_count == 2, "Should have 2 R-denoisers"
        assert s_count == 1, "Should have 1 S-denoiser"
        assert x_count == 4, "Should have 4 X-denoisers"

    def test_t5_standard(self):
        """t5_standard() should have single span denoiser."""
        config = UL25Config.t5_standard()
        assert len(config.denoisers) == 1
        assert config.denoisers[0].task == Task.SPAN

    def test_flan_ul2_finetune(self):
        """flan_ul2_finetune() should have no prefixes."""
        config = UL25Config.flan_ul2_finetune()
        for d in config.denoisers:
            assert d.prefix == "", (
                f"Denoiser should have empty prefix, got '{d.prefix}'"
            )

    def test_minimal(self):
        """minimal() should be minimal but valid."""
        config = UL25Config.minimal()
        assert len(config.denoisers) == 2
        assert abs(sum(config.weights) - 1.0) < 1e-6

    def test_all_features(self):
        """all_features() should have boundary snapping enabled."""
        config = UL25Config.all_features()
        assert len(config.denoisers) > 0
        assert abs(sum(config.weights) - 1.0) < 1e-6
        assert config.enable_boundary_snapping is True

    def test_boundary_snapping_disabled_by_default(self):
        """Boundary snapping should be disabled by default."""
        config = UL25Config(
            denoisers=[DenoiserSpec(task=Task.SPAN)],
            weights=[1.0],
        )
        assert config.enable_boundary_snapping is False

    def test_standard_presets_boundary_snapping_disabled(self):
        """Standard presets should have boundary snapping disabled."""
        presets = [
            UL25Config.recommended(),
            UL25Config.recommended_with_curriculum(),
            UL25Config.ul2_original(),
            UL25Config.t5_standard(),
            UL25Config.minimal(),
            UL25Config.span_heavy(),
            UL25Config.flan_ul2_finetune(),
        ]
        for config in presets:
            assert config.enable_boundary_snapping is False, (
                "Preset should have boundary snapping disabled"
            )
