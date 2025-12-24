"""UL2.5 Configuration classes."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class Task(IntEnum):
    """Denoising task types."""

    SPAN = 0  # Standard T5 span corruption
    SPAN_MIDDLE = 1  # Position-biased (middle-heavy) with actual spans
    PREFIX_RANDOM = 2  # Random prefix/suffix split
    PREFIX_SHORT = 3  # Long prefix, short target (QA-like)
    PREFIX_LONG = 4  # Short prefix, long target (generation)
    INFILLING = 5  # Middle-out (bidirectional context)


class DenoiserSpec(BaseModel):
    """Single denoiser configuration with validation."""

    task: Task
    mu: float = Field(default=3.0, gt=0, description="Mean span length")
    r: float = Field(default=0.15, ge=0.01, le=0.99, description="Noise density")
    max_spans: int = Field(default=512, gt=0, description="Max corruption spans")
    prefix: str = Field(default="", description="Task prefix token")
    variable_r: bool = Field(default=False, description="Sample r from bounds")
    r_bounds: tuple[float, float] = Field(
        default=(0.05, 0.50), description="Bounds for variable r"
    )

    model_config = {"frozen": False, "extra": "forbid"}

    @field_validator("r_bounds")
    @classmethod
    def validate_r_bounds(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError("r_bounds[0] must be less than r_bounds[1]")
        if v[0] < 0.01 or v[1] > 0.99:
            raise ValueError("r_bounds must be within [0.01, 0.99]")
        return v

    def __repr__(self):
        return f"DenoiserSpec({self.task.name}, r={self.r}, μ={self.mu}, prefix='{self.prefix}')"


class UL25Config(BaseModel):
    """
    UL2.5 mixture configuration with validation.

    Attributes:
        denoisers: List of DenoiserSpec defining the task mixture
        weights: Sampling weights (normalized automatically)
        curriculum_start: Optional starting weights for curriculum
        curriculum_end: Optional ending weights for curriculum
        enable_length_adaptive: Enable length-adaptive task selection
        enable_boundary_snapping: Enable span boundary snapping
    """

    denoisers: list[DenoiserSpec] = Field(default_factory=list)
    weights: list[float] = Field(default_factory=list)
    curriculum_start: list[float] | None = Field(
        default=None, description="Starting weights for curriculum"
    )
    curriculum_end: list[float] | None = Field(
        default=None, description="Ending weights for curriculum"
    )
    enable_length_adaptive: bool = Field(
        default=True, description="Enable length-adaptive task selection"
    )
    enable_boundary_snapping: bool = Field(
        default=False,
        description="Enable span boundary snapping (CPU-only, adds overhead)",
    )

    model_config = {"frozen": False, "extra": "forbid"}

    @model_validator(mode="after")
    def validate_config(self) -> UL25Config:
        if self.denoisers and not self.weights:
            raise ValueError("weights must be provided when denoisers are set")

        # Normalize and validate weights
        if self.weights:
            if any(w < 0 for w in self.weights):
                raise ValueError("weights must be non-negative")
            total = sum(self.weights)
            if total <= 0:
                raise ValueError("weights must sum to > 0")
            self.weights = [w / total for w in self.weights]

        if (self.curriculum_start is None) ^ (self.curriculum_end is None):
            raise ValueError("curriculum_start and curriculum_end must both be set")

        if self.curriculum_start:
            if any(w < 0 for w in self.curriculum_start):
                raise ValueError("curriculum_start must be non-negative")
            total = sum(self.curriculum_start)
            if total <= 0:
                raise ValueError("curriculum_start must sum to > 0")
            self.curriculum_start = [w / total for w in self.curriculum_start]
        if self.curriculum_end:
            if any(w < 0 for w in self.curriculum_end):
                raise ValueError("curriculum_end must be non-negative")
            total = sum(self.curriculum_end)
            if total <= 0:
                raise ValueError("curriculum_end must sum to > 0")
            self.curriculum_end = [w / total for w in self.curriculum_end]

        # Validate lengths match
        n = len(self.denoisers)
        if self.weights and len(self.weights) != n:
            raise ValueError(
                f"weights length ({len(self.weights)}) must match denoisers ({n})"
            )
        if self.curriculum_start and len(self.curriculum_start) != n:
            raise ValueError(f"curriculum_start length must match denoisers ({n})")
        if self.curriculum_end and len(self.curriculum_end) != n:
            raise ValueError(f"curriculum_end length must match denoisers ({n})")

        return self

    def get_weights(self, progress: float = 0.0) -> list[float]:
        """Get interpolated weights based on training progress."""
        if self.curriculum_start is None or self.curriculum_end is None:
            return self.weights

        progress = max(0.0, min(1.0, progress))
        return [
            (1 - progress) * s + progress * e
            for s, e in zip(self.curriculum_start, self.curriculum_end)
        ]

    @classmethod
    def recommended(cls) -> UL25Config:
        """
        Recommended UL2.5 configuration with balanced mixture.

        Prefix semantics (per UL2 paper):
        - [R] = Regular denoising (r < 50%, mu < 12)
        - [S] = Sequential / Prefix LM
        - [X] = eXtreme denoising (r >= 50% OR mu >= 12)
        - [I] = Infilling (UL2.5 extension, not in original paper)

        Mixture:
        - 30% span denoising ([R] standard + [X] middle-heavy with mu=12)
        - 50% prefix LM variants ([S])
        - 20% infilling ([I])

        Note: SPAN_MIDDLE with mu=12 uses [X] because mu >= 12 qualifies as extreme.
        """
        return cls(
            denoisers=[
                DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                DenoiserSpec(task=Task.SPAN, mu=8.0, r=0.25, prefix="[R]"),
                DenoiserSpec(task=Task.SPAN_MIDDLE, mu=12.0, r=0.20, prefix="[X]"),
                DenoiserSpec(task=Task.PREFIX_RANDOM, prefix="[S]"),
                DenoiserSpec(task=Task.PREFIX_SHORT, prefix="[S]"),
                DenoiserSpec(task=Task.PREFIX_LONG, prefix="[S]"),
                DenoiserSpec(task=Task.INFILLING, r=0.30, prefix="[I]"),
            ],
            weights=[0.10, 0.10, 0.10, 0.20, 0.15, 0.15, 0.20],
        )

    @classmethod
    def recommended_with_curriculum(cls) -> UL25Config:
        """
        Recommended config with curriculum learning.

        Early: More span denoising (easier)
        Late: More prefix LM (matches inference)
        """
        return cls(
            denoisers=[
                DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                DenoiserSpec(task=Task.SPAN, mu=8.0, r=0.25, prefix="[R]"),
                DenoiserSpec(task=Task.SPAN_MIDDLE, mu=12.0, r=0.20, prefix="[X]"),
                DenoiserSpec(task=Task.PREFIX_RANDOM, prefix="[S]"),
                DenoiserSpec(task=Task.PREFIX_SHORT, prefix="[S]"),
                DenoiserSpec(task=Task.PREFIX_LONG, prefix="[S]"),
                DenoiserSpec(task=Task.INFILLING, r=0.30, prefix="[I]"),
            ],
            weights=[0.10, 0.10, 0.10, 0.20, 0.15, 0.15, 0.20],
            curriculum_start=[0.25, 0.25, 0.10, 0.15, 0.10, 0.10, 0.05],
            curriculum_end=[0.05, 0.05, 0.10, 0.25, 0.20, 0.20, 0.15],
        )

    @classmethod
    def ul2_original(cls) -> UL25Config:
        """
        Original UL2 7-denoiser mixture (Table 2 from the paper).

        Prefix semantics:
        - [R] = Regular (r=15%, mu in {3, 8})
        - [S] = Sequential (Prefix LM)
        - [X] = eXtreme (r=50% OR mu=64)

        Weights match paper: R=33%, S=34%, X=33%

        Note: Original UL2 used max_length=512. For Flan-UL2 (2048 context),
        use flan_ul2_finetune() which omits mode tokens as Flan-UL2 was trained
        to "forget" them.
        """
        return cls(
            denoisers=[
                DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                DenoiserSpec(task=Task.SPAN, mu=8.0, r=0.15, prefix="[R]"),
                DenoiserSpec(task=Task.PREFIX_RANDOM, prefix="[S]"),
                DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.50, prefix="[X]"),
                DenoiserSpec(task=Task.SPAN, mu=8.0, r=0.15, prefix="[X]"),
                DenoiserSpec(task=Task.SPAN, mu=64.0, r=0.15, prefix="[X]"),
                DenoiserSpec(task=Task.SPAN, mu=64.0, r=0.50, prefix="[X]"),
            ],
            weights=[0.165, 0.165, 0.34, 0.0825, 0.0825, 0.0825, 0.0825],
        )

    @classmethod
    def t5_standard(cls) -> UL25Config:
        """Standard T5 span corruption."""
        return cls(
            denoisers=[DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15)],
            weights=[1.0],
        )

    @classmethod
    def span_heavy(cls) -> UL25Config:
        """Original UL2-style with more span denoising."""
        return cls(
            denoisers=[
                DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                DenoiserSpec(task=Task.SPAN, mu=8.0, r=0.15, prefix="[R]"),
                DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.50, prefix="[X]"),
                DenoiserSpec(task=Task.SPAN, mu=64.0, r=0.50, prefix="[X]"),
                DenoiserSpec(task=Task.PREFIX_RANDOM, prefix="[S]"),
            ],
            weights=[0.20, 0.20, 0.15, 0.15, 0.30],
        )

    @classmethod
    def minimal(cls) -> UL25Config:
        """Minimal config for testing."""
        return cls(
            denoisers=[
                DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15),
                DenoiserSpec(task=Task.PREFIX_RANDOM),
            ],
            weights=[0.5, 0.5],
        )

    @classmethod
    def flan_ul2_finetune(cls) -> UL25Config:
        """
        Configuration for fine-tuning from Flan-UL2 checkpoint.

        Flan-UL2 was trained to "forget" mode tokens, so this preset
        omits all [R]/[S]/[X]/[I] prefixes. Use this when:
        - Fine-tuning google/flan-ul2 checkpoint
        - Training models that will be instruction-tuned later
        - Simpler deployment (no prefix handling at inference)

        Same mixture as recommended() but with empty prefixes.
        Supports 2048 context length (Flan-UL2 default).
        """
        return cls(
            denoisers=[
                DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15, prefix=""),
                DenoiserSpec(task=Task.SPAN, mu=8.0, r=0.25, prefix=""),
                DenoiserSpec(task=Task.SPAN_MIDDLE, mu=12.0, r=0.20, prefix=""),
                DenoiserSpec(task=Task.PREFIX_RANDOM, prefix=""),
                DenoiserSpec(task=Task.PREFIX_SHORT, prefix=""),
                DenoiserSpec(task=Task.PREFIX_LONG, prefix=""),
                DenoiserSpec(task=Task.INFILLING, r=0.30, prefix=""),
            ],
            weights=[0.10, 0.10, 0.10, 0.20, 0.15, 0.15, 0.20],
        )

    @classmethod
    def all_features(cls) -> UL25Config:
        """
        Full-featured config with all optional enhancements enabled.

        Same mixture as recommended() but with boundary snapping enabled.
        Use when training quality is prioritized over throughput.

        Note: Boundary snapping adds CPU overhead for aligning span starts
        to word boundaries. Only use when semantic alignment matters more
        than training speed.
        """
        return cls(
            denoisers=[
                DenoiserSpec(task=Task.SPAN, mu=3.0, r=0.15, prefix="[R]"),
                DenoiserSpec(task=Task.SPAN, mu=8.0, r=0.25, prefix="[R]"),
                DenoiserSpec(task=Task.SPAN_MIDDLE, mu=12.0, r=0.20, prefix="[X]"),
                DenoiserSpec(task=Task.PREFIX_RANDOM, prefix="[S]"),
                DenoiserSpec(task=Task.PREFIX_SHORT, prefix="[S]"),
                DenoiserSpec(task=Task.PREFIX_LONG, prefix="[S]"),
                DenoiserSpec(task=Task.INFILLING, r=0.30, prefix="[I]"),
            ],
            weights=[0.10, 0.10, 0.10, 0.20, 0.15, 0.15, 0.20],
            enable_boundary_snapping=True,
        )
