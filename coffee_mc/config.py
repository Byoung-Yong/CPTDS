"""Configuration objects for the Monte Carlo study."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class PhysicalConstants:
    """Phenomenological reference constants for scaling the simulated index."""

    lambda0: float = 1.0
    k0: float = 1.0
    reference_index: float = 0.013237
    lambda0_medium_exponent: float = 0.20
    lambda0_mobility_exponent: float = 0.25
    lambda0_mobility_floor: float = 0.85


@dataclass(frozen=True)
class StudyConfig:
    """Top-level numerical settings for the study."""

    output_dir: Path = Path("outputs")
    mode_name: str = "default"
    random_seed: int = 20260328
    n_parameter_samples: int = 220
    n_trials_per_sample: int = 18
    sweep_points: int = 17
    sweep_trials: int = 24
    convergence_trials: tuple[int, ...] = (4, 8, 16, 32, 48)
    convergence_ion_counts: tuple[int, ...] = (40, 60, 80, 120, 160)
    dimensions_to_test: tuple[int, ...] = (2, 3)
    targeted_dimensional_trials: int = 36
    noise_fraction: float = 0.06
    top_matches_per_state: int = 5
    bootstrap_samples: int = 400
    confidence_level: float = 0.95
    n_generalized_samples: int = 420
    generalized_trials_per_sample: int = 20
    codilution_factors: tuple[float, ...] = (1.00, 0.85, 0.70, 0.55, 0.40, 0.25)
    codilution_trials: int = 28
    physical: PhysicalConstants = field(default_factory=PhysicalConstants)

    @classmethod
    def for_mode(cls, mode_name: str) -> "StudyConfig":
        """Return a configuration tuned for the requested rigor mode."""
        base = cls(mode_name=mode_name)
        if mode_name == "high-rigor":
            return replace(
                base,
                n_parameter_samples=420,
                n_trials_per_sample=30,
                sweep_points=21,
                sweep_trials=36,
                bootstrap_samples=700,
                n_generalized_samples=1600,
                generalized_trials_per_sample=44,
                targeted_dimensional_trials=72,
                codilution_trials=56,
            )
        return base


MODEL_NAMES = {
    "A": "Random geometric blocking",
    "B": "Blocking + caging",
    "C": "Blocking + caging + clustering",
    "D": "Medium renormalization only",
    "E": "Combined blocking + caging + medium renormalization",
}
