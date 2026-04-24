"""Model definitions and per-model parameter activation."""

from __future__ import annotations

from dataclasses import dataclass

from .config import MODEL_NAMES


@dataclass(frozen=True)
class ModelDefinition:
    """Logical definition of a phenomenological screening model."""

    code: str
    name: str
    use_blocking: bool
    use_caging: bool
    use_clustering: bool
    use_medium: bool

    @property
    def active_parameters(self) -> tuple[str, ...]:
        names = ["n_ions", "interaction_cutoff"]
        if self.use_blocking:
            names.extend(["blocker_count", "blocker_radius_mean", "blocker_radius_cv"])
        if self.use_caging:
            names.append("trap_radius")
        if self.use_clustering:
            names.extend(["cluster_strength", "cluster_scale"])
        if self.use_medium:
            names.append("g")
        return tuple(names)


MODEL_DEFINITIONS = {
    code: ModelDefinition(
        code=code,
        name=MODEL_NAMES[code],
        use_blocking=code in {"A", "B", "C", "E"},
        use_caging=code in {"B", "C", "E"},
        use_clustering=code in {"C", "E"},
        use_medium=code in {"D", "E"},
    )
    for code in MODEL_NAMES
}
