"""Core Monte Carlo simulation engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import PhysicalConstants
from .models import ModelDefinition


EPS = 1e-12


@dataclass(frozen=True)
class SimulationParameters:
    """Single parameter set for one Monte Carlo condition."""

    n_ions: int = 96
    interaction_cutoff: float = 0.22
    blocker_count: int = 18
    blocker_radius_mean: float = 0.055
    blocker_radius_cv: float = 0.25
    trap_radius: float = 0.06
    cluster_strength: float = 0.0
    cluster_scale: float = 0.05
    g: float = 1.0
    domain_count: int = 0
    domain_radius_mean: float = 0.10
    blocker_mix_weight: float = 0.50
    blocker_radius_mean_large: float = 0.12
    medium_heterogeneity: float = 0.0
    distance_decay: float = 0.0
    sequestration_softness: float = 0.0
    charge_bias: float = 0.0
    dilution_factor: float = 1.0
    radius_mode_code: int = 0
    cluster_mode_code: int = 0
    medium_mode_code: int = 0
    box_length: float = 1.0
    dimension: int = 3

    def __post_init__(self) -> None:
        """Normalize integer-like fields after pandas-based construction."""
        object.__setattr__(self, "n_ions", int(round(self.n_ions)))
        object.__setattr__(self, "blocker_count", int(round(self.blocker_count)))
        object.__setattr__(self, "domain_count", int(round(self.domain_count)))
        object.__setattr__(self, "radius_mode_code", int(round(self.radius_mode_code)))
        object.__setattr__(self, "cluster_mode_code", int(round(self.cluster_mode_code)))
        object.__setattr__(self, "medium_mode_code", int(round(self.medium_mode_code)))
        object.__setattr__(self, "dimension", int(round(self.dimension)))

    @classmethod
    def from_series(cls, series: pd.Series, dimension: int = 3) -> "SimulationParameters":
        """Build parameters from a pandas row."""
        values = series.to_dict()
        values["dimension"] = dimension
        return cls(**values)

    def to_dict(self) -> Dict[str, float]:
        """Convert to a regular dictionary."""
        return asdict(self)


def _sample_radii(rng: np.random.Generator, count: int, mean: float, cv: float) -> np.ndarray:
    """Sample blocker radii from a lognormal distribution with a target mean and CV."""
    if count <= 0 or mean <= 0:
        return np.zeros(0)
    sigma2 = np.log(cv * cv + 1.0)
    sigma = np.sqrt(max(sigma2, EPS))
    mu = np.log(mean) - 0.5 * sigma2
    return rng.lognormal(mean=mu, sigma=sigma, size=count)


def _sample_domain_centers(
    rng: np.random.Generator,
    params: SimulationParameters,
) -> np.ndarray:
    """Sample centers for generalized heterogeneous domains."""
    if params.domain_count <= 0:
        return np.zeros((0, params.dimension))
    return rng.uniform(0.0, params.box_length, size=(params.domain_count, params.dimension))


def _apply_codilution(params: SimulationParameters) -> SimulationParameters:
    """Apply a phenomenological co-dilution rule to ions and non-ionic matrix components."""
    phi = float(np.clip(params.dilution_factor, 0.05, 1.0))
    if abs(phi - 1.0) < 1e-12:
        return params
    updated = params.to_dict()
    updated["n_ions"] = max(2, int(round(params.n_ions * phi)))
    updated["blocker_count"] = max(0, int(round(params.blocker_count * phi)))
    updated["domain_count"] = max(0, int(round(params.domain_count * phi)))
    updated["g"] = 1.0 - phi * (1.0 - params.g)
    updated["medium_heterogeneity"] = params.medium_heterogeneity * phi
    updated["sequestration_softness"] = params.sequestration_softness * phi
    updated["dilution_factor"] = phi
    return SimulationParameters(**updated)


def _sample_ions(
    rng: np.random.Generator,
    n_ions: int,
    dimension: int,
    box_length: float,
    charge_bias: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ion positions and balanced charges."""
    positions = rng.uniform(0.0, box_length, size=(n_ions, dimension))
    positive_fraction = np.clip(0.5 + 0.5 * charge_bias, 0.25, 0.75)
    positive_count = int(round(n_ions * positive_fraction))
    positive_count = min(max(1, positive_count), n_ions - 1)
    charges = -np.ones(n_ions, dtype=int)
    charges[:positive_count] = 1
    rng.shuffle(charges)
    return positions, charges


def _sample_blockers(
    rng: np.random.Generator,
    params: SimulationParameters,
    ions: np.ndarray,
    model: ModelDefinition,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample blocker centers and radii, optionally clustered around ions."""
    count = params.blocker_count if model.use_blocking else 0
    if params.radius_mode_code == 1 and count > 0:
        small_count = int(round(count * params.blocker_mix_weight))
        small_count = min(max(1, small_count), count - 1) if count > 1 else count
        large_count = count - small_count
        radii_small = _sample_radii(rng, small_count, params.blocker_radius_mean, params.blocker_radius_cv)
        radii_large = _sample_radii(rng, large_count, params.blocker_radius_mean_large, params.blocker_radius_cv)
        radii = np.concatenate([radii_small, radii_large])
        rng.shuffle(radii)
    else:
        radii = _sample_radii(rng, count, params.blocker_radius_mean, params.blocker_radius_cv)
    if count == 0:
        return np.zeros((0, params.dimension)), radii

    centers = np.empty((count, params.dimension))
    domain_centers = _sample_domain_centers(rng, params)
    for i in range(count):
        if model.use_clustering and params.cluster_mode_code in {1, 3} and rng.random() < params.cluster_strength:
            anchor = ions[rng.integers(0, len(ions))]
            offset = rng.normal(0.0, params.cluster_scale, size=params.dimension)
            centers[i] = np.clip(anchor + offset, 0.0, params.box_length)
        elif model.use_clustering and params.cluster_mode_code in {2, 3} and len(domain_centers) > 0 and rng.random() < params.cluster_strength:
            anchor = domain_centers[rng.integers(0, len(domain_centers))]
            offset = rng.normal(0.0, params.cluster_scale, size=params.dimension)
            centers[i] = np.clip(anchor + offset, 0.0, params.box_length)
        else:
            centers[i] = rng.uniform(0.0, params.box_length, size=params.dimension)
    return centers, radii


def _line_point_distance(points_a: np.ndarray, points_b: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Distance from a point to each finite line segment joining A and B."""
    ab = points_b - points_a
    denom = np.sum(ab * ab, axis=1)
    denom = np.where(denom < EPS, EPS, denom)
    t = np.sum((center - points_a) * ab, axis=1) / denom
    t = np.clip(t, 0.0, 1.0)
    projection = points_a + ab * t[:, None]
    return np.linalg.norm(center - projection, axis=1)


def _candidate_pairs(
    positions: np.ndarray,
    charges: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find opposite-charge ion pairs within the interaction cutoff."""
    n_ions = len(charges)
    if n_ions < 2:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0)
    pair_i, pair_j = np.triu_indices(n_ions, k=1)
    diff = positions[pair_i] - positions[pair_j]
    distances = np.linalg.norm(diff, axis=1)
    mask = (charges[pair_i] != charges[pair_j]) & (distances <= cutoff)
    return pair_i[mask], pair_j[mask], distances[mask]


def _local_medium_field(
    positions: np.ndarray,
    params: SimulationParameters,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct a local medium field on ions for heterogeneous-medium scenarios."""
    if params.medium_mode_code == 0 or params.medium_heterogeneity <= 0 or params.domain_count <= 0:
        return np.full(len(positions), params.g, dtype=float)
    domain_centers = _sample_domain_centers(rng, params)
    if len(domain_centers) == 0:
        return np.full(len(positions), params.g, dtype=float)
    distances = np.linalg.norm(positions[:, None, :] - domain_centers[None, :, :], axis=2)
    profile = np.exp(-0.5 * (distances / max(params.domain_radius_mean, EPS)) ** 2)
    local_modifier = 1.0 - params.medium_heterogeneity * profile.max(axis=1)
    return np.clip(params.g * local_modifier, 0.01, 1.0)


def _baseline_degree(n_ions: int, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    """Degree count before any blocking or caging."""
    degree = np.zeros(n_ions, dtype=float)
    np.add.at(degree, pair_i, 1.0)
    np.add.at(degree, pair_j, 1.0)
    return degree


def _largest_component_fraction(n_ions: int, pair_i: np.ndarray, pair_j: np.ndarray) -> float:
    """Compute the largest connected-component size fraction."""
    if n_ions == 0:
        return 0.0
    parent = np.arange(n_ions)
    size = np.ones(n_ions, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for a, b in zip(pair_i, pair_j):
        union(int(a), int(b))

    if len(pair_i) == 0:
        return 0.0
    component_sizes = {}
    for node in np.unique(np.concatenate([pair_i, pair_j])):
        root = find(int(node))
        component_sizes[root] = component_sizes.get(root, 0) + 1
    return max(component_sizes.values()) / n_ions


def run_single_trial(
    rng: np.random.Generator,
    params: SimulationParameters,
    model: ModelDefinition,
    physical: PhysicalConstants,
) -> dict:
    """Run one Monte Carlo realization and compute accessibility metrics."""
    effective_params = _apply_codilution(params)
    ions, charges = _sample_ions(rng, effective_params.n_ions, effective_params.dimension, effective_params.box_length, effective_params.charge_bias)
    blockers, radii = _sample_blockers(rng, effective_params, ions, model)
    local_g = _local_medium_field(ions, effective_params, rng)

    pair_i, pair_j, _ = _candidate_pairs(ions, charges, effective_params.interaction_cutoff)
    baseline_pairs = len(pair_i)
    baseline_degree = _baseline_degree(effective_params.n_ions, pair_i, pair_j)
    mean_baseline_degree = baseline_degree.mean() if baseline_degree.size else 0.0
    baseline_giant = _largest_component_fraction(effective_params.n_ions, pair_i, pair_j)

    active = np.ones(effective_params.n_ions, dtype=bool)
    if model.use_caging and len(blockers) > 0 and effective_params.trap_radius > 0:
        for center, radius in zip(blockers, radii):
            distance = np.linalg.norm(ions - center[None, :], axis=1)
            shell = radius + effective_params.trap_radius
            if effective_params.sequestration_softness > 0:
                probability = np.clip(
                    np.exp(-np.maximum(distance - radius, 0.0) / max(effective_params.trap_radius, EPS))
                    * effective_params.sequestration_softness,
                    0.0,
                    1.0,
                )
                deactivate = rng.random(len(ions)) < probability
                deactivate &= distance <= shell
            else:
                deactivate = distance <= shell
            active &= ~deactivate

    accessible = np.ones(baseline_pairs, dtype=bool)
    if model.use_blocking and baseline_pairs > 0 and len(blockers) > 0:
        points_a = ions[pair_i]
        points_b = ions[pair_j]
        for center, radius in zip(blockers, radii):
            accessible &= _line_point_distance(points_a, points_b, center) > radius

    if baseline_pairs > 0:
        accessible &= active[pair_i] & active[pair_j]
        pair_i_acc = pair_i[accessible]
        pair_j_acc = pair_j[accessible]
        pair_distances = np.linalg.norm(ions[pair_i] - ions[pair_j], axis=1)
        pair_weights = np.exp(-effective_params.distance_decay * pair_distances / max(effective_params.interaction_cutoff, EPS))
        accessible_weights = pair_weights[accessible]
    else:
        pair_i_acc = pair_i
        pair_j_acc = pair_j
        pair_weights = np.zeros(0)
        accessible_weights = np.zeros(0)

    accessible_pairs = len(pair_i_acc)
    accessible_degree = _baseline_degree(effective_params.n_ions, pair_i_acc, pair_j_acc)
    mean_accessible_degree = accessible_degree.mean() if accessible_degree.size else 0.0

    xi_pair = accessible_pairs / baseline_pairs if baseline_pairs > 0 else 0.0
    xi_degree = mean_accessible_degree / mean_baseline_degree if mean_baseline_degree > 0 else 0.0
    accessible_giant = _largest_component_fraction(effective_params.n_ions, pair_i_acc, pair_j_acc)
    xi_giant = accessible_giant / baseline_giant if baseline_giant > 0 else 0.0
    xi_weighted = accessible_weights.sum() / pair_weights.sum() if pair_weights.sum() > 0 else 0.0

    metrics = np.clip(np.array([xi_pair, xi_degree, xi_giant, xi_weighted], dtype=float), 0.0, 1.0)
    xi_composite = float(np.exp(np.mean(np.log(np.clip(metrics, EPS, 1.0)))))

    active_fraction = float(active.mean())

    if model.use_medium:
        if accessible_pairs > 0:
            pair_local_g = np.sqrt(local_g[pair_i_acc] * local_g[pair_j_acc])
            medium_factor = float(pair_local_g.mean())
        elif active.any():
            medium_factor = float(local_g[active].mean())
        else:
            medium_factor = float(local_g.mean())
    else:
        medium_factor = 1.0

    # The limiting-conductivity term is allowed to soften under screening,
    # but with a bounded floor so that intercept collapse remains milder than slope collapse.
    mobility_factor = physical.lambda0_mobility_floor + (
        1.0 - physical.lambda0_mobility_floor
    ) * np.clip(active_fraction, 0.0, 1.0) ** physical.lambda0_mobility_exponent
    lambda0_eff = (
        physical.lambda0
        * medium_factor ** physical.lambda0_medium_exponent
        * mobility_factor
    )
    k_eff = physical.k0 * medium_factor * np.sqrt(xi_composite)
    index = physical.reference_index * (k_eff / physical.k0) ** 2 / max(lambda0_eff, EPS) ** 3

    return {
        "baseline_pairs": baseline_pairs,
        "accessible_pairs": accessible_pairs,
        "active_fraction": active_fraction,
        "xi_pair": xi_pair,
        "xi_degree": xi_degree,
        "xi_giant": xi_giant,
        "xi_weighted": xi_weighted,
        "xi_composite": xi_composite,
        "k_eff_over_k0": k_eff / physical.k0,
        "lambda0_eff_over_lambda0": lambda0_eff / physical.lambda0,
        "index_pred": index,
        "blocked_fraction": 1.0 - accessible_pairs / baseline_pairs if baseline_pairs > 0 else 1.0,
        "density": effective_params.n_ions / effective_params.box_length ** effective_params.dimension,
        "medium_factor_effective": medium_factor,
        "mobility_factor_effective": mobility_factor,
        "local_g_mean": local_g.mean(),
        "local_g_std": local_g.std(ddof=0),
        "effective_n_ions": effective_params.n_ions,
        "effective_blocker_count": effective_params.blocker_count,
        "effective_domain_count": effective_params.domain_count,
        "effective_g_param": effective_params.g,
        "dilution_factor": effective_params.dilution_factor,
    }


def summarize_trials(records: list[dict]) -> dict:
    """Summarize repeated realizations into mean and confidence intervals."""
    frame = pd.DataFrame(records)
    summary = {}
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        summary[f"{column}_mean"] = values.mean()
        summary[f"{column}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        summary[f"{column}_q025"] = np.quantile(values, 0.025)
        summary[f"{column}_q975"] = np.quantile(values, 0.975)
    return summary


def run_replicated_condition(
    params: SimulationParameters,
    model: ModelDefinition,
    physical: PhysicalConstants,
    n_trials: int,
    seed: int,
) -> dict:
    """Run repeated Monte Carlo trials for a single parameter condition."""
    records = []
    for trial in range(n_trials):
        rng = np.random.default_rng(seed + 1009 * trial)
        records.append(run_single_trial(rng, params, model, physical))
    summary = summarize_trials(records)
    summary.update(params.to_dict())
    summary["model_code"] = model.code
    summary["model_name"] = model.name
    summary["n_trials"] = n_trials
    return summary
