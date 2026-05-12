"""Generic conductance-manifold tools for user data.

This module contains no manuscript-specific coffee measurements. It exposes
the three-coordinate Monte Carlo grid, interpolation, and coefficient-projection
utilities used to apply the CPTDS conductance framework to an arbitrary user
dataset. The default retained manifold varies ion count, obstructant count, and
obstructant size while holding clustering and bulk medium renormalization off.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .config import PhysicalConstants
from .models import MODEL_DEFINITIONS
from .simulation import SimulationParameters, run_replicated_condition


@dataclass(frozen=True)
class ManifoldConfig:
    """Configuration for the generic three-coordinate conductance manifold."""

    n_ions_values: tuple[int, ...] = (48, 64, 80, 96, 112, 136, 160)
    blocker_count_values: tuple[int, ...] = (0, 4, 10, 16, 22, 32, 42)
    blocker_size_values: tuple[float, ...] = (
        0.005,
        0.010,
        0.015,
        0.020,
        0.030,
        0.040,
        0.060,
        0.080,
        0.095,
        0.110,
        0.140,
        0.170,
        0.200,
        0.250,
        0.300,
        0.400,
    )
    interaction_cutoff: float = 0.24
    blocker_radius_cv: float = 0.20
    trap_radius: float = 0.10
    cluster_strength: float = 0.0
    cluster_scale: float = 0.05
    g: float = 1.0
    grid_trials: int = 96
    final_trials: int = 288
    sigma_slope_rel: float = 0.06
    sigma_intercept_rel: float = 0.02
    ridge_strength: float = 0.01
    seed: int = 20260405
    max_workers: int = 6
    dimension: int = 3


def quick_manifold_config(grid_trials: int = 4, final_trials: int = 8) -> ManifoldConfig:
    """Return a small configuration intended only for smoke tests."""

    return ManifoldConfig(
        n_ions_values=(64, 112, 160),
        blocker_count_values=(0, 16, 42),
        blocker_size_values=(0.010, 0.080, 0.200, 0.400),
        grid_trials=grid_trials,
        final_trials=final_trials,
        max_workers=2,
    )


def shared_parameter_dict(config: ManifoldConfig) -> dict[str, float]:
    """Return the fixed Monte Carlo parameters shared across grid states."""

    return {
        "interaction_cutoff": config.interaction_cutoff,
        "blocker_radius_cv": config.blocker_radius_cv,
        "trap_radius": config.trap_radius,
        "cluster_strength": config.cluster_strength,
        "cluster_scale": config.cluster_scale,
        "g": config.g,
        "domain_count": 0,
        "domain_radius_mean": 0.10,
        "blocker_mix_weight": 0.50,
        "blocker_radius_mean_large": 0.12,
        "medium_heterogeneity": 0.0,
        "distance_decay": 0.0,
        "sequestration_softness": 0.0,
        "charge_bias": 0.0,
        "dilution_factor": 1.0,
        "radius_mode_code": 0,
        "cluster_mode_code": 0,
        "medium_mode_code": 0,
        "box_length": 1.0,
        "dimension": config.dimension,
    }


def _grid_task(task: tuple[int, int, float, dict[str, float], int, int]) -> dict:
    n_ions, blocker_count, blocker_radius_mean, shared, n_trials, seed = task
    params = SimulationParameters(
        n_ions=int(n_ions),
        blocker_count=int(blocker_count),
        blocker_radius_mean=float(blocker_radius_mean),
        **shared,
    )
    summary = run_replicated_condition(
        params=params,
        model=MODEL_DEFINITIONS["E"],
        physical=PhysicalConstants(),
        n_trials=n_trials,
        seed=seed,
    )
    summary["n_ions_grid"] = int(n_ions)
    summary["blocker_count_grid"] = int(blocker_count)
    summary["blocker_size_grid"] = float(blocker_radius_mean)
    return summary


def evaluate_manifold_grid(config: ManifoldConfig) -> pd.DataFrame:
    """Evaluate the generic 3D Monte Carlo conductance grid."""

    shared = shared_parameter_dict(config)
    tasks: list[tuple[int, int, float, dict[str, float], int, int]] = []
    seed_cursor = config.seed + 5000
    for n_ions in config.n_ions_values:
        for blocker_count in config.blocker_count_values:
            for blocker_size in config.blocker_size_values:
                tasks.append((n_ions, blocker_count, blocker_size, shared, config.grid_trials, seed_cursor))
                seed_cursor += 37

    rows: list[dict] = []
    max_workers = min(config.max_workers, max(1, (os.cpu_count() or 1) - 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_grid_task, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())

    return (
        pd.DataFrame(rows)
        .sort_values(["n_ions_grid", "blocker_count_grid", "blocker_size_grid"])
        .reset_index(drop=True)
    )


def prepare_interpolators(grid_df: pd.DataFrame) -> dict[str, object]:
    """Build interpolators for the effective conductance response surface."""

    n_values = np.array(sorted(grid_df["n_ions_grid"].unique()), dtype=float)
    count_values = np.array(sorted(grid_df["blocker_count_grid"].unique()), dtype=float)
    size_values = np.array(sorted(grid_df["blocker_size_grid"].unique()), dtype=float)
    shape = (len(n_values), len(count_values), len(size_values))

    k_grid = np.zeros(shape, dtype=float)
    lambda_grid = np.zeros(shape, dtype=float)

    n_index = {value: idx for idx, value in enumerate(n_values)}
    count_index = {value: idx for idx, value in enumerate(count_values)}
    size_index = {value: idx for idx, value in enumerate(size_values)}

    for row in grid_df.itertuples(index=False):
        i = n_index[float(row.n_ions_grid)]
        j = count_index[float(row.blocker_count_grid)]
        k = size_index[float(row.blocker_size_grid)]
        k_grid[i, j, k] = float(row.k_eff_over_k0_mean)
        lambda_grid[i, j, k] = float(row.lambda0_eff_over_lambda0_mean)

    return {
        "n_values": n_values,
        "count_values": count_values,
        "size_values": size_values,
        "k_interp": RegularGridInterpolator((n_values, count_values, size_values), k_grid, bounds_error=False, fill_value=None),
        "lambda_interp": RegularGridInterpolator((n_values, count_values, size_values), lambda_grid, bounds_error=False, fill_value=None),
    }


def solve_scale(feature: np.ndarray, observed: np.ndarray, sigma: float) -> float:
    """Solve a one-parameter weighted least-squares scale."""

    feature = np.asarray(feature, dtype=float)
    observed = np.asarray(observed, dtype=float)
    weights = np.full_like(feature, 1.0 / max(float(sigma) ** 2, 1e-12), dtype=float)
    numerator = float(np.sum(weights * feature * observed))
    denominator = float(np.sum(weights * feature * feature))
    return max(numerator / max(denominator, 1e-12), 1e-12)
