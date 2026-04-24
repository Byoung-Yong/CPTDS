"""Literature-anchored particle Monte Carlo benchmark for electrolyte conductivity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AVOGADRO = 6.02214076e23
EPS = 1e-12


@dataclass(frozen=True)
class LiteratureMCBenchmarkConfig:
    """Configuration for the literature-anchored KMC benchmark."""

    electrolyte: str = "KCl"
    lambda_infinite_dilution: float = 149.79
    literature_table: tuple[tuple[float, float], ...] = (
        (0.005, 143.48),
        (0.010, 141.20),
        (0.020, 138.27),
        (0.050, 133.30),
        (0.100, 128.90),
    )
    simulation_concentrations_m: tuple[float, ...] = (
        0.005,
        0.010,
        0.020,
        0.050,
        0.100,
        0.200,
        0.500,
        1.000,
    )
    n_ions_total: int = 64
    dimension: int = 3
    interaction_cutoff_nm: float = 1.2
    field_strength_nm: float = 0.03
    step_scale_nm: float = 0.08
    interaction_scale: float = 0.04
    acceptance_temperature: float = 0.07
    screening_prefactor_nm_mhalf: float = 0.30
    ion_diameter_nm: float = 0.28
    calibration_mode: str = "affine"
    n_steps: int = 320
    burn_in_steps: int = 100
    n_replicates: int = 8
    random_seed: int = 20260328
    output_prefix: str = "kcl_literature_mc_benchmark"


def literature_dataframe(config: LiteratureMCBenchmarkConfig) -> pd.DataFrame:
    """Return the literature benchmark table as a DataFrame."""

    rows = []
    for concentration_m, lambda_value in config.literature_table:
        rows.append(
            {
                "electrolyte": config.electrolyte,
                "concentration_m": concentration_m,
                "sqrt_concentration_m_half": np.sqrt(concentration_m),
                "lambda_literature": lambda_value,
                "lambda_literature_norm": lambda_value / config.lambda_infinite_dilution,
            }
        )
    return pd.DataFrame(rows).sort_values("concentration_m").reset_index(drop=True)


def box_length_nm_from_molarity(n_ions_total: int, concentration_m: float) -> float:
    """Convert a target molarity into a cubic-box length in nm."""

    total_ion_density_nm3 = 2.0 * concentration_m * AVOGADRO / 1.0e24
    volume_nm3 = n_ions_total / max(total_ion_density_nm3, EPS)
    return float(volume_nm3 ** (1.0 / 3.0))


def _sample_balanced_ions(
    rng: np.random.Generator,
    n_ions: int,
    dimension: int,
    box_length_nm: float,
    ion_diameter_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample uniformly distributed ions with balanced charges."""

    positions = np.empty((n_ions, dimension), dtype=float)
    for ion_idx in range(n_ions):
        for _ in range(5000):
            candidate = rng.uniform(0.0, box_length_nm, size=dimension)
            if ion_idx == 0:
                positions[ion_idx] = candidate
                break
            diff = _minimal_image(positions[:ion_idx] - candidate[None, :], box_length_nm)
            distances = np.linalg.norm(diff, axis=1)
            if np.all(distances >= max(0.85 * ion_diameter_nm, 1e-6)):
                positions[ion_idx] = candidate
                break
        else:
            positions[ion_idx] = candidate
    charges = -np.ones(n_ions, dtype=int)
    charges[: n_ions // 2] = 1
    rng.shuffle(charges)
    return positions, charges


def _minimal_image(diff: np.ndarray, box_length_nm: float) -> np.ndarray:
    """Apply the minimum-image convention."""

    return diff - box_length_nm * np.round(diff / box_length_nm)


def _pair_energy_periodic(
    point: np.ndarray,
    charge: float,
    other_points: np.ndarray,
    other_charges: np.ndarray,
    concentration_m: float,
    cutoff_nm: float,
    interaction_scale: float,
    screening_prefactor_nm_mhalf: float,
    box_length_nm: float,
) -> float:
    """Screened Coulomb-like pair energy in a periodic box."""

    if len(other_points) == 0 or interaction_scale <= 0.0:
        return 0.0
    diff = _minimal_image(other_points - point[None, :], box_length_nm)
    distances = np.linalg.norm(diff, axis=1)
    mask = (distances > 1e-6) & (distances < cutoff_nm)
    if not np.any(mask):
        return 0.0
    distances = distances[mask]
    charge_products = charge * other_charges[mask]
    screening_length_nm = max(screening_prefactor_nm_mhalf / np.sqrt(max(concentration_m, EPS)), 0.05)
    screening = np.exp(-distances / screening_length_nm)
    return float(interaction_scale * np.sum(charge_products * screening / np.maximum(distances, 0.015)))


def _violates_hard_core(
    point: np.ndarray,
    other_points: np.ndarray,
    ion_diameter_nm: float,
    box_length_nm: float,
) -> bool:
    """Return whether the point overlaps another ion under periodic boundaries."""

    if len(other_points) == 0 or ion_diameter_nm <= 0.0:
        return False
    diff = _minimal_image(other_points - point[None, :], box_length_nm)
    distances = np.linalg.norm(diff, axis=1)
    return bool(np.any(distances < ion_diameter_nm))


def run_literature_kmc_replicate(
    concentration_m: float,
    seed: int,
    config: LiteratureMCBenchmarkConfig,
) -> dict[str, float]:
    """Run one particle Monte Carlo replicate at the target molarity."""

    box_length_nm = box_length_nm_from_molarity(config.n_ions_total, concentration_m)
    rng = np.random.default_rng(seed)
    positions, charges = _sample_balanced_ions(
        rng,
        config.n_ions_total,
        config.dimension,
        box_length_nm,
        config.ion_diameter_nm,
    )
    field_axis = np.zeros(config.dimension, dtype=float)
    field_axis[0] = 1.0
    beta = 1.0 / max(config.acceptance_temperature, EPS)
    current_history_raw = []
    indices = np.arange(config.n_ions_total)
    accepted_moves = 0
    attempted_moves = 0

    for _ in range(config.n_steps):
        sweep_current = 0.0
        for ion_idx in range(config.n_ions_total):
            current = positions[ion_idx].copy()
            charge = float(charges[ion_idx])
            proposal_raw = current + config.step_scale_nm * rng.normal(size=config.dimension)
            proposal_raw += config.field_strength_nm * charge * field_axis
            proposal = np.mod(proposal_raw, box_length_nm)
            dx = proposal_raw[0] - current[0]

            mask = indices != ion_idx
            attempted_moves += 1
            if _violates_hard_core(proposal, positions[mask], config.ion_diameter_nm, box_length_nm):
                continue
            energy_before = _pair_energy_periodic(
                current,
                charge,
                positions[mask],
                charges[mask],
                concentration_m,
                config.interaction_cutoff_nm,
                config.interaction_scale,
                config.screening_prefactor_nm_mhalf,
                box_length_nm,
            )
            energy_after = _pair_energy_periodic(
                proposal,
                charge,
                positions[mask],
                charges[mask],
                concentration_m,
                config.interaction_cutoff_nm,
                config.interaction_scale,
                config.screening_prefactor_nm_mhalf,
                box_length_nm,
            )
            delta_energy = (energy_after - energy_before) - config.field_strength_nm * charge * dx
            accept = delta_energy <= 0.0 or rng.random() < np.exp(-delta_energy * beta)
            if accept:
                positions[ion_idx] = proposal
                sweep_current += charge * dx
                accepted_moves += 1
        current_history_raw.append(sweep_current)

    production = np.array(current_history_raw[config.burn_in_steps :], dtype=float)
    n_pairs = config.n_ions_total / 2.0
    current_density_proxy = production.mean() / max(box_length_nm**config.dimension, EPS)
    molar_conductivity_proxy = production.mean() / max(n_pairs * config.field_strength_nm, EPS)

    return {
        "concentration_m": concentration_m,
        "sqrt_concentration_m_half": np.sqrt(concentration_m),
        "box_length_nm": box_length_nm,
        "current_density_proxy": float(current_density_proxy),
        "molar_conductivity_proxy": float(molar_conductivity_proxy),
        "raw_sweep_current_mean": float(production.mean()),
        "raw_sweep_current_std": float(production.std(ddof=1)) if len(production) > 1 else 0.0,
        "acceptance_fraction": float(accepted_moves / max(attempted_moves, 1)),
    }


def summarize_replicates(replicates_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicate-level KMC outputs by concentration."""

    rows = []
    for concentration_m, group in replicates_df.groupby("concentration_m", sort=True):
        proxy_values = group["molar_conductivity_proxy"].to_numpy(dtype=float)
        current_values = group["current_density_proxy"].to_numpy(dtype=float)
        rows.append(
            {
                "concentration_m": concentration_m,
                "sqrt_concentration_m_half": np.sqrt(concentration_m),
                "box_length_nm_mean": float(group["box_length_nm"].mean()),
                "n_replicates": int(len(group)),
                "molar_conductivity_proxy_mean": float(proxy_values.mean()),
                "molar_conductivity_proxy_std": float(proxy_values.std(ddof=1)) if len(group) > 1 else 0.0,
                "molar_conductivity_proxy_sem": float(proxy_values.std(ddof=1) / np.sqrt(len(group))) if len(group) > 1 else 0.0,
                "current_density_proxy_mean": float(current_values.mean()),
                "current_density_proxy_std": float(current_values.std(ddof=1)) if len(group) > 1 else 0.0,
                "current_density_proxy_sem": float(current_values.std(ddof=1) / np.sqrt(len(group))) if len(group) > 1 else 0.0,
                "acceptance_fraction_mean": float(group["acceptance_fraction"].mean()),
                "acceptance_fraction_std": float(group["acceptance_fraction"].std(ddof=1)) if len(group) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("concentration_m").reset_index(drop=True)


def fit_dilute_kohlrausch(
    summary_df: pd.DataFrame,
    concentration_min_m: float = 0.005,
    concentration_max_m: float = 0.1,
) -> pd.DataFrame:
    """Fit a Kohlrausch-style straight line on the literature-covered range."""

    dilute = summary_df[
        (summary_df["concentration_m"] >= concentration_min_m)
        & (summary_df["concentration_m"] <= concentration_max_m)
    ].copy()
    x = dilute["sqrt_concentration_m_half"].to_numpy(dtype=float)
    y = dilute["molar_conductivity_proxy_mean"].to_numpy(dtype=float)
    weights = 1.0 / np.maximum(dilute["molar_conductivity_proxy_sem"].to_numpy(dtype=float), 1e-8)
    slope, intercept = np.polyfit(x, y, deg=1, w=weights)
    prediction = intercept + slope * x
    residual = y - prediction
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return pd.DataFrame(
        [
            {
                "concentration_max_m": concentration_max_m,
                "concentration_min_m": concentration_min_m,
                "intercept_proxy": float(intercept),
                "slope_proxy": float(slope),
                "rmse_proxy": float(np.sqrt(np.mean(residual**2))),
                "r_squared": float(1.0 - ss_res / max(ss_tot, EPS)),
            }
        ]
    )


def calibrate_to_literature(
    summary_df: pd.DataFrame,
    literature_df: pd.DataFrame,
    lambda_infinite_dilution: float,
    calibration_mode: str = "affine",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map proxy units to literature units using a global calibration."""

    merged = summary_df.merge(
        literature_df[["concentration_m", "lambda_literature", "lambda_literature_norm"]],
        on="concentration_m",
        how="left",
    )
    overlap = merged[merged["lambda_literature"].notna()].copy()
    x = overlap["molar_conductivity_proxy_mean"].to_numpy(dtype=float)
    y = overlap["lambda_literature"].to_numpy(dtype=float)
    if calibration_mode == "through_origin":
        offset = 0.0
        scale = float(np.sum(x * y) / max(np.sum(x * x), EPS))
    elif calibration_mode == "affine":
        design = np.column_stack([np.ones_like(x), x])
        offset, scale = np.linalg.lstsq(design, y, rcond=None)[0]
        offset = float(offset)
        scale = float(scale)
    else:
        raise ValueError(f"Unsupported calibration_mode: {calibration_mode}")

    merged["lambda_mc_scaled"] = offset + scale * merged["molar_conductivity_proxy_mean"]
    merged["lambda_mc_scaled_sem"] = scale * merged["molar_conductivity_proxy_sem"]
    merged["lambda_mc_scaled_norm"] = merged["lambda_mc_scaled"] / lambda_infinite_dilution

    overlap = merged[merged["lambda_literature"].notna()].copy()
    overlap["absolute_error"] = overlap["lambda_mc_scaled"] - overlap["lambda_literature"]
    overlap["absolute_error_norm"] = overlap["lambda_mc_scaled_norm"] - overlap["lambda_literature_norm"]

    summary = pd.DataFrame(
        [
            {
                "scale_proxy_to_literature": float(scale),
                "offset_proxy_to_literature": float(offset),
                "calibration_mode": calibration_mode,
                "rmse_literature": float(np.sqrt(np.mean(overlap["absolute_error"] ** 2))),
                "rmse_literature_norm": float(np.sqrt(np.mean(overlap["absolute_error_norm"] ** 2))),
                "mae_literature": float(np.mean(np.abs(overlap["absolute_error"]))),
                "mae_literature_norm": float(np.mean(np.abs(overlap["absolute_error_norm"]))),
            }
        ]
    )
    return merged, summary


def plot_literature_mc_benchmark(
    merged_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    config: LiteratureMCBenchmarkConfig,
    output_path: Path,
    x_max: float | None = None,
) -> None:
    """Plot literature and Monte Carlo benchmark curves on normalized axes."""

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    literature = merged_df[merged_df["lambda_literature"].notna()].copy()
    simulation_only = merged_df[merged_df["lambda_literature"].isna()].copy()

    ax.errorbar(
        merged_df["sqrt_concentration_m_half"],
        merged_df["lambda_mc_scaled_norm"],
        yerr=1.96 * merged_df["lambda_mc_scaled_sem"] / config.lambda_infinite_dilution,
        fmt="o",
        ms=5,
        color="#1f77b4",
        ecolor="#8fb7e8",
        capsize=3,
        label="Monte Carlo benchmark",
    )
    ax.scatter(
        literature["sqrt_concentration_m_half"],
        literature["lambda_literature_norm"],
        s=54,
        color="#d62728",
        marker="s",
        label="Literature KCl table",
        zorder=4,
    )
    if len(simulation_only) > 0:
        ax.scatter(
            simulation_only["sqrt_concentration_m_half"],
            simulation_only["lambda_mc_scaled_norm"],
            s=50,
            facecolors="none",
            edgecolors="#1f77b4",
            label="MC extension beyond table",
            zorder=3,
        )

    fit_row = fit_df.iloc[0]
    x = np.linspace(0.0, float(merged_df["sqrt_concentration_m_half"].max()), 300)
    y_proxy = fit_row["intercept_proxy"] + fit_row["slope_proxy"] * x
    y_norm = (
        fit_row["offset_proxy_to_literature"] + fit_row["scale_proxy_to_literature"] * y_proxy
    ) / config.lambda_infinite_dilution
    ax.plot(x, y_norm, color="#444444", lw=1.8, ls="--", label="Dilute MC fit")

    ax.set_xlabel(r"$\sqrt{c}$ (M$^{1/2}$)")
    ax.set_ylabel(r"Normalized equivalent conductivity, $\Lambda / \Lambda^\circ$")
    ax.set_title(f"{config.electrolyte} literature-anchored Monte Carlo benchmark")
    ax.legend(loc="lower left")

    if x_max is not None:
        ax.set_xlim(0.0, x_max)
        visible = merged_df[merged_df["sqrt_concentration_m_half"] <= x_max].copy()
        visible_lit = visible.loc[visible["lambda_literature"].notna(), "lambda_literature_norm"]
        y_min = float(min(visible["lambda_mc_scaled_norm"].min(), visible_lit.min()))
        y_max = float(max(visible["lambda_mc_scaled_norm"].max(), visible_lit.max()))
        padding = max(0.02, 0.08 * (y_max - y_min))
        ax.set_ylim(y_min - padding, min(1.02, y_max + padding))

    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: np.square(np.maximum(x, 0.0)), lambda c: np.sqrt(np.maximum(c, 0.0))),
    )
    secax.set_xlabel(r"$c$ (M)")
    if x_max is not None:
        secax.set_xticks([0.005, 0.01, 0.1])
    else:
        secax.set_xticks([0.005, 0.01, 0.1, 0.5, 1.0])

    fit_info = (
        f"literature-covered range: 0.005-0.100 M\n"
        f"MC dilute fit R$^2$ = {fit_row['r_squared']:.3f}\n"
        f"normalized RMSE vs literature = {fit_row['rmse_literature_norm']:.4f}"
    )
    ax.text(
        0.03,
        0.03,
        fit_info,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.90, edgecolor="none"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_literature_mc_benchmark_actual(
    merged_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    config: LiteratureMCBenchmarkConfig,
    output_path: Path,
    x_max: float | None = None,
) -> None:
    """Plot literature and Monte Carlo benchmark curves in actual conductivity units."""

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    literature = merged_df[merged_df["lambda_literature"].notna()].copy()
    simulation_only = merged_df[merged_df["lambda_literature"].isna()].copy()

    ax.errorbar(
        merged_df["sqrt_concentration_m_half"],
        merged_df["lambda_mc_scaled"],
        yerr=1.96 * merged_df["lambda_mc_scaled_sem"],
        fmt="o",
        ms=5,
        color="#1f77b4",
        ecolor="#8fb7e8",
        capsize=3,
        label="Monte Carlo benchmark",
    )
    ax.scatter(
        literature["sqrt_concentration_m_half"],
        literature["lambda_literature"],
        s=54,
        color="#d62728",
        marker="s",
        label="Literature KCl table",
        zorder=4,
    )
    if len(simulation_only) > 0:
        ax.scatter(
            simulation_only["sqrt_concentration_m_half"],
            simulation_only["lambda_mc_scaled"],
            s=50,
            facecolors="none",
            edgecolors="#1f77b4",
            label="MC extension beyond table",
            zorder=3,
        )

    fit_row = fit_df.iloc[0]
    x = np.linspace(0.0, float(merged_df["sqrt_concentration_m_half"].max()), 300)
    y_actual = fit_row["intercept_literature_units"] + fit_row["slope_literature_units"] * x
    ax.plot(x, y_actual, color="#444444", lw=1.8, ls="--", label="Dilute MC fit")

    ax.set_xlabel(r"$\sqrt{c}$ (M$^{1/2}$)")
    ax.set_ylabel(r"Equivalent conductivity, $\Lambda$ ($10^{-4}$ m$^2$ S mol$^{-1}$)")
    ax.set_title(f"{config.electrolyte} literature-anchored Monte Carlo benchmark")
    ax.legend(loc="lower left")

    if x_max is not None:
        ax.set_xlim(0.0, x_max)
        visible = merged_df[merged_df["sqrt_concentration_m_half"] <= x_max].copy()
        visible_lit = visible.loc[visible["lambda_literature"].notna(), "lambda_literature"]
        y_min = float(min(visible["lambda_mc_scaled"].min(), visible_lit.min()))
        y_max = float(max(visible["lambda_mc_scaled"].max(), visible_lit.max()))
        padding = max(1.0, 0.08 * (y_max - y_min))
        ax.set_ylim(y_min - padding, y_max + padding)

    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: np.square(np.maximum(x, 0.0)), lambda c: np.sqrt(np.maximum(c, 0.0))),
    )
    secax.set_xlabel(r"$c$ (M)")
    if x_max is not None:
        secax.set_xticks([0.005, 0.01, 0.1])
    else:
        secax.set_xticks([0.005, 0.01, 0.1, 0.5, 1.0])

    fit_info = (
        f"literature-covered range: 0.005-0.100 M\n"
        f"MC dilute fit R$^2$ = {fit_row['r_squared']:.3f}\n"
        f"RMSE vs literature = {fit_row['rmse_literature']:.2f}"
    )
    ax.text(
        0.03,
        0.03,
        fit_info,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.90, edgecolor="none"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_literature_mc_residuals(overlap_df: pd.DataFrame, output_path: Path) -> None:
    """Plot normalized residuals relative to the literature table."""

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.scatter(overlap_df["concentration_m"], overlap_df["absolute_error_norm"], color="#444444", s=42)
    ax.set_xscale("log")
    ax.set_xlabel(r"$c$ (M)")
    ax.set_ylabel(r"MC - literature in $\Lambda / \Lambda^\circ$")
    ax.set_title("Residuals on the literature-covered range")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    output_path: Path,
    merged_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    config: LiteratureMCBenchmarkConfig,
) -> None:
    """Write a markdown summary for the literature-anchored benchmark."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fit_row = fit_df.iloc[0]
    text = f"""# {config.electrolyte} Literature-Anchored Monte Carlo Benchmark

## Purpose
This benchmark uses an actual particle Monte Carlo simulation, rather than a closed-form conductivity equation, to test whether an ions-only periodic transport model can reproduce the concentration dependence of equivalent conductivity for a representative strong 1:1 electrolyte.

## Literature Table
- Electrolyte: {config.electrolyte}
- Source: `equivalent conductivity of electrolytes.pdf`
- Literature-covered concentration range: 0.005 to 0.100 M
- Infinite-dilution conductivity: {config.lambda_infinite_dilution:.2f} in units of `10^-4 m^2 S mol^-1`

## Simulation Setup
- Simulated concentration range: {config.simulation_concentrations_m}
- Total ions per box: {config.n_ions_total}
- Geometry: periodic {config.dimension}D cubic box
- Interaction cutoff: {config.interaction_cutoff_nm:.2f} nm
- Proposal step scale: {config.step_scale_nm:.2f} nm
- Field bias: {config.field_strength_nm:.3f} nm
- Interaction scale: {config.interaction_scale:.3f}
- Acceptance temperature: {config.acceptance_temperature:.3f}
- Calibration mode: {config.calibration_mode}
- Sweeps per replicate: {config.n_steps} with burn-in {config.burn_in_steps}
- Replicates per concentration: {config.n_replicates}

## Benchmark Outcome
- Monte Carlo dilute-fit R^2 over 0.005-0.100 M: {fit_row['r_squared']:.6f}
- Absolute RMSE versus literature table: {fit_row['rmse_literature']:.3f}
- Normalized RMSE versus literature table: {fit_row['rmse_literature_norm']:.4f}
- Absolute MAE versus literature table: {fit_row['mae_literature']:.3f}
- Normalized MAE versus literature table: {fit_row['mae_literature_norm']:.4f}

## Interpretation
The horizontal axis is the actual concentration in molarity, reported through `sqrt(c)` on the main axis and `c` on the top axis.
The literature table is used only on the range 0.005-0.100 M for validation, because the 0.001 M point was intentionally excluded from the fitting set and the provided source does not tabulate values beyond 0.1 M.
Monte Carlo points at 0.2, 0.5, and 1.0 M are therefore simulation extensions and should not be interpreted as literature-validated values from this source.
The Monte Carlo curve is converted into the literature conductivity units by a global {config.calibration_mode} calibration fitted over the literature-covered range. This benchmark is therefore a true particle Monte Carlo consistency check and is more appropriate as the foundational validator than the previous formula-based DHO surrogate."""
    output_path.write_text(text, encoding="utf-8")


def write_methods_draft(output_path: Path, config: LiteratureMCBenchmarkConfig) -> None:
    """Write a manuscript-style methods paragraph."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Literature Monte Carlo Benchmark Draft

As the primary bare-electrolyte validation, an ions-only periodic Monte Carlo benchmark was carried out for KCl using literature equivalent-conductivity data at 25 degrees C as the external reference. A balanced population of {config.n_ions_total} ions was placed in a periodic three-dimensional cubic box, and the box length was adjusted at each target concentration so that the total ion density corresponded to the desired molarity for a 1:1 electrolyte. Single-particle Monte Carlo proposals combined a Gaussian displacement with a weak field-biased drift term along one axis. The moves were accepted or rejected using a screened Coulomb-like pair energy evaluated under periodic boundary conditions with an interaction cutoff of {config.interaction_cutoff_nm:.2f} nm. The pair interaction included a concentration-dependent screening length and a hard-core exclusion diameter of {config.ion_diameter_nm:.2f} nm. For each concentration, {config.n_replicates} independent replicates of {config.n_steps} Monte Carlo sweeps were generated after a burn-in of {config.burn_in_steps} sweeps. A molar-conductivity proxy was defined from the mean signed field-direction charge displacement per sweep, normalized by the number of electrolyte formula units and the field amplitude. The proxy curve was then converted into the literature conductivity units using a global {config.calibration_mode} calibration fitted over the literature-covered range of 0.005-0.100 M, while the same simulation conditions were extended to 1.0 M for qualitative continuation beyond the tabulated range."""
    output_path.write_text(text, encoding="utf-8")


def run_literature_mc_benchmark(
    config: LiteratureMCBenchmarkConfig | None = None,
    root_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the literature-anchored Monte Carlo benchmark and save outputs."""

    config = config or LiteratureMCBenchmarkConfig()
    root_dir = root_dir or Path(".")
    output_dir = root_dir / "outputs"
    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"
    text_dir = output_dir / "text"

    replicate_rows = []
    for concentration_idx, concentration_m in enumerate(config.simulation_concentrations_m):
        for replicate in range(config.n_replicates):
            seed = config.random_seed + 1000 * concentration_idx + 37 * replicate
            row = run_literature_kmc_replicate(concentration_m, seed, config)
            row["replicate"] = replicate
            replicate_rows.append(row)

    replicates_df = pd.DataFrame(replicate_rows)
    summary_df = summarize_replicates(replicates_df)
    literature_df = literature_dataframe(config)
    fit_proxy_df = fit_dilute_kohlrausch(summary_df, concentration_min_m=0.005, concentration_max_m=0.1)
    merged_df, fit_quality_df = calibrate_to_literature(
        summary_df,
        literature_df,
        config.lambda_infinite_dilution,
        calibration_mode=config.calibration_mode,
    )
    fit_df = pd.concat([fit_proxy_df, fit_quality_df], axis=1)
    fit_df["intercept_literature_units"] = (
        fit_df["offset_proxy_to_literature"] + fit_df["intercept_proxy"] * fit_df["scale_proxy_to_literature"]
    )
    fit_df["slope_literature_units"] = fit_df["slope_proxy"] * fit_df["scale_proxy_to_literature"]
    overlap_df = merged_df[merged_df["lambda_literature"].notna()].copy()
    overlap_df["absolute_error"] = overlap_df["lambda_mc_scaled"] - overlap_df["lambda_literature"]
    overlap_df["absolute_error_norm"] = overlap_df["lambda_mc_scaled_norm"] - overlap_df["lambda_literature_norm"]

    table_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.output_prefix
    literature_df.to_csv(table_dir / f"{prefix}_literature.csv", index=False)
    replicates_df.to_csv(table_dir / f"{prefix}_replicates.csv", index=False)
    summary_df.to_csv(table_dir / f"{prefix}_summary.csv", index=False)
    merged_df.to_csv(table_dir / f"{prefix}_merged.csv", index=False)
    fit_df.to_csv(table_dir / f"{prefix}_fit.csv", index=False)
    overlap_df.to_csv(table_dir / f"{prefix}_overlap.csv", index=False)

    plot_literature_mc_benchmark(merged_df, fit_df, config, fig_dir / f"{prefix}_lambda_vs_sqrtc.png")
    plot_literature_mc_benchmark(
        merged_df,
        fit_df,
        config,
        fig_dir / f"{prefix}_lambda_vs_sqrtc_zoom.png",
        x_max=0.4,
    )
    plot_literature_mc_benchmark_actual(merged_df, fit_df, config, fig_dir / f"{prefix}_actual_units.png")
    plot_literature_mc_benchmark_actual(
        merged_df,
        fit_df,
        config,
        fig_dir / f"{prefix}_actual_units_zoom.png",
        x_max=0.4,
    )
    plot_literature_mc_residuals(overlap_df, fig_dir / f"{prefix}_residuals.png")
    write_summary(text_dir / f"{prefix}_summary.md", merged_df, fit_df, config)
    write_methods_draft(root_dir / "manuscript" / "Literature_MC_Benchmark_Draft.md", config)

    return {
        "literature": literature_df,
        "replicates": replicates_df,
        "summary": summary_df,
        "merged": merged_df,
        "fit": fit_df,
        "overlap": overlap_df,
    }
