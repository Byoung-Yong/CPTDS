"""Salt-specific particle Monte Carlo benchmark for monovalent electrolytes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from coffee_mc.literature_mc_benchmark import (
    AVOGADRO,
    EPS,
    calibrate_to_literature,
    fit_dilute_kohlrausch,
    plot_literature_mc_benchmark,
    plot_literature_mc_benchmark_actual,
    plot_literature_mc_residuals,
)


ION_PROPERTIES = {
    "Na+": {"lambda_inf": 50.11, "diameter_nm": 0.204},
    "K+": {"lambda_inf": 73.50, "diameter_nm": 0.266},
    "Cl-": {"lambda_inf": 76.35, "diameter_nm": 0.362},
    "I-": {"lambda_inf": 76.84, "diameter_nm": 0.440},
}


ELECTROLYTE_LIBRARY = {
    "KCl": {
        "cation": "K+",
        "anion": "Cl-",
        "lambda_infinite_dilution": 149.79,
        "literature_table": (
            (0.005, 143.48),
            (0.010, 141.20),
            (0.020, 138.27),
            (0.050, 133.30),
            (0.100, 128.90),
        ),
    },
    "NaCl": {
        "cation": "Na+",
        "anion": "Cl-",
        "lambda_infinite_dilution": 126.39,
        "literature_table": (
            (0.005, 120.59),
            (0.010, 118.45),
            (0.020, 115.70),
            (0.050, 111.01),
            (0.100, 106.69),
        ),
    },
    "KI": {
        "cation": "K+",
        "anion": "I-",
        "lambda_infinite_dilution": 150.31,
        "literature_table": (
            (0.005, 144.30),
            (0.010, 142.11),
            (0.020, 139.38),
            (0.050, 134.90),
            (0.100, 131.05),
        ),
    },
}


@dataclass(frozen=True)
class SaltSpecificBenchmarkConfig:
    """Salt-specific benchmark configuration for a 1:1 electrolyte."""

    electrolyte: str
    cation: str
    anion: str
    lambda_infinite_dilution: float
    literature_table: tuple[tuple[float, float], ...]
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
    n_ions_total: int = 128
    dimension: int = 3
    interaction_cutoff_nm: float = 1.32
    base_field_strength_nm: float = 0.025
    base_step_scale_nm: float = 0.081
    interaction_scale: float = 0.042
    acceptance_temperature: float = 0.048
    screening_prefactor_nm_mhalf: float = 0.36
    hard_core_scale: float = 0.90
    calibration_mode: str = "affine"
    n_steps: int = 520
    burn_in_steps: int = 170
    n_replicates: int = 12
    random_seed: int = 20260328
    output_prefix: str = "salt_specific_mc_benchmark"
    mobility_reference: float = 74.925

    @property
    def cation_lambda_inf(self) -> float:
        return float(ION_PROPERTIES[self.cation]["lambda_inf"])

    @property
    def anion_lambda_inf(self) -> float:
        return float(ION_PROPERTIES[self.anion]["lambda_inf"])

    @property
    def cation_diameter_nm(self) -> float:
        return float(ION_PROPERTIES[self.cation]["diameter_nm"])

    @property
    def anion_diameter_nm(self) -> float:
        return float(ION_PROPERTIES[self.anion]["diameter_nm"])


def build_salt_config(electrolyte: str, **overrides) -> SaltSpecificBenchmarkConfig:
    """Create a salt-specific config from the built-in electrolyte library."""

    spec = ELECTROLYTE_LIBRARY[electrolyte]
    params = {
        "electrolyte": electrolyte,
        "cation": spec["cation"],
        "anion": spec["anion"],
        "lambda_infinite_dilution": spec["lambda_infinite_dilution"],
        "literature_table": spec["literature_table"],
        "output_prefix": f"{electrolyte.lower()}_salt_specific_mc_benchmark",
    }
    params.update(overrides)
    return SaltSpecificBenchmarkConfig(**params)


def literature_dataframe(config: SaltSpecificBenchmarkConfig) -> pd.DataFrame:
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


def _minimal_image(diff: np.ndarray, box_length_nm: float) -> np.ndarray:
    """Apply the minimum-image convention."""

    return diff - box_length_nm * np.round(diff / box_length_nm)


def _sample_ions_with_species(
    rng: np.random.Generator,
    config: SaltSpecificBenchmarkConfig,
    box_length_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample balanced ions with salt-specific mobilities and diameters."""

    n_cations = config.n_ions_total // 2
    n_anions = config.n_ions_total - n_cations
    charges = np.concatenate([np.ones(n_cations, dtype=int), -np.ones(n_anions, dtype=int)])
    mobility_factors = np.concatenate(
        [
            np.full(n_cations, config.cation_lambda_inf / config.mobility_reference, dtype=float),
            np.full(n_anions, config.anion_lambda_inf / config.mobility_reference, dtype=float),
        ]
    )
    diameters = np.concatenate(
        [
            np.full(n_cations, config.cation_diameter_nm, dtype=float),
            np.full(n_anions, config.anion_diameter_nm, dtype=float),
        ]
    )

    permutation = rng.permutation(config.n_ions_total)
    charges = charges[permutation]
    mobility_factors = mobility_factors[permutation]
    diameters = diameters[permutation]

    positions = np.empty((config.n_ions_total, config.dimension), dtype=float)
    for ion_idx in range(config.n_ions_total):
        diameter_i = diameters[ion_idx]
        for _ in range(8000):
            candidate = rng.uniform(0.0, box_length_nm, size=config.dimension)
            if ion_idx == 0:
                positions[ion_idx] = candidate
                break
            diff = _minimal_image(positions[:ion_idx] - candidate[None, :], box_length_nm)
            distances = np.linalg.norm(diff, axis=1)
            min_allowed = config.hard_core_scale * 0.5 * (diameter_i + diameters[:ion_idx])
            if np.all(distances >= np.maximum(min_allowed, 1e-6)):
                positions[ion_idx] = candidate
                break
        else:
            positions[ion_idx] = candidate

    return positions, charges, mobility_factors, diameters


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
    diameter_nm: float,
    other_points: np.ndarray,
    other_diameters_nm: np.ndarray,
    hard_core_scale: float,
    box_length_nm: float,
) -> bool:
    """Return whether the proposed position overlaps another ion."""

    if len(other_points) == 0:
        return False
    diff = _minimal_image(other_points - point[None, :], box_length_nm)
    distances = np.linalg.norm(diff, axis=1)
    min_allowed = hard_core_scale * 0.5 * (diameter_nm + other_diameters_nm)
    return bool(np.any(distances < np.maximum(min_allowed, 1e-6)))


def run_salt_specific_replicate(
    concentration_m: float,
    seed: int,
    config: SaltSpecificBenchmarkConfig,
) -> dict[str, float]:
    """Run one salt-specific particle Monte Carlo replicate."""

    box_length_nm = box_length_nm_from_molarity(config.n_ions_total, concentration_m)
    rng = np.random.default_rng(seed)
    positions, charges, mobility_factors, diameters = _sample_ions_with_species(rng, config, box_length_nm)
    field_axis = np.zeros(config.dimension, dtype=float)
    field_axis[0] = 1.0
    beta = 1.0 / max(config.acceptance_temperature, EPS)
    indices = np.arange(config.n_ions_total)
    current_history_raw = []
    accepted_moves = 0
    attempted_moves = 0

    for _ in range(config.n_steps):
        sweep_current = 0.0
        for ion_idx in range(config.n_ions_total):
            current = positions[ion_idx].copy()
            charge = float(charges[ion_idx])
            mobility_factor = float(mobility_factors[ion_idx])
            proposal_raw = current + config.base_step_scale_nm * np.sqrt(mobility_factor) * rng.normal(size=config.dimension)
            proposal_raw += config.base_field_strength_nm * mobility_factor * charge * field_axis
            proposal = np.mod(proposal_raw, box_length_nm)
            dx = proposal_raw[0] - current[0]

            mask = indices != ion_idx
            attempted_moves += 1
            if _violates_hard_core(
                proposal,
                diameters[ion_idx],
                positions[mask],
                diameters[mask],
                config.hard_core_scale,
                box_length_nm,
            ):
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
            delta_energy = (energy_after - energy_before) - config.base_field_strength_nm * mobility_factor * charge * dx
            accept = delta_energy <= 0.0 or rng.random() < np.exp(-delta_energy * beta)
            if accept:
                positions[ion_idx] = proposal
                sweep_current += charge * dx
                accepted_moves += 1
        current_history_raw.append(sweep_current)

    production = np.array(current_history_raw[config.burn_in_steps :], dtype=float)
    n_pairs = config.n_ions_total / 2.0
    current_density_proxy = production.mean() / max(box_length_nm**config.dimension, EPS)
    molar_conductivity_proxy = production.mean() / max(n_pairs * config.base_field_strength_nm, EPS)

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
    """Aggregate replicate-level outputs by concentration."""

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


def write_summary(output_path: Path, merged_df: pd.DataFrame, fit_df: pd.DataFrame, config: SaltSpecificBenchmarkConfig) -> None:
    """Write a markdown summary for the salt-specific benchmark."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fit_row = fit_df.iloc[0]
    text = f"""# {config.electrolyte} Salt-Specific Monte Carlo Benchmark

## Purpose
This benchmark tests whether a salt-specific particle Monte Carlo model with ion-dependent mobility and diameter priors can reproduce the equivalent conductivity trend for {config.electrolyte}.

## Electrolyte Definition
- Electrolyte: {config.electrolyte}
- Cation: {config.cation}
- Anion: {config.anion}
- Cation limiting ionic conductivity: {config.cation_lambda_inf:.2f}
- Anion limiting ionic conductivity: {config.anion_lambda_inf:.2f}
- Cation diameter prior: {config.cation_diameter_nm:.3f} nm
- Anion diameter prior: {config.anion_diameter_nm:.3f} nm

## Benchmark Outcome
- Validation range: 0.005 to 0.100 M
- Monte Carlo dilute-fit R^2: {fit_row['r_squared']:.6f}
- Absolute RMSE versus literature table: {fit_row['rmse_literature']:.3f}
- Normalized RMSE versus literature table: {fit_row['rmse_literature_norm']:.4f}
- Absolute MAE versus literature table: {fit_row['mae_literature']:.3f}
- Normalized MAE versus literature table: {fit_row['mae_literature_norm']:.4f}

## Interpretation
The salt-specific model keeps the same particle Monte Carlo structure as the KCl benchmark, but assigns ion-dependent mobility factors from limiting ionic conductivities and ion-dependent hard-core diameters from physically motivated size priors. The Monte Carlo conductivity proxy is converted into literature units by a global {config.calibration_mode} calibration over the literature-covered range only."""
    output_path.write_text(text, encoding="utf-8")


def run_salt_specific_benchmark(
    config: SaltSpecificBenchmarkConfig,
    root_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run a salt-specific literature-anchored Monte Carlo benchmark."""

    root_dir = root_dir or Path(".")
    output_dir = root_dir / "outputs"
    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"
    text_dir = output_dir / "text"

    replicate_rows = []
    for concentration_idx, concentration_m in enumerate(config.simulation_concentrations_m):
        for replicate in range(config.n_replicates):
            seed = config.random_seed + 1000 * concentration_idx + 37 * replicate
            row = run_salt_specific_replicate(concentration_m, seed, config)
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

    return {
        "literature": literature_df,
        "replicates": replicates_df,
        "summary": summary_df,
        "merged": merged_df,
        "fit": fit_df,
        "overlap": overlap_df,
    }
