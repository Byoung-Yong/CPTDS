"""Universal strong-electrolyte particle Monte Carlo benchmark."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coffee_mc.literature_mc_benchmark import EPS
from coffee_mc.literature_mc_benchmark import fit_dilute_kohlrausch
from coffee_mc.salt_specific_benchmark import build_salt_config
from coffee_mc.salt_specific_benchmark import literature_dataframe
from coffee_mc.salt_specific_benchmark import run_salt_specific_replicate
from coffee_mc.salt_specific_benchmark import summarize_replicates


TARGET_ELECTROLYTES = ("KCl", "NaCl", "KI")

ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_J_PER_K = 1.380649e-23
EPSILON_0_F_PER_M = 8.8541878128e-12
AVOGADRO_PER_MOL = 6.02214076e23
WATER_RELATIVE_PERMITTIVITY_25C = 78.37
WATER_TEMPERATURE_25C_K = 298.15


# Selected final values are patched after the global search is run.
SELECTED_UNIVERSAL_PARAMS = {
    "interaction_cutoff_nm": 1.32,
    "base_field_strength_nm": 0.025,
    "base_step_scale_nm": 0.081,
    "interaction_scale": 0.042,
    "acceptance_temperature": 0.046,
    "screening_prefactor_nm_mhalf": 0.35,
    "hard_core_scale": 0.92,
}


@dataclass(frozen=True)
class UniversalElectrolyteBenchmarkConfig:
    """Shared Monte Carlo transport parameters for multiple 1:1 electrolytes."""

    electrolytes: tuple[str, ...] = TARGET_ELECTROLYTES
    validation_concentrations_m: tuple[float, ...] = (0.005, 0.010, 0.020, 0.050, 0.100)
    simulation_concentrations_m: tuple[float, ...] = (0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500, 1.000)
    interaction_cutoff_nm: float = 1.32
    base_field_strength_nm: float = 0.025
    base_step_scale_nm: float = 0.081
    interaction_scale: float = 0.042
    acceptance_temperature: float = 0.046
    screening_prefactor_nm_mhalf: float = 0.37
    hard_core_scale: float = 0.90
    calibration_mode: str = "affine"
    n_ions_total: int = 64
    n_steps: int = 120
    burn_in_steps: int = 40
    n_replicates: int = 2
    random_seed: int = 20260329
    output_prefix: str = "universal_electrolyte_benchmark"


def bjerrum_length_nm(
    temperature_k: float = WATER_TEMPERATURE_25C_K,
    relative_permittivity: float = WATER_RELATIVE_PERMITTIVITY_25C,
) -> float:
    """Return the Bjerrum length in water-like media in nm."""

    numerator = ELEMENTARY_CHARGE_C**2
    denominator = 4.0 * pi * EPSILON_0_F_PER_M * relative_permittivity * BOLTZMANN_J_PER_K * temperature_k
    return 1.0e9 * numerator / denominator


def debye_prefactor_nm_mhalf(
    temperature_k: float = WATER_TEMPERATURE_25C_K,
    relative_permittivity: float = WATER_RELATIVE_PERMITTIVITY_25C,
) -> float:
    """Return the Debye-length prefactor A for lambda_D = A / sqrt(I[M]) in nm."""

    numerator = relative_permittivity * EPSILON_0_F_PER_M * BOLTZMANN_J_PER_K * temperature_k
    denominator = 2.0 * (ELEMENTARY_CHARGE_C**2) * AVOGADRO_PER_MOL * 1000.0
    return 1.0e9 * np.sqrt(numerator / denominator)


def build_physically_anchored_universal_config(
    coupling_scale: float = 1.0,
    interaction_cutoff_nm: float = 1.32,
    base_field_strength_nm: float = 0.025,
    base_step_scale_nm: float = 0.081,
    hard_core_scale: float = 0.92,
    n_ions_total: int = 64,
    n_steps: int = 120,
    burn_in_steps: int = 40,
    n_replicates: int = 2,
    output_prefix: str = "physically_anchored_universal_benchmark",
) -> UniversalElectrolyteBenchmarkConfig:
    """Build a shared benchmark config with interaction terms tied to water 25 C constants."""

    return UniversalElectrolyteBenchmarkConfig(
        interaction_cutoff_nm=interaction_cutoff_nm,
        base_field_strength_nm=base_field_strength_nm,
        base_step_scale_nm=base_step_scale_nm,
        interaction_scale=bjerrum_length_nm() * coupling_scale,
        acceptance_temperature=1.0,
        screening_prefactor_nm_mhalf=debye_prefactor_nm_mhalf(),
        hard_core_scale=hard_core_scale,
        calibration_mode="through_origin",
        n_ions_total=n_ions_total,
        n_steps=n_steps,
        burn_in_steps=burn_in_steps,
        n_replicates=n_replicates,
        output_prefix=output_prefix,
    )


def build_selected_universal_config(**overrides) -> UniversalElectrolyteBenchmarkConfig:
    """Return the selected universal electrolyte benchmark configuration."""

    params = {**SELECTED_UNIVERSAL_PARAMS}
    params.update(overrides)
    return UniversalElectrolyteBenchmarkConfig(**params)


def build_universal_salt_config(
    electrolyte: str,
    config: UniversalElectrolyteBenchmarkConfig,
    concentrations: tuple[float, ...] | None = None,
) -> object:
    """Build one salt-specific config using shared transport parameters."""

    return build_salt_config(
        electrolyte,
        simulation_concentrations_m=concentrations or config.simulation_concentrations_m,
        n_ions_total=config.n_ions_total,
        interaction_cutoff_nm=config.interaction_cutoff_nm,
        base_field_strength_nm=config.base_field_strength_nm,
        base_step_scale_nm=config.base_step_scale_nm,
        interaction_scale=config.interaction_scale,
        acceptance_temperature=config.acceptance_temperature,
        screening_prefactor_nm_mhalf=config.screening_prefactor_nm_mhalf,
        hard_core_scale=config.hard_core_scale,
        calibration_mode=config.calibration_mode,
        n_steps=config.n_steps,
        burn_in_steps=config.burn_in_steps,
        n_replicates=config.n_replicates,
        random_seed=config.random_seed,
    )


def run_universal_raw_data(
    config: UniversalElectrolyteBenchmarkConfig,
    validation_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run raw Monte Carlo replicates and summary curves for all electrolytes."""

    summary_blocks = []
    literature_blocks = []
    proxy_fit_rows = []
    concentrations = config.validation_concentrations_m if validation_only else config.simulation_concentrations_m

    for salt_idx, electrolyte in enumerate(config.electrolytes):
        salt_seed = config.random_seed + 100000 * salt_idx
        salt_config = replace(
            build_universal_salt_config(electrolyte, config, concentrations=concentrations),
            random_seed=salt_seed,
        )
        literature_df = literature_dataframe(salt_config)

        replicate_rows = []
        for concentration_idx, concentration_m in enumerate(salt_config.simulation_concentrations_m):
            for replicate in range(salt_config.n_replicates):
                seed = salt_config.random_seed + 1000 * concentration_idx + 37 * replicate
                row = run_salt_specific_replicate(concentration_m, seed, salt_config)
                row["replicate"] = replicate
                row["electrolyte"] = electrolyte
                replicate_rows.append(row)

        replicate_df = pd.DataFrame(replicate_rows)
        summary_df = summarize_replicates(replicate_df)
        summary_df["electrolyte"] = electrolyte
        summary_df["lambda_infinite_dilution"] = salt_config.lambda_infinite_dilution
        summary_df["cation"] = salt_config.cation
        summary_df["anion"] = salt_config.anion
        summary_df["cation_lambda_inf"] = salt_config.cation_lambda_inf
        summary_df["anion_lambda_inf"] = salt_config.anion_lambda_inf
        summary_df["cation_diameter_nm"] = salt_config.cation_diameter_nm
        summary_df["anion_diameter_nm"] = salt_config.anion_diameter_nm
        summary_df["validation_point"] = summary_df["concentration_m"].isin(config.validation_concentrations_m)

        fit_proxy = fit_dilute_kohlrausch(summary_df, concentration_min_m=0.005, concentration_max_m=0.1).iloc[0].to_dict()
        fit_proxy["electrolyte"] = electrolyte

        literature_df["lambda_infinite_dilution"] = salt_config.lambda_infinite_dilution
        literature_df["cation"] = salt_config.cation
        literature_df["anion"] = salt_config.anion

        summary_blocks.append(summary_df)
        literature_blocks.append(literature_df)
        proxy_fit_rows.append(fit_proxy)

    summary_all = pd.concat(summary_blocks, ignore_index=True)
    literature_all = pd.concat(literature_blocks, ignore_index=True)
    proxy_fit_df = pd.DataFrame(proxy_fit_rows).sort_values("electrolyte").reset_index(drop=True)
    return summary_all, literature_all, proxy_fit_df


def fit_global_calibration(
    summary_df: pd.DataFrame,
    literature_df: pd.DataFrame,
    calibration_mode: str = "affine",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit one global proxy-to-conductivity calibration across all electrolytes."""

    merged = summary_df.merge(
        literature_df[
            [
                "electrolyte",
                "concentration_m",
                "lambda_literature",
                "lambda_literature_norm",
                "lambda_infinite_dilution",
            ]
        ],
        on=["electrolyte", "concentration_m", "lambda_infinite_dilution"],
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
    merged["lambda_mc_scaled_norm"] = merged["lambda_mc_scaled"] / merged["lambda_infinite_dilution"]

    overlap = merged[merged["lambda_literature"].notna()].copy()
    overlap["absolute_error"] = overlap["lambda_mc_scaled"] - overlap["lambda_literature"]
    overlap["absolute_error_norm"] = overlap["lambda_mc_scaled_norm"] - overlap["lambda_literature_norm"]

    pooled = {
        "calibration_mode": calibration_mode,
        "scale_proxy_to_literature": scale,
        "offset_proxy_to_literature": offset,
        "pooled_rmse_actual": float(np.sqrt(np.mean(overlap["absolute_error"] ** 2))),
        "pooled_rmse_norm": float(np.sqrt(np.mean(overlap["absolute_error_norm"] ** 2))),
        "pooled_mae_actual": float(np.mean(np.abs(overlap["absolute_error"]))),
        "pooled_mae_norm": float(np.mean(np.abs(overlap["absolute_error_norm"]))),
    }
    return merged, pooled


def compute_per_salt_metrics(calibrated_df: pd.DataFrame, proxy_fit_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-electrolyte validation metrics after global calibration."""

    rows = []
    for electrolyte, group in calibrated_df.groupby("electrolyte", sort=True):
        group = group.sort_values("concentration_m").reset_index(drop=True)
        overlap = group[group["lambda_literature"].notna()].copy()
        proxy_fit = proxy_fit_df.loc[proxy_fit_df["electrolyte"] == electrolyte].iloc[0]
        monotonic_violations = int(np.sum(np.diff(group["lambda_mc_scaled"].to_numpy(dtype=float)) > 0.0))
        rows.append(
            {
                "electrolyte": electrolyte,
                "lambda_infinite_dilution": float(group["lambda_infinite_dilution"].iloc[0]),
                "rmse_actual": float(np.sqrt(np.mean((overlap["lambda_mc_scaled"] - overlap["lambda_literature"]) ** 2))),
                "rmse_norm": float(np.sqrt(np.mean((overlap["lambda_mc_scaled_norm"] - overlap["lambda_literature_norm"]) ** 2))),
                "mae_actual": float(np.mean(np.abs(overlap["lambda_mc_scaled"] - overlap["lambda_literature"]))),
                "mae_norm": float(np.mean(np.abs(overlap["lambda_mc_scaled_norm"] - overlap["lambda_literature_norm"]))),
                "worst_abs_error_actual": float(np.max(np.abs(overlap["lambda_mc_scaled"] - overlap["lambda_literature"]))),
                "worst_abs_error_norm": float(np.max(np.abs(overlap["lambda_mc_scaled_norm"] - overlap["lambda_literature_norm"]))),
                "r_squared_proxy": float(proxy_fit["r_squared"]),
                "acceptance_fraction_mean": float(group["acceptance_fraction_mean"].mean()),
                "monotonic_violations": monotonic_violations,
            }
        )
    return pd.DataFrame(rows).sort_values("electrolyte").reset_index(drop=True)


def evaluate_universal_candidate(config: UniversalElectrolyteBenchmarkConfig) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Evaluate one shared-parameter candidate on the validation range only."""

    summary_df, literature_df, proxy_fit_df = run_universal_raw_data(config, validation_only=True)
    calibrated_df, pooled = fit_global_calibration(summary_df, literature_df, calibration_mode=config.calibration_mode)
    per_salt_df = compute_per_salt_metrics(calibrated_df, proxy_fit_df)

    min_r2 = float(per_salt_df["r_squared_proxy"].min())
    worst_norm = float(per_salt_df["rmse_norm"].max())
    total_monotonic_violations = int(per_salt_df["monotonic_violations"].sum())
    score = (
        pooled["pooled_rmse_norm"]
        + 0.65 * worst_norm
        + 0.25 * pooled["pooled_mae_norm"]
        + 0.60 * max(0.0, 0.965 - min_r2)
        + 0.020 * total_monotonic_violations
        + 0.001 * abs(pooled["offset_proxy_to_literature"])
    )

    result = {
        **asdict(config),
        **pooled,
        "min_r_squared_proxy": min_r2,
        "worst_rmse_norm": worst_norm,
        "worst_rmse_actual": float(per_salt_df["rmse_actual"].max()),
        "total_monotonic_violations": total_monotonic_violations,
        "score": float(score),
    }
    return result, per_salt_df, calibrated_df


def build_stage1_candidates(base: UniversalElectrolyteBenchmarkConfig) -> list[UniversalElectrolyteBenchmarkConfig]:
    """Construct the coarse global search grid around physically plausible values."""

    candidates = []
    for screening in (0.35, 0.37, 0.39):
        for temperature in (0.044, 0.046, 0.048):
            for hard_core in (0.88, 0.90, 0.92):
                candidates.append(
                    replace(
                        base,
                        screening_prefactor_nm_mhalf=screening,
                        acceptance_temperature=temperature,
                        hard_core_scale=hard_core,
                    )
                )
    return candidates


def build_stage2_candidates(base: UniversalElectrolyteBenchmarkConfig) -> list[UniversalElectrolyteBenchmarkConfig]:
    """Construct a local refinement set around the best stage-1 point."""

    screening_values = sorted(
        {
            round(max(0.30, base.screening_prefactor_nm_mhalf - 0.01), 3),
            round(max(0.30, base.screening_prefactor_nm_mhalf - 0.005), 3),
            round(base.screening_prefactor_nm_mhalf, 3),
            round(min(0.42, base.screening_prefactor_nm_mhalf + 0.005), 3),
            round(min(0.42, base.screening_prefactor_nm_mhalf + 0.01), 3),
        }
    )
    temperature_values = sorted(
        {
            round(max(0.040, base.acceptance_temperature - 0.002), 3),
            round(max(0.040, base.acceptance_temperature - 0.001), 3),
            round(base.acceptance_temperature, 3),
            round(min(0.052, base.acceptance_temperature + 0.001), 3),
            round(min(0.052, base.acceptance_temperature + 0.002), 3),
        }
    )
    hard_core_values = sorted(
        {
            round(max(0.84, base.hard_core_scale - 0.02), 3),
            round(max(0.84, base.hard_core_scale - 0.01), 3),
            round(base.hard_core_scale, 3),
            round(min(0.96, base.hard_core_scale + 0.01), 3),
            round(min(0.96, base.hard_core_scale + 0.02), 3),
        }
    )

    candidates = []
    for screening in screening_values:
        candidates.append(replace(base, screening_prefactor_nm_mhalf=screening))
    for temperature in temperature_values:
        candidates.append(replace(base, acceptance_temperature=temperature))
    for hard_core in hard_core_values:
        candidates.append(replace(base, hard_core_scale=hard_core))

    unique = {}
    for candidate in candidates:
        key = (
            candidate.screening_prefactor_nm_mhalf,
            candidate.acceptance_temperature,
            candidate.hard_core_scale,
        )
        unique[key] = candidate
    return list(unique.values())


def plot_universal_comparison_actual(
    calibrated_df: pd.DataFrame,
    output_path: Path,
    x_max: float | None = None,
) -> None:
    """Plot actual conductivity curves for all electrolytes under one global calibration."""

    electrolytes = list(calibrated_df["electrolyte"].drop_duplicates())
    fig, axes = plt.subplots(1, len(electrolytes), figsize=(5.2 * len(electrolytes), 4.6), constrained_layout=True, sharex=False)
    if len(electrolytes) == 1:
        axes = [axes]

    for ax, electrolyte in zip(axes, electrolytes):
        group = calibrated_df[calibrated_df["electrolyte"] == electrolyte].sort_values("concentration_m")
        literature = group[group["lambda_literature"].notna()].copy()
        extension = group[group["lambda_literature"].isna()].copy()
        ax.errorbar(
            group["sqrt_concentration_m_half"],
            group["lambda_mc_scaled"],
            yerr=1.96 * group["lambda_mc_scaled_sem"],
            fmt="o-",
            ms=4,
            lw=1.5,
            color="#1f77b4",
            ecolor="#8fb7e8",
            capsize=2,
            label="Monte Carlo",
        )
        ax.scatter(
            literature["sqrt_concentration_m_half"],
            literature["lambda_literature"],
            s=52,
            color="#d62728",
            marker="s",
            zorder=4,
            label="Literature",
        )
        if len(extension) > 0:
            ax.scatter(
                extension["sqrt_concentration_m_half"],
                extension["lambda_mc_scaled"],
                s=42,
                facecolors="none",
                edgecolors="#1f77b4",
                zorder=3,
                label="MC extension",
            )
        ax.set_title(electrolyte)
        ax.set_xlabel(r"$\sqrt{c}$ (M$^{1/2}$)")
        if x_max is not None:
            ax.set_xlim(0.0, x_max)
            visible = group[group["sqrt_concentration_m_half"] <= x_max].copy()
            visible_lit = visible["lambda_literature"].dropna()
            y_min = float(min(visible["lambda_mc_scaled"].min(), visible_lit.min()))
            y_max = float(max(visible["lambda_mc_scaled"].max(), visible_lit.max()))
            ax.set_ylim(y_min - max(1.0, 0.08 * (y_max - y_min)), y_max + max(1.0, 0.08 * (y_max - y_min)))
        secax = ax.secondary_xaxis(
            "top",
            functions=(lambda x: np.square(np.maximum(x, 0.0)), lambda c: np.sqrt(np.maximum(c, 0.0))),
        )
        secax.set_xlabel(r"$c$ (M)")
        secax.set_xticks([0.005, 0.01, 0.1] if x_max is not None else [0.005, 0.01, 0.1, 0.5, 1.0])
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(r"Equivalent conductivity, $\Lambda$ ($10^{-4}$ m$^2$ S mol$^{-1}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Universal strong-electrolyte Monte Carlo benchmark", y=1.10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_universal_residuals(per_salt_df: pd.DataFrame, output_path: Path) -> None:
    """Plot pooled and per-salt normalized RMSE values."""

    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    ax.bar(per_salt_df["electrolyte"], per_salt_df["rmse_norm"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Normalized RMSE")
    ax.set_title("Universal benchmark: per-salt normalized RMSE")
    ax.grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_universal_summary(
    output_path: Path,
    config: UniversalElectrolyteBenchmarkConfig,
    pooled_metrics: dict[str, float],
    per_salt_df: pd.DataFrame,
    through_origin_metrics: dict[str, float] | None = None,
) -> None:
    """Write a manuscript-style markdown summary."""

    per_salt_text = per_salt_df.to_string(index=False)
    through_origin_text = ""
    if through_origin_metrics is not None:
        through_origin_text = (
            "\n## Through-Origin Sensitivity\n"
            f"- pooled RMSE (actual): {through_origin_metrics['pooled_rmse_actual']:.3f}\n"
            f"- pooled RMSE (normalized): {through_origin_metrics['pooled_rmse_norm']:.4f}\n"
            f"- offset fixed to: {through_origin_metrics['offset_proxy_to_literature']:.3f}\n"
            f"- scale: {through_origin_metrics['scale_proxy_to_literature']:.3f}\n"
        )

    text = f"""# Universal Strong-Electrolyte Monte Carlo Benchmark

## Purpose
This benchmark tests whether a single shared particle Monte Carlo transport engine can reproduce the equivalent-conductivity trends of multiple strong 1:1 electrolytes using only ion-specific physical inputs and one globally shared proxy-to-unit calibration.

## Physical Inputs
- Electrolytes: {", ".join(config.electrolytes)}
- Ion-specific mobility factors from limiting ionic conductivities
- Ion-specific hard-core diameters from physically motivated size priors
- Shared transport parameters:
  - interaction_cutoff_nm = {config.interaction_cutoff_nm:.3f}
  - base_field_strength_nm = {config.base_field_strength_nm:.3f}
  - base_step_scale_nm = {config.base_step_scale_nm:.3f}
  - interaction_scale = {config.interaction_scale:.3f}
  - acceptance_temperature = {config.acceptance_temperature:.3f}
  - screening_prefactor_nm_mhalf = {config.screening_prefactor_nm_mhalf:.3f}
  - hard_core_scale = {config.hard_core_scale:.3f}

## Global Calibration
- calibration mode: {pooled_metrics['calibration_mode']}
- global scale = {pooled_metrics['scale_proxy_to_literature']:.3f}
- global offset = {pooled_metrics['offset_proxy_to_literature']:.3f}
- pooled RMSE (actual) = {pooled_metrics['pooled_rmse_actual']:.3f}
- pooled RMSE (normalized) = {pooled_metrics['pooled_rmse_norm']:.4f}
- pooled MAE (actual) = {pooled_metrics['pooled_mae_actual']:.3f}
- pooled MAE (normalized) = {pooled_metrics['pooled_mae_norm']:.4f}

## Per-Salt Metrics
{per_salt_text}

## Interpretation
The same strong-electrolyte Monte Carlo engine was applied to KCl, NaCl, and KI without salt-by-salt re-tuning of the transport parameters. Salt-to-salt differences entered only through ion-specific limiting ionic conductivities and diameter priors, while the conductivity scale conversion was fitted once globally across the pooled literature points. This provides a stronger foundation for subsequent matrix-screening simulations than a separately tuned salt-specific benchmark.{through_origin_text}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def run_final_universal_benchmark(
    config: UniversalElectrolyteBenchmarkConfig,
    root_dir: Path | None = None,
) -> dict[str, object]:
    """Run the final universal benchmark and save outputs."""

    root_dir = root_dir or Path(".")
    output_dir = root_dir / "outputs"
    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"
    text_dir = output_dir / "text"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    summary_df, literature_df, proxy_fit_df = run_universal_raw_data(config, validation_only=False)
    calibrated_df, pooled_metrics = fit_global_calibration(summary_df, literature_df, calibration_mode=config.calibration_mode)
    per_salt_df = compute_per_salt_metrics(calibrated_df, proxy_fit_df)
    _, through_origin_metrics = fit_global_calibration(summary_df, literature_df, calibration_mode="through_origin")

    overlap_df = calibrated_df[calibrated_df["lambda_literature"].notna()].copy()
    overlap_df["absolute_error"] = overlap_df["lambda_mc_scaled"] - overlap_df["lambda_literature"]
    overlap_df["absolute_error_norm"] = overlap_df["lambda_mc_scaled_norm"] - overlap_df["lambda_literature_norm"]
    pooled_summary_df = pd.DataFrame([{**pooled_metrics, **asdict(config)}])
    through_origin_df = pd.DataFrame([{**through_origin_metrics, **asdict(config)}])

    prefix = config.output_prefix
    summary_df.to_csv(table_dir / f"{prefix}_summary.csv", index=False)
    literature_df.to_csv(table_dir / f"{prefix}_literature.csv", index=False)
    calibrated_df.to_csv(table_dir / f"{prefix}_merged.csv", index=False)
    overlap_df.to_csv(table_dir / f"{prefix}_overlap.csv", index=False)
    proxy_fit_df.to_csv(table_dir / f"{prefix}_proxy_fit.csv", index=False)
    per_salt_df.to_csv(table_dir / f"{prefix}_per_salt_metrics.csv", index=False)
    pooled_summary_df.to_csv(table_dir / f"{prefix}_pooled_metrics.csv", index=False)
    through_origin_df.to_csv(table_dir / f"{prefix}_through_origin_metrics.csv", index=False)

    plot_universal_comparison_actual(calibrated_df, fig_dir / f"{prefix}_comparison_actual_units.png")
    plot_universal_comparison_actual(calibrated_df, fig_dir / f"{prefix}_comparison_actual_units_zoom.png", x_max=0.4)
    plot_universal_residuals(per_salt_df, fig_dir / f"{prefix}_per_salt_rmse_norm.png")
    write_universal_summary(text_dir / f"{prefix}_summary.md", config, pooled_metrics, per_salt_df, through_origin_metrics)

    return {
        "config": config,
        "summary": summary_df,
        "literature": literature_df,
        "merged": calibrated_df,
        "overlap": overlap_df,
        "proxy_fit": proxy_fit_df,
        "per_salt": per_salt_df,
        "pooled_metrics": pooled_metrics,
        "through_origin_metrics": through_origin_metrics,
    }
