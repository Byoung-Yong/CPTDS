"""Fit conductance-equivalent Monte Carlo parameters to user CPTDS coefficients.

Input CSV columns:
    sample_id, slope, intercept

Optional columns are preserved in the output. If index_exp is absent, it is
computed as slope**2 / intercept**3.

The fitted variables are conductance-equivalent coordinates:
    fitted_n_ions
    fitted_blocker_count
    fitted_blocker_size

They are not direct analytical concentrations or molecular radii.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from coffee_mc.config import PhysicalConstants
from coffee_mc.generic_manifold import (
    ManifoldConfig,
    evaluate_manifold_grid,
    prepare_interpolators,
    quick_manifold_config,
    shared_parameter_dict,
    solve_scale,
)
from coffee_mc.models import MODEL_DEFINITIONS
from coffee_mc.si_physicalization import EffectiveSIConfig, augment_with_si_units
from coffee_mc.simulation import SimulationParameters, run_replicated_condition


REQUIRED_COLUMNS = {"sample_id", "slope", "intercept"}


def _bounded_from_latent(latent: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + (high - low) * expit(np.asarray(latent, dtype=float))


def _initial_latent_for_center(values: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = (np.asarray(values, dtype=float) - low) / max(high - low, 1e-12)
    return logit(np.clip(scaled, 1e-4, 1.0 - 1e-4))


def _prepare_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["slope"] = pd.to_numeric(df["slope"], errors="coerce")
    df["intercept"] = pd.to_numeric(df["intercept"], errors="coerce")
    df = df.dropna(subset=["sample_id", "slope", "intercept"])
    if df.empty:
        raise ValueError("No valid coefficient rows were found.")
    if "index_exp" not in df.columns:
        df["index_exp"] = df["slope"] ** 2 / np.maximum(df["intercept"] ** 3, 1e-12)
    else:
        df["index_exp"] = pd.to_numeric(df["index_exp"], errors="coerce")
        df["index_exp"] = df["index_exp"].fillna(df["slope"] ** 2 / np.maximum(df["intercept"] ** 3, 1e-12))
    return df.reset_index(drop=True)


def _decode_theta(theta: np.ndarray, n_states: int, config: ManifoldConfig) -> pd.DataFrame:
    theta = np.asarray(theta, dtype=float).reshape(n_states, 3)
    n_min, n_max = min(config.n_ions_values), max(config.n_ions_values)
    c_min, c_max = min(config.blocker_count_values), max(config.blocker_count_values)
    s_min, s_max = min(config.blocker_size_values), max(config.blocker_size_values)
    return pd.DataFrame(
        {
            "fitted_n_ions": _bounded_from_latent(theta[:, 0], n_min, n_max),
            "fitted_blocker_count": _bounded_from_latent(theta[:, 1], c_min, c_max),
            "fitted_blocker_size": _bounded_from_latent(theta[:, 2], s_min, s_max),
        }
    )


def _feature_dataframe(local_df: pd.DataFrame, interpolators: dict[str, object], config: ManifoldConfig) -> pd.DataFrame:
    points = local_df[["fitted_n_ions", "fitted_blocker_count", "fitted_blocker_size"]].to_numpy(dtype=float)
    k_rel = np.asarray(interpolators["k_interp"](points), dtype=float)
    lambda_rel = np.asarray(interpolators["lambda_interp"](points), dtype=float)
    n_ref = float(max(config.n_ions_values))
    out = local_df.copy()
    out["k_rel"] = k_rel
    out["lambda_rel"] = lambda_rel
    out["slope_feature"] = (out["fitted_n_ions"].to_numpy(dtype=float) / n_ref) ** 1.5 * k_rel
    out["intercept_feature"] = (out["fitted_n_ions"].to_numpy(dtype=float) / n_ref) * lambda_rel
    return out


def fit_coefficients(df: pd.DataFrame, grid_df: pd.DataFrame, config: ManifoldConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    interpolators = prepare_interpolators(grid_df)
    n_states = len(df)
    slope_obs = df["slope"].to_numpy(dtype=float)
    intercept_obs = df["intercept"].to_numpy(dtype=float)
    sigma_slope = config.sigma_slope_rel * max(float(np.max(slope_obs)), 1e-12)
    sigma_intercept = config.sigma_intercept_rel * max(float(np.max(intercept_obs)), 1e-12)

    def objective(theta: np.ndarray) -> float:
        local = _decode_theta(theta, n_states, config)
        features = _feature_dataframe(local, interpolators, config)
        slope_scale = solve_scale(features["slope_feature"].to_numpy(float), slope_obs, sigma_slope)
        intercept_scale = solve_scale(features["intercept_feature"].to_numpy(float), intercept_obs, sigma_intercept)
        slope_pred = slope_scale * features["slope_feature"].to_numpy(float)
        intercept_pred = intercept_scale * features["intercept_feature"].to_numpy(float)
        fit = np.sum(((slope_obs - slope_pred) / sigma_slope) ** 2)
        fit += np.sum(((intercept_obs - intercept_pred) / sigma_intercept) ** 2)
        ridge = config.ridge_strength * float(np.sum(np.square(theta)))
        return float(fit + ridge)

    n_mid = np.full(n_states, np.mean(config.n_ions_values), dtype=float)
    c_mid = np.full(n_states, np.mean(config.blocker_count_values), dtype=float)
    s_mid = np.full(n_states, np.mean(config.blocker_size_values), dtype=float)
    base = np.column_stack(
        [
            _initial_latent_for_center(n_mid, min(config.n_ions_values), max(config.n_ions_values)),
            _initial_latent_for_center(c_mid, min(config.blocker_count_values), max(config.blocker_count_values)),
            _initial_latent_for_center(s_mid, min(config.blocker_size_values), max(config.blocker_size_values)),
        ]
    ).ravel()
    starts = [
        base,
        np.zeros_like(base),
        base + 0.5,
        base - 0.5,
    ]
    bounds = [(-8.0, 8.0)] * len(base)
    fits = [minimize(objective, x0=start, method="L-BFGS-B", bounds=bounds) for start in starts]
    result = min(fits, key=lambda res: res.fun)

    fitted = _decode_theta(result.x, n_states, config)
    features = _feature_dataframe(fitted, interpolators, config)
    slope_scale = solve_scale(features["slope_feature"].to_numpy(float), slope_obs, sigma_slope)
    intercept_scale = solve_scale(features["intercept_feature"].to_numpy(float), intercept_obs, sigma_intercept)

    out = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    out["slope_scale"] = slope_scale
    out["intercept_scale"] = intercept_scale
    out["slope_pred"] = slope_scale * out["slope_feature"]
    out["intercept_pred"] = intercept_scale * out["intercept_feature"]
    out["index_pred"] = out["slope_pred"] ** 2 / np.maximum(out["intercept_pred"] ** 3, 1e-12)
    out["slope_error"] = out["slope_pred"] - out["slope"]
    out["intercept_error"] = out["intercept_pred"] - out["intercept"]
    out["index_error"] = out["index_pred"] - out["index_exp"]

    summary = pd.DataFrame(
        [
            {
                "objective": float(result.fun),
                "n_states": int(n_states),
                "n_latent_parameters": int(len(base)),
                "slope_rmse": float(np.sqrt(np.mean(np.square(out["slope_error"])))),
                "intercept_rmse": float(np.sqrt(np.mean(np.square(out["intercept_error"])))),
                "index_rmse": float(np.sqrt(np.mean(np.square(out["index_error"])))),
                "slope_scale": float(slope_scale),
                "intercept_scale": float(intercept_scale),
            }
        ]
    )
    return out, summary


def direct_reevaluate(fit_df: pd.DataFrame, config: ManifoldConfig) -> pd.DataFrame:
    if config.final_trials <= 0:
        return fit_df.copy()

    shared = shared_parameter_dict(config)
    rows: list[dict] = []
    for idx, row in enumerate(fit_df.itertuples(index=False)):
        n_ions = int(round(float(row.fitted_n_ions)))
        blocker_count = int(round(float(row.fitted_blocker_count)))
        blocker_size = float(row.fitted_blocker_size)
        params = SimulationParameters(
            n_ions=n_ions,
            blocker_count=blocker_count,
            blocker_radius_mean=blocker_size,
            **shared,
        )
        summary = run_replicated_condition(
            params=params,
            model=MODEL_DEFINITIONS["E"],
            physical=PhysicalConstants(),
            n_trials=config.final_trials,
            seed=config.seed + 200000 + 101 * idx,
        )
        rows.append(
            {
                "sample_id": row.sample_id,
                "simulation_n_ions": n_ions,
                "simulation_blocker_count": blocker_count,
                "simulation_blocker_size": blocker_size,
                "k_eff_over_k0_final": summary["k_eff_over_k0_mean"],
                "lambda0_eff_over_lambda0_final": summary["lambda0_eff_over_lambda0_mean"],
            }
        )
    direct = pd.DataFrame(rows)
    out = fit_df.merge(direct, on="sample_id", how="left")
    n_ref = float(max(config.n_ions_values))
    out["slope_feature_final"] = (out["simulation_n_ions"].to_numpy(float) / n_ref) ** 1.5 * out[
        "k_eff_over_k0_final"
    ].to_numpy(float)
    out["intercept_feature_final"] = (out["simulation_n_ions"].to_numpy(float) / n_ref) * out[
        "lambda0_eff_over_lambda0_final"
    ].to_numpy(float)
    slope_scale = solve_scale(out["slope_feature_final"].to_numpy(float), out["slope"].to_numpy(float), 1.0)
    intercept_scale = solve_scale(out["intercept_feature_final"].to_numpy(float), out["intercept"].to_numpy(float), 1.0)
    out["slope_pred_final"] = slope_scale * out["slope_feature_final"]
    out["intercept_pred_final"] = intercept_scale * out["intercept_feature_final"]
    out["index_pred_final"] = out["slope_pred_final"] ** 2 / np.maximum(out["intercept_pred_final"] ** 3, 1e-12)
    return out


def load_or_generate_grid(grid_path: Path, config: ManifoldConfig, force: bool = False) -> pd.DataFrame:
    if grid_path.exists() and not force:
        return pd.read_csv(grid_path)
    grid_df = evaluate_manifold_grid(config)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(grid_path, index=False)
    return grid_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Coefficient CSV with sample_id, slope, intercept.")
    parser.add_argument("--output-dir", default="outputs/user_fit", help="Directory for output CSV files.")
    parser.add_argument("--grid-path", default=None, help="Optional path for the Monte Carlo grid CSV.")
    parser.add_argument("--quick", action="store_true", help="Use a small grid for a fast smoke test.")
    parser.add_argument("--force-grid", action="store_true", help="Regenerate the grid even if grid-path exists.")
    parser.add_argument("--grid-trials", type=int, default=None, help="Replicates per grid condition.")
    parser.add_argument("--final-trials", type=int, default=None, help="Direct reevaluation replicates per fitted state.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum parallel workers for grid generation.")
    parser.add_argument("--box-length-nm", type=float, default=15.0, help="Effective reporting box length.")
    args = parser.parse_args()

    config = quick_manifold_config() if args.quick else ManifoldConfig()
    updates = {}
    if args.grid_trials is not None:
        updates["grid_trials"] = args.grid_trials
    if args.final_trials is not None:
        updates["final_trials"] = args.final_trials
    if args.max_workers is not None:
        updates["max_workers"] = args.max_workers
    if updates:
        config = ManifoldConfig(**{**config.__dict__, **updates})

    output_dir = Path(args.output_dir)
    grid_path = Path(args.grid_path) if args.grid_path else output_dir / "effective_conductance_grid.csv"
    df = _prepare_input(Path(args.input))
    grid_df = load_or_generate_grid(grid_path, config, force=args.force_grid)
    fit_df, summary_df = fit_coefficients(df, grid_df, config)
    final_df = direct_reevaluate(fit_df, config)
    si_df = augment_with_si_units(
        final_df,
        mean_total_ions=float(final_df["fitted_n_ions"].mean()),
        config=EffectiveSIConfig(fixed_box_length_nm=float(args.box_length_nm)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fit_df.to_csv(output_dir / "fit_interpolated.csv", index=False)
    final_df.to_csv(output_dir / "fit_direct.csv", index=False)
    si_df.to_csv(output_dir / "fit_direct_si.csv", index=False)
    summary_df.to_csv(output_dir / "fit_summary.csv", index=False)
    print(f"Saved {output_dir / 'fit_interpolated.csv'}")
    print(f"Saved {output_dir / 'fit_direct.csv'}")
    print(f"Saved {output_dir / 'fit_direct_si.csv'}")
    print(f"Saved {output_dir / 'fit_summary.csv'}")


if __name__ == "__main__":
    main()
