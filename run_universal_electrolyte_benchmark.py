"""Search and run a universal strong-electrolyte Monte Carlo benchmark."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from coffee_mc.universal_electrolyte_benchmark import UniversalElectrolyteBenchmarkConfig
from coffee_mc.universal_electrolyte_benchmark import build_stage1_candidates
from coffee_mc.universal_electrolyte_benchmark import build_stage2_candidates
from coffee_mc.universal_electrolyte_benchmark import evaluate_universal_candidate
from coffee_mc.universal_electrolyte_benchmark import run_final_universal_benchmark


def _evaluate_candidates(
    candidates: list[UniversalElectrolyteBenchmarkConfig],
    output_csv: Path,
    label: str,
) -> pd.DataFrame:
    """Evaluate candidate configurations serially and persist progress."""

    rows = []
    for idx, candidate in enumerate(candidates, start=1):
        result, _, _ = evaluate_universal_candidate(candidate)
        result["stage"] = label
        result["candidate_index"] = idx
        rows.append(result)
        pd.DataFrame(rows).sort_values("score").to_csv(output_csv, index=False)
        print(
            f"[{label}] {idx:02d}/{len(candidates):02d} "
            f"score={result['score']:.5f} pooled_norm_rmse={result['pooled_rmse_norm']:.5f} "
            f"worst_norm_rmse={result['worst_rmse_norm']:.5f}"
        )
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def main() -> None:
    """Run the global search and the final universal benchmark."""

    root_dir = Path(".")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = root_dir / "outputs" / "temp_search" / "universal_electrolyte_benchmark" / stamp
    temp_dir.mkdir(parents=True, exist_ok=True)

    stage_base = UniversalElectrolyteBenchmarkConfig(
        n_ions_total=64,
        n_steps=120,
        burn_in_steps=40,
        n_replicates=2,
        simulation_concentrations_m=(0.005, 0.010, 0.020, 0.050, 0.100),
    )
    stage1_candidates = build_stage1_candidates(stage_base)
    stage1_df = _evaluate_candidates(stage1_candidates, temp_dir / "stage1_results.csv", "stage1")
    best_stage1 = stage1_df.iloc[0]
    best_stage1_config = replace(
        stage_base,
        screening_prefactor_nm_mhalf=float(best_stage1["screening_prefactor_nm_mhalf"]),
        acceptance_temperature=float(best_stage1["acceptance_temperature"]),
        hard_core_scale=float(best_stage1["hard_core_scale"]),
    )

    stage2_candidates = build_stage2_candidates(best_stage1_config)
    stage2_df = _evaluate_candidates(stage2_candidates, temp_dir / "stage2_results.csv", "stage2")
    best_stage2 = stage2_df.iloc[0]

    final_config = UniversalElectrolyteBenchmarkConfig(
        interaction_cutoff_nm=float(best_stage2["interaction_cutoff_nm"]),
        base_field_strength_nm=float(best_stage2["base_field_strength_nm"]),
        base_step_scale_nm=float(best_stage2["base_step_scale_nm"]),
        interaction_scale=float(best_stage2["interaction_scale"]),
        acceptance_temperature=float(best_stage2["acceptance_temperature"]),
        screening_prefactor_nm_mhalf=float(best_stage2["screening_prefactor_nm_mhalf"]),
        hard_core_scale=float(best_stage2["hard_core_scale"]),
        calibration_mode=str(best_stage2["calibration_mode"]),
        n_ions_total=128,
        n_steps=520,
        burn_in_steps=170,
        n_replicates=10,
        simulation_concentrations_m=(0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500, 1.000),
    )
    final_result = run_final_universal_benchmark(final_config, root_dir=root_dir)

    pd.DataFrame([asdict(final_config)]).to_csv(
        root_dir / "outputs" / "tables" / "universal_electrolyte_benchmark_selected_config.csv",
        index=False,
    )
    pd.DataFrame([asdict(final_config)]).to_csv(temp_dir / "selected_config.csv", index=False)

    print("\nSelected universal configuration:")
    print(pd.DataFrame([asdict(final_config)]).to_string(index=False))
    print("\nPer-salt metrics:")
    print(final_result["per_salt"].to_string(index=False))
    print("\nPooled metrics:")
    print(pd.DataFrame([final_result["pooled_metrics"]]).to_string(index=False))


if __name__ == "__main__":
    main()
