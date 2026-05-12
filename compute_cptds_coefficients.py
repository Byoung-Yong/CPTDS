"""Compute CPTDS slope, intercept, and Index from user dilution data.

Input CSV columns:
    sample_id, tds, conductivity

Optional columns such as coffee, method, roast_level, or notes are preserved by
taking the first non-null value within each sample_id group.

The exported slope uses the concentration-penalty convention
CPTDS = intercept - slope * sqrt(TDS). Samples whose CPTDS increases with
sqrt(TDS) are outside this retained model and are rejected by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"sample_id", "tds", "conductivity"}
SIGN_TOL = 1e-12


def coefficient_table(raw_df: pd.DataFrame, allow_nonpenalty: bool = False) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = raw_df.copy()
    df["tds"] = pd.to_numeric(df["tds"], errors="coerce")
    df["conductivity"] = pd.to_numeric(df["conductivity"], errors="coerce")
    df = df.dropna(subset=["sample_id", "tds", "conductivity"])
    df = df.loc[df["tds"] > 0].copy()
    if df.empty:
        raise ValueError("No valid positive-TDS rows were found.")

    rows: list[dict] = []
    metadata_cols = [col for col in df.columns if col not in {"tds", "conductivity"}]
    for sample_id, group in df.groupby("sample_id", sort=False):
        if len(group) < 2:
            raise ValueError(f"Sample {sample_id!r} has fewer than two valid dilution points.")
        x = np.sqrt(group["tds"].to_numpy(dtype=float))
        y = (group["conductivity"].to_numpy(dtype=float) / group["tds"].to_numpy(dtype=float))
        slope_raw, intercept = np.polyfit(x, y, 1)
        slope = -float(slope_raw)
        intercept = float(intercept)
        if intercept <= 0:
            raise ValueError(f"Sample {sample_id!r} has a non-positive fitted intercept.")
        if slope < -SIGN_TOL and not allow_nonpenalty:
            raise ValueError(
                f"Sample {sample_id!r} has CPTDS increasing with sqrt(TDS). "
                "This is outside the concentration-penalty model; use "
                "--allow-nonpenalty only to export signed diagnostic rows."
            )
        if slope < 0 and abs(slope) <= SIGN_TOL:
            slope = 0.0
        trend_status = "penalty" if slope > 0 else "flat"
        index_exp = float(slope**2 / intercept**3)
        if slope < 0:
            trend_status = "nonpenalty_increasing_cptds"
            index_exp = np.nan
        row = {
            "sample_id": sample_id,
            "n_points": int(len(group)),
            "slope": slope,
            "intercept": intercept,
            "index_exp": index_exp,
            "sqrt_tds_coefficient": float(slope_raw),
            "trend_status": trend_status,
        }
        for col in metadata_cols:
            if col == "sample_id":
                continue
            values = group[col].dropna()
            if len(values) > 0:
                row[col] = values.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw dilution CSV.")
    parser.add_argument("--output", required=True, help="Output coefficient CSV.")
    parser.add_argument(
        "--allow-nonpenalty",
        action="store_true",
        help="Export signed diagnostic rows for samples outside CPTDS = intercept - slope*sqrt(TDS).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    raw_df = pd.read_csv(input_path)
    out_df = coefficient_table(raw_df, allow_nonpenalty=args.allow_nonpenalty)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
