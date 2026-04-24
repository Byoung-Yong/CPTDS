"""Physicalization helpers for converting effective dimensionless coffee variables to SI-like units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


AVOGADRO_PER_NM3_PER_M = 0.602214076  # number density corresponding to 1 M in nm^-3


@dataclass(frozen=True)
class EffectiveSIConfig:
    """Reference box-length setting used to physicalize the effective coffee model."""

    fixed_box_length_nm: float = 15.0
    interaction_cutoff_reduced: float = 0.24
    trap_radius_reduced: float = 0.10


def effective_box_length_nm(mean_total_ions: float, config: EffectiveSIConfig | None = None) -> float:
    """Return the effective box side length in nm under the fixed reporting scale."""

    config = config or EffectiveSIConfig()
    return float(config.fixed_box_length_nm)


def augment_with_si_units(
    df: pd.DataFrame,
    mean_total_ions: float,
    config: EffectiveSIConfig | None = None,
) -> pd.DataFrame:
    """Add effective SI columns to a fitted coffee-state dataframe."""

    config = config or EffectiveSIConfig()
    l_box_nm = effective_box_length_nm(mean_total_ions, config=config)
    volume_nm3 = l_box_nm**3

    out = df.copy()
    out["effective_box_length_nm"] = l_box_nm
    out["effective_box_volume_nm3"] = volume_nm3
    out["ion_particle_concentration_mM"] = out["fitted_n_ions"].to_numpy(dtype=float) / (
        AVOGADRO_PER_NM3_PER_M * volume_nm3
    ) * 1000.0
    out["salt_equivalent_concentration_mM"] = out["ion_particle_concentration_mM"] / 2.0
    out["ideal_1to1_electrolyte_concentration_mM"] = out["salt_equivalent_concentration_mM"]
    if "fitted_blocker_count" in out.columns:
        out["blocker_concentration_mM"] = out["fitted_blocker_count"].to_numpy(dtype=float) / (
            AVOGADRO_PER_NM3_PER_M * volume_nm3
        ) * 1000.0
    if "fitted_blocker_size" in out.columns:
        out["blocker_radius_nm"] = out["fitted_blocker_size"].to_numpy(dtype=float) * l_box_nm
        out["blocker_diameter_nm"] = 2.0 * out["blocker_radius_nm"]
        out["blocker_volume_fraction"] = (
            out["fitted_blocker_count"].to_numpy(dtype=float)
            * (4.0 * np.pi / 3.0)
            * np.power(out["blocker_radius_nm"].to_numpy(dtype=float), 3)
            / volume_nm3
        )
    out["interaction_cutoff_nm"] = config.interaction_cutoff_reduced * l_box_nm
    out["trap_radius_nm"] = config.trap_radius_reduced * l_box_nm
    return out
