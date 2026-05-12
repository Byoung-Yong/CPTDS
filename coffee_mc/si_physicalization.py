"""Helpers for optional SI-like reporting of dimensionless coffee variables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


AVOGADRO_PER_NM3_PER_M = 0.602214076  # number density corresponding to 1 M in nm^-3


@dataclass(frozen=True)
class EffectiveSIConfig:
    """Reference box-length setting for conductance-equivalent reporting."""

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
    """Add conductance-equivalent SI-like reporting columns to a fitted dataframe."""

    config = config or EffectiveSIConfig()
    l_box_nm = effective_box_length_nm(mean_total_ions, config=config)
    volume_nm3 = l_box_nm**3

    out = df.copy()
    out["effective_box_length_nm"] = l_box_nm
    out["effective_box_volume_nm3"] = volume_nm3
    out["effective_ion_particle_concentration_mM"] = out["fitted_n_ions"].to_numpy(dtype=float) / (
        AVOGADRO_PER_NM3_PER_M * volume_nm3
    ) * 1000.0
    out["effective_salt_equivalent_concentration_mM"] = out["effective_ion_particle_concentration_mM"] / 2.0
    out["ideal_1to1_electrolyte_concentration_mM"] = out["effective_salt_equivalent_concentration_mM"]
    if "fitted_blocker_count" in out.columns:
        out["effective_obstructant_concentration_mM"] = out["fitted_blocker_count"].to_numpy(dtype=float) / (
            AVOGADRO_PER_NM3_PER_M * volume_nm3
        ) * 1000.0
    if "fitted_blocker_size" in out.columns:
        out["obstruction_length_nm"] = out["fitted_blocker_size"].to_numpy(dtype=float) * l_box_nm
        out["obstruction_diameter_equivalent_nm"] = 2.0 * out["obstruction_length_nm"]
        out["obstruction_reporting_volume_fraction"] = (
            out["fitted_blocker_count"].to_numpy(dtype=float)
            * (4.0 * np.pi / 3.0)
            * np.power(out["obstruction_length_nm"].to_numpy(dtype=float), 3)
            / volume_nm3
        )
    out["interaction_cutoff_reporting_nm"] = config.interaction_cutoff_reduced * l_box_nm
    out["trap_radius_reporting_nm"] = config.trap_radius_reduced * l_box_nm
    return out
