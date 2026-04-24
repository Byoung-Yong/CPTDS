# CPTDS Conductance Monte Carlo: Public Method Code

This folder contains public method code for applying the CPTDS conductance framework to user-provided coffee or liquid samples.

It does not contain the manuscript coffee datasets, manuscript output tables, or manuscript figure-generation scripts. Those data and results are part of the submitted publication record and are not distributed here.

## What Is Included

- Monte Carlo conductance simulation engine.
- Generic 3D conductance-manifold generation.
- CPTDS coefficient extraction from raw dilution measurements.
- Generic fitting of user CPTDS coefficients to conductance-equivalent parameters.
- Optional idealized 1:1 electrolyte benchmark code.
- Input templates for user data.

## What Is Not Included

- Manuscript coffee dilution data.
- Manuscript application datasets.
- Manuscript output tables.
- Manuscript figure scripts.
- Cached manuscript simulation results.

## Installation

Use Python 3.11 or later.

```bash
python -m pip install -r requirements.txt
```

Required packages:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

## Workflow A: Starting From Raw Dilution Measurements

Prepare a CSV file with at least:

```text
sample_id,tds,conductivity
```

Optional columns such as `coffee`, `method`, or `notes` are preserved in the coefficient table.

Example:

```bash
python compute_cptds_coefficients.py \
  --input examples/raw_dilution_template.csv \
  --output outputs/user_coefficients.csv
```

This fits:

```text
CPTDS = conductivity / TDS
CPTDS = intercept - slope * sqrt(TDS)
Index = slope**2 / intercept**3
```

## Workflow B: Starting From Existing Coefficients

Prepare a CSV file with at least:

```text
sample_id,slope,intercept
```

If `index_exp` is absent, it is calculated automatically.

Example:

```bash
python fit_user_coefficients.py \
  --input examples/coefficient_template.csv \
  --output-dir outputs/example_fit \
  --quick
```

The `--quick` flag uses a small grid for a fast smoke test. It is useful for checking installation only.

For a real analysis, omit `--quick`:

```bash
python fit_user_coefficients.py \
  --input outputs/user_coefficients.csv \
  --output-dir outputs/user_fit \
  --grid-trials 96 \
  --final-trials 288
```

This generates a new Monte Carlo grid in:

```text
outputs/user_fit/effective_conductance_grid.csv
```

If the grid already exists, it is reused. To regenerate it:

```bash
python fit_user_coefficients.py \
  --input outputs/user_coefficients.csv \
  --output-dir outputs/user_fit \
  --force-grid
```

## Output Files

The fitting script writes:

- `fit_interpolated.csv`: continuous interpolation fit on the Monte Carlo manifold.
- `fit_direct.csv`: direct Monte Carlo reevaluation at rounded fitted particle counts.
- `fit_direct_si.csv`: conductance-equivalent parameters converted to the fixed reporting length scale.
- `fit_summary.csv`: objective value, RMSE values, and fitted global scale factors.
- `effective_conductance_grid.csv`: generated Monte Carlo grid, unless an existing grid is reused.

## Interpretation

The fitted quantities are conductance-equivalent coordinates:

- `fitted_n_ions`
- `fitted_blocker_count`
- `fitted_blocker_size`
- effective ion concentration in mM
- effective obstructant concentration in mM
- effective obstructant size in nm

These are not direct analytical concentrations or molecular radii. They describe where the user's sample lies on the Monte Carlo conductance manifold.

## Optional Electrolyte Benchmark

The optional script:

```bash
python run_universal_electrolyte_benchmark.py
```

runs the idealized 1:1 electrolyte benchmark machinery. It is slower than the user-coefficient workflow and is not required for fitting user data.

## Notes For Reuse

Use the same TDS and conductivity units consistently within one dataset. The CPTDS slope and intercept depend on the units of the concentration axis, so different unit conventions should not be mixed in one fit.

For publication or comparison across labs, report:

- dilution protocol,
- TDS unit,
- conductivity unit,
- fitted slope/intercept convention,
- Index definition,
- Monte Carlo grid settings,
- number of grid and final replicates,
- effective box length used for SI-like reporting.
