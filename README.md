# CPTDS Conductance Monte Carlo: Public Method Code

This folder contains public method code for applying the CPTDS conductance framework to user-provided coffee or liquid samples.

It does not contain the manuscript coffee datasets, manuscript output tables, or manuscript figure-generation scripts. Those data and results are part of the submitted publication record and are not distributed here.

## What Is Included

- Monte Carlo conductance simulation engine.
- Generic three-coordinate conductance-manifold generation.
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
The `tds` column should be an independent operational concentration coordinate,
for example refractometric or gravimetric TDS. Do not use a conductivity-meter
ppm/TDS value calculated from electrical conductivity, because that would make
`conductivity / TDS` partly circular.

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

Here `slope` is the signed concentration-penalty coefficient, `-d(CPTDS)/d(sqrt(TDS))`.
Rows where CPTDS increases with `sqrt(TDS)` are rejected by default because they are
outside the retained concentration-penalty model. Use `--allow-nonpenalty` only when
you want to export signed diagnostic rows rather than fit them as concentration
penalties.

## Workflow B: Starting From Existing Coefficients

Prepare a CSV file with at least:

```text
sample_id,slope,intercept
```

The `slope` column must use the non-negative concentration-penalty convention above.
Negative slopes are treated as outside-model data, not as stronger penalties.

If `index_exp` is absent, it is calculated automatically.

Example:

```bash
python fit_user_coefficients.py \
  --input examples/coefficient_template.csv \
  --output-dir outputs/example_fit \
  --quick
```

The `--quick` flag uses a small grid for a fast smoke test. It is useful for checking installation only.

## Workflow C: Browser-Only Use

Readers who do not want to run Python can open:

```text
web/index.html
```

in a web browser. The page accepts either raw dilution data or fitted coefficient data, generates a small Monte Carlo conductance grid in the browser, and projects the user's samples onto that grid.

The browser version is intended for interactive use and teaching. It does not contain manuscript data or manuscript results. It also does not replace the Python workflow for publication-quality calculations, because browser execution is slower and normally uses a smaller grid.
It uses the same retained 3D public-model convention of balanced positive/negative ions and
opposite-charge candidate pairs, but remains a compact in-browser approximation.

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

The public Monte Carlo network samples a bounded three-dimensional box with balanced
positive and negative ions. Candidate conductance links are opposite-charge pairs
within a fixed interaction cutoff; same-charge pairs are not part of the ion network.
The retained default manifold varies ion count, obstructant count, and obstructant
size over the expanded `0.005-0.400` reduced-length obstruction grid, while clustering and bulk medium-renormalization parameters are fixed off
(`cluster_strength = 0`, `g = 1`).

The fitted quantities are conductance-equivalent coordinates:

- `fitted_n_ions`
- `fitted_blocker_count`
- `fitted_blocker_size`
- optional reporting-scale ion concentration equivalent in mM
- optional reporting-scale obstructant concentration equivalent in mM
- optional conductance-equivalent obstruction length in nm

These are not direct analytical concentrations or molecular radii. They describe where the user's sample lies on the Monte Carlo conductance manifold.
Because each sample is assigned independent coordinates while only slope and
intercept are observed, these coordinates are not uniquely identifiable physical
state variables unless additional shared constraints or external measurements are
introduced. Report them as conductance-equivalent projections.

## Optional Electrolyte Benchmark

The optional script:

```bash
python run_universal_electrolyte_benchmark.py
```

runs the idealized 1:1 electrolyte benchmark machinery. It is slower than the user-coefficient workflow and is not required for fitting user data.

## Notes For Reuse

Use the same TDS and conductivity units consistently within one dataset. The CPTDS slope and intercept depend on the units of the concentration axis, so different unit conventions should not be mixed in one fit. TDS should be measured independently of conductivity, such as by a coffee refractometer or gravimetric solids measurement, rather than imported from an EC/TDS meter conversion.

For publication or comparison across labs, report:

- dilution protocol,
- TDS unit,
- conductivity unit,
- fitted slope/intercept convention,
- Index definition,
- Monte Carlo grid settings,
- number of grid and final replicates,
- effective box length used for SI-like reporting.
