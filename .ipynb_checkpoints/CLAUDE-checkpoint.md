# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is the analysis code accompanying Gale et al. (2026b) (preprint: https://essopenarchive.org/doi/full/10.22541/essoar.15002817/v1), a climate-science study using a CNN and ML-PLS to separate internally-forced from externally-forced Arctic surface temperature trends. It is a Jupyter-notebook research pipeline, not an application — there is no build, lint, or automated test suite. "Development" here means editing/running notebooks and the one shared Python module.

## Environment setup

Two conda environments are defined at the repo root; which one to use depends on which stage of the pipeline you're touching:

```bash
conda env create -f environment-cnn.yml        # arctic-cnn: TensorFlow + core scientific stack
conda env create -f environment-analysis.yml    # arctic-monthly-internal: same + cartopy, xesmf, gcsfs, cdsapi
```

- `environment-cnn.yml` → only `3_methods/Monthly_CNN_7090.ipynb` and `3_methods/Monthly_CNN_Global.ipynb`.
- `environment-analysis.yml` → everything else (download, preprocessing, ML-PLS, dynamical adjustment, PInudge, vertical trends, figures).

There is no `pip install -e .` / package — notebooks import `figure_utils.py` directly via relative path (`6_figures/figure_utils.py`), and `sys.path`/relative-path tweaks may be needed if a notebook is moved.

File paths inside notebooks are hardcoded to local/NCAR GLADE storage and often need adjusting when running outside this filesystem.

## Pipeline architecture

The repo is a strictly ordered, numbered pipeline. Each stage reads outputs of the previous one from `data/`; there's no orchestration script, so the run order matters and is enforced only by convention:

1. **`1_download/`** — pulls raw ERA5 (Copernicus CDS via `cdsapi`), CMIP6 (Google Cloud archive via `gcsfs`, and ESGF), CESM2/SMBB, and E3SMv2 data. One notebook per source.
2. **`2_preprocess/`** — `organize_cmip6_data.ipynb` restructures raw CMIP6 output; `calculate_all_trends.ipynb` computes 43-year rolling trend maps used as CNN/PLS training inputs.
3. **`3_methods/`** — the core statistical/ML methods, each independent of the others:
   - `Monthly_CNN_7090.ipynb` / `Monthly_CNN_Global.ipynb` — CNN training+eval for the Arctic (poleward of 20°N, 28×144×2 grid) vs. global (72×144×2) domains. Same architecture in both: `Conv2D(16, 1×3, ReLU) → MaxPool(1×1) → Dropout(0.5) → Flatten → Dense(2, linear)`, MSE loss, Adam (lr=1e-4), 10 epochs, batch 32.
   - `Monthly_PLS_7090.ipynb` / `Monthly_PLS_Global.ipynb` — partial least squares equivalent of the CNN, same domains.
   - `Monthly_Dynamical_Adjustment.ipynb` — dynamical-adjustment baseline.
   - `CNN_PLS_MMM_Uncertainty.ipynb`, `Plot_RMSE.ipynb` — cross-method comparison/uncertainty diagnostics, run after the above.
4. **`4_nudged/`** — `Nudged_Data.ipynb` processes the PInudge wind-nudging ensemble (Gilbert et al. 2025) stored under `data/PInudge/`.
5. **`5_vertical/`** — `MMM_Vert_Trends.ipynb` computes vertical temperature/zonal-wind trend profiles for the multi-model mean; depends on `data/ERA5_Vertical_*.nc`, which are **not** in the Zenodo archive and exist only on NCAR Casper (regenerate via `1_download/download_ERA5.ipynb` if missing).
6. **`6_figures/`** — `make_figures.ipynb` produces every manuscript/supplementary figure by `from figure_utils import *`; all shared plotting/statistics logic lives in `6_figures/figure_utils.py` (see below).

## CNN methodology (both `Monthly_CNN_*.ipynb`)

Both notebooks follow the same internal shape — when editing one, check whether the equivalent change is needed in the other (there's no shared module between them, logic is duplicated):

- A **Configuration** cell near the top sets `region`, `MONTH_IDX` (2/3/4 = March/April/May), `SAT_SLP` (whether SLP is stacked with SAT as a second input channel), and `GEOMETRY_WEIGHTING` (`None` / `'cos'` / `'sqrt_cos'` — an area-weighting sensitivity test added for reviewer response; `'cos'`/`'sqrt_cos'` runs write `.npy` outputs with a suffix tag so they don't clobber the `None` baseline).
- **Validation is leave-one-model-out CV**: iterates over CMIP6 large-ensemble members, training on the rest and testing on the held-out model, with 50 random weight reinitializations per fold to average out initialization noise. The model is built and compiled **once**; `reinitialize_weights()` re-randomizes weights in place between randomizations instead of rebuilding the model (keeps the TF graph traced once — deliberate perf choice, don't "simplify" this back into rebuilding per-loop).
- Predictions/observations per fold are saved as `.npy` files under `3_methods/preds_and_vals/<month>_<region>/`, then reloaded by later cells/notebooks for comparison and by `6_figures/`.

## `figure_utils.py`

Single shared module (`6_figures/figure_utils.py`), imported with `from figure_utils import *`. Grouped roughly into:
- **Formatting constants**: `MONTHS`, `MONTHS_ALL`, `PANEL_LABELS`, colormap levels (`SAT_LEVELS`, `SLP_LEVELS`, `VERT_LEVELS`, `U_LEVELS`), font sizes (`FS_*`), `DPI`, colorbar defaults — change once here to restyle every figure.
- **Spatial statistics**: `spatial_average` (cosine-latitude weighted mean, handles 2–5D arrays), `regional_average`, `weighted_corrcoef[_pvalue]`, `significance_mask`, `mahalanobis_pvalue`.
- **Plotting helpers**: `polar_ax`, `plot_region`, `add_stippling`, `add_colorbar`.
- **Data loaders**: `load_obs_trends`, `load_sim_trends`, `load_nudged`, `load_obs_grids`, `load_model_grids` — these encode the glob patterns and slicing conventions (e.g. `month_slice`, `year_slice`) used to pull processed data out of `data/`; check these first when a figure notebook needs a new data source rather than re-deriving loading logic inline.

## Data layout (`data/`, gitignored)

`data/` is excluded via `.gitignore` and lives only on GLADE/Zenodo, not in git history. Key subdirectories referenced throughout the notebooks:
- `data/monthly-trends/{hist,obs,spliced}` — 43-year rolling trend maps per source.
- `data/spliced/`, `data/sim-trends/` — per-model (e.g. `CESM2.nc`, `CanESM5.nc`) processed output.
- `data/training-data/{monthly,seasonal,annual}/{hist,observations,spliced}` — CNN/PLS training inputs.
- `data/obs-grids/`, `data/obs-trends/` — regridded and trend-computed observational products (BerkeleyEarth, GISTv4, HadCRUTv5, NOAAv6 for SAT; ERA5, JRA-55, MERRA-2 for SLP).
- `data/PInudge/` — raw PInudge wind-nudging ensemble members.

When a notebook can't find its inputs, it's almost always because these paths point at NCAR Casper/GLADE and need to be repointed at the Zenodo data archive (see README's Data Availability section) rather than a code bug.
