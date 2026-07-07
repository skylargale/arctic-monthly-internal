# arctic-monthly-internal

Code accompanying [Gale et al. (2026b)](https://essopenarchive.org/doi/full/10.22541/essoar.15002817/v1) (preprint).

## Repository Structure
arctic-monthly-internal/
├── README.md

├── LICENSE

├── figure_utils.py              # Shared utilities: data loaders, plot helpers, statistics

│

├── 1_download/                  # Data acquisition

│   ├── download_era5.ipynb

│   ├── download_era5_OG.ipynb

│   ├── download_CMIP6_hist.ipynb

│   ├── download_CMIP6_ssp370.ipynb

│   ├── download_CMIP6_ssp585.ipynb

│   ├── google_archive_download.ipynb

│   ├── cesm2_and_smbb_download.ipynb

│   └── e3smv2_download.ipynb

│

├── 2_preprocess/                # Data organization and trend computation

│   ├── organize_cmip6_data.ipynb

│   └── calculate_all_trends.ipynb

│

├── 3_methods/                   # CNN, ML-PLS, and dynamical adjustment

│   ├── Monthly_CNN_7090.ipynb       # Arctic CNN training and evaluation

│   ├── Monthly_CNN_Global.ipynb     # Global CNN training and evaluation

│   ├── Monthly_PLS_7090-explore.ipynb

│   ├── Monthly_PLS_Global-explore.ipynb

│   ├── My_Obs_Dyn_Adj_Monthly.ipynb

│   ├── CNN_PLS_MMM_Uncertainty.ipynb

│   └── save_plot_rmse.ipynb

│

├── 4_nudged/                    # PInudge wind-nudging analysis

│   └── Nudged_Data.ipynb

│

├── 5_vertical/                  # Vertical temperature and wind trend analysis

│   └── MMM_Vert_Trends.ipynb

│

└── 6_figures/                   # Figure generation

└── make_figures.ipynb

## Reproducing the Analysis

The notebooks are numbered in the order they should be run:

1. **Download** (`1_download/`) — acquire ERA5, CMIP6, CESM2, and E3SMv2 data from their respective sources (see Data Availability below)
2. **Preprocess** (`2_preprocess/`) — organize CMIP6 output and compute 43-year rolling trend maps
3. **Methods** (`3_methods/`) — train and evaluate the CNN, ML-PLS, and dynamical adjustment; CNN code is in `Monthly_CNN_7090.ipynb` (Arctic) and `Monthly_CNN_Global.ipynb` (global)
4. **Nudged** (`4_nudged/`) — process PInudge wind-nudging experiments
5. **Vertical** (`5_vertical/`) — compute vertical temperature and zonal wind trend profiles for the MMM
6. **Figures** (`6_figures/`) — generate all manuscript and supplementary figures using `make_figures.ipynb`, which imports shared utilities from `figure_utils.py`

## CNN Architecture

The convolutional neural network used in this study is defined in `3_methods/Monthly_CNN_7090.ipynb` and `3_methods/Monthly_CNN_Global.ipynb`. Both CNNs share the same architecture:

- **Input:** SAT and SLP 43-year trend maps (2.5° × 2.5° resolution)
  - Arctic CNN: 28 × 144 × 2 (poleward of 20°N); 129,138 trainable parameters
  - Global CNN: 72 × 144 × 2; 332,466 trainable parameters
- **Architecture:** Conv2D(16 filters, 1×3 kernel, ReLU) → MaxPool(1×1) → Dropout(0.5) → Flatten → Dense(2, linear)
- **Training:** MSE loss, Adam optimizer (lr = 1×10⁻⁴), 10 epochs, batch size 32, 50 random initializations per cross-validation fold
- **Validation:** Leave-one-model-out cross-validation across 11 CMIP6 large-ensemble models

## Setup

This repository was developed using a standard Python environment with commonly used scientific libraries. Key dependencies include NumPy, Xarray, Matplotlib, SciPy, Cartopy, TensorFlow, and xESMF. Users may install these manually within their preferred Python environment.

File paths within notebooks may need to be updated depending on local data storage locations.

## Data Availability

Processed data and a code archive are available on Zenodo:

- Code archive: [https://zenodo.org/records/20040166](https://zenodo.org/records/20040166)
- Data archive: [https://zenodo.org/records/18842089](https://zenodo.org/records/18842089)

Raw input data are available from the following sources:
- **ERA5:** [Copernicus Climate Data Store](https://cds.climate.copernicus.eu)
- **CMIP6:** [Google Cloud CMIP6 archive](https://console.cloud.google.com/marketplace/details/noaa-public/cmip6) and [ESGF](https://esgf-node.llnl.gov)
- **CESM2 and E3SMv2:** [ESGF](https://esgf-node.llnl.gov)
- **PInudge wind-nudging simulations:** Gilbert et al. (2025)

## Notes

File paths within notebooks may need to be modified depending on local data storage locations.

## Citation

If you use this code, please cite:

```text
Gale, S. (2026). arctic-monthly-internal (Version 1.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20040166
```
