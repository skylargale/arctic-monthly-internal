# arctic-monthly-internal

Code accompanying [Gale et al. (2026b)](https://essopenarchive.org/doi/full/10.22541/essoar.15002817/v1) (preprint; in review).

## Repository Structure

```
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
```

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
 
Two conda environments are provided:
 
- **`environment-cnn.yml`:** For `3_methods/Monthly_CNN_7090.ipynb` and `3_methods/Monthly_CNN_Global.ipynb` (CNN training/evaluation). Installs NumPy, Xarray, Matplotlib, SciPy, scikit-learn, and TensorFlow.
- **`environment-analysis.yml`:** For all other notebooks (download, preprocessing, ML-PLS, dynamical adjustment, PInudge, vertical trends, figures). Installs the above plus Cartopy, xESMF, gcsfs, cdsapi, and requests.

To install the environments, copy and paste the following in an open session terminal:

```bash
conda env create -f environment-cnn.yml
conda env create -f environment-analysis.yml
```

## Data Availability

Processed data and code used in this study are available on Zenodo:
- Data: [https://doi.org/10.5281/zenodo.21285590](https://doi.org/10.5281/zenodo.21285590)
- Code (Version 2.0): [https://doi.org/10.5281/zenodo.21230841](https://doi.org/10.5281/zenodo.21230841)

**Note:** The following folders found within `data/` exceed Zenodo's upload size limits and are not included in the Zenodo data archive: `monthly-trends/`, `PInudge/`, `spliced/`, and `training-data/`. These are available via the NCAR GLADE Globus share linked below. ERA5 source files can also be regenerated using `1_download/download_era5.ipynb` with the Copernicus Climate Data Store. The `PInudge/` folder contains wind-nudging simulation output produced by Gilbert et al. (2025), hosted here with permission. Please see [Gilbert et al. (2025)](https://iopscience.iop.org/article/10.1088/2752-5295/ae11cb/meta) for the original archive and citation.

The full `data/` directory, including the folders above, is shared via NCAR GLADE (no NCAR account required): [Globus Share](https://app.globus.org/file-manager?origin_id=313510e9-f39e-45ba-8c55-8eceff415e73&origin_path=%2F)

Raw input data are also available from their original public sources:
- **ERA5:** [Copernicus Climate Data Store](https://cds.climate.copernicus.eu)
- **CMIP6:** [Google Cloud CMIP6 archive](https://console.cloud.google.com/marketplace/details/noaa-public/cmip6) and [ESGF](https://esgf-node.llnl.gov)
- **CESM2 and E3SMv2:** [ESGF](https://esgf-node.llnl.gov)
- **PInudge wind-nudging simulations:** See [Gilbert et al. (2025)](https://iopscience.iop.org/article/10.1088/2752-5295/ae11cb/meta)

### Citation

If you use this code or data, please cite:

Gale, S. (2026a). Why April Stands Out: Monthly Impacts of Internal Variability on Arctic Amplification [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.21285590

Gale, S. (2026b). arctic-monthly-internal (Version 2.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21230841

## Notes
 
File paths within notebooks may need to be modified depending on local data storage locations.
