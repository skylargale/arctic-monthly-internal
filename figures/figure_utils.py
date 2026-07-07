"""
figure_utils.py
===============
Shared utilities for Gale et al. (JGR) figure notebook.
Import at the top of Make_Figures.ipynb with:
    from figure_utils import *
"""

import re
import os
import glob
import numpy as np
import xarray as xr
from scipy import stats
from scipy.stats import gaussian_kde, chi2, pearsonr
from scipy.signal import detrend
import cartopy.crs as ccrs
from xesmf import Regridder
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from cartopy.util import add_cyclic_point
from collections import defaultdict

# ============================================================
# Universal formatting
# ============================================================

MONTHS      = ["March", "April", "May"]
MONTHS_ALL  = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PANEL_LABELS = list("abcdefghijklmnopqrstuvwxyz")

# Colormaps and levels
SAT_LEVELS  = np.linspace(-2, 2, 20)
SLP_LEVELS  = np.linspace(-4, 4, 17)
VERT_LEVELS = np.linspace(-2, 2, 20)
U_LEVELS    = np.linspace(-1, 1, 20)

# Font sizes — change here to apply everywhere
FS_TITLE    = 16   # panel title / label
FS_LABEL    = 14   # axis label / colorbar label
FS_TICK     = 12   # tick labels
FS_ROWCOL   = 16   # row/column headers

# Figure DPI
DPI = 300

# Colorbar defaults
CBAR_SHRINK = 0.8
CBAR_ASPECT = 15
CBAR_PAD    = 0.02

# Circular boundary for polar plots
_theta  = np.linspace(0, 2 * np.pi, 100)
_verts  = np.vstack([np.sin(_theta), np.cos(_theta)]).T * 0.5 + 0.5
circle_path = Path(_verts)


# ============================================================
# Spatial utilities
# ============================================================

def spatial_average(array, lats, num_lons=None):
    """
    Cosine-latitude weighted spatial mean.
    Accepts 2-D through 5-D arrays.
    """
    weights = np.cos(np.deg2rad(lats))
    ndim = len(array.shape)

    if ndim == 2:
        w = weights[:, np.newaxis]
        return np.nansum(array * w) / (np.nansum(weights) * array.shape[1])
    if ndim == 3:
        w = weights[np.newaxis, :, np.newaxis]
        weighted = array * w
        s = weighted.shape
        return np.nansum(weighted.reshape(s[0], -1), axis=1) / (np.nansum(weights) * s[2])
    if ndim == 4:
        w = weights[np.newaxis, np.newaxis, :, np.newaxis]
        weighted = array * w
        s = weighted.shape
        return np.nansum(weighted.reshape(s[0], s[1], -1), axis=2) / (np.nansum(weights) * s[3])
    if ndim == 5:
        w = weights[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        weighted = array * w
        s = weighted.shape
        return np.nansum(weighted.reshape(s[0], s[1], s[2], -1), axis=3) / (np.nansum(weights) * s[4])
    raise ValueError(f"spatial_average: unsupported ndim={ndim}")


def regional_average(array, lats, lons, lat_min, lat_max, lon_min, lon_max):
    """Cosine-weighted area mean over a rectangular region. array: (..., lat, lon)."""
    lat_idx = (lats >= lat_min) & (lats <= lat_max)
    lon_idx = (lons >= lon_min) & (lons <= lon_max)
    sub = array[..., lat_idx, :][..., lon_idx]
    return spatial_average(sub, lats[lat_idx], np.sum(lon_idx))


def weighted_corrcoef(x, y, lats):
    """
    Area-weighted Pearson correlation between two (lat, lon) arrays.
    """
    w = np.cos(np.deg2rad(lats))[:, np.newaxis]
    w = np.broadcast_to(w, x.shape).ravel()
    mask = np.isfinite(x.ravel()) & np.isfinite(y.ravel())
    w, xf, yf = w[mask], x.ravel()[mask], y.ravel()[mask]
    w = w / w.sum()
    xm = np.sum(w * xf)
    ym = np.sum(w * yf)
    cov  = np.sum(w * (xf - xm) * (yf - ym))
    sx   = np.sqrt(np.sum(w * (xf - xm) ** 2))
    sy   = np.sqrt(np.sum(w * (yf - ym) ** 2))
    return cov / (sx * sy)


# ============================================================
# Significance testing
# ============================================================

def significance_mask(timeseries, alpha=0.05):
    """
    Grid-point-wise OLS trend t-test with effective N from lag-1 autocorrelation.

    Parameters
    ----------
    timeseries : (n_years, lat, lon) array — full annual time series
    alpha      : significance level (default 0.05 → 95%)

    Returns
    -------
    Boolean mask — True where trend is NOT significant (→ stipple)
    """
    d = timeseries
    n = d.shape[0]
    t_vec = np.arange(n)

    # Lag-1 autocorrelation
    r1 = np.nanmean(d[:-1] * d[1:], axis=0) / (np.nanvar(d, axis=0) + 1e-10)
    r1 = np.clip(r1, -1, 1)
    n_eff = np.maximum(2, n * (1 - r1) / (1 + r1))

    t_vals = np.full(d.shape[1:], np.nan)
    for i in range(d.shape[1]):
        for j in range(d.shape[2]):
            y = d[:, i, j]
            if np.sum(np.isfinite(y)) >= 4:
                slope, intercept = np.polyfit(t_vec, y, 1)
                resid = y - (slope * t_vec + intercept)
                denom = np.sum((t_vec - t_vec.mean()) ** 2)
                n_eff_ij = max(float(n_eff[i, j]), 3.0)
                if denom > 0 and (n_eff_ij - 2) > 0:
                    se = np.sqrt(np.sum(resid ** 2) / (n_eff_ij - 2) / denom)
                else:
                    continue
                t_vals[i, j] = slope / se if se > 0 else np.nan

    p_vals = 2 * stats.t.sf(np.abs(t_vals), df=n_eff - 2)
    return p_vals >= alpha   # True = NOT significant


def weighted_corrcoef_pvalue(x, y, lats):
    """
    Area-weighted Pearson correlation with p-value.
    Effective N accounts for spatial autocorrelation following
    Bretherton et al. (1999).
    """
    r = weighted_corrcoef(x, y, lats)

    # Flatten and weight
    w = np.cos(np.deg2rad(lats))[:, np.newaxis]
    w = np.broadcast_to(w, x.shape).ravel()
    mask = np.isfinite(x.ravel()) & np.isfinite(y.ravel())
    w = w[mask]
    xf = x.ravel()[mask]
    yf = y.ravel()[mask]
    w = w / w.sum()

    # Weighted means and deviations
    xm = np.sum(w * xf)
    ym = np.sum(w * yf)
    xd = xf - xm
    yd = yf - ym

    # Lag-1 spatial autocorrelation of each field (treats flattened spatial
    # points as a 1-D sequence — approximation, but standard in practice)
    r1x = np.corrcoef(xd[:-1], xd[1:])[0, 1]
    r1y = np.corrcoef(yd[:-1], yd[1:])[0, 1]

    # Bretherton et al. (1999) effective N
    n_raw = np.sum(mask)
    n_eff = n_raw * (1 - r1x * r1y) / (1 + r1x * r1y)
    n_eff = max(n_eff, 3)

    # t-statistic and p-value
    t = r * np.sqrt(n_eff - 2) / np.sqrt(1 - r ** 2 + 1e-10)
    p = 2 * stats.t.sf(np.abs(t), df=n_eff - 2)

    return r, p


def mahalanobis_pvalue(x_samp, y_samp, obs_x, obs_y):
    """
    Squared Mahalanobis distance of (obs_x, obs_y) from 2-D sample distribution.
    p-value from chi-squared with 2 dof.
    """
    data  = np.vstack([x_samp, y_samp]).T
    mu    = data.mean(axis=0)
    cov   = np.cov(data.T)
    delta = np.array([obs_x, obs_y]) - mu
    D2    = delta @ np.linalg.solve(cov, delta)
    p     = chi2.sf(D2, df=2)
    return D2, p


# ============================================================
# Plot helpers
# ============================================================

def polar_ax(ax, extent=(-180, 180, 54, 90)):
    """Standard polar stereographic formatting."""
    ax.coastlines(resolution='110m')
    ax.set_boundary(circle_path, transform=ax.transAxes)
    ax.set_extent(list(extent), crs=ccrs.PlateCarree())


def plot_region(ax, lon_min, lon_max, lat_min, lat_max, n_points=100, **kwargs):
    """Draw a rectangular region outline on a polar stereographic axis."""
    lons = np.concatenate([
        np.linspace(lon_min, lon_max, n_points), lon_max * np.ones(n_points),
        np.linspace(lon_max, lon_min, n_points), lon_min * np.ones(n_points)
    ])
    lats = np.concatenate([
        lat_min * np.ones(n_points), np.linspace(lat_min, lat_max, n_points),
        lat_max * np.ones(n_points), np.linspace(lat_max, lat_min, n_points)
    ])
    ax.plot(lons, lats, transform=ccrs.PlateCarree(), **kwargs)


def add_stippling(ax, insig_mask, lon, lat, density=3):
    """
    Overlay dots where insig_mask is True (not significant at 95%).
    density : subsample every N grid points to avoid over-crowding.
    """
    # Build coordinate arrays matching the mask shape exactly
    nlat, nlon = insig_mask.shape
    lon_use = lon[:nlon]
    lat_use = lat[:nlat]
    lon2d, lat2d = np.meshgrid(lon_use, lat_use)

    mask_sub = insig_mask[::density, ::density]
    lon_sub  = lon2d[::density, ::density]
    lat_sub  = lat2d[::density, ::density]

    ax.scatter(
        lon_sub[mask_sub], lat_sub[mask_sub],
        s=0.8, color='dimgray', alpha=0.5,
        transform=ccrs.PlateCarree(), zorder=5, linewidths=0
    )


def add_colorbar(fig, im, ax_array, label, ticks, orientation='vertical'):
    """Standardised colorbar using universal formatting constants."""
    cbar = fig.colorbar(
        im, ax=ax_array, orientation=orientation,
        ticks=ticks, shrink=CBAR_SHRINK, aspect=CBAR_ASPECT, pad=CBAR_PAD
    )
    cbar.set_label(label, fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)
    return cbar


# ============================================================
# Data loading helpers
# ============================================================

def load_obs_trends(sat_glob, slp_glob, month_slice=slice(2, 5)):
    """
    Load and average observed SAT and SLP trend maps.

    Returns
    -------
    obs_sat        : (3, lat, lon) — MAM mean across datasets
    obs_slp        : (3, lat, lon) — MAM mean across reanalyses, hPa
    obs_sat_all    : (n_datasets, 3, lat, lon) — per-dataset (for stippling)
    obs_slp_all    : (n_reanalyses, 3, lat, lon) — per-reanalysis (for stippling)
    lat, lon       : 1-D coordinate arrays
    """
    sat_paths = glob.glob(sat_glob)
    slp_paths = glob.glob(slp_glob)

    sat_data = [xr.open_dataset(p).DATA for p in sat_paths]
    slp_data = [xr.open_dataset(p).DATA for p in slp_paths]

    obs_sat_all = np.array([np.array(d)[month_slice] for d in sat_data])
    obs_slp_all = np.array([np.array(d)[month_slice] / 100 for d in slp_data])

    obs_sat = np.nanmean(obs_sat_all, axis=0)
    obs_slp = np.nanmean(obs_slp_all, axis=0)

    lat = sat_data[-1].lat.values
    lon = sat_data[-1].lon.values

    return obs_sat, obs_slp, obs_sat_all, obs_slp_all, lat, lon


def load_sim_trends(sim_glob, month_slice=slice(2, 4), scale=0.213/0.260):
    """
    Load simulated SAT and SLP trend maps from CMIP6 large-ensemble files.

    Returns
    -------
    mod_sat         : (3, lat, lon) — MMM SAT
    mod_slp         : (3, lat, lon) — MMM SLP, hPa
    all_members_sat : (total_members, 3, lat, lon) — for stippling
    """
    sim_paths = glob.glob(sim_glob)
    sim_sat, sim_slp, all_members_sat = [], [], []

    for path in sim_paths:
        ds   = xr.open_dataset(path).X_SAT_SLP.isel(period=-6).sel(month=month_slice)
        sat  = ds[:, :, :, :, 0].values * scale
        slp  = ds[:, :, :, :, 1].values / 100
        sim_sat.append(np.nanmean(sat, axis=0))
        sim_slp.append(np.nanmean(slp, axis=0))
        all_members_sat.append(sat)

    mod_sat = np.nanmean(sim_sat, axis=0)
    mod_slp = np.nanmean(sim_slp, axis=0)
    all_members_sat = np.concatenate(all_members_sat, axis=0)

    return mod_sat, mod_slp, all_members_sat


def load_nudged(nudged_glob, target_lat, target_lon,
                time_slice=("1980-01-01", "2022-12-31"), months=(4, 5, 6)):
    """
    Load PInudge wind-nudged ensemble data, compute trends, and regrid.

    Returns
    -------
    nudged_sat      : (3, lat, lon)         — ensemble-mean SAT trend (native grid)
    nudged_slp      : (3, lat, lon)         — ensemble-mean SLP trend (native grid), hPa
    nudged_sat_ens  : (ens, 3, lat, lon)    — per-member SAT trend
    nudged_slp_ens  : (ens, 3, lat, lon)    — per-member SLP trend, hPa
    nudged_sat_rg   : (3, lat, lon)         — ensemble-mean SAT regridded to 2.5°
    nudged_slp_rg   : (3, lat, lon)         — ensemble-mean SLP regridded to 2.5°
    nudged_t_ens    : (ens, 3, lev, lat)    — per-member T trend (zonal mean, native grid)
    nudged_t        : (3, lev, lat)         — ensemble-mean T trend (zonal mean, native grid)
    nudged_sat_ts   : (ens, years, 3, lat, lon) — full SAT time series (native grid)
    nudged_slp_ts   : (ens, years, 3, lat, lon) — full SLP time series (native grid)
    nudged_lat, nudged_lon : native grid coordinates
    """
    nudged_paths = sorted(glob.glob(nudged_glob))
    file_dict = defaultdict(list)
    for path in nudged_paths:
        m = re.search(r'(PSL|TREFHT|\.T\.)', os.path.basename(path))
        if m:
            file_dict[m.group(1)].append(path)

    arrays = {}
    for varname, files in file_dict.items():
        datasets = []
        for path in files:
            em = re.search(r'\.(\d{3})\.h0', os.path.basename(path))
            ens = int(em.group(1)) if em else 0
            ds = xr.open_dataset(path, chunks={'time': 12}).assign_coords(ensemble=ens)
            datasets.append(ds)
        ds_all = xr.concat(datasets, dim='ensemble')
        ds_all = ds_all.sel(time=slice(*time_slice))
        ds_all = ds_all.sel(time=ds_all['time'].dt.month.isin(list(months)))
        # strip the leading dot from .T. key so it works as a dict key
        key = varname.strip('.')
        arrays[key] = ds_all[varname.strip('.') if varname.strip('.') in ds_all else varname]

    # ---- Surface variables (PSL, TREFHT) ----
    ds_surf = xr.concat([arrays['PSL'], arrays['TREFHT']], dim='variable')
    years    = np.unique(ds_surf['time'].dt.year)
    months_u = np.unique(ds_surf['time'].dt.month)

    df = ds_surf.groupby('time.year').apply(
        lambda d: d.groupby('time.month').mean('time')
    ).assign_coords(year=years, month=months_u).transpose(
        'ensemble', 'variable', 'year', 'month', 'lat', 'lon'
    )

    # Full time series — (ens, years, months, lat, lon)
    nudged_slp_ts = df[:, 0, :, :, :, :].values / 100   # hPa
    nudged_sat_ts = df[:, 1, :, :, :, :].values

    # Trends — (ens, months, lat, lon)
    nudged_trends  = df.polyfit(dim='year', deg=1)['polyfit_coefficients'].sel(degree=1) * 10
    nudged_slp_ens = nudged_trends[:, 0].values / 100
    nudged_sat_ens = nudged_trends[:, 1].values
    nudged_sat     = nudged_sat_ens.mean(axis=0)
    nudged_slp     = nudged_slp_ens.mean(axis=0)

    # Regrid surface to 2.5°
    target        = xr.Dataset({'lat': (['lat'], target_lat), 'lon': (['lon'], target_lon)})
    rg            = Regridder(ds_surf, target, 'bilinear', periodic=True, reuse_weights=False)
    nudged_sat_rg = rg(nudged_sat)
    nudged_slp_rg = rg(nudged_slp)

    # ---- 3-D temperature (T) ----
    ds_t     = arrays['T']   # (ens, time, lev, lat, lon)
    years_t  = np.unique(ds_t['time'].dt.year)
    months_t = np.unique(ds_t['time'].dt.month)

    df_t = ds_t.groupby('time.year').apply(
        lambda d: d.groupby('time.month').mean('time')
    ).assign_coords(year=years_t, month=months_t).transpose(
        'ensemble', 'year', 'month', 'lev', 'lat', 'lon'
    )

    # Zonal mean before computing trend — (ens, years, months, lev, lat)
    df_t_zonal = df_t.mean(dim='lon')

    t_trends = df_t_zonal.polyfit(
        dim='year', deg=1
    )['polyfit_coefficients'].sel(degree=1) * 10   # (ens, months, lev, lat)

    nudged_t_ens = t_trends          # (ens, 3, lev, lat)
    nudged_t     = nudged_t_ens.mean(axis=0)  # (3, lev, lat)

    return (nudged_sat, nudged_slp,
            nudged_sat_ens, nudged_slp_ens,
            nudged_sat_rg, nudged_slp_rg,
            nudged_t_ens, nudged_t,
            nudged_sat_ts, nudged_slp_ts,
            ds_surf.lat.values, ds_surf.lon.values,
            nudged_trends)


def load_obs_grids(sat_glob, slp_glob):
    """Load gridded monthly obs time series for interannual analysis."""
    sat_paths = glob.glob(sat_glob)
    slp_paths = glob.glob(slp_glob)
    SAT_maps = np.nanmean([xr.open_dataset(p).DATA.values for p in sat_paths], axis=0)
    SLP_maps = np.nanmean([xr.open_dataset(p).DATA.values for p in slp_paths], axis=0) / 100
    lat = xr.open_dataset(sat_paths[-1]).DATA.lat.values
    lon = xr.open_dataset(sat_paths[-1]).DATA.lon.values
    return SAT_maps, SLP_maps, lat, lon


def load_model_grids(spliced_glob, year_slice=(1980, 2022), month_slice=(2, 4)):
    """Load gridded monthly model time series for interannual analysis."""
    spliced = glob.glob(spliced_glob)
    mod_sat_list, mod_slp_list = [], []
    lat = lon = None
    for path in spliced:
        ds = xr.open_dataset(path).SAT_SLP
        ds = ds.sel(year=slice(*year_slice), month=slice(*month_slice), lat=slice(20, None))
        ds = ds.mean(dim='member')
        arr = np.array(ds)
        if 'CESM' in path:
            mod_sat_list.append(arr[:, :, :, :, 1])
            mod_slp_list.append(arr[:, :, :, :, 0] / 100)
        else:
            mod_sat_list.append(arr[:, :, :, :, 0])
            mod_slp_list.append(arr[:, :, :, :, 1] / 100)
        if lat is None:
            lat = ds.lat.values
            lon = ds.lon.values
    return np.nanmean(mod_sat_list, axis=0), np.nanmean(mod_slp_list, axis=0), lat, lon
