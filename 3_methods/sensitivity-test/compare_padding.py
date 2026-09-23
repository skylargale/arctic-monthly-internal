#!/usr/bin/env python
# coding: utf-8
"""Compare North Pole padding sensitivity runs from Monthly_CNN_7090.py.

Loads the `.npy` outputs that Monthly_CNN_7090.py writes for a given month
(untagged for the baseline run, `_pole` for the reviewer-suggested padding run)
and reports the mean predicted Sum (Internal + External) trend for each, plus
the difference from baseline.

python compare_padding.py --month april
"""
import argparse
import glob
import os

import numpy as np

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']
region = 'Arctic'

month_lookup = {
    "march": 2,
    "april": 3,
    "may": 4,
}

# Saved Arctic_obs*.npy arrays are (cv_fold, component, obs_product) = (8, 3, 12).
# The component axis (axis 1) is Internal/External/Sum -- NOT the last axis.
component_idx = {"internal": 0, "external": 1, "sum": 2}


def load_run(output_dir, tag):
    """Load the obs predictions (and per-model predictions) for one padding tag.

    tag is '' for the baseline (unpadded) run, or '_pole' for the padded run.
    Returns (None, None) if the obs file for that tag doesn't exist yet (e.g.
    the corresponding qsub job hasn't finished/run).

    Per-model prediction files are discovered by globbing rather than reading
    a hardcoded model list, so this script works regardless of run order or
    which models were included in a given training run.
    """
    obs_path = os.path.join(output_dir, f'{region}_obs{tag}.npy')
    if not os.path.exists(obs_path):
        return None, None

    obs = np.load(obs_path)

    all_pred_paths = sorted(glob.glob(os.path.join(output_dir, f'{region.lower()}_*.npy')))
    known_tags = ['_cos', '_sqrt_cos', '_pole', '_periodic']

    if tag:
        pred_paths = [p for p in all_pred_paths if p.endswith(f'{tag}.npy')]
    else:
        # Untagged (baseline) run: exclude files that belong to a *padded* tag.
        pred_paths = [
            p for p in all_pred_paths
            if not any(p.endswith(f'{t}.npy') for t in known_tags)
        ]

    preds = [np.load(p) for p in pred_paths]
    return obs, preds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--month",
        choices=["march", "april", "may"],
        required=True,
        help="Month to compare (must match the --month used for the training runs).",
    )
    parser.add_argument(
        "--component",
        choices=["internal", "external", "sum"],
        default="sum",
        help="Which predicted component to report (matches the Internal/External/Sum figure panels).",
    )
    args = parser.parse_args()

    month_idx = month_lookup[args.month]
    idx = component_idx[args.component]
    output_dir = f'./preds_and_vals/{args.month}_{region.lower()}/'

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Output directory not found: {output_dir}\n"
            f"Run at least the baseline job first: qsub -v MONTH={args.month},PADDING=none submit_cnn_7090.sh"
        )

    baseline_obs, baseline_preds = load_run(output_dir, '')
    if baseline_obs is None:
        raise FileNotFoundError(
            f"No baseline (unpadded) run found for {args.month} in {output_dir}\n"
            f"Run: qsub -v MONTH={args.month},PADDING=none submit_cnn_7090.sh"
        )

    baseline_mean = np.nanmean(baseline_obs[:, idx, :])
    print(f'Baseline (no padding)      {months[month_idx]} mean predicted {args.component}: {baseline_mean:.3f} K/dec')

    pole_obs, pole_preds = load_run(output_dir, '_pole')
    if pole_obs is None:
        print(f"\n[pole] not found yet -- run: qsub -v MONTH={args.month},PADDING=pole submit_cnn_7090.sh")
        return

    pole_mean = np.nanmean(pole_obs[:, idx, :])
    label = 'North Pole padded'
    print(f'{label:<27} {months[month_idx]} mean predicted {args.component}: {pole_mean:.3f} K/dec')
    print(f'  Difference vs baseline:  {pole_mean - baseline_mean:+.3f} K/dec')


if __name__ == "__main__":
    main()
