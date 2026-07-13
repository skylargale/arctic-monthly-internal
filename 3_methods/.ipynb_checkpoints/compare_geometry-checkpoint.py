#!/usr/bin/env python
# coding: utf-8
"""Compare geometry-weighting sensitivity runs from Monthly_CNN_7090.py.

Loads the `.npy` outputs that Monthly_CNN_7090.py writes for a given month
(one file per geometry mode: untagged for 'none', `_cos`, `_sqrt_cos`) and
reports the mean predicted Sum (Internal + External) trend for the baseline
run alongside each weighted run found, plus the difference from baseline.

python compare_geometry.py --month april
python compare_geometry.py --month april --weighted cos
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


def load_run(output_dir, tag):
    """Load the obs predictions (and per-model predictions) for one geometry tag.

    tag is '' for the baseline (unweighted) run, or '_cos' / '_sqrt_cos' for a
    weighted run. Returns (None, None) if the obs file for that tag doesn't
    exist yet (e.g. the corresponding qsub job hasn't finished/run).

    Per-model prediction files are discovered by globbing rather than reading
    a hardcoded model list, so this script works regardless of run order or
    which models were included in a given training run.
    """
    obs_path = os.path.join(output_dir, f'{region}_obs{tag}.npy')
    if not os.path.exists(obs_path):
        return None, None

    obs = np.load(obs_path)

    all_pred_paths = sorted(glob.glob(os.path.join(output_dir, f'{region.lower()}_*.npy')))
    known_tags = ['_cos', '_sqrt_cos']

    if tag:
        pred_paths = [p for p in all_pred_paths if p.endswith(f'{tag}.npy')]
    else:
        # Untagged (baseline) run: exclude files that belong to a *weighted* tag.
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
        "--weighted",
        nargs="+",
        choices=["cos", "sqrt_cos"],
        default=["cos", "sqrt_cos"],
        help="Which weighted run(s) to compare against baseline. Missing files are skipped, not errors.",
    )
    args = parser.parse_args()

    month_idx = month_lookup[args.month]
    output_dir = f'./preds_and_vals/{args.month}_{region.lower()}/'

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Output directory not found: {output_dir}\n"
            f"Run at least the baseline job first: qsub -v MONTH={args.month},GEOMETRY=none submit_cnn_7090.sh"
        )

    baseline_obs, baseline_preds = load_run(output_dir, '')
    if baseline_obs is None:
        raise FileNotFoundError(
            f"No baseline (unweighted) run found for {args.month} in {output_dir}\n"
            f"Run: qsub -v MONTH={args.month},GEOMETRY=none submit_cnn_7090.sh"
        )

    # Column index 2 is the 'Sum' (Internal + External) trend.
    baseline_mean = np.nanmean(baseline_obs[..., 2])
    print(f'Baseline (unweighted)      {months[month_idx]} mean predicted trend: {baseline_mean:.3f} K/dec')

    found_any = False
    for mode in args.weighted:
        tag = f'_{mode}'
        weighted_obs, weighted_preds = load_run(output_dir, tag)

        if weighted_obs is None:
            print(f"  [{mode}] not found yet -- run: qsub -v MONTH={args.month},GEOMETRY={mode} submit_cnn_7090.sh")
            continue

        found_any = True
        weighted_mean = np.nanmean(weighted_obs[..., 2])
        label = f'{mode}-latitude weighted'
        print(f'{label:<27} {months[month_idx]} mean predicted trend: {weighted_mean:.3f} K/dec')
        print(f'  Difference vs baseline:  {weighted_mean - baseline_mean:+.3f} K/dec')

    if not found_any:
        print("\nNo weighted runs found yet for this month -- nothing to compare.")


if __name__ == "__main__":
    main()
