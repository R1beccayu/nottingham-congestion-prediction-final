# -*- coding: utf-8 -*-
"""
Historical Average baseline.

Predictions use the average training-set speed for the same node and
within-day time slot, independent of forecasting horizon. Statistics are
estimated only from the training period, without using validation or test data.
"""
import argparse
import os

import numpy as np
import pandas as pd

from metrics import evaluate, show
from windows import WindowIndex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_profile(w):
    """Average speed for each node and within-day time slot in the training period, using only evaluable cells."""
    rows = np.isin(w.day_of_row, np.where(w.day_split == "train")[0])
    prof = np.full((w.steps_per_day, w.n_node), np.nan, dtype=np.float32)
    for slot in range(w.steps_per_day):
        sel = rows & (w.tod_of_row == slot)
        if not sel.any():
            continue
        vals = np.where(w.eval_mask[sel], w.speed[sel], np.nan)
        with np.errstate(invalid="ignore"):
            prof[slot] = np.nanmean(vals, axis=0)
    blank = np.isnan(prof)
    if blank.any():
        node_mean = np.nanmean(prof, axis=0)
        prof[blank] = np.take(node_mean, np.where(blank)[1])
        print(f"Node time-slot combinations with no training observations: {int(blank.sum())}; filled by node means")
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="eps0.3")
    ap.add_argument("--group", default=None,
                    help="Result subdirectory used to isolate different datasets")
    args = ap.parse_args()

    path = os.path.join(ROOT, "data", "processed", f"dataset_{args.tag}.npz")
    w = WindowIndex(path)
    print(w.describe())
    prof = build_profile(w)

    out = []
    for split in ("val", "test"):
        idx = w.samples[split]
        y, m, tod = w.target(idx)
        pred = prof[tod]                      # (n, n_horizon, n_node)
        out.append(evaluate(y, pred, m, tod, w.horizons, "HistoricalAverage", split))
    df = pd.concat(out, ignore_index=True)
    df["seed"] = -1
    df["dataset_tag"] = args.tag

    res_dir = (os.path.join(ROOT, "results", args.group)
               if args.group else os.path.join(ROOT, "results"))
    os.makedirs(res_dir, exist_ok=True)
    csv = os.path.join(res_dir, "metrics_historical_average.csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    print("\nAll day")
    print(show(df, "all"))
    print("\nAM peak")
    print(show(df, "am_peak"))
    print("\nPM peak")
    print(show(df, "pm_peak"))
    ex = int(df["mape_excluded"].max())
    print(f"\nMaximum low-speed cells excluded from MAPE: {ex}; wrote {csv}")


if __name__ == "__main__":
    main()
