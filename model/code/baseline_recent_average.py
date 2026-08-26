# -*- coding: utf-8 -*-
"""
Recent Average baseline.

Predictions use the average of the most recent 12 five-minute speed values for
each sample. This baseline uses only recent history before the prediction
origin, without training-set time-slot profiles, weather variables, or the road
network graph.
"""
import argparse
import os

import numpy as np
import pandas as pd

from metrics import evaluate, show
from windows import WindowIndex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def predict_recent_average(w, idx):
    """Predict all target horizons using the mean of the most recent 12 input steps."""
    recent = w.recent(idx).astype(np.float32)       # (n, recent_len, n_node)
    avg = recent.mean(axis=1)                       # (n, n_node)
    return np.repeat(avg[:, None, :], len(w.horizons), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="novfeb_eps0.3")
    ap.add_argument("--group", default=None,
                    help="Result subdirectory used to isolate different datasets")
    args = ap.parse_args()

    path = os.path.join(ROOT, "data", "processed", f"dataset_{args.tag}.npz")
    w = WindowIndex(path)
    print(w.describe())

    out = []
    for split in ("val", "test"):
        idx = w.samples[split]
        y, m, tod = w.target(idx)
        pred = predict_recent_average(w, idx)
        out.append(evaluate(y, pred, m, tod, w.horizons, "RecentAverage", split))
    df = pd.concat(out, ignore_index=True)
    df["seed"] = -1
    df["dataset_tag"] = args.tag

    res_dir = (os.path.join(ROOT, "results", args.group)
               if args.group else os.path.join(ROOT, "results"))
    os.makedirs(res_dir, exist_ok=True)
    csv = os.path.join(res_dir, "metrics_recent_average.csv")
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
