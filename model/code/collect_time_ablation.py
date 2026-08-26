# -*- coding: utf-8 -*-
"""Summarise the three-factor ablation results with node-level time-slot baselines."""
import argparse
import glob
import os

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ["MAE", "RMSE", "WMAPE", "MAPE"]
DESIGN = {
    "A0": ("B0", 0, 0, 0),
    "A1": ("B1", 1, 0, 0),
    "A2": ("B2", 0, 1, 0),
    "A3": ("B3", 1, 1, 0),
    "B4": ("B4", 0, 0, 1),
    "B5": ("B5", 1, 0, 1),
    "B6": ("B6", 0, 1, 1),
    "B7": ("B7", 1, 1, 1),
}


def load_metrics(res_dir, tag):
    frames = []
    for path in sorted(glob.glob(os.path.join(res_dir, "metrics_*.csv"))):
        frame = pd.read_csv(path)
        if "dataset_tag" in frame.columns:
            frame = frame[frame.dataset_tag == tag]
        if not frame.empty and frame.model.iloc[0] in DESIGN:
            frames.append(frame)
    if not frames:
        raise SystemExit("No metric files available for the B-series ablation")
    return pd.concat(frames, ignore_index=True)


def aggregate(df, split, scope):
    d = df[(df.split == split) & (df.scope == scope)].copy()
    d["configuration"] = d.model.map(lambda x: DESIGN[x][0])
    grouped = d.groupby(["configuration", "horizon_min"])[METRICS]
    mean = grouped.mean()
    std = grouped.std(ddof=0)
    count = d.groupby(["configuration", "horizon_min"])["seed"].nunique()
    out = mean.copy()
    for metric in METRICS:
        out[f"{metric}_std"] = std[metric]
    out["n_seed"] = count
    out = out.reset_index()
    factors = {v[0]: v[1:] for v in DESIGN.values()}
    out["st_enhance"] = out.configuration.map(lambda x: factors[x][0])
    out["weather_gate"] = out.configuration.map(lambda x: factors[x][1])
    out["node_time_baseline"] = out.configuration.map(lambda x: factors[x][2])
    return out.sort_values(["configuration", "horizon_min"])


def add_improvement(table):
    table = table.copy()
    table["MAE_improvement_vs_B0_pct"] = np.nan
    for horizon in table.horizon_min.unique():
        base = table[(table.configuration == "B0") &
                     (table.horizon_min == horizon)]
        if base.empty:
            continue
        value = float(base.MAE.iloc[0])
        rows = table.horizon_min == horizon
        table.loc[rows, "MAE_improvement_vs_B0_pct"] = (
            (value - table.loc[rows, "MAE"]) / value * 100
        )
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    res_dir = os.path.join(ROOT, "results", args.group)
    df = load_metrics(res_dir, args.tag)
    have = {DESIGN[m][0] for m in df.model.unique() if m in DESIGN}
    print(f"Available B-series configurations {sorted(have)}")
    for scope in ("all", "am_peak", "pm_peak"):
        table = add_improvement(aggregate(df, args.split, scope))
        path = os.path.join(res_dir, f"table_time_baseline_{scope}_{args.split}.csv")
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
