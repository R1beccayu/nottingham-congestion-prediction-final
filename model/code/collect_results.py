# -*- coding: utf-8 -*-
"""
Summarise model results, generate baseline comparison and ablation tables, and
compute relative improvements for the acceptance checks.

The acceptance baseline is the independently implemented STGCN. Ablation
settings A0 to A3 use the same model framework and are switched by two flags, so
A0 and STGCN are close but not identical; both are reported.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_ORDER = ["HistoricalAverage", "RecentAverage", "lstm", "cnn_lstm",
                  "stgcn", "stsgcn", "staeformer", "itransformer", "A3"]
ABLATION_ORDER = ["A0", "A1", "A2", "A3", "A4"]
DISPLAY = {"HistoricalAverage": "Historical Average", "lstm": "LSTM",
           "RecentAverage": "Recent Average", "cnn_lstm": "CNN-LSTM",
           "stgcn": "STGCN", "stsgcn": "STSGCN",
           "staeformer": "STAEformer", "itransformer": "iTransformer",
           "A0": "A0 no enhancement or gate", "A1": "A1 spatio-temporal only",
           "A2": "A2 weather gate only", "A3": "A3 full proposed method",
           "A4": "A4 full method with spatial attention"}
METRICS = ["MAE", "RMSE", "WMAPE", "MAPE"]


def load_all(res_dir):
    frames = []
    for f in sorted(glob.glob(os.path.join(res_dir, "metrics_*.csv"))):
        d = pd.read_csv(f)
        if "seed" not in d.columns:
            d["seed"] = -1
        frames.append(d)
    if not frames:
        raise SystemExit("No metric files found under the results directory")
    return pd.concat(frames, ignore_index=True)


def aggregate(df, split="test", scope="all"):
    """Compute means and standard deviations across multiple seeds for each model."""
    d = df[(df.split == split) & (df.scope == scope)]
    g = d.groupby(["model", "horizon_min"])[METRICS]
    out = g.mean().round(4)
    std = g.std(ddof=0).round(4)
    cnt = d.groupby(["model", "horizon_min"])["seed"].nunique()
    out.columns = [f"{c}" for c in out.columns]
    for c in METRICS:
        out[f"{c}_std"] = std[c]
    out["n_seed"] = cnt
    return out.reset_index()


def table(agg, order, title):
    rows = []
    for m in order:
        d = agg[agg.model == m]
        if d.empty:
            continue
        for h in sorted(d.horizon_min.unique()):
            r = d[d.horizon_min == h].iloc[0]
            rows.append({
                "model": DISPLAY.get(m, m),
                "horizon": "mean" if h < 0 else f"{int(h)}min",
                "MAE": r.MAE, "RMSE": r.RMSE,
                "WMAPE": r.WMAPE, "MAPE": r.MAPE,
                "MAE_std": r.MAE_std if r.n_seed > 1 else np.nan,
                "n_seed": int(r.n_seed),
            })
    t = pd.DataFrame(rows)
    print(f"\n{title}")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}",
                      na_rep="-"))
    return t


def rel(new, base):
    """Relative change; positive values mean the metric increased and therefore worsened."""
    return (new - base) / base * 100


def acceptance(agg):
    def pick(model, h=-1):
        d = agg[(agg.model == model) & (agg.horizon_min == h)]
        return None if d.empty else d.iloc[0]

    base = pick("stgcn")
    full = pick("A3")
    if base is None or full is None:
        print("\nMissing STGCN or A3 results; skipping acceptance checks")
        return
    print("\nAcceptance checks, using the independently implemented STGCN as baseline")
    mae_gain = -rel(full.MAE, base.MAE)
    print(f"  Condition 1: mean MAE reduction {mae_gain:.2f}%, required at least 1.00%  "
          f"{'passed' if mae_gain >= 1.0 else 'failed'}")

    down = []
    for h in (15, 30, 45, 60):
        b, f = pick("stgcn", h), pick("A3", h)
        if b is None or f is None:
            continue
        g = -rel(f.MAE, b.MAE)
        down.append(g > 0)
        print(f"    {h}min MAE {b.MAE:.4f} -> {f.MAE:.4f}  {g:+.2f}%")
    print(f"  Condition 2: MAE decreased at {sum(down)} horizons, required at least 2  "
          f"{'passed' if sum(down) >= 2 else 'failed'}")

    r_rmse, r_wmape = rel(full.RMSE, base.RMSE), rel(full.WMAPE, base.WMAPE)
    ok3 = ((r_rmse < 0 and r_wmape <= 0.5) or (r_wmape < 0 and r_rmse <= 0.5))
    print(f"  Condition 3: mean RMSE {r_rmse:+.2f}%, WMAPE {r_wmape:+.2f}%  "
          f"{'passed' if ok3 else 'failed'}")

    for tag, cond, what in (("A2", "Condition 4", "spatio-temporal enhancement"),
                            ("A1", "Condition 5", "weather gate")):
        d = pick(tag)
        if d is None:
            continue
        worse_mae = rel(d.MAE, full.MAE)
        worse_wmape = rel(d.WMAPE, full.WMAPE)
        ok = worse_mae >= 0.3 or worse_wmape >= 0.3
        print(f"  {cond}: after removing {what}, MAE {worse_mae:+.2f}%, WMAPE {worse_wmape:+.2f}%  "
              f"{'passed' if ok else 'failed'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default=None, help="Summarise only the specified dataset tag")
    ap.add_argument("--group", default=None,
                    help="Read from and write summary tables to the specified results subdirectory")
    args = ap.parse_args()
    res_dir = (os.path.join(ROOT, "results", args.group)
               if args.group else os.path.join(ROOT, "results"))
    df = load_all(res_dir)
    if args.tag:
        if "dataset_tag" not in df.columns:
            raise SystemExit("Metric files do not contain dataset_tag, so filtering by dataset tag is unavailable")
        df = df[df.dataset_tag == args.tag]
        if df.empty:
            raise SystemExit(f"No metrics found for dataset tag {args.tag}")
    have = sorted(df.model.unique())
    print(f"Models with available results {have}")

    for scope, label in (("all", "all day"), ("am_peak", "AM peak"), ("pm_peak", "PM peak")):
        agg = aggregate(df, args.split, scope)
        t1 = table(agg, BASELINE_ORDER, f"Baseline comparison {label} {args.split}")
        t1.to_csv(os.path.join(res_dir, f"table_baseline_{scope}_{args.split}.csv"),
                  index=False, encoding="utf-8-sig")
        t2 = table(agg, ABLATION_ORDER, f"Ablation {label} {args.split}")
        t2.to_csv(os.path.join(res_dir, f"table_ablation_{scope}_{args.split}.csv"),
                  index=False, encoding="utf-8-sig")
        if scope == "all":
            acceptance(agg)


if __name__ == "__main__":
    main()
