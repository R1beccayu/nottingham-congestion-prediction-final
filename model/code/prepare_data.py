# -*- coding: utf-8 -*-
"""
Build the modelling input matrix.

Compared with the established preprocessing specification, all statistic-based
steps, including same-time-slot medians, node medians, and normalisation
parameters, are estimated only from the training period to avoid validation or
test information leaking into training. Linear interpolation is also restricted
within each calendar day, because the time axis is no longer continuous after
weekends and holidays are removed.

Output: data/processed/dataset_<tag>.npz
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "data_processed")
IN_COMMON = os.path.join(RAW, "model_inputs", "weekday_no_bank_holiday", "common")
DIAG = os.path.join(RAW, "speed", "model_input_diagnostics_weekday_no_bank_holiday")
ADJ_DIR = os.path.join(RAW, "adjacency", "osm")
OUT = os.path.join(ROOT, "data", "processed")

TRAIN_END = "2024-12-10"
VAL_END = "2025-01-07"
TEST_END = "2025-01-31"

STEPS_PER_DAY = 288
MAX_GAP = 3
IQR_MULT = 3.0
IQR_MIN_OBS = 100

# Traffic timestamps are in UTC. This was identified from the within-day speed
# curves: the curves during and after British Summer Time have nearly the same
# shape but are shifted by one hour, and commute peaks should not move simply
# because daylight saving time ends. In 2024, British Summer Time ended at
# 02:00 local time on 27 October, corresponding to 01:00 UTC.
BST_END_UTC = "2024-10-27 01:00:00"
# Visual Crossing timestamps are provided in local time, so they must be
# converted to UTC before alignment with traffic data; otherwise October would
# be shifted by one hour.
BST_END_LOCAL = "2024-10-27 02:00:00"
WEATHER_DIR = "weather"
WEATHER_NUM = ["temp", "humidity", "precip", "windspeed", "visibility"]


def utc_to_local(ts_utc):
    """Convert UTC timestamps to UK local time."""
    off = pd.Series(pd.Timedelta(0), index=range(len(ts_utc)))
    off[(ts_utc < pd.Timestamp(BST_END_UTC)).to_numpy()] = pd.Timedelta(hours=1)
    return ts_utc.reset_index(drop=True) + off


def local_to_utc(ts_local):
    """Convert UK local timestamps to UTC."""
    off = pd.Series(pd.Timedelta(0), index=range(len(ts_local)))
    off[(ts_local < pd.Timestamp(BST_END_LOCAL)).to_numpy()] = pd.Timedelta(hours=1)
    return ts_local.reset_index(drop=True) - off


def build_dynamic(ts_utc, raw_root, holiday_src):
    """
    Rebuild external features. Weather is read from raw hourly files, converted
    from local time to UTC, and backward-matched to each five-minute traffic
    timestamp. Hour and weekday encodings are generated in local time because
    commuting patterns follow the local clock rather than UTC.
    """
    wdir = os.path.join(raw_root, "data_external", WEATHER_DIR)
    files = sorted(f for f in os.listdir(wdir) if f.lower().endswith(".csv"))
    parts = []
    for f in files:
        w = pd.read_csv(os.path.join(wdir, f))
        w["local"] = pd.to_datetime(w["datetime"])
        parts.append(w)
    w = pd.concat(parts, ignore_index=True).sort_values("local")
    w["utc"] = local_to_utc(w["local"]).to_numpy()
    w = w.drop_duplicates("utc").sort_values("utc").reset_index(drop=True)

    for c in WEATHER_NUM:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w[WEATHER_NUM] = (w.set_index("utc")[WEATHER_NUM]
                      .interpolate(method="time").ffill().bfill().to_numpy())
    w["icon"] = w["icon"].astype(str).str.strip().str.replace("-", "_")

    left = pd.DataFrame({"utc": ts_utc.to_numpy()})
    m = pd.merge_asof(left, w[["utc", "icon"] + WEATHER_NUM], on="utc",
                      direction="backward")
    m[WEATHER_NUM] = m[WEATHER_NUM].ffill().bfill()
    m["icon"] = m["icon"].ffill().bfill()

    local = utc_to_local(ts_utc)
    out = m[WEATHER_NUM].copy()
    out["is_rain"] = (((m["precip"].fillna(0) > 0) |
                       m["icon"].str.contains("rain")).astype(np.float32))
    hour = local.dt.hour.to_numpy()
    minute = local.dt.minute.to_numpy()
    wday = local.dt.dayofweek.to_numpy()
    tod = hour * 12 + minute // 5
    out["hour_sin"] = np.sin(2 * np.pi * tod / STEPS_PER_DAY)
    out["hour_cos"] = np.cos(2 * np.pi * tod / STEPS_PER_DAY)
    out["weekday_sin"] = np.sin(2 * np.pi * wday / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * wday / 7)

    # Term-time and Christmas-period date ranges follow the established source
    # and are applied by local date.
    hs = pd.read_csv(holiday_src, usecols=["timestamp", "is_school_holiday",
                                           "is_christmas_period"])
    hs["date"] = pd.to_datetime(hs["timestamp"]).dt.normalize()
    flags = hs.groupby("date")[["is_school_holiday", "is_christmas_period"]].max()
    ld = local.dt.normalize()
    for c in ("is_school_holiday", "is_christmas_period"):
        out[c] = ld.map(flags[c]).fillna(0).to_numpy().astype(np.float32)

    icons = sorted(m["icon"].dropna().unique())
    for ic in icons:
        out[f"icon_{ic}"] = (m["icon"] == ic).astype(np.float32)
    return out.astype(np.float32), tod.astype(np.int16), icons


def load_matrix(path, dtype=np.float32):
    return pd.read_csv(path).to_numpy(dtype=dtype)


def split_by_date(dates, train_end=TRAIN_END, val_end=VAL_END):
    """Return the split label assigned to each calendar day."""
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    labels = np.empty(len(dates), dtype="<U5")
    for i, d in enumerate(dates):
        if d <= train_end:
            labels[i] = "train"
        elif d <= val_end:
            labels[i] = "val"
        else:
            labels[i] = "test"
    return labels


def interp_within_day(speed, day_of_row, max_gap=MAX_GAP):
    """Apply linear interpolation within each calendar day, filling only gaps no longer than max_gap and never crossing days."""
    filled = speed.copy()
    method = np.zeros(speed.shape, dtype=np.int8)
    for day in np.unique(day_of_row):
        rows = np.where(day_of_row == day)[0]
        block = filled[rows]
        for j in range(block.shape[1]):
            col = block[:, j]
            nan_pos = np.isnan(col)
            if not nan_pos.any() or nan_pos.all():
                continue
            # Find consecutive gaps; interpolate only when both ends are
            # observed and the gap length is within the limit.
            idx = 0
            n = len(col)
            while idx < n:
                if not nan_pos[idx]:
                    idx += 1
                    continue
                start = idx
                while idx < n and nan_pos[idx]:
                    idx += 1
                end = idx - 1
                if start == 0 or end == n - 1:
                    continue
                if end - start + 1 > max_gap:
                    continue
                left, right = col[start - 1], col[end + 1]
                span = end - start + 2
                for k in range(start, end + 1):
                    w = (k - start + 1) / span
                    col[k] = left + (right - left) * w
                method[rows[start]:rows[end] + 1, j] = 1
            block[:, j] = col
        filled[rows] = block
    return filled, method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0.3", help="OSM adjacency-matrix threshold level")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--source_root", default=os.path.join(ROOT, "data", "raw"),
                    help="Root directory of the data package; should contain data_processed and data_external")
    ap.add_argument(
        "--common_rel",
        default=os.path.join("data_processed", "model_inputs",
                             "weekday_no_bank_holiday", "common"),
        help="Common modelling-input directory relative to source_root",
    )
    ap.add_argument(
        "--diag_rel",
        default=os.path.join("data_processed", "speed",
                             "model_input_diagnostics_weekday_no_bank_holiday"),
        help="Speed diagnostic directory relative to source_root",
    )
    ap.add_argument(
        "--adj_rel",
        default=os.path.join("data_processed", "adjacency", "osm"),
        help="OSM adjacency directory relative to source_root",
    )
    ap.add_argument("--train_end", default=TRAIN_END)
    ap.add_argument("--val_end", default=VAL_END)
    ap.add_argument("--test_end", default=TEST_END)
    ap.add_argument("--iqr_scope", default="train", choices=["train", "full"],
                    help="Period used to estimate the IQR upper bound: train uses only the training period, full uses the full period")
    args = ap.parse_args()
    tag = args.tag or f"eps{args.eps}"
    os.makedirs(OUT, exist_ok=True)

    source_root = os.path.abspath(args.source_root)
    in_common = os.path.join(source_root, args.common_rel)
    diag_dir = os.path.join(source_root, args.diag_rel)
    adj_dir = os.path.join(source_root, args.adj_rel)
    for label, path in (("common input", in_common), ("speed diagnostics", diag_dir),
                        ("OSM adjacency", adj_dir)):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{label} directory does not exist: {path}")

    ts = pd.read_csv(os.path.join(in_common, "timestamps.csv"))["datetime"]
    ts = pd.to_datetime(ts)
    nodes = pd.read_csv(os.path.join(in_common, "node_list.csv"))
    nodes = nodes.sort_values("node_order").reset_index(drop=True)

    speed_raw = load_matrix(os.path.join(diag_dir, "speed_raw.csv"))
    obs = load_matrix(os.path.join(diag_dir, "speed_observed_mask.csv"), np.int8).astype(bool)
    iqr_src = load_matrix(os.path.join(diag_dir, "iqr_outlier_mask.csv"), np.int8).astype(bool)

    n_t, n_v = speed_raw.shape
    assert n_t == len(ts), f"Inconsistent number of time steps: {n_t} vs {len(ts)}"
    assert n_v == len(nodes), f"Inconsistent number of nodes: {n_v} vs {len(nodes)}"

    col_order = list(pd.read_csv(os.path.join(diag_dir, "speed_raw.csv"), nrows=0).columns)
    assert col_order == nodes["node_id"].tolist(), "Speed matrix column order is inconsistent with node_list"

    # Time-axis structure.
    dates = ts.dt.normalize()
    uniq_days = pd.DatetimeIndex(sorted(dates.unique()))
    day_of_row = dates.map({d: i for i, d in enumerate(uniq_days)}).to_numpy()
    # Days are still segmented in UTC so the time axis remains continuous. The
    # within-day time slot is converted to local time because commuting patterns,
    # same-time-slot imputation statistics, historical averages, and peak-period
    # definitions should all follow the local clock.
    tod_utc = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy().astype(np.int16)
    dyn_df, tod_of_row, icon_list = build_dynamic(
        ts, source_root, os.path.join(in_common, "dynamic_features.csv"))
    shifted = int((tod_of_row != tod_utc).sum())
    print(f"Within-day slots corrected to local time; affected time steps: {shifted:,} "
          f"({shifted / n_t * 100:.1f}%). Weather has been converted from local time to UTC and realigned.")
    day_split = split_by_date(uniq_days, args.train_end, args.val_end)
    row_split = day_split[day_of_row]

    if uniq_days.max() > pd.Timestamp(args.test_end):
        raise ValueError(f"Data contain dates after test_end: {uniq_days.max().date()}")
    if uniq_days.max() < pd.Timestamp(args.test_end):
        raise ValueError(f"Data do not cover test_end: {uniq_days.max().date()} < {args.test_end}")

    counts = pd.Series(day_split).value_counts()
    print(f"Nodes {n_v}, time steps {n_t}, calendar days {len(uniq_days)}")
    print(f"Split days train={counts.get('train',0)} val={counts.get('val',0)} test={counts.get('test',0)}")

    # IQR outlier removal: node-wise, upper-tail only, multiplier 3.0, minimum
    # observations 100. Quantiles are taken from all speed-observed times for
    # each node. The difference here is that the estimation period is restricted
    # to training, so this step no longer uses validation or test distributions.
    valid = ~np.isnan(speed_raw)
    train_rows_pre = row_split == "train"
    scope_rows = train_rows_pre if args.iqr_scope == "train" else np.ones(n_t, bool)
    iqr_mask = np.zeros_like(valid)
    upper = np.full(n_v, np.nan, dtype=np.float64)
    for j in range(n_v):
        v = speed_raw[scope_rows, j]
        v = v[~np.isnan(v)]
        if v.size < IQR_MIN_OBS:
            continue
        q1, q3 = np.percentile(v.astype(np.float64), [25, 75])
        upper[j] = q3 + IQR_MULT * (q3 - q1)
        iqr_mask[:, j] = valid[:, j] & (speed_raw[:, j] > upper[j])
    agree = float((iqr_mask == iqr_src).mean()) * 100
    print(f"IQR upper bound estimated from {args.iqr_scope}; removed {int(iqr_mask.sum()):,} cells "
          f"({iqr_mask.sum()/valid.sum()*100:.4f}%). Source-data mask has {int(iqr_src.sum()):,} cells; "
          f"cell-wise agreement {agree:.4f}%")

    # Evaluable cells: speed is observed and not flagged as an IQR outlier.
    eval_mask = valid & ~iqr_mask
    print(f"Evaluable cells {eval_mask.sum():,} ({eval_mask.mean()*100:.2f}%), "
          f"to impute {(~eval_mask).sum():,} ({(~eval_mask).mean()*100:.2f}%)")

    # Keep only evaluable cells as known values; reconstruct the rest from
    # training-set statistics.
    work = np.where(eval_mask, speed_raw, np.nan).astype(np.float32)
    filled, method = interp_within_day(work, day_of_row)

    train_rows = row_split == "train"
    # Same-time-slot node medians, using only the training set. Some node-slot
    # combinations may have no observations in the full training period and will
    # become NaN here; node medians below provide the fallback.
    tod_median = np.full((STEPS_PER_DAY, n_v), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for slot in range(STEPS_PER_DAY):
            sel = train_rows & (tod_of_row == slot)
            if sel.any():
                tod_median[slot] = np.nanmedian(filled[sel], axis=0)
        node_median = np.nanmedian(filled[train_rows], axis=0).astype(np.float32)
        global_median = float(np.nanmedian(filled[train_rows]))
    n_blank_slot = int(np.isnan(tod_median).sum())
    if n_blank_slot:
        print(f"Node time-slot combinations with no training observations: {n_blank_slot}; filled by node medians")

    miss = np.isnan(filled)
    r, c = np.where(miss)
    cand = tod_median[tod_of_row[r], c]
    use_tod = ~np.isnan(cand)
    filled[r[use_tod], c[use_tod]] = cand[use_tod]
    method[r[use_tod], c[use_tod]] = 2

    miss = np.isnan(filled)
    r, c = np.where(miss)
    cand = node_median[c]
    use_node = ~np.isnan(cand)
    filled[r[use_node], c[use_node]] = cand[use_node]
    method[r[use_node], c[use_node]] = 3

    still = np.isnan(filled)
    filled[still] = global_median
    method[still] = 4

    assert not np.isnan(filled).any(), "Missing values remain after imputation"
    names = {0: "observed", 1: "short_gap_interp", 2: "train_tod_median",
             3: "train_node_median", 4: "train_global_median"}
    for k, name in names.items():
        cnt = int((method == k).sum())
        if cnt:
            print(f"  {name:<20} {cnt:>9,} ({cnt/method.size*100:.2f}%)")

    # Normalisation parameters are estimated only from evaluable training cells.
    tr_vals = filled[train_rows][eval_mask[train_rows]]
    sp_mean, sp_std = float(tr_vals.mean()), float(tr_vals.std())
    print(f"Training-period speed mean={sp_mean:.4f} std={sp_std:.4f} mph "
          f"min={tr_vals.min():.2f} max={tr_vals.max():.2f}")

    # Dynamic features.
    const = [c for c in dyn_df.columns if dyn_df[c].nunique() <= 1]
    if const:
        print(f"Removed constant columns {const}")
    keep = [c for c in dyn_df.columns if c not in const]
    cont = [c for c in WEATHER_NUM if c in keep]
    dyn_x = dyn_df[keep].to_numpy(dtype=np.float32)
    ci = [keep.index(c) for c in cont]
    dyn_mean = dyn_x[train_rows][:, ci].mean(axis=0)
    dyn_std = dyn_x[train_rows][:, ci].std(axis=0)
    dyn_std[dyn_std < 1e-6] = 1.0
    dyn_x[:, ci] = (dyn_x[:, ci] - dyn_mean) / dyn_std
    print(f"Kept {len(keep)} dynamic feature columns; standardised columns: {cont}")

    # Adjacency matrix.
    adj = np.load(os.path.join(adj_dir, f"adj_osm_eps{args.eps}.npy")).astype(np.float32)
    assert adj.shape == (n_v, n_v), f"Adjacency matrix shape {adj.shape}"
    assert np.allclose(adj, adj.T, atol=1e-6), "Adjacency matrix is not symmetric"
    assert np.abs(np.diag(adj)).max() == 0, "Adjacency matrix diagonal is non-zero"
    same_cl = nodes["Countline"].to_numpy()
    pair = (same_cl[:, None] == same_cl[None, :]) & ~np.eye(n_v, dtype=bool)
    assert adj[pair].max() == 0, "Opposite directions of the same countline still have non-zero weights"
    deg = (adj > 0).sum(axis=1)
    print(f"Adjacency eps={args.eps}, non-zero edges {int((adj>0).sum())}, "
          f"average neighbours {deg.mean():.1f}, isolated nodes {int((deg==0).sum())}")

    # Day-level mappings for daily and weekly periodic branches.
    dow = uniq_days.dayofweek.to_numpy()
    day_pos = {d: i for i, d in enumerate(uniq_days)}
    prev_workday = np.full(len(uniq_days), -1, dtype=np.int32)
    prev_workday_gap = np.zeros(len(uniq_days), dtype=np.int16)
    prev_week = np.full(len(uniq_days), -1, dtype=np.int32)
    for i, d in enumerate(uniq_days):
        if i > 0:
            prev_workday[i] = i - 1
            prev_workday_gap[i] = (d - uniq_days[i - 1]).days
        w = d - pd.Timedelta(days=7)
        if w in day_pos:
            prev_week[i] = day_pos[w]
    long_gap = [(uniq_days[i].strftime("%Y-%m-%d"), int(g))
                for i, g in enumerate(prev_workday_gap) if g > 3]
    print(f"Days missing previous workday {int((prev_workday<0).sum())}, "
          f"days missing same weekday in previous week {int((prev_week<0).sum())}")
    if long_gap:
        print(f"Dates with a gap greater than 3 days from the previous available workday: {long_gap}")

    zero_cells = int(((filled == 0) & eval_mask).sum())
    low_cells = int(((filled < 1.0) & eval_mask).sum())
    print(f"Zero-speed observed cells {zero_cells}, observed cells below 1 mph {low_cells}; "
          f"MAPE must be restricted to cells with true speed at least 1 mph")

    meta = dict(
        tag=tag, eps=args.eps, n_time=int(n_t), n_node=int(n_v),
        n_day=int(len(uniq_days)), steps_per_day=STEPS_PER_DAY,
        train_end=args.train_end, val_end=args.val_end, test_end=args.test_end,
        source_root=source_root,
        speed_unit="mph", speed_mean=sp_mean, speed_std=sp_std,
        dyn_columns=keep, dyn_cont=cont, icons=icon_list,
        timestamp_zone="UTC", tod_zone="Europe/London",
        tod_shifted_steps=shifted,
        imputation_counts={names[k]: int((method == k).sum()) for k in names},
        eval_cells=int(eval_mask.sum()), zero_speed_cells=zero_cells,
        iqr_scope=args.iqr_scope, iqr_mult=IQR_MULT,
        iqr_removed=int(iqr_mask.sum()),
        iqr_agree_with_source=round(agree, 4),
        sub_one_mph_cells=low_cells, mape_min_speed=1.0,
        blank_train_slots=n_blank_slot,
        adj_nonzero_edges=int((adj > 0).sum()), adj_isolated_nodes=int((deg == 0).sum()),
        days_missing_prev_workday=int((prev_workday < 0).sum()),
        days_missing_prev_week=int((prev_week < 0).sum()),
        long_workday_gaps=long_gap,
    )
    out = os.path.join(OUT, f"dataset_{tag}.npz")
    np.savez_compressed(
        out,
        speed=filled, eval_mask=eval_mask, impute_method=method,
        dyn=dyn_x, adj=adj,
        day_of_row=day_of_row.astype(np.int32), tod_of_row=tod_of_row.astype(np.int16),
        row_split=row_split.astype("<U5"), day_split=day_split.astype("<U5"),
        day_dow=dow.astype(np.int8), prev_workday=prev_workday,
        prev_workday_gap=prev_workday_gap, prev_week=prev_week,
        node_id=nodes["node_id"].to_numpy().astype("<U40"),
        days=uniq_days.strftime("%Y-%m-%d").to_numpy().astype("<U10"),
    )
    with open(os.path.join(OUT, f"meta_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
