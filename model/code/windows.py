# -*- coding: utf-8 -*-
"""
Sliding-window construction.

After weekends and holidays are removed, the time axis is not continuous.
Windows are therefore generated separately within each calendar day, with inputs
and labels restricted to the same day and never crossing split boundaries. The
daily branch uses the same time slot from the previous available workday, and
the weekly branch uses the same weekday in the previous week. Both branches
carry availability flags and are masked by the model when unavailable.
"""
import json
import os

import numpy as np

RECENT_LEN = 12          # Recent input length: past 60 minutes.
HORIZONS = (3, 6, 9, 12)  # Future 15, 30, 45, and 60 minutes.
PERIOD_LEN = 12          # Segment length for daily and weekly branches.
AM_PEAK = (7 * 12, 10 * 12)   # AM peak: 07:00 to 10:00.
PM_PEAK = (16 * 12, 19 * 12)  # PM peak: 16:00 to 19:00.


class WindowIndex:
    """Wrap an npz dataset as sample-indexed window collections."""

    def __init__(self, path, recent_len=RECENT_LEN, horizons=HORIZONS,
                 period_len=PERIOD_LEN):
        d = np.load(path, allow_pickle=False)
        self.speed = d["speed"]
        self.eval_mask = d["eval_mask"]
        self.dyn = d["dyn"]
        self.adj = d["adj"]
        self.day_of_row = d["day_of_row"]
        self.tod_of_row = d["tod_of_row"]
        self.day_split = d["day_split"]
        self.day_dow = d["day_dow"]
        self.prev_workday = d["prev_workday"]
        self.prev_week = d["prev_week"]
        self.node_id = d["node_id"]
        self.days = d["days"]
        self.steps_per_day = 288
        self.recent_len = recent_len
        self.horizons = tuple(horizons)
        self.period_len = period_len
        self.n_node = self.speed.shape[1]

        meta_path = path.replace("dataset_", "meta_").replace(".npz", ".json")
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.speed_mean = self.meta["speed_mean"]
        self.speed_std = self.meta["speed_std"]

        # Start row of each day on the global time axis.
        self.day_start = np.zeros(len(self.days), dtype=np.int64)
        for i in range(len(self.days)):
            self.day_start[i] = np.searchsorted(self.day_of_row, i)

        self._build()

    def _build(self):
        max_h = max(self.horizons)
        lo = self.recent_len - 1
        hi = self.steps_per_day - max_h - 1
        self.samples = {}
        for split in ("train", "val", "test"):
            days = np.where(self.day_split == split)[0]
            rows = []
            for day in days:
                for t in range(lo, hi + 1):
                    rows.append((day, t))
            self.samples[split] = np.array(rows, dtype=np.int64)

    def describe(self):
        out = []
        for split in ("train", "val", "test"):
            s = self.samples[split]
            days = np.unique(s[:, 0])
            out.append(f"{split}: {len(s):,} samples / {len(days)} days")
        return "  ".join(out)

    def recent(self, idx):
        """Recent branch, shape (n, recent_len, n_node)."""
        day, t = idx[:, 0], idx[:, 1]
        base = self.day_start[day] + t
        offs = np.arange(-self.recent_len + 1, 1)
        rows = base[:, None] + offs[None, :]
        return self.speed[rows]

    def target(self, idx):
        """Targets, shape (n, n_horizon, n_node), plus evaluable mask and target time slots."""
        day, t = idx[:, 0], idx[:, 1]
        base = self.day_start[day] + t
        hs = np.array(self.horizons)
        rows = base[:, None] + hs[None, :]
        return self.speed[rows], self.eval_mask[rows], self.tod_of_row[rows]

    def _period(self, idx, ref_day):
        """Fetch periodic branch segments by day-level mapping, returning the segment and availability flag."""
        day, t = idx[:, 0], idx[:, 1]
        src = ref_day[day]
        ok = src >= 0
        offs = np.arange(-self.period_len + 1, 1)
        out = np.zeros((len(idx), self.period_len, self.n_node), dtype=np.float32)
        if ok.any():
            base = self.day_start[src[ok]] + t[ok]
            rows = base[:, None] + offs[None, :]
            out[ok] = self.speed[rows]
        return out, ok

    def daily(self, idx):
        return self._period(idx, self.prev_workday)

    def weekly(self, idx):
        return self._period(idx, self.prev_week)

    def dynamic(self, idx):
        """External features at target times, shape (n, n_horizon, n_dyn)."""
        day, t = idx[:, 0], idx[:, 1]
        base = self.day_start[day] + t
        hs = np.array(self.horizons)
        rows = base[:, None] + hs[None, :]
        return self.dyn[rows]


def peak_mask(tod):
    """Create AM and PM peak masks using target time slots."""
    am = (tod >= AM_PEAK[0]) & (tod < AM_PEAK[1])
    pm = (tod >= PM_PEAK[0]) & (tod < PM_PEAK[1])
    return am, pm


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    w = WindowIndex(os.path.join(root, "data", "processed",
                                 "dataset_novfeb_eps0.3.npz"))
    print(w.describe())
    for split in ("train", "val", "test"):
        idx = w.samples[split]
        y, m, tod = w.target(idx)
        _, d_ok = w.daily(idx)
        _, k_ok = w.weekly(idx)
        am, pm = peak_mask(tod)
        print(f"{split}: target cells {y.size:,}, evaluable {m.mean()*100:.2f}%  "
              f"daily available {d_ok.mean()*100:.1f}%  weekly available {k_ok.mean()*100:.1f}%  "
              f"AM-peak targets {int(am.sum()):,}, PM-peak targets {int(pm.sum()):,}")
    x = w.recent(w.samples['val'][:64])
    print("recent branch shape", x.shape, "target shape", w.target(w.samples['val'][:64])[0].shape,
          "external feature shape", w.dynamic(w.samples['val'][:64]).shape)
