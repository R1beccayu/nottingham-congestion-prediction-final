# -*- coding: utf-8 -*-
"""
Batch construction. The dataset is only a few dozen megabytes, so it is kept on
the GPU and windows are gathered by index instead of using a DataLoader, which
avoids making the CPU side a bottleneck.

Each batch provides the recent, daily, and weekly branches, together with
external features at the target times. Baseline models use whichever fields they
need. Missing periodic branches are zero-filled and accompanied by availability
flags.
"""
import numpy as np
import torch


class Batcher:
    def __init__(self, w, split, device, batch_size=256, shuffle=False, seed=0,
                 keep_days=None):
        """keep_days lists the calendar-day indices allowed for model selection on selected dates."""
        self.w = w
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
        self.speed = t(w.speed, torch.float32)
        self.mask = t(w.eval_mask, torch.bool)
        self.dyn = t(w.dyn, torch.float32)
        self.tod = t(w.tod_of_row, torch.long)
        self.day_dow = t(w.day_dow, torch.long)
        self.day_start = t(w.day_start, torch.long)
        self.prev_workday = t(w.prev_workday, torch.long)
        self.prev_week = t(w.prev_week, torch.long)
        sel = w.samples[split]
        if keep_days is not None:
            sel = sel[np.isin(sel[:, 0], np.asarray(keep_days))]
        self.idx = t(sel, torch.long)

        self.recent_off = torch.arange(-w.recent_len + 1, 1, device=device)
        self.period_off = torch.arange(-w.period_len + 1, 1, device=device)
        self.h = torch.as_tensor(w.horizons, dtype=torch.long, device=device)
        self.mean, self.std = w.speed_mean, w.speed_std
        self.n_node = w.n_node
        self.n_dyn = w.dyn.shape[1]

        # Training-period node time-slot means, used as an optional periodic
        # prior. Validation and test information is not used in estimation.
        train_days = np.where(w.day_split == "train")[0]
        train_rows = np.isin(w.day_of_row, train_days)
        profile = np.full((w.steps_per_day, w.n_node), np.nan, dtype=np.float32)
        for slot in range(w.steps_per_day):
            use = train_rows & (w.tod_of_row == slot)
            if use.any():
                valid = w.eval_mask[use]
                count = valid.sum(axis=0)
                total = np.where(valid, w.speed[use], 0.0).sum(axis=0)
                profile[slot] = np.divide(
                    total, count,
                    out=np.full(w.n_node, np.nan, dtype=np.float32),
                    where=count > 0,
                )
        missing = np.isnan(profile)
        if missing.any():
            count = np.isfinite(profile).sum(axis=0)
            node_mean = np.divide(
                np.nansum(profile, axis=0), count,
                out=np.full(w.n_node, np.nan, dtype=np.float32),
                where=count > 0,
            )
            node_mean = np.where(np.isfinite(node_mean), node_mean,
                                 np.nanmean(profile))
            profile[missing] = np.take(node_mean, np.where(missing)[1])
        self.ha_profile = t(profile, torch.float32)

    def __len__(self):
        return (len(self.idx) + self.batch_size - 1) // self.batch_size

    def n_sample(self):
        return len(self.idx)

    def _period(self, ref, day, t):
        src = ref[day]
        ok = src >= 0
        base = self.day_start[src.clamp(min=0)] + t
        rows = base[:, None] + self.period_off[None, :]
        seg = self.speed[rows]
        return seg * ok[:, None, None], ok

    def _batch(self, sel):
        day, t = sel[:, 0], sel[:, 1]
        base = self.day_start[day] + t
        rows = base[:, None] + self.recent_off[None, :]
        recent = self.speed[rows]
        recent_tod = self.tod[rows]
        recent_dow = self.day_dow[day][:, None].expand_as(recent_tod)

        trows = base[:, None] + self.h[None, :]
        y = self.speed[trows]
        m = self.mask[trows]
        tod = self.tod[trows]
        dyn = self.dyn[trows]
        ha = self.ha_profile[tod]

        daily, d_ok = self._period(self.prev_workday, day, t)
        weekly, w_ok = self._period(self.prev_week, day, t)

        z = lambda x: (x - self.mean) / self.std
        return dict(
            recent=z(recent), daily=z(daily) * d_ok[:, None, None],
            weekly=z(weekly) * w_ok[:, None, None],
            recent_tod=recent_tod, recent_dow=recent_dow,
            daily_ok=d_ok.float(), weekly_ok=w_ok.float(),
            dyn=dyn, y=y, y_norm=z(y), ha_norm=z(ha), mask=m, tod=tod,
        )

    def __iter__(self):
        order = (torch.randperm(len(self.idx), generator=self.gen).to(self.device)
                 if self.shuffle else torch.arange(len(self.idx), device=self.device))
        for i in range(0, len(order), self.batch_size):
            yield self._batch(self.idx[order[i:i + self.batch_size]])


def masked_mse(pred, target, mask):
    """Backpropagate error only on evaluable cells; imputed values do not contribute to training."""
    d = (pred - target) ** 2
    d = d * mask
    n = mask.sum()
    return d.sum() / n.clamp(min=1)


def masked_mae(pred, target, mask):
    d = (pred - target).abs() * mask
    return d.sum() / mask.sum().clamp(min=1)


def masked_huber(pred, target, mask, delta=0.5):
    d = (pred - target).abs()
    loss = torch.where(d <= delta, 0.5 * d.square() / delta, d - 0.5 * delta)
    return (loss * mask).sum() / mask.sum().clamp(min=1)
