# -*- coding: utf-8 -*-
"""
Unified evaluation. All models use this implementation so the numbers are
comparable.

Scores are computed only on evaluable cells, meaning original observations that
were not flagged as IQR outliers. Imputed values are excluded. Speed is measured
in mph, so MAE and RMSE are also in mph; WMAPE and MAPE are ratios. When the
true speed is below 1 mph, the MAPE denominator is too small and can inflate the
error, so those cells are excluded from MAPE and the exclusion count is recorded.
"""
import numpy as np
import pandas as pd

from windows import peak_mask

MAPE_MIN = 1.0


def _score(y, p):
    err = np.abs(y - p)
    mae = float(err.mean())
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    denom = float(np.abs(y).sum())
    wmape = float(err.sum() / denom) if denom > 0 else float("nan")
    ok = y >= MAPE_MIN
    mape = float((err[ok] / y[ok]).mean()) if ok.any() else float("nan")
    return dict(n=int(y.size), MAE=mae, RMSE=rmse,
                WMAPE=wmape * 100, MAPE=mape * 100,
                mape_excluded=int((~ok).sum()))


def evaluate(y_true, y_pred, mask, tod, horizons, model, split):
    """
    y_true / y_pred / mask have shape (n_sample, n_horizon, n_node).
    tod has shape (n_sample, n_horizon) and gives the within-day target slot.
    """
    am, pm = peak_mask(tod)
    rows = []
    scopes = {"all": None, "am_peak": am, "pm_peak": pm}
    for hi, h in enumerate(horizons):
        for scope, sel in scopes.items():
            m = mask[:, hi, :]
            if sel is not None:
                m = m & sel[:, hi][:, None]
            if not m.any():
                continue
            r = _score(y_true[:, hi, :][m], y_pred[:, hi, :][m])
            r.update(model=model, split=split, horizon_min=h * 5, scope=scope)
            rows.append(r)
    # Add an average across the four horizons; acceptance checks use this mean.
    for scope, sel in scopes.items():
        ys, ps = [], []
        for hi in range(len(horizons)):
            m = mask[:, hi, :]
            if sel is not None:
                m = m & sel[:, hi][:, None]
            ys.append(y_true[:, hi, :][m])
            ps.append(y_pred[:, hi, :][m])
        y, p = np.concatenate(ys), np.concatenate(ps)
        if y.size:
            r = _score(y, p)
            r.update(model=model, split=split, horizon_min=-1, scope=scope)
            rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["model", "split", "horizon_min", "scope", "n",
            "MAE", "RMSE", "WMAPE", "MAPE", "mape_excluded"]
    return df[cols]


def show(df, scope="all"):
    d = df[df["scope"] == scope].copy()
    d["horizon"] = d["horizon_min"].map(lambda x: "mean" if x < 0 else f"{x}min")
    return d[["model", "split", "horizon", "MAE", "RMSE", "WMAPE", "MAPE"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}")
