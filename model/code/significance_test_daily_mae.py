# -*- coding: utf-8 -*-
"""Calculate paired daily MAE tests from the delivered prediction arrays."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_METHODS = ("STGCN", "A3", "A4", "B5", "STAEformer")
FOLDERS = {
    "HA": "historical_average",
    "HistoricalAverage": "historical_average",
    "LSTM": "lstm",
    "CNN_LSTM": "cnn_lstm",
    "CNN-LSTM": "cnn_lstm",
    "STGCN": "stgcn",
    "STSGCN": "stsgcn",
    "iTransformer": "itransformer",
    "A0": "A0",
    "A1": "A1",
    "A2": "A2",
    "A3": "A3",
    "A4": "A4",
    "B4": "B4",
    "B5": "B5",
    "B6": "B6",
    "B7": "B7",
    "ProposedDF": "proposed_df",
    "STAEformer": "staeformer",
}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                 n_boot: int) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def holm_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def load_method(package: Path, method: str) -> tuple[np.ndarray, ...]:
    if method not in FOLDERS:
        raise ValueError(
            f"Unknown method {method!r}. Available methods: {', '.join(sorted(FOLDERS))}"
        )
    folder = package / FOLDERS[method]
    matches = sorted(folder.glob("*_predictions.npz"))
    if not matches:
        raise FileNotFoundError(f"No *_predictions.npz found for {method}: {folder}")
    path = matches[0]
    with np.load(path, allow_pickle=False) as data:
        predictions = data["prediction_speed_mph_by_seed"].copy()
        actual = data["actual_speed_mph"].copy()
        mask = data["evaluation_mask"].astype(bool)
        seeds = data["seed"].copy()
        sample_date = np.asarray(
            [value[:10] for value in data["sample_origin_time_utc"].astype("U20")],
            dtype="U10",
        )
    return predictions, actual, mask, seeds, sample_date


def calculate(package: Path, output: Path, methods: tuple[str, ...], bootstrap: int,
              bootstrap_seed: int) -> None:
    rows = []
    reference_actual = reference_mask = reference_dates = None
    for method in methods:
        prediction, actual, mask, seeds, dates = load_method(package, method)
        if reference_actual is None:
            reference_actual, reference_mask, reference_dates = actual, mask, dates
        elif not (np.array_equal(actual, reference_actual)
                  and np.array_equal(mask, reference_mask)
                  and np.array_equal(dates, reference_dates)):
            raise ValueError(f"{method} does not share the same targets")
        for seed_index, seed in enumerate(seeds):
            for date in np.unique(dates):
                sample_selected = dates == date
                selected = mask[sample_selected]
                error = np.abs(prediction[seed_index, sample_selected] - actual[sample_selected])
                rows.append({
                    "date": date,
                    "model": method,
                    "seed": int(seed),
                    "MAE": float(error[selected].mean()),
                    "n_evaluable_cells": int(selected.sum()),
                })
    daily = pd.DataFrame(rows)
    if daily["date"].nunique() != 16:
        raise ValueError("Expected 16 paired test workdays")
    method_daily = (daily.groupby(["date", "model"], as_index=False)
                    .agg(MAE=("MAE", "mean"),
                         MAE_seed_std=("MAE", lambda x: x.std(ddof=0)),
                         n_evaluable_cells=("n_evaluable_cells", "max")))
    wide = method_daily.pivot(index="date", columns="model", values="MAE")
    rng = np.random.default_rng(bootstrap_seed)
    tests = []
    pairs = [
        (methods[left], methods[right])
        for left in range(len(methods))
        for right in range(left + 1, len(methods))
    ]
    for baseline, candidate in pairs:
        base = wide[baseline].to_numpy()
        current = wide[candidate].to_numpy()
        difference = base - current
        statistic, p_value = wilcoxon(base, current, alternative="two-sided",
                                      method="auto")
        low, high = bootstrap_ci(difference, rng, bootstrap)
        tests.append({
            "baseline": baseline,
            "candidate": candidate,
            "n_days": len(difference),
            "baseline_mean_MAE": float(base.mean()),
            "candidate_mean_MAE": float(current.mean()),
            "mean_MAE_difference": float(difference.mean()),
            "median_MAE_difference": float(np.median(difference)),
            "relative_MAE_improvement_pct": float(difference.mean() / base.mean() * 100),
            "improved_days": int((difference > 0).sum()),
            "worsened_days": int((difference < 0).sum()),
            "tied_days": int((difference == 0).sum()),
            "bootstrap_mean_diff_ci95_low": float(low),
            "bootstrap_mean_diff_ci95_high": float(high),
            "wilcoxon_statistic": float(statistic),
            "p_value_two_sided": float(p_value),
        })
    tests = pd.DataFrame(tests)
    tests["p_value_holm"] = holm_adjust(tests["p_value_two_sided"].to_numpy())
    tests["significant_0.05_holm"] = tests["p_value_holm"] < 0.05
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "significance_daily_mae_by_seed.csv", index=False,
                 encoding="utf-8-sig", float_format="%.8f")
    method_daily.to_csv(output / "significance_daily_mae_by_method.csv", index=False,
                        encoding="utf-8-sig", float_format="%.8f")
    tests.to_csv(output / "significance_wilcoxon_summary.csv", index=False,
                 encoding="utf-8-sig", float_format="%.8f")
    print(tests.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help=(
            "Methods to include. The script runs all pairwise comparisons among "
            "the selected methods. Default: %(default)s"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    args = parser.parse_args()
    calculate(args.package, args.output, tuple(args.methods), args.bootstrap, args.bootstrap_seed)


if __name__ == "__main__":
    main()
