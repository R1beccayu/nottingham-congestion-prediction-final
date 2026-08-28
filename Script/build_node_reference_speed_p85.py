#!/usr/bin/env python3
"""Calculate node-level 85th percentile reference speed from observed training data.

The default settings target the Nov-Feb 155-node dataset used by the current
forecast exports. Reference speed is calculated from non-imputed observed
5-minute records only, with a positive motorised vehicle count.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATE_RE = re.compile(r"speed_(\d{4}-\d{2}-\d{2})_5min_node_level\.csv$")
DEFAULT_BANK_HOLIDAYS = [
    "2024-12-25",
    "2024-12-26",
    "2025-01-01",
]
REQUIRED_COLUMNS = [
    "time_5min",
    "node_id",
    "speed_observation_status",
    "motorised_mean_speed",
    "motorised_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data_interim" / "speed_5min_v2_nov_feb",
        help="Folder containing speed_YYYY-MM-DD_5min_node_level.csv files.",
    )
    parser.add_argument(
        "--node-list",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "model_inputs" / "nov_feb" / "common" / "node_list.csv",
        help="Canonical node list. node_order and node_id define output order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data_processed"
        / "speed"
        / "reference_speed"
        / "node_reference_speed_p85_training_novfeb_155.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--start-date",
        default="2024-11-01",
        help="Inclusive training start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default="2025-01-14",
        help="Inclusive training end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.85,
        help="Reference speed percentile expressed as a fraction, e.g. 0.85.",
    )
    parser.add_argument(
        "--bank-holidays",
        nargs="*",
        default=DEFAULT_BANK_HOLIDAYS,
        help="Bank holidays to exclude by default, in YYYY-MM-DD format.",
    )
    parser.set_defaults(exclude_weekends=True, exclude_bank_holidays=True)
    parser.add_argument(
        "--include-weekends",
        action="store_false",
        dest="exclude_weekends",
        help="Include Saturdays and Sundays in the reference-speed sample.",
    )
    parser.add_argument(
        "--include-bank-holidays",
        action="store_false",
        dest="exclude_bank_holidays",
        help="Include configured bank holidays in the reference-speed sample.",
    )
    return parser.parse_args()


def file_date(path: Path) -> pd.Timestamp:
    match = DATE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected input filename: {path.name}")
    return pd.Timestamp(match.group(1))


def selected_files(
    input_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    exclude_weekends: bool,
    exclude_bank_holidays: bool,
    bank_holidays: set[pd.Timestamp],
) -> list[Path]:
    files: list[Path] = []
    for path in sorted(input_dir.glob("speed_*_5min_node_level.csv")):
        day = file_date(path)
        if day < start_date or day > end_date:
            continue
        if exclude_weekends and day.dayofweek >= 5:
            continue
        if exclude_bank_holidays and day in bank_holidays:
            continue
        files.append(path)
    return files


def load_node_list(path: Path) -> pd.DataFrame:
    node_list = pd.read_csv(path)
    missing = {"node_order", "node_id"} - set(node_list.columns)
    if missing:
        raise ValueError(f"Node list is missing required columns: {sorted(missing)}")
    return node_list[["node_order", "node_id"]].copy()


def load_observed_speeds(path: Path, node_ids: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=REQUIRED_COLUMNS, parse_dates=["time_5min"])
    observed = (
        df["node_id"].isin(node_ids)
        & df["speed_observation_status"].eq("observed")
        & df["motorised_mean_speed"].notna()
        & df["motorised_count"].gt(0)
    )
    return df.loc[observed, ["node_id", "motorised_mean_speed", "motorised_count"]]


def main() -> None:
    args = parse_args()
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    bank_holidays = {pd.Timestamp(day) for day in args.bank_holidays}

    if not 0 < args.percentile < 1:
        raise ValueError("--percentile must be between 0 and 1, e.g. 0.85")

    node_list = load_node_list(args.node_list)
    node_ids = set(node_list["node_id"].astype(str))
    files = selected_files(
        args.input_dir,
        start_date,
        end_date,
        args.exclude_weekends,
        args.exclude_bank_holidays,
        bank_holidays,
    )
    if not files:
        raise FileNotFoundError("No daily speed files matched the selected training period.")

    observed_parts = [load_observed_speeds(path, node_ids) for path in files]
    observed = pd.concat(observed_parts, ignore_index=True)
    if observed.empty:
        raise ValueError("No observed speed records remained after filtering.")

    summary = (
        observed.groupby("node_id", sort=False)
        .agg(
            reference_speed_mph=("motorised_mean_speed", lambda s: s.quantile(args.percentile)),
            observed_records=("motorised_mean_speed", "size"),
            total_motorised_count=("motorised_count", "sum"),
            mean_training_speed_mph=("motorised_mean_speed", "mean"),
            median_training_speed_mph=("motorised_mean_speed", "median"),
            min_training_speed_mph=("motorised_mean_speed", "min"),
            max_training_speed_mph=("motorised_mean_speed", "max"),
        )
        .reset_index()
    )
    summary["percentile"] = args.percentile
    summary["reference_source"] = "training_observed_motorised_mean_speed"
    summary["training_start_date"] = args.start_date
    summary["training_end_date"] = args.end_date
    summary["weekends_excluded"] = args.exclude_weekends
    summary["bank_holidays_excluded"] = args.exclude_bank_holidays
    summary["training_days_used"] = len(files)

    out = node_list.merge(summary, on="node_id", how="left")
    missing_nodes = out["reference_speed_mph"].isna()
    if missing_nodes.any():
        missing = out.loc[missing_nodes, "node_id"].tolist()
        raise ValueError(f"Missing reference speed for {len(missing)} nodes: {missing[:10]}")

    ordered_cols = [
        "node_order",
        "node_id",
        "reference_speed_mph",
        "percentile",
        "observed_records",
        "total_motorised_count",
        "mean_training_speed_mph",
        "median_training_speed_mph",
        "min_training_speed_mph",
        "max_training_speed_mph",
        "reference_source",
        "training_start_date",
        "training_end_date",
        "training_days_used",
        "weekends_excluded",
        "bank_holidays_excluded",
    ]
    out = out[ordered_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} node reference speeds to {args.output}")
    print(f"Training days used: {len(files)}")
    print(
        "Reference speed range: "
        f"{out['reference_speed_mph'].min():.3f} - {out['reference_speed_mph'].max():.3f} mph"
    )


if __name__ == "__main__":
    main()
