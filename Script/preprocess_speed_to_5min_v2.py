#!/usr/bin/env python3
"""Preprocess Nottingham Vivacity speed data into 5-minute node-level records.

The raw speed files are treated as 1-minute observation-level or semi-aggregated
records. This script:

1. keeps motorised vehicle types only;
2. combines repeated Countline + timestamp + Direction + vType records using
   vehicle-count weighting;
3. aggregates records from 1-minute to 5-minute Countline + Direction nodes;
4. distinguishes observed speed, no-vehicle intervals, and raw-missing intervals;
5. writes model-ready CSV files and observation-status summaries.

Raw input files are never modified.

This is the v2 preprocessing script. It intentionally keeps the v1 script and
the v1 outputs unchanged.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd


BIN_COLS = ["0.00", "10.00", "20.00", "30.00", "40.00", "50.00", "60.00", "70.00"]
RAW_VALUE_COLS = ["Mean", "85th Percentile"]
RAW_KEY_COLS = ["Countline", "From", "Direction", "vType"]
MOTORISED_TYPES = ["bus", "car", "minibus", "motorbike", "rigid", "taxi", "truck", "van"]
VEHICLE_GROUPS = {
    "car": ["car"],
    "bus_minibus": ["bus", "minibus"],
    "heavy_goods": ["rigid", "truck"],
    "van": ["van"],
    "taxi": ["taxi"],
    "motorbike": ["motorbike"],
}
OUTPUT_COUNT_COLS = [f"{name}_count" for name in VEHICLE_GROUPS]
OUTPUT_SHARE_COLS = [f"{name}_share" for name in VEHICLE_GROUPS]
SPEED_FILE_RE = re.compile(r"speed_(\d{4}-\d{2}-\d{2})\.csv$")


@dataclass
class PreprocessStats:
    input_file: str
    raw_rows: int = 0
    exact_full_duplicate_rows_removed: int = 0
    non_motorised_rows_removed: int = 0
    invalid_timestamp_rows_removed: int = 0
    invalid_countline_rows_removed: int = 0
    motorised_rows_used: int = 0
    one_minute_vtype_rows: int = 0
    output_rows: int = 0
    unique_timestamps_5min: int = 0
    unique_nodes: int = 0
    null_speed_rows: int = 0
    observed_speed_rows: int = 0
    no_vehicle_rows: int = 0
    raw_missing_rows: int = 0

    @property
    def null_speed_rate(self) -> float:
        if self.output_rows == 0:
            return 0.0
        return self.null_speed_rows / self.output_rows

    @property
    def observed_speed_rate(self) -> float:
        if self.output_rows == 0:
            return 0.0
        return self.observed_speed_rows / self.output_rows

    @property
    def no_vehicle_rate(self) -> float:
        if self.output_rows == 0:
            return 0.0
        return self.no_vehicle_rows / self.output_rows

    @property
    def raw_missing_rate(self) -> float:
        if self.output_rows == 0:
            return 0.0
        return self.raw_missing_rows / self.output_rows


def normalise_direction(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()


def parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_from_speed_filename(path: Path) -> date | None:
    match = SPEED_FILE_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d").date()


def read_and_aggregate_to_vtype_5min(path: Path, chunksize: int) -> tuple[pd.DataFrame, PreprocessStats]:
    stats = PreprocessStats(input_file=path.name)
    usecols = RAW_KEY_COLS + BIN_COLS + RAW_VALUE_COLS
    partials = []

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        stats.raw_rows += len(chunk)

        before = len(chunk)
        chunk = chunk.drop_duplicates()
        stats.exact_full_duplicate_rows_removed += before - len(chunk)

        chunk["vType"] = chunk["vType"].astype(str).str.lower().str.strip()
        before = len(chunk)
        chunk = chunk[chunk["vType"].isin(MOTORISED_TYPES)].copy()
        stats.non_motorised_rows_removed += before - len(chunk)

        chunk["datetime"] = pd.to_datetime(chunk["From"], format="%Y-%m-%d %H:%M", errors="coerce")
        before = len(chunk)
        chunk = chunk[chunk["datetime"].notna()].copy()
        stats.invalid_timestamp_rows_removed += before - len(chunk)

        chunk["Countline"] = pd.to_numeric(chunk["Countline"], errors="coerce")
        before = len(chunk)
        chunk = chunk[chunk["Countline"].notna()].copy()
        stats.invalid_countline_rows_removed += before - len(chunk)
        chunk["Countline"] = chunk["Countline"].astype("int64")

        chunk["Direction"] = normalise_direction(chunk["Direction"])

        for col in BIN_COLS + RAW_VALUE_COLS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

        chunk["vehicle_count"] = chunk[BIN_COLS].sum(axis=1)
        chunk["mean_speed_x_count"] = chunk["Mean"] * chunk["vehicle_count"]
        chunk["p85_speed_x_count"] = chunk["85th Percentile"] * chunk["vehicle_count"]
        chunk["time_5min"] = chunk["datetime"].dt.floor("5min")
        stats.motorised_rows_used += len(chunk)

        # This combines repeated records at the intended model resolution. A full
        # one-minute intermediate is not required because 5-minute aggregation uses
        # the same additive sufficient statistics.
        grouped = (
            chunk.groupby(["time_5min", "Countline", "Direction", "vType"], dropna=False)
            .agg(
                vehicle_count=("vehicle_count", "sum"),
                mean_speed_x_count=("mean_speed_x_count", "sum"),
                p85_speed_x_count=("p85_speed_x_count", "sum"),
            )
            .reset_index()
        )
        partials.append(grouped)

    if not partials:
        return pd.DataFrame(), stats

    combined = pd.concat(partials, ignore_index=True)
    vtype_5min = (
        combined.groupby(["time_5min", "Countline", "Direction", "vType"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    stats.one_minute_vtype_rows = len(vtype_5min)
    return vtype_5min, stats


def build_node_level_features(vtype_5min: pd.DataFrame) -> pd.DataFrame:
    if vtype_5min.empty:
        return pd.DataFrame()

    node_base = (
        vtype_5min.groupby(["time_5min", "Countline", "Direction"], dropna=False)
        .agg(
            motorised_count=("vehicle_count", "sum"),
            mean_speed_x_count=("mean_speed_x_count", "sum"),
            p85_speed_x_count=("p85_speed_x_count", "sum"),
        )
        .reset_index()
    )
    node_base["has_raw_record_5min"] = True
    node_base["motorised_mean_speed"] = node_base["mean_speed_x_count"] / node_base["motorised_count"]
    node_base["motorised_p85_speed"] = node_base["p85_speed_x_count"] / node_base["motorised_count"]
    node_base.loc[node_base["motorised_count"] == 0, ["motorised_mean_speed", "motorised_p85_speed"]] = pd.NA
    node_base = node_base.drop(columns=["mean_speed_x_count", "p85_speed_x_count"])

    count_pivot = (
        vtype_5min.pivot_table(
            index=["time_5min", "Countline", "Direction"],
            columns="vType",
            values="vehicle_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for vtype in MOTORISED_TYPES:
        if vtype not in count_pivot.columns:
            count_pivot[vtype] = 0

    group_features = count_pivot[["time_5min", "Countline", "Direction"]].copy()
    for group_name, members in VEHICLE_GROUPS.items():
        group_features[f"{group_name}_count"] = count_pivot[members].sum(axis=1)

    node = node_base.merge(group_features, on=["time_5min", "Countline", "Direction"], how="left")
    for group_name in VEHICLE_GROUPS:
        count_col = f"{group_name}_count"
        share_col = f"{group_name}_share"
        node[share_col] = node[count_col] / node["motorised_count"]
        node.loc[node["motorised_count"] == 0, share_col] = 0

    node["node_id"] = node["Countline"].astype(str) + "_" + node["Direction"]

    full_times = pd.date_range(node["time_5min"].min(), node["time_5min"].max(), freq="5min")
    nodes = node[["Countline", "Direction", "node_id"]].drop_duplicates()
    full_grid = (
        pd.MultiIndex.from_product([full_times, nodes.index], names=["time_5min", "node_index"])
        .to_frame(index=False)
        .merge(nodes.reset_index().rename(columns={"index": "node_index"}), on="node_index", how="left")
        .drop(columns=["node_index"])
    )
    node = full_grid.merge(node, on=["time_5min", "Countline", "Direction", "node_id"], how="left")

    node["has_raw_record_5min"] = node["has_raw_record_5min"].fillna(False).astype(bool)

    count_cols = ["motorised_count"] + OUTPUT_COUNT_COLS
    node[count_cols] = node[count_cols].fillna(0)
    node[OUTPUT_SHARE_COLS] = node[OUTPUT_SHARE_COLS].fillna(0)

    observed_mask = (
        node["has_raw_record_5min"]
        & (node["motorised_count"] > 0)
        & node["motorised_mean_speed"].notna()
    )
    no_vehicle_mask = node["has_raw_record_5min"] & (node["motorised_count"] == 0)
    raw_missing_mask = ~node["has_raw_record_5min"]

    node["speed_observation_status"] = "raw_missing_or_not_reported"
    node.loc[no_vehicle_mask, "speed_observation_status"] = "no_vehicle_observed"
    node.loc[observed_mask, "speed_observation_status"] = "observed"

    node["speed_observed_mask"] = observed_mask.astype("int8")
    node["no_vehicle_flag"] = no_vehicle_mask.astype("int8")
    node["raw_missing_flag"] = raw_missing_mask.astype("int8")
    node["interpolated_flag"] = 0

    ordered_cols = [
        "time_5min",
        "Countline",
        "Direction",
        "node_id",
        "has_raw_record_5min",
        "speed_observation_status",
        "speed_observed_mask",
        "no_vehicle_flag",
        "raw_missing_flag",
        "interpolated_flag",
        "motorised_mean_speed",
        "motorised_p85_speed",
        "motorised_count",
    ] + OUTPUT_COUNT_COLS + OUTPUT_SHARE_COLS
    return node[ordered_cols].sort_values(["time_5min", "node_id"]).reset_index(drop=True)


def longest_missing_run(values: pd.Series) -> int:
    max_run = 0
    current = 0
    for is_missing in values.astype(bool):
        if is_missing:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def classify_observation_row(
    raw_missing_rate: float,
    max_raw_missing_run: int,
    short_gap_limit: int,
) -> str:
    if raw_missing_rate == 0:
        return "complete"
    if raw_missing_rate >= 0.5:
        return "high_raw_missingness_review_node"
    if max_raw_missing_run > short_gap_limit:
        return "long_raw_gap_keep_missing_or_mask"
    return "short_raw_gap_candidate_for_interpolation"


def build_missing_summary(node_5min: pd.DataFrame, short_gap_limit: int) -> pd.DataFrame:
    if node_5min.empty:
        return pd.DataFrame()

    data = node_5min.copy()
    data["date"] = pd.to_datetime(data["time_5min"]).dt.date
    data["is_null_speed"] = data["motorised_mean_speed"].isna()
    data["is_observed_speed"] = data["speed_observation_status"].eq("observed")
    data["is_no_vehicle"] = data["speed_observation_status"].eq("no_vehicle_observed")
    data["is_raw_missing"] = data["speed_observation_status"].eq("raw_missing_or_not_reported")

    rows = []
    for (node_id, date), group in data.groupby(["node_id", "date"], sort=True):
        group = group.sort_values("time_5min")
        total_intervals = len(group)
        observed_intervals = int(group["is_observed_speed"].sum())
        no_vehicle_intervals = int(group["is_no_vehicle"].sum())
        raw_missing_intervals = int(group["is_raw_missing"].sum())
        null_speed_intervals = int(group["is_null_speed"].sum())
        max_raw_missing_run = longest_missing_run(group["is_raw_missing"])
        raw_missing_rate = raw_missing_intervals / total_intervals if total_intervals else 0.0
        no_vehicle_rate = no_vehicle_intervals / total_intervals if total_intervals else 0.0
        observed_speed_rate = observed_intervals / total_intervals if total_intervals else 0.0
        null_speed_rate = null_speed_intervals / total_intervals if total_intervals else 0.0
        rows.append(
            {
                "node_id": node_id,
                "date": date,
                "total_intervals": total_intervals,
                "observed_speed_intervals": observed_intervals,
                "no_vehicle_intervals": no_vehicle_intervals,
                "raw_missing_intervals": raw_missing_intervals,
                "null_speed_intervals": null_speed_intervals,
                "observed_speed_rate": observed_speed_rate,
                "no_vehicle_rate": no_vehicle_rate,
                "raw_missing_rate": raw_missing_rate,
                "null_speed_rate": null_speed_rate,
                "max_consecutive_raw_missing_intervals": max_raw_missing_run,
                "suggested_handling": classify_observation_row(
                    raw_missing_rate,
                    max_raw_missing_run,
                    short_gap_limit,
                ),
            }
        )
    return pd.DataFrame(rows)


def fill_stats_from_output(stats: PreprocessStats, node_5min: pd.DataFrame) -> PreprocessStats:
    stats.output_rows = len(node_5min)
    if not node_5min.empty:
        stats.unique_timestamps_5min = int(node_5min["time_5min"].nunique())
        stats.unique_nodes = int(node_5min["node_id"].nunique())
        stats.null_speed_rows = int(node_5min["motorised_mean_speed"].isna().sum())
        stats.observed_speed_rows = int(node_5min["speed_observation_status"].eq("observed").sum())
        stats.no_vehicle_rows = int(node_5min["speed_observation_status"].eq("no_vehicle_observed").sum())
        stats.raw_missing_rows = int(node_5min["speed_observation_status"].eq("raw_missing_or_not_reported").sum())
    return stats


def write_summary(output_csv: Path, stats: PreprocessStats, missing_summary: pd.DataFrame) -> Path:
    summary_path = output_csv.with_suffix(".summary.md")
    missing_actions = {}
    if not missing_summary.empty:
        missing_actions = missing_summary["suggested_handling"].value_counts().to_dict()

    lines = [
        f"# Preprocessing Summary: {output_csv.name}",
        "",
        "## Raw Row Handling",
        "",
        f"- Input file: `{stats.input_file}`",
        f"- Raw rows read: {stats.raw_rows:,}",
        f"- Exact full duplicate rows removed: {stats.exact_full_duplicate_rows_removed:,}",
        f"- Non-motorised rows removed: {stats.non_motorised_rows_removed:,}",
        f"- Invalid timestamp rows removed: {stats.invalid_timestamp_rows_removed:,}",
        f"- Invalid countline rows removed: {stats.invalid_countline_rows_removed:,}",
        f"- Motorised rows used for aggregation: {stats.motorised_rows_used:,}",
        "",
        "## 5-Minute Node-Level Output",
        "",
        f"- Output rows: {stats.output_rows:,}",
        f"- Unique 5-minute timestamps: {stats.unique_timestamps_5min:,}",
        f"- Unique countline-direction nodes: {stats.unique_nodes:,}",
        f"- Rows with observed motorised speed: {stats.observed_speed_rows:,}",
        f"- Observed motorised speed proportion: {stats.observed_speed_rate:.4f}",
        f"- Rows with no motorised vehicles observed: {stats.no_vehicle_rows:,}",
        f"- No-vehicle proportion: {stats.no_vehicle_rate:.4f}",
        f"- Rows with raw missing / not reported records: {stats.raw_missing_rows:,}",
        f"- Raw missing / not reported proportion: {stats.raw_missing_rate:.4f}",
        f"- Rows with null mean speed after aggregation: {stats.null_speed_rows:,}",
        f"- Null speed proportion: {stats.null_speed_rate:.4f}",
        "",
        "## Aggregation Rules Applied",
        "",
        "- Retained vehicle types: motorised vehicles only.",
        "- Excluded vehicle types: `pedestrian`, `cyclist`.",
        "- Speed-bin columns were used only to calculate vehicle counts and weights.",
        "- Final output does not retain speed-bin columns.",
        "- Mean speed aggregation: vehicle-count-weighted aggregation of raw `Mean`.",
        "- 85th percentile aggregation: vehicle-count-weighted approximation using raw `85th Percentile`.",
        "- Intervals with raw records but `motorised_count = 0` were labelled `no_vehicle_observed`.",
        "- Full-grid intervals without raw records were labelled `raw_missing_or_not_reported`.",
        "- Speed is kept null for both `no_vehicle_observed` and `raw_missing_or_not_reported`; these categories are separated by flags.",
        "- No interpolation is applied in this script. Short true raw-missing gaps can be interpolated later using the flags.",
        "",
        "## Missing Summary Actions",
        "",
    ]
    if missing_actions:
        for action, count in sorted(missing_actions.items()):
            lines.append(f"- {action}: {count:,} node-day records")
    else:
        lines.append("- No missing summary records produced.")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def preprocess_file(input_csv: Path, output_dir: Path, chunksize: int, short_gap_limit: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_dir = output_dir / "missing_summaries"
    missing_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"{input_csv.stem}_5min_node_level.csv"
    missing_csv = missing_dir / f"{input_csv.stem}_missing_summary.csv"

    vtype_5min, stats = read_and_aggregate_to_vtype_5min(input_csv, chunksize)
    node_5min = build_node_level_features(vtype_5min)
    stats = fill_stats_from_output(stats, node_5min)
    missing_summary = build_missing_summary(node_5min, short_gap_limit)

    node_5min.to_csv(output_csv, index=False)
    missing_summary.to_csv(missing_csv, index=False)
    summary_md = write_summary(output_csv, stats, missing_summary)

    return {
        "input_file": str(input_csv),
        "output_csv": str(output_csv),
        "missing_summary_csv": str(missing_csv),
        "summary_md": str(summary_md),
        "output_rows": str(stats.output_rows),
        "unique_nodes": str(stats.unique_nodes),
        "observed_speed_rate": f"{stats.observed_speed_rate:.4f}",
        "no_vehicle_rate": f"{stats.no_vehicle_rate:.4f}",
        "raw_missing_rate": f"{stats.raw_missing_rate:.4f}",
        "null_speed_rate": f"{stats.null_speed_rate:.4f}",
    }


def resolve_input_files(
    input_path: Path,
    pattern: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[Path]:
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob(pattern))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if start_date is None and end_date is None:
        return files

    filtered = []
    for file_path in files:
        file_date = date_from_speed_filename(file_path)
        if file_date is None:
            continue
        if start_date is not None and file_date < start_date:
            continue
        if end_date is not None and file_date > end_date:
            continue
        filtered.append(file_path)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Vivacity speed CSV files to 5-minute node-level data using v2 rules.")
    parser.add_argument("--input", required=True, help="Raw daily speed CSV file or folder containing CSV files.")
    parser.add_argument("--output-dir", default="data_interim/speed_5min_v2", help="Folder for processed CSV outputs.")
    parser.add_argument("--pattern", default="speed_*.csv", help="Filename pattern when --input is a folder.")
    parser.add_argument("--start-date", default=None, help="Optional inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive end date, YYYY-MM-DD.")
    parser.add_argument("--chunksize", type=int, default=500_000, help="Rows per pandas chunk.")
    parser.add_argument(
        "--short-gap-limit",
        type=int,
        default=2,
        help="Maximum consecutive missing 5-minute intervals considered a short gap.",
    )
    args = parser.parse_args()

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if start_date and end_date and end_date < start_date:
        raise ValueError("--end-date must be on or after --start-date")

    input_files = resolve_input_files(Path(args.input), args.pattern, start_date, end_date)
    if not input_files:
        raise FileNotFoundError(f"No files matched {args.pattern!r} under {args.input!r}")

    output_dir = Path(args.output_dir)
    run_rows = []
    for input_csv in input_files:
        print(f"Preprocessing {input_csv} ...", flush=True)
        run_rows.append(preprocess_file(input_csv, output_dir, args.chunksize, args.short_gap_limit))

    run_summary = pd.DataFrame(run_rows)
    run_summary_path = output_dir / "preprocessing_run_summary.csv"
    run_summary.to_csv(run_summary_path, index=False)
    print(f"Wrote run summary: {run_summary_path}")


if __name__ == "__main__":
    main()
