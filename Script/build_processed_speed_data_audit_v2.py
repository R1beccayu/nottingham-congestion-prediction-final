#!/usr/bin/env python3
"""Build audit tables for v2 processed 5-minute speed data.

The v2 preprocessing separates no-vehicle intervals from true raw-missing
intervals. This audit script summarizes those categories at daily and node
levels. It does not modify the processed speed files.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DATE_RE = re.compile(r"speed_(\d{4}-\d{2}-\d{2})_5min_node_level\.csv$")


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False, floatfmt: str = ".4f") -> str:
    """Render a small DataFrame as markdown without pandas' optional tabulate dependency."""
    table = df.reset_index() if index else df.copy()
    if table.empty:
        return "_No rows._"

    formatted = table.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(
                lambda value: "" if pd.isna(value) else f"{value:{floatfmt}}"
            )
        else:
            formatted[col] = formatted[col].map(lambda value: "" if pd.isna(value) else str(value))

    header = "| " + " | ".join(str(col) for col in formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in formatted.to_numpy(dtype=object)
    ]
    return "\n".join([header, separator, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data_interim/speed_5min_v2"),
        help="Folder containing v2 daily speed_*_5min_node_level.csv files.",
    )
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=Path("data_interim/speed_5min_v2/preprocessing_run_summary.csv"),
        help="v2 preprocessing run summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_processed/speed/audit_v2"),
        help="Output folder for v2 audit tables.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("outputs/tables"),
        help="Folder for copies of key human-readable tables.",
    )
    return parser.parse_args()


def date_from_file(path: Path) -> str:
    match = DATE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected processed filename: {path.name}")
    return match.group(1)


def load_files(run_summary_path: Path) -> list[Path]:
    run_summary = pd.read_csv(run_summary_path)
    files = [Path(path) for path in run_summary["output_csv"]]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Processed files listed in run summary are missing: {missing[:5]}")
    return files


def summarize_daily(path: Path) -> dict[str, object]:
    usecols = [
        "time_5min",
        "Countline",
        "Direction",
        "node_id",
        "speed_observation_status",
        "motorised_mean_speed",
        "motorised_p85_speed",
        "motorised_count",
        "car_count",
        "bus_minibus_count",
        "heavy_goods_count",
        "van_count",
        "taxi_count",
        "motorbike_count",
    ]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["time_5min"])
    observed = df["speed_observation_status"].eq("observed")
    no_vehicle = df["speed_observation_status"].eq("no_vehicle_observed")
    raw_missing = df["speed_observation_status"].eq("raw_missing_or_not_reported")
    date = date_from_file(path)
    return {
        "date": date,
        "output_file": str(path),
        "rows": len(df),
        "unique_timestamps": df["time_5min"].nunique(),
        "unique_nodes": df["node_id"].nunique(),
        "unique_countlines": df["Countline"].nunique(),
        "observed_speed_intervals": int(observed.sum()),
        "no_vehicle_intervals": int(no_vehicle.sum()),
        "raw_missing_intervals": int(raw_missing.sum()),
        "null_speed_intervals": int(df["motorised_mean_speed"].isna().sum()),
        "observed_speed_rate": float(observed.mean()),
        "no_vehicle_rate": float(no_vehicle.mean()),
        "raw_missing_rate": float(raw_missing.mean()),
        "null_speed_rate": float(df["motorised_mean_speed"].isna().mean()),
        "total_motorised_count": int(df["motorised_count"].sum()),
        "mean_observed_speed": float(df.loc[observed, "motorised_mean_speed"].mean()),
        "median_observed_speed": float(df.loc[observed, "motorised_mean_speed"].median()),
        "p95_observed_speed": float(df.loc[observed, "motorised_mean_speed"].quantile(0.95)),
        "max_observed_speed": float(df.loc[observed, "motorised_mean_speed"].max()),
        "car_count": int(df["car_count"].sum()),
        "bus_minibus_count": int(df["bus_minibus_count"].sum()),
        "heavy_goods_count": int(df["heavy_goods_count"].sum()),
        "van_count": int(df["van_count"].sum()),
        "taxi_count": int(df["taxi_count"].sum()),
        "motorbike_count": int(df["motorbike_count"].sum()),
    }


def build_node_daily(path: Path) -> pd.DataFrame:
    usecols = [
        "time_5min",
        "Countline",
        "Direction",
        "node_id",
        "speed_observation_status",
        "motorised_mean_speed",
        "motorised_count",
        "car_count",
        "bus_minibus_count",
        "heavy_goods_count",
        "van_count",
        "taxi_count",
        "motorbike_count",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df["date"] = date_from_file(path)
    df["observed_flag"] = df["speed_observation_status"].eq("observed")
    df["no_vehicle_flag"] = df["speed_observation_status"].eq("no_vehicle_observed")
    df["raw_missing_flag"] = df["speed_observation_status"].eq("raw_missing_or_not_reported")
    grouped = (
        df.groupby(["node_id", "Countline", "Direction", "date"], dropna=False)
        .agg(
            intervals=("speed_observation_status", "size"),
            observed_speed_intervals=("observed_flag", "sum"),
            no_vehicle_intervals=("no_vehicle_flag", "sum"),
            raw_missing_intervals=("raw_missing_flag", "sum"),
            total_motorised_count=("motorised_count", "sum"),
            mean_observed_speed=("motorised_mean_speed", "mean"),
            median_observed_speed=("motorised_mean_speed", "median"),
            car_count=("car_count", "sum"),
            bus_minibus_count=("bus_minibus_count", "sum"),
            heavy_goods_count=("heavy_goods_count", "sum"),
            van_count=("van_count", "sum"),
            taxi_count=("taxi_count", "sum"),
            motorbike_count=("motorbike_count", "sum"),
        )
        .reset_index()
    )
    return grouped


def build_node_coverage(node_daily: pd.DataFrame, expected_days: int) -> pd.DataFrame:
    coverage = (
        node_daily.groupby(["node_id", "Countline", "Direction"], dropna=False)
        .agg(
            days_present=("date", "nunique"),
            total_intervals=("intervals", "sum"),
            observed_speed_intervals=("observed_speed_intervals", "sum"),
            no_vehicle_intervals=("no_vehicle_intervals", "sum"),
            raw_missing_intervals=("raw_missing_intervals", "sum"),
            total_motorised_count=("total_motorised_count", "sum"),
            mean_observed_speed=("mean_observed_speed", "mean"),
            median_observed_speed=("median_observed_speed", "median"),
            car_count=("car_count", "sum"),
            bus_minibus_count=("bus_minibus_count", "sum"),
            heavy_goods_count=("heavy_goods_count", "sum"),
            van_count=("van_count", "sum"),
            taxi_count=("taxi_count", "sum"),
            motorbike_count=("motorbike_count", "sum"),
        )
        .reset_index()
    )
    coverage["missing_days"] = expected_days - coverage["days_present"]
    coverage["expected_intervals"] = expected_days * 288
    coverage["absent_day_intervals"] = coverage["missing_days"] * 288
    coverage["observed_speed_rate_present_days"] = (
        coverage["observed_speed_intervals"] / coverage["total_intervals"]
    )
    coverage["no_vehicle_rate_present_days"] = coverage["no_vehicle_intervals"] / coverage["total_intervals"]
    coverage["raw_missing_rate_present_days"] = coverage["raw_missing_intervals"] / coverage["total_intervals"]
    coverage["observed_speed_rate_expected_period"] = (
        coverage["observed_speed_intervals"] / coverage["expected_intervals"]
    )
    coverage["node_presence_rate"] = coverage["days_present"] / expected_days
    vehicle_group_cols = [
        "car_count",
        "bus_minibus_count",
        "heavy_goods_count",
        "van_count",
        "taxi_count",
        "motorbike_count",
    ]
    for col in vehicle_group_cols:
        coverage[col.replace("_count", "_share_total")] = (
            coverage[col] / coverage["total_motorised_count"].replace(0, pd.NA)
        )
    return coverage.sort_values(
        ["days_present", "observed_speed_rate_present_days", "total_motorised_count", "node_id"],
        ascending=[False, False, False, True],
    )


def build_rule_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    expected_all_days = coverage["missing_days"].eq(0)
    rules = [
        ("present_all_days_only", expected_all_days),
        ("present_all_days_observed_ge_40pct", expected_all_days & coverage["observed_speed_rate_present_days"].ge(0.40)),
        ("present_all_days_observed_ge_50pct", expected_all_days & coverage["observed_speed_rate_present_days"].ge(0.50)),
        ("present_all_days_observed_ge_60pct", expected_all_days & coverage["observed_speed_rate_present_days"].ge(0.60)),
        ("present_all_days_count_ge_100000", expected_all_days & coverage["total_motorised_count"].ge(100_000)),
        (
            "present_all_days_observed_ge_50pct_count_ge_100000",
            expected_all_days
            & coverage["observed_speed_rate_present_days"].ge(0.50)
            & coverage["total_motorised_count"].ge(100_000),
        ),
    ]
    rows = []
    for name, mask in rules:
        selected = coverage[mask].copy()
        rows.append(
            {
                "selection_rule": name,
                "selected_nodes": len(selected),
                "selected_countlines": selected["Countline"].nunique(),
                "mean_observed_speed_rate_present_days": selected[
                    "observed_speed_rate_present_days"
                ].mean(),
                "mean_no_vehicle_rate_present_days": selected["no_vehicle_rate_present_days"].mean(),
                "total_motorised_count": selected["total_motorised_count"].sum(),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_summary(
    output_path: Path,
    daily_summary: pd.DataFrame,
    node_coverage: pd.DataFrame,
    rule_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Processed Speed Data Audit v2",
        "",
        "## Study Period",
        "",
        f"- Daily files processed: {len(daily_summary):,}",
        f"- Date range: {daily_summary['date'].min()} to {daily_summary['date'].max()}",
        "- Resolution: 5 minutes, 288 intervals per complete day.",
        "",
        "## Key v2 Finding",
        "",
        (
            "- For nodes present in each daily file, `raw_missing_or_not_reported` intervals were not observed "
            "in the current v2 processed outputs. Null speed values are therefore mainly caused by "
            "`no_vehicle_observed` intervals rather than true within-day raw data gaps."
        ),
        "",
        "## Daily Output Summary",
        "",
        f"- Mean nodes per day: {daily_summary['unique_nodes'].mean():.1f}",
        f"- Node range per day: {daily_summary['unique_nodes'].min():,} to {daily_summary['unique_nodes'].max():,}",
        f"- Mean observed speed rate: {daily_summary['observed_speed_rate'].mean():.4f}",
        f"- Mean no-vehicle rate: {daily_summary['no_vehicle_rate'].mean():.4f}",
        f"- Mean raw-missing rate: {daily_summary['raw_missing_rate'].mean():.4f}",
        "",
        "## Cross-Day Node Coverage",
        "",
        f"- Unique countline-direction nodes across the period: {node_coverage['node_id'].nunique():,}",
        f"- Unique countlines across the period: {node_coverage['Countline'].nunique():,}",
        f"- Nodes present on all days: {int(node_coverage['missing_days'].eq(0).sum()):,}",
        "",
        "## Candidate Selection Rules",
        "",
        dataframe_to_markdown(rule_summary, index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    files = load_files(args.run_summary)
    daily_summary = pd.DataFrame(summarize_daily(path) for path in files)
    node_daily = pd.concat([build_node_daily(path) for path in files], ignore_index=True)
    node_coverage = build_node_coverage(node_daily, expected_days=len(daily_summary))
    rule_summary = build_rule_summary(node_coverage)

    outputs = {
        "v2_daily_summary.csv": daily_summary,
        "v2_node_daily_summary.csv": node_daily,
        "v2_node_coverage_report.csv": node_coverage,
        "v2_node_selection_rule_summary.csv": rule_summary,
    }
    for filename, df in outputs.items():
        df.to_csv(args.output_dir / filename, index=False)
        df.to_csv(args.tables_dir / filename, index=False)

    write_markdown_summary(
        args.output_dir / "processed_speed_data_audit_v2.md",
        daily_summary,
        node_coverage,
        rule_summary,
    )
    print(f"Wrote v2 speed audit outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
