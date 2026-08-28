"""
Build 5-minute dynamic feature inputs for the Nottingham STGCN workflow.

Inputs:
- data_processed/model_inputs/common/timestamps.csv
- hourly weather CSV files in data_external/weather

Outputs:
- data_processed/model_inputs/common/dynamic_features.csv
- data_processed/dynamic_features/dynamic_features_summary.md
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TIMESTAMPS = (
    PROJECT_ROOT / "data_processed" / "model_inputs" / "common" / "timestamps.csv"
)
DEFAULT_WEATHER_DIR = PROJECT_ROOT / "data_external" / "weather"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "model_inputs" / "common"
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "data_processed" / "dynamic_features"


BANK_HOLIDAYS = {
    "2024-01-01",
    "2024-03-29",
    "2024-04-01",
    "2024-05-06",
    "2024-05-27",
    "2024-08-26",
    "2024-12-25",
    "2024-12-26",
    "2025-01-01",
    "2025-04-18",
    "2025-04-21",
    "2025-05-05",
    "2025-05-26",
    "2025-08-25",
    "2025-12-25",
    "2025-12-26",
}


SCHOOL_HOLIDAY_RANGES = [
    # Nottingham City Council School Terms and Holidays Calendar 2024/25.
    # Inset days are not included here. Bank holidays are handled separately
    # through is_public_holiday, although the date ranges may overlap.
    ("2024-08-01", "2024-08-28"),  # Summer holiday continuation
    ("2024-10-21", "2024-11-01"),  # Autumn half-term
    ("2024-12-23", "2025-01-03"),  # Christmas school holiday
    ("2025-02-17", "2025-02-21"),  # Spring half-term
    ("2025-04-07", "2025-04-21"),  # Easter school holiday
    ("2025-05-26", "2025-05-30"),  # Summer half-term
    ("2025-07-21", "2025-07-31"),  # Summer holiday starts
]


CHRISTMAS_PERIOD_RANGES = [
    ("2024-12-23", "2025-01-03"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dynamic weather/calendar features.")
    parser.add_argument("--timestamps", type=Path, default=DEFAULT_TIMESTAMPS)
    parser.add_argument("--weather_dir", type=Path, default=DEFAULT_WEATHER_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary_dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def in_ranges(dates: pd.Series, ranges: list[tuple[str, str]]) -> pd.Series:
    mask = pd.Series(False, index=dates.index)
    for start, end in ranges:
        mask |= (dates >= pd.Timestamp(start).date()) & (dates <= pd.Timestamp(end).date())
    return mask


def load_weather(weather_dir: Path) -> pd.DataFrame:
    weather_files = sorted(weather_dir.glob("*.csv"))
    if not weather_files:
        raise FileNotFoundError(f"No weather CSV files found in {weather_dir}")

    frames = []
    for path in weather_files:
        df = pd.read_csv(path)
        missing = {"datetime", "temp", "humidity", "precip", "windspeed", "visibility", "icon"} - set(
            df.columns
        )
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
        df = df[["datetime", "temp", "humidity", "precip", "windspeed", "visibility", "icon"]].copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        frames.append(df)

    weather = pd.concat(frames, ignore_index=True)
    weather = weather.dropna(subset=["datetime"])
    weather = weather.sort_values("datetime").drop_duplicates("datetime", keep="last")
    weather = weather.set_index("datetime")

    numeric_cols = ["temp", "humidity", "precip", "windspeed", "visibility"]
    for col in numeric_cols:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    # Weather is hourly. Interpolate occasional missing numeric weather values on the
    # hourly index, then carry the latest hourly observation to each 5-minute timestamp.
    weather[numeric_cols] = weather[numeric_cols].interpolate(method="time").ffill().bfill()
    weather["icon"] = weather["icon"].astype(str).str.strip().str.lower().replace({"nan": "unknown"})
    weather["icon"] = weather["icon"].replace("", "unknown").ffill().bfill()
    return weather


def build_features(timestamps_path: Path, weather_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    timestamps = pd.read_csv(timestamps_path)
    if "datetime" not in timestamps.columns:
        raise ValueError(f"{timestamps_path} must contain a 'datetime' column.")

    features = timestamps.copy()
    features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")
    features = features.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    weather = load_weather(weather_dir)
    merged = pd.merge_asof(
        features[["datetime"]],
        weather.reset_index().sort_values("datetime"),
        on="datetime",
        direction="backward",
    )

    # If the first timestamp predates the first weather row, use the nearest following row.
    weather_cols = ["temp", "humidity", "precip", "windspeed", "visibility", "icon"]
    if merged[weather_cols].isna().any(axis=None):
        nearest = pd.merge_asof(
            features[["datetime"]],
            weather.reset_index().sort_values("datetime"),
            on="datetime",
            direction="nearest",
        )
        for col in weather_cols:
            merged[col] = merged[col].combine_first(nearest[col])

    out = features[["datetime"]].copy()
    out["timestamp"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ["temp", "humidity", "precip", "windspeed", "visibility"]:
        out[col] = pd.to_numeric(merged[col], errors="coerce")

    out["icon"] = merged["icon"].fillna("unknown").astype(str).map(safe_name)
    out["is_rain"] = ((out["precip"].fillna(0) > 0) | out["icon"].str.contains("rain")).astype(int)

    icon_dummies = pd.get_dummies(out["icon"], prefix="icon", dtype=int)
    out = pd.concat([out, icon_dummies], axis=1)

    dt = out["datetime"]
    dates = dt.dt.date
    out["hour"] = dt.dt.hour
    out["weekday"] = dt.dt.weekday
    out["month"] = dt.dt.month
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)
    out["is_public_holiday"] = dates.astype(str).isin(BANK_HOLIDAYS).astype(int)
    out["is_school_holiday"] = in_ranges(dates, SCHOOL_HOLIDAY_RANGES).astype(int)
    out["is_christmas_period"] = in_ranges(dates, CHRISTMAS_PERIOD_RANGES).astype(int)

    out["hour_sin"] = np.sin(2 * math.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * math.pi * out["hour"] / 24)
    out["weekday_sin"] = np.sin(2 * math.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * math.pi * out["weekday"] / 7)

    # Keep datetime for checking and timestamp as a simple string key.
    ordered_cols = [
        "timestamp",
        "temp",
        "humidity",
        "precip",
        "windspeed",
        "visibility",
        "is_rain",
        "is_weekend",
        "is_public_holiday",
        "is_school_holiday",
        "is_christmas_period",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "hour",
        "weekday",
        "month",
        "icon",
    ]
    icon_cols = sorted([c for c in out.columns if c.startswith("icon_")])
    out = out[ordered_cols + icon_cols]

    summary = {
        "timestamp_rows": len(out),
        "start": out["timestamp"].iloc[0] if len(out) else None,
        "end": out["timestamp"].iloc[-1] if len(out) else None,
        "weather_files": [p.name for p in sorted(weather_dir.glob("*.csv"))],
        "icons": sorted(out["icon"].dropna().unique().tolist()),
        "numeric_missing_after_processing": int(out[["temp", "humidity", "precip", "windspeed", "visibility"]].isna().sum().sum()),
        "public_holiday_rows": int(out["is_public_holiday"].sum()),
        "school_holiday_rows": int(out["is_school_holiday"].sum()),
        "christmas_period_rows": int(out["is_christmas_period"].sum()),
    }
    return out, summary


def write_summary(path: Path, summary: dict[str, object], feature_cols: list[str]) -> None:
    lines = [
        "# Dynamic Feature Build Summary",
        "",
        "## Inputs",
        "",
        "- Timestamps: `data_processed/model_inputs/common/timestamps.csv`",
        "- Weather: hourly Visual Crossing CSV files from `data_external/weather/`",
        "- Public holidays: GOV.UK England and Wales bank holiday dates for 2024 and 2025",
        "- School holidays: Nottingham City Council School Terms and Holidays Calendar 2024/25",
        "",
        "## Output",
        "",
        "- `data_processed/model_inputs/common/dynamic_features.csv`",
        "- `data_processed/dynamic_features/dynamic_features_summary.md`",
        "",
        "## Processing Rules",
        "",
        "- Weather data are hourly and are aligned to each 5-minute model timestamp using backward as-of matching.",
        "- Occasional missing numeric weather values are filled by time interpolation on the hourly weather series, followed by forward/backward fill.",
        "- `is_rain` is set to 1 when precipitation is greater than 0 or the weather icon contains `rain`.",
        "- Public holiday, school holiday, and Christmas-period indicators are generated from date ranges.",
        "- Hour-of-day and day-of-week are encoded using sine/cosine cyclic features.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['timestamp_rows']}",
        f"- Start: {summary['start']}",
        f"- End: {summary['end']}",
        f"- Weather files: {', '.join(summary['weather_files'])}",
        f"- Weather icons: {', '.join(summary['icons'])}",
        f"- Numeric missing values after processing: {summary['numeric_missing_after_processing']}",
        f"- Public-holiday 5-minute rows: {summary['public_holiday_rows']}",
        f"- School-holiday 5-minute rows: {summary['school_holiday_rows']}",
        f"- Christmas-period 5-minute rows: {summary['christmas_period_rows']}",
        "",
        "## Feature Columns",
        "",
    ]
    lines.extend(f"- `{col}`" for col in feature_cols)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    features, summary = build_features(args.timestamps, args.weather_dir)

    output_csv = args.output_dir / "dynamic_features.csv"
    output_summary = args.summary_dir / "dynamic_features_summary.md"
    features.to_csv(output_csv, index=False)
    write_summary(output_summary, summary, [c for c in features.columns if c != "timestamp"])

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_summary}")
    print(f"Rows: {summary['timestamp_rows']}")
    print(f"Weather icons: {', '.join(summary['icons'])}")


if __name__ == "__main__":
    main()
