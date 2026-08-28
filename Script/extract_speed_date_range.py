#!/usr/bin/env python3
"""Extract selected daily speed CSV files from the Nottingham speed zip archive.

The script extracts only files named like ``speed_YYYY-MM-DD.csv`` that fall
within the requested inclusive date range. It writes a manifest so the extracted
raw-data subset is auditable.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


SPEED_FILE_RE = re.compile(r"(^|/)speed_(\d{4}-\d{2}-\d{2})\.csv$")


@dataclass(frozen=True)
class ZipMember:
    name: str
    file_date: date
    uncompressed_size: int


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def expected_dates(start_date: date, end_date: date) -> set[date]:
    days = set()
    current = start_date
    while current <= end_date:
        days.add(current)
        current += timedelta(days=1)
    return days


def find_matching_members(zip_path: Path, start_date: date, end_date: date) -> list[ZipMember]:
    members = []
    with ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            match = SPEED_FILE_RE.search(info.filename)
            if not match:
                continue
            file_date = parse_date(match.group(2))
            if start_date <= file_date <= end_date:
                members.append(
                    ZipMember(
                        name=info.filename,
                        file_date=file_date,
                        uncompressed_size=info.file_size,
                    )
                )
    return sorted(members, key=lambda item: item.file_date)


def extract_member(archive: ZipFile, member: ZipMember, output_dir: Path, overwrite: bool) -> Path:
    output_path = output_dir / Path(member.name).name
    if output_path.exists() and not overwrite and output_path.stat().st_size == member.uncompressed_size:
        return output_path
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    with archive.open(member.name) as source, tmp_path.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    actual_size = tmp_path.stat().st_size
    if actual_size != member.uncompressed_size:
        tmp_path.unlink(missing_ok=True)
        raise IOError(
            f"Extracted size mismatch for {member.name}: "
            f"expected {member.uncompressed_size}, got {actual_size}"
        )
    tmp_path.replace(output_path)
    return output_path


def write_manifest(
    manifest_path: Path,
    members: list[ZipMember],
    extracted_paths: list[Path],
    start_date: date,
    end_date: date,
) -> None:
    expected = expected_dates(start_date, end_date)
    extracted_dates = {member.file_date for member in members}
    missing_dates = sorted(expected - extracted_dates)

    rows = []
    for member, extracted_path in zip(members, extracted_paths, strict=True):
        rows.append(
            {
                "date": member.file_date.isoformat(),
                "zip_member": member.name,
                "extracted_path": str(extracted_path),
                "uncompressed_size_mb": round(member.uncompressed_size / 1024 / 1024, 2),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)

    summary_path = manifest_path.with_suffix(".summary.md")
    total_size_gb = sum(member.uncompressed_size for member in members) / 1024 / 1024 / 1024
    lines = [
        f"# Speed Raw Extraction Summary: {start_date.isoformat()} to {end_date.isoformat()}",
        "",
        f"- Files extracted or already present: {len(members):,}",
        f"- Expected calendar days: {len(expected):,}",
        f"- Missing dates in zip for requested range: {len(missing_dates):,}",
        f"- Total uncompressed size of selected files: {total_size_gb:.2f} GB",
        f"- Manifest CSV: `{manifest_path}`",
        "",
    ]
    if missing_dates:
        lines.append("## Missing Dates")
        lines.append("")
        lines.extend(f"- {item.isoformat()}" for item in missing_dates)
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a date range of daily speed CSVs from speed_Ntt.zip.")
    parser.add_argument("--zip", required=True, help="Path to speed_Ntt.zip")
    parser.add_argument("--start-date", required=True, help="Inclusive start date, YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Inclusive end date, YYYY-MM-DD")
    parser.add_argument("--output-dir", default="data_raw", help="Directory for extracted daily speed CSV files")
    parser.add_argument("--manifest", default=None, help="Optional manifest CSV path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted CSV files")
    args = parser.parse_args()

    zip_path = Path(args.zip).expanduser()
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if end_date < start_date:
        raise ValueError("--end-date must be on or after --start-date")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / (
        f"speed_extraction_manifest_{start_date.isoformat()}_{end_date.isoformat()}.csv"
    )

    members = find_matching_members(zip_path, start_date, end_date)
    if not members:
        raise FileNotFoundError(f"No speed_YYYY-MM-DD.csv files found for {start_date} to {end_date}")

    extracted_paths = []
    with ZipFile(zip_path) as archive:
        for index, member in enumerate(members, start=1):
            output_path = extract_member(archive, member, output_dir, args.overwrite)
            extracted_paths.append(output_path)
            print(f"[{index}/{len(members)}] {member.name} -> {output_path}", flush=True)

    write_manifest(manifest_path, members, extracted_paths, start_date, end_date)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {manifest_path.with_suffix('.summary.md')}")


if __name__ == "__main__":
    main()
