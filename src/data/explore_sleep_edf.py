#!/usr/bin/env python3
"""Build a reproducible, dependency-free inventory of Sleep-EDF files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


STAGE_RE = re.compile(r"Sleep stage\s+([WR1234M?])", re.IGNORECASE)
MOVEMENT_RE = re.compile(r"Movement time", re.IGNORECASE)


def text_field(value: bytes) -> str:
    return value.decode("ascii", errors="replace").strip().strip("\x00")


def integer_field(value: bytes, default: int = 0) -> int:
    try:
        return int(text_field(value) or default)
    except ValueError:
        return default


def float_field(value: bytes, default: float = 0.0) -> float:
    try:
        return float(text_field(value) or default)
    except ValueError:
        return default


def read_edf_header(path: Path) -> dict:
    with path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256:
            raise ValueError("short EDF fixed header")
        number_of_records = integer_field(fixed[236:244], -1)
        record_duration_seconds = float_field(fixed[244:252])
        signal_count = integer_field(fixed[252:256])
        if signal_count <= 0:
            raise ValueError("invalid EDF signal count")
        signal_header = stream.read(signal_count * 256)
        if len(signal_header) != signal_count * 256:
            raise ValueError("short EDF signal header")

    labels = [
        text_field(signal_header[index * 16 : (index + 1) * 16])
        for index in range(signal_count)
    ]
    samples = [
        integer_field(
            signal_header[216 * signal_count + index * 8 : 216 * signal_count + (index + 1) * 8]
        )
        for index in range(signal_count)
    ]
    return {
        "number_of_records": number_of_records,
        "record_duration_seconds": record_duration_seconds,
        "signal_count": signal_count,
        "duration_seconds": (
            number_of_records * record_duration_seconds
            if number_of_records >= 0
            else None
        ),
        "channels": labels,
        "samples_per_record": samples,
        "sample_rates_hz": [
            sample / record_duration_seconds if record_duration_seconds else None
            for sample in samples
        ],
        "header_bytes": 256 + signal_count * 256,
    }


def hypnogram_counts(path: Path, header: dict) -> dict[str, int]:
    annotation_indexes = [
        index
        for index, label in enumerate(header["channels"])
        if "annotation" in label.lower()
    ]
    if not annotation_indexes or header["number_of_records"] < 0:
        return {}

    annotation_index = annotation_indexes[0]
    samples = header["samples_per_record"]
    record_size = 2 * sum(samples)
    annotation_offset = 2 * sum(samples[:annotation_index])
    annotation_bytes = 2 * samples[annotation_index]
    counts: Counter[str] = Counter()
    with path.open("rb") as stream:
        stream.seek(header["header_bytes"])
        for _ in range(header["number_of_records"]):
            record = stream.read(record_size)
            if len(record) != record_size:
                break
            payload = record[annotation_offset : annotation_offset + annotation_bytes]
            text = payload.replace(b"\x00", b"").decode("latin-1", errors="ignore")
            counts.update(match.group(1).upper() for match in STAGE_RE.finditer(text))
            counts.update("M" for _ in MOVEMENT_RE.finditer(text))
    return dict(sorted(counts.items()))


def record_key(path: Path) -> str:
    """Return the subject/night key shared by PSG and hypnogram names."""

    stem = path.name.replace("-PSG.edf", "").replace("-Hypnogram.edf", "")
    return stem[:7]


def inventory(data_root: Path) -> dict:
    psg_files = sorted(data_root.rglob("*-PSG.edf"))
    hypnogram_files = sorted(data_root.rglob("*-Hypnogram.edf"))
    hypnograms_by_key = {record_key(path): path for path in hypnogram_files}
    records: list[dict] = []
    errors: list[dict] = []
    stage_counts: Counter[str] = Counter()

    for psg in psg_files:
        try:
            header = read_edf_header(psg)
            hypnogram = hypnograms_by_key.get(record_key(psg))
            stages = (
                hypnogram_counts(hypnogram, read_edf_header(hypnogram))
                if hypnogram is not None
                else {}
            )
            stage_counts.update(stages)
            records.append(
                {
                    "relative_path": str(psg.relative_to(data_root)).replace("\\", "/"),
                    "paired_hypnogram": (
                        str(hypnogram.relative_to(data_root)).replace("\\", "/")
                        if hypnogram is not None
                        else None
                    ),
                    "size_bytes": psg.stat().st_size,
                    "duration_seconds": header["duration_seconds"],
                    "signal_count": header["signal_count"],
                    "channels": header["channels"],
                    "sample_rates_hz": header["sample_rates_hz"],
                    "stage_counts": stages,
                }
            )
        except Exception as exc:  # keep inventorying other recordings
            errors.append({"relative_path": str(psg.relative_to(data_root)), "error": str(exc)})

    psg_keys = {record_key(path) for path in psg_files}
    unmatched_hypnograms = [
        str(path.relative_to(data_root)).replace("\\", "/")
        for path in hypnogram_files
        if record_key(path) not in psg_keys
    ]
    return {
        "data_root": str(data_root),
        "psg_files": len(psg_files),
        "hypnogram_files": len(hypnogram_files),
        "paired_records": sum(record["paired_hypnogram"] is not None for record in records),
        "unmatched_hypnograms": unmatched_hypnograms,
        "stage_counts": dict(sorted(stage_counts.items())),
        "records_with_errors": errors,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()
    result = inventory(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sleep_edf_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "sleep_edf_records.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "relative_path", "paired_hypnogram", "size_bytes", "duration_seconds",
            "signal_count", "channels", "sample_rates_hz", "stage_counts",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in result["records"]:
            row = dict(record)
            row["channels"] = ";".join(record["channels"])
            row["sample_rates_hz"] = ";".join(
                "" if rate is None else f"{rate:g}" for rate in record["sample_rates_hz"]
            )
            row["stage_counts"] = json.dumps(record["stage_counts"], sort_keys=True)
            writer.writerow(row)
    keys = ("psg_files", "hypnogram_files", "paired_records", "stage_counts", "records_with_errors")
    print(json.dumps({key: result[key] for key in keys}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
