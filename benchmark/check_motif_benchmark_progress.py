#!/usr/bin/env python3
"""Report progress for the per-design motif benchmark outputs.

The benchmark has one output directory per design. A design is counted as
completed when its processing CSV contains the final expected cycle, even if
that cycle contains a failed model status. This measures execution progress;
it is deliberately different from the later scientific success criterion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def inspect_design(path: Path, expected_cycles: int) -> dict:
    csv_path = path / "processing_results.csv"
    if not csv_path.exists():
        return {
            "started": False,
            "completed": False,
            "cycles": 0,
            "max_cycle": -1,
            "statuses": {},
        }

    try:
        frame = pd.read_csv(csv_path)
    except Exception as error:
        return {
            "started": True,
            "completed": False,
            "cycles": 0,
            "max_cycle": -1,
            "statuses": {"CSV_ERROR": str(error)},
        }

    if "cycle" not in frame.columns:
        return {
            "started": True,
            "completed": False,
            "cycles": 0,
            "max_cycle": -1,
            "statuses": {"MISSING_CYCLE_COLUMN": len(frame)},
        }

    cycles = pd.to_numeric(frame["cycle"], errors="coerce").dropna()
    cycles = {int(value) for value in cycles if int(value) >= 0}
    max_cycle = max(cycles, default=-1)
    statuses = {}
    if "HalluDesign_Status" in frame.columns:
        statuses = frame["HalluDesign_Status"].fillna("NaN").value_counts().to_dict()
    return {
        "started": True,
        "completed": max_cycle >= expected_cycles - 1,
        "cycles": len(cycles),
        "max_cycle": max_cycle,
        "statuses": statuses,
    }


def percentage(value: int, total: int) -> float:
    return 100.0 * value / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("benchmark/motif_protenix/outputs"),
        help="Root containing <contig>/design_<N>/ directories.",
    )
    parser.add_argument(
        "--expected-cycles",
        type=int,
        default=11,
        help="Expected CSV cycles, including the final evaluation cycle.",
    )
    args = parser.parse_args()
    if args.expected_cycles < 1:
        parser.error("--expected-cycles must be positive")

    design_dirs = sorted(args.root.glob("*/design_*"))
    if not design_dirs:
        raise SystemExit(f"No design directories found below {args.root}")

    records = []
    for design_dir in design_dirs:
        record = inspect_design(design_dir, args.expected_cycles)
        record["contig"] = design_dir.parent.name
        record["design"] = design_dir.name
        records.append(record)

    total = len(records)
    started = sum(record["started"] for record in records)
    completed = sum(record["completed"] for record in records)
    attempted_cycles = sum(record["cycles"] for record in records)
    expected_total_cycles = total * args.expected_cycles

    print(f"root: {args.root.resolve()}")
    print(f"designs: {total}")
    print(
        f"CSV written: {started}/{total} "
        f"({percentage(started, total):.2f}%)"
    )
    print(
        f"final cycle reached: {completed}/{total} "
        f"({percentage(completed, total):.2f}%)"
    )
    print(
        f"cycle progress: {attempted_cycles}/{expected_total_cycles} "
        f"({percentage(attempted_cycles, expected_total_cycles):.2f}%)"
    )
    print()
    print("contig       CSV       final     cycles       status_counts")
    print("------------ --------- --------- ------------ ------------------------------")

    by_contig = {}
    for record in records:
        by_contig.setdefault(record["contig"], []).append(record)
    for contig in sorted(by_contig):
        group = by_contig[contig]
        group_started = sum(item["started"] for item in group)
        group_completed = sum(item["completed"] for item in group)
        group_cycles = sum(item["cycles"] for item in group)
        status_counts = {}
        for item in group:
            for status, count in item["statuses"].items():
                if status in {"CSV_ERROR", "MISSING_CYCLE_COLUMN"}:
                    continue
                status_counts[status] = status_counts.get(status, 0) + count
        status_text = ", ".join(
            f"{status}={count}" for status, count in sorted(status_counts.items())
        ) or "-"
        print(
            f"{contig:<12} "
            f"{group_started:>3}/100 "
            f"{group_completed:>3}/100 "
            f"{group_cycles:>4}/{len(group) * args.expected_cycles:<5} "
            f"{status_text}"
        )


if __name__ == "__main__":
    main()
