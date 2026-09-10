#!/usr/bin/env python3
"""Estimate per-design wall-clock time from matched consecutive recycles.

For a scaffold tag present in both recycle_(k-1) and recycle_k, the PDB mtime
difference estimates the time required to advance that scaffold by one cycle.
The file system does not retain creation time for these files, so mtime is used.
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm


LENGTH_RE = re.compile(r"^protein_length_(\d+)_sample_\d+")
RECYCLE_SUFFIX_RE = re.compile(r"_recycle_\d+")


def scaffold_key(path: Path) -> tuple[str, int] | None:
    """Return an invariant scaffold identifier and its requested length."""
    match = LENGTH_RE.match(path.stem)
    if match is None:
        return None
    return RECYCLE_SUFFIX_RE.sub("", path.stem), int(match.group(1))


def pdb_times(directory: Path) -> dict[str, tuple[int, float]]:
    """Map each scaffold tag to (length, PDB modification time)."""
    result = {}
    for path in directory.glob("*.pdb"):
        parsed = scaffold_key(path)
        if parsed is not None:
            key, length = parsed
            result[key] = (length, path.stat().st_mtime)
    return result


def collect_intervals(
    root: Path, n_recycles: int, workers: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle_directories = [root / f"recycle_{cycle}" for cycle in range(1, n_recycles + 1)]

    # Each directory is independent, so metadata reads can be parallelized.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        cycle_times = list(tqdm(
            executor.map(pdb_times, cycle_directories),
            total=len(cycle_directories),
            desc="Reading recycle PDB timestamps",
            unit="cycle",
        ))

    rows = []
    for cycle in range(2, n_recycles + 1):
        previous = cycle_times[cycle - 2]
        current = cycle_times[cycle - 1]
        for key in previous.keys() & current.keys():
            length, previous_time = previous[key]
            current_length, current_time = current[key]
            if length != current_length:
                raise ValueError(f"Length mismatch for {key}")
            delta_seconds = current_time - previous_time
            if delta_seconds > 0:
                rows.append({
                    "tag": key,
                    "length": length,
                    "previous_cycle": cycle - 1,
                    "cycle": cycle,
                    "seconds_per_design": delta_seconds,
                })

    intervals = pd.DataFrame(rows)
    if intervals.empty:
        return intervals, pd.DataFrame()

    summary = (
        intervals.groupby("length", as_index=False)["seconds_per_design"]
        .agg(n_matched_intervals="count", mean_seconds="mean", median_seconds="median",
             std_seconds="std", min_seconds="min", max_seconds="max")
    )
    summary["designs_per_day_mean"] = 86400 / summary["mean_seconds"]
    summary["designs_per_day_median"] = 86400 / summary["median_seconds"]
    return intervals, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", type=Path,
        help="Absolute experiment directory containing recycle_1, recycle_2, ...",
    )
    parser.add_argument("--n-recycles", type=int, default=20)
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of threads used to read PDB metadata (default: 8)",
    )
    parser.add_argument(
        "--output-prefix", type=Path,
        default=Path("/fangminchao/for_develop/HalluDesign/recycle_throughput"),
        help="Absolute output prefix (default: /fangminchao/for_develop/HalluDesign/recycle_throughput)",
    )
    args = parser.parse_args()

    root = args.root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    intervals, summary = collect_intervals(root, args.n_recycles, args.workers)
    interval_path = output_prefix.with_name(output_prefix.name + "_intervals.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_by_length.csv")
    intervals.to_csv(interval_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Intervals: {interval_path}")
    print(f"Summary:   {summary_path}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
