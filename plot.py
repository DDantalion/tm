#!/usr/bin/env python3
"""
batch_cycle_plots.py ― generate a trimmed-distribution histogram
for every *.log* file in a folder and save the plots elsewhere.

Example
-------
$ python batch_cycle_plots.py ./logs ./plots --trim 5 --bins auto
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


CYCLE_RE = re.compile(r"\bCycle:\s*(\d+)")

# ---------- helpers ---------------------------------------------------------


def find_logs(folder: Path, recursive: bool = False) -> Iterable[Path]:
    """Yield *.log files in *folder* (optionally recurse)."""
    if recursive:
        yield from folder.rglob("*.log")
    else:
        yield from folder.glob("*.log")


def parse_cycles(path: Path) -> np.ndarray:
    """Return all cycle counts found in *path* (may be empty)."""
    values: List[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = CYCLE_RE.search(line)
            if m:
                values.append(int(m.group(1)))
    return np.asarray(values, dtype=np.int64)


def trimmed(values: np.ndarray, pct: float) -> np.ndarray:
    """Return *values* with the lowest/highest *pct* percent removed."""
    if values.size == 0:
        return values
    lo = np.percentile(values, pct)
    hi = np.percentile(values, 100 - pct)
    return values[(values >= lo) & (values <= hi)]


def plot_histogram(
    data: np.ndarray, title: str, outfile: Path, bins="auto"
) -> None:
    """Draw and save a histogram for *data*."""
    plt.figure(figsize=(8, 4.5))
    plt.hist(data, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel("Cycle count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# ---------- main ------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create trimmed cycle-count histograms for each log file."
    )
    p.add_argument("src", type=Path, help="Folder that contains *.log files")
    p.add_argument("dst", type=Path, help="Folder to write PNGs to")
    p.add_argument(
        "--bins",
        default="auto",
        help="Histogram bin rule/number (passed to numpy; default: auto)",
    )
    p.add_argument(
        "--trim",
        type=float,
        default=5.0,
        metavar="PCT",
        help="Trim PCT%% from each tail before plotting (default: 5)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search for logs recursively (src/**/*.log)",
    )
    args = p.parse_args()

    src: Path = args.src.expanduser().resolve()
    dst: Path = args.dst.expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    if not src.is_dir():
        raise SystemExit(f"Source folder {src} does not exist or is not a dir")

    logs = list(find_logs(src, args.recursive))
    if not logs:
        raise SystemExit(f"No *.log files found in {src}")

    # convert --bins to int if user passed a number
    bins = int(args.bins) if str(args.bins).isdigit() else args.bins

    for log in logs:
        data = parse_cycles(log)
        if data.size == 0:
            print(f"[skip] {log.name:30s}  (no Cycle: lines)")
            continue

        data_trimmed = trimmed(data, args.trim)
        out_name = log.with_suffix(".png").name
        out_file = dst / out_name

        plot_histogram(
            data_trimmed,
            f"{log.name}  (trim {args.trim:.0f} % tails)",
            out_file,
            bins=bins,
        )
        print(f"[ ok ] {log.name:30s} → {out_file.name}")

    print(f"All done!  Plots written to {dst}")


if __name__ == "__main__":
    main()
