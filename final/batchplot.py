#!/usr/bin/env python3
"""
batchplot.py  ―  draw per-GPU latency & bandwidth curves for each log file.

Example
-------
$ python batchplot.py ./logs ./plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import DefaultDict, List

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
#  Parsing helpers
# --------------------------------------------------------------------------- #

LINE_RE = re.compile(
    r"GPU([1-7])Cycle:\s*(\d+)\s+Estimated bandwidth:\s*([\d.]+)"
)


def parse_log(path: Path):
    """
    Return two dictionaries:
      cycles[g]     -> list[int]
      bandwidth[g]  -> list[float]
    where g is the integer GPU index (1..7).
    """
    from collections import defaultdict

    cycles: DefaultDict[int, List[int]] = defaultdict(list)
    bandwidth: DefaultDict[int, List[float]] = defaultdict(list)

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = LINE_RE.search(line)
            if m:
                gpu = int(m.group(1))
                cycles[gpu].append(int(m.group(2)))
                bandwidth[gpu].append(float(m.group(3)))

    return cycles, bandwidth


# --------------------------------------------------------------------------- #
#  Plotting helpers
# --------------------------------------------------------------------------- #


def _plot_series(
    data: dict[int, list],
    ylabel: str,
    title: str,
    outfile: Path,
    *,
    x_is_index: bool = True,
):
    """Draw a multi-GPU line plot and save it to *outfile*."""
    if not data:
        print(f"[skip] {outfile.name}  (no matching lines)")
        return

    plt.figure(figsize=(8, 4.5))

    # Matplotlib’s default colour cycle already has enough distinct colours;
    # plotting GPUs in sorted order keeps colours consistent across files.
    for gpu in sorted(data):
        y = np.asarray(data[gpu])
        x = np.arange(len(y)) if x_is_index else y[:, 0]
        plt.plot(x, y, marker="o", linewidth=1.2, label=f"GPU {gpu}")

    plt.title(title)
    plt.xlabel("Sample index" if x_is_index else "x")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[ ok ] {outfile.name}")


def plot_file(log_path: Path, out_dir: Path):
    """Generate latency & bandwidth plots for one log file."""
    cycles, bw = parse_log(log_path)

    # --- latency plot -------------------------------------------------------
    lat_png = out_dir / (log_path.stem + "_latency.png")
    _plot_series(
        cycles,
        ylabel="Cycles",
        title=f"{log_path.name} – Latency per sample",
        outfile=lat_png,
    )

    # --- bandwidth plot -----------------------------------------------------
    bw_png = out_dir / (log_path.stem + "_bandwidth.png")
    _plot_series(
        bw,
        ylabel="Bandwidth [GB/s]",
        title=f"{log_path.name} – Bandwidth per sample",
        outfile=bw_png,
    )


# --------------------------------------------------------------------------- #
#  Main driver
# --------------------------------------------------------------------------- #


def find_logs(folder: Path, recursive: bool = False):
    """Yield *.log files in *folder*."""
    yield from (folder.rglob("*.log") if recursive else folder.glob("*.log"))


def main():
    ap = argparse.ArgumentParser(
        description="Create latency & bandwidth plots for every log in a folder"
    )
    ap.add_argument("src", type=Path, help="Folder containing *.log files")
    ap.add_argument("dst", type=Path, help="Folder to write PNGs to")
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into sub-directories of SRC",
    )
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    logs = list(find_logs(src, args.recursive))
    if not logs:
        raise SystemExit(f"No .log files found in {src}")

    for log in logs:
        plot_file(log, dst)

    print(f"Finished – all plots written to {dst}")


if __name__ == "__main__":
    main()
