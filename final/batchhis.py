#!/usr/bin/env python3
"""
batch_gpu_hists.py  –  make per-GPU latency & bandwidth HISTOGRAMS
                         for every *.log file in a folder.

Example
-------
$ python batch_gpu_hists.py ./logs ./plots --bins 30
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import DefaultDict, List

import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Regex for the new line format
#     GPU2Cycle: 721597    Estimated bandwidth: 0.258825 GB/s
# ──────────────────────────────────────────────────────────────────────────────
LINE_RE = re.compile(
    r"GPU([1-7])Cycle:\s*(\d+)\s+Estimated bandwidth:\s*([\d.]+)"
)

# ──────────────────────────────────────────────────────────────────────────────
#  Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_log(path: Path):
    """
    Return two dicts keyed by GPU id:
        cycles[gpu]    -> List[int]
        bw[gpu]        -> List[float]
    """
    from collections import defaultdict

    cycles: DefaultDict[int, List[int]] = defaultdict(list)
    bw: DefaultDict[int, List[float]] = defaultdict(list)

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = LINE_RE.search(line)
            if m:
                gpu = int(m.group(1))
                cycles[gpu].append(int(m.group(2)))
                bw[gpu].append(float(m.group(3)))

    return cycles, bw


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def _plot_histograms(
    data: dict[int, list],
    title: str,
    ylabel: str,
    outfile: Path,
    bins,
):
    """Overlay histograms (one colour per GPU) and save to *outfile*."""
    if not data:
        print(f"[skip] {outfile.name:30s}  (no matching data)")
        return

    plt.figure(figsize=(8, 4.5))

    # Use common bin edges so GPUs are comparable
    concatenated = np.concatenate(list(map(np.asarray, data.values())))
    bin_edges = np.histogram_bin_edges(concatenated, bins=bins)

    # Cycle colours automatically; alpha gives some transparency for overlaps
    for gpu, values in sorted(data.items()):
        plt.hist(
            values,
            bins=bin_edges,
            alpha=0.6,
            label=f"GPU {gpu}",
            edgecolor="black",
        )

    plt.title(title)
    plt.xlabel(ylabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[ ok ] {outfile.name}")


def plot_file(log_path: Path, out_dir: Path, bins):
    cycles, bw = parse_log(log_path)

    _plot_histograms(
        cycles,
        title=f"{log_path.name} – Latency distribution per GPU",
        ylabel="Cycles",
        outfile=out_dir / (log_path.stem + "_latency_hist.png"),
        bins=bins,
    )

    _plot_histograms(
        bw,
        title=f"{log_path.name} – Bandwidth distribution per GPU",
        ylabel="Bandwidth [GB/s]",
        outfile=out_dir / (log_path.stem + "_bandwidth_hist.png"),
        bins=bins,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────────────────────
def find_logs(folder: Path, recursive: bool = False):
    yield from (folder.rglob("*.log") if recursive else folder.glob("*.log"))


def main():
    ap = argparse.ArgumentParser(
        description="Create per-GPU latency & bandwidth histograms for *.log files"
    )
    ap.add_argument("src", type=Path, help="Folder containing logs")
    ap.add_argument("dst", type=Path, help="Folder to save PNGs")
    ap.add_argument("--bins", default="auto", help="Bin rule/number (default: auto)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into sub-dirs")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    logs = list(find_logs(src, args.recursive))
    if not logs:
        raise SystemExit(f"No .log files found in {src}")

    bins = int(args.bins) if str(args.bins).isdigit() else args.bins

    for log in logs:
        plot_file(log, dst, bins)

    print(f"Done – histograms written to {dst}")


if __name__ == "__main__":
    main()
