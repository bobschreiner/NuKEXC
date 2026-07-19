#!/usr/bin/env python3
"""
Plot the adaptive-grid accuracy-per-point study produced by
tests/convergence_adaptive.cxx: the four combinations of the two make_flat_grid
knobs (pruning x per_element) on water, for the core-Hamiltonian occupied-orbital
energy sum.

The CSV starts with '#'-prefixed provenance lines (documenting the reference
energy E_ref) followed by the columns:
combo, nrad, nang_order, npts, band_sum, abs_error.

The figure plots |error| vs the TOTAL number of grid points on a log-log axis:
each grid "level" grows nrad and the Lebedev order together, so a curve that
sits lower-and-left reaches a given accuracy with fewer points (better
accuracy-per-point). A curve that flattens above zero has hit a systematic bias
(e.g. angular pruning of the core shells) that refinement does not remove.

What E_ref is
-------------
E_ref is the observable on a fine UNIFORM UNPRUNED grid (nrad_ref, nang_ref),
finer than any swept level. All four combinations are measured against it, so
error = |E - E_ref| is a fair common yardstick (grid self-convergence, not an
analytic value).

Accessibility: curves are distinguished by line style + marker shape (color is
only a redundant cue), so the plot is readable in grayscale or with color-vision
deficiency.

Usage:
    python tests/plot_adaptive.py [convergence_adaptive.csv]
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe; writes PNGs
import matplotlib.pyplot as plt


ERR_FLOOR = 1e-16

# Distinct (style, marker, color) per combination -- color is a redundant cue.
STYLE = {
    "uniform": {"linestyle": "-", "marker": "o", "color": "#000000"},
    "pruned": {"linestyle": "--", "marker": "s", "color": "#E69F00"},
    "per-element": {"linestyle": "-.", "marker": "^", "color": "#56B4E9"},
    "both": {"linestyle": ":", "marker": "D", "color": "#D55E00"},
}
PLOT_ORDER = ["uniform", "pruned", "per-element", "both"]


def load(csv_path):
    comment_lines, data_lines = [], []
    with open(csv_path, newline="") as fh:
        for line in fh:
            (comment_lines if line.lstrip().startswith("#") else data_lines).append(line)

    rows = []
    for row in csv.DictReader(data_lines):
        rows.append(
            {
                "combo": row["combo"],
                "npts": int(row["npts"]),
                "err": max(float(row["abs_error"]), ERR_FLOOR),
            }
        )
    if not rows:
        raise SystemExit(f"No data rows found in {csv_path}")
    return rows, parse_reference(comment_lines)


def parse_reference(comment_lines):
    text = " ".join(l.lstrip("#").strip() for l in comment_lines)

    def num(key):
        m = re.search(key + r"=(-?\d+\.\d+(?:[eE][-+]?\d+)?)", text)
        return float(m.group(1)) if m else None

    def integer(key):
        m = re.search(key + r"=(\d+)", text)
        return int(m.group(1)) if m else None

    return {
        "E_ref": num("E_ref"),
        "nrad_ref": integer("nrad_ref"),
        "nang_ref": integer("nang_order_ref"),
        "npts_ref": integer("npts_ref"),
    }


def reference_caption(meta):
    if meta.get("E_ref") is None:
        return "E_ref provenance not found in CSV header."
    where = ""
    if meta.get("nrad_ref") and meta.get("nang_ref"):
        where = f" (nrad={meta['nrad_ref']}, nang_order={meta['nang_ref']}"
        if meta.get("npts_ref"):
            where += f", {meta['npts_ref']:,} pts"
        where += ")"
    return (
        f"error = |E - E_ref|.   E_ref = {meta['E_ref']:.10f} Ha on a fine "
        f"uniform unpruned grid{where}.\n"
        f"Lower-and-left = better accuracy per point; a flat tail = a residual "
        f"bias refinement cannot remove."
    )


def series(rows, combo):
    pts = sorted((r for r in rows if r["combo"] == combo), key=lambda r: r["npts"])
    return np.array([r["npts"] for r in pts]), np.array([r["err"] for r in pts])


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("convergence_adaptive.csv")
    if not csv_path.exists():
        raise SystemExit(
            f"CSV not found: {csv_path}\n"
            "Run the study first (build/tests/convergence_adaptive), then pass "
            "the path to convergence_adaptive.csv."
        )

    rows, meta = load(csv_path)
    combos = [c for c in PLOT_ORDER if any(r["combo"] == c for r in rows)]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for combo in combos:
        x, y = series(rows, combo)
        ax.loglog(x, y, label=combo, markersize=7, linewidth=1.9, **STYLE[combo])

    ax.set_xlabel("total grid points")
    ax.set_ylabel("|E - E_ref|  (Ha)")
    ax.set_title("Adaptive grids on water: accuracy per point\n"
                 "core-Hamiltonian occupied-orbital energy sum")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(title="pruning x radial sizing", fontsize=9, loc="upper right")

    fig.text(0.5, 0.01, reference_caption(meta), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    print("Reference: " + reference_caption(meta).replace("\n", " "))


if __name__ == "__main__":
    main()
