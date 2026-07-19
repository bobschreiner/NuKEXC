#!/usr/bin/env python3
"""
Plot the angular pruning-scheme comparison produced by
tests/convergence_pruning.cxx: Unpruned vs Treutler vs Robust on water
(core-Hamiltonian occupied-orbital energy sum), with per_element held OFF so
only the pruning differs.

The CSV starts with '#'-prefixed provenance lines (documenting the reference
energy E_ref) followed by the columns:
scheme, nrad, nang_order, npts, band_sum, abs_error.

The figure plots |error| vs the TOTAL number of grid points (log-log). Each grid
"level" grows nrad and the Lebedev order together, so a curve that sits
lower-and-left reaches a given accuracy with fewer points. A curve that flattens
above zero has hit a systematic pruning bias that refinement cannot remove.

The point of the plot:
  * Treutler uses FIXED low/medium angular orders (7, 11) on the inner shells,
    so it plateaus once the base order passes 11.
  * Robust's medium order is base-6, i.e. it refines WITH the base order, so it
    tracks the Unpruned curve while still using fewer points.

Accessibility: curves are distinguished by line style + marker shape (color is
only a redundant cue), so the plot is readable in grayscale or with color-vision
deficiency.

Usage:
    python tests/plot_pruning.py [convergence_pruning.csv]
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

# Distinct (style, marker, color) per scheme -- color is only a redundant cue.
STYLE = {
    "Unpruned": {"linestyle": "-", "marker": "o", "color": "#000000"},
    "Treutler": {"linestyle": "--", "marker": "s", "color": "#E69F00"},
    "Robust": {"linestyle": "-.", "marker": "^", "color": "#56B4E9"},
}
PLOT_ORDER = ["Unpruned", "Treutler", "Robust"]


def load(csv_path):
    comment_lines, data_lines = [], []
    with open(csv_path, newline="") as fh:
        for line in fh:
            (comment_lines if line.lstrip().startswith("#") else data_lines).append(line)

    rows = []
    for row in csv.DictReader(data_lines):
        rows.append(
            {
                "scheme": row["scheme"],
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
        f"pruning bias refinement cannot remove."
    )


def series(rows, scheme):
    pts = sorted((r for r in rows if r["scheme"] == scheme), key=lambda r: r["npts"])
    return np.array([r["npts"] for r in pts]), np.array([r["err"] for r in pts])


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("convergence_pruning.csv")
    if not csv_path.exists():
        raise SystemExit(
            f"CSV not found: {csv_path}\n"
            "Run the study first (build/tests/convergence_pruning), then pass "
            "the path to convergence_pruning.csv."
        )

    rows, meta = load(csv_path)
    schemes = [s for s in PLOT_ORDER if any(r["scheme"] == s for r in rows)]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for scheme in schemes:
        x, y = series(rows, scheme)
        ax.loglog(x, y, label=scheme, markersize=7, linewidth=1.9, **STYLE[scheme])
        ax.annotate(
            scheme, xy=(x[-1], y[-1]), xytext=(x[-1] * 1.06, y[-1]),
            va="center", fontsize=9, annotation_clip=False,
        )

    ax.set_xlabel("total grid points")
    ax.set_ylabel("|E - E_ref|  (Ha)")
    ax.set_title("Angular pruning schemes on water: accuracy per point\n"
                 "core-Hamiltonian occupied-orbital energy sum")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(title="pruning scheme", fontsize=9, loc="lower left")
    ax.margins(x=0.12)

    fig.text(0.5, 0.01, reference_caption(meta), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    print("Reference: " + reference_caption(meta).replace("\n", " "))


if __name__ == "__main__":
    main()
