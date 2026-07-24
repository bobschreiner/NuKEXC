#!/usr/bin/env python3
"""
Plot the adaptive-grid accuracy-per-point study produced by
tests/convergence_adaptive.cxx: the four combinations of the two make_flat_grid
knobs (pruning x per_element) on water, for the unrestricted Hartree-Fock energy.

The CSV starts with '#'-prefixed provenance lines (documenting the reference
energies) followed by the columns:
combo, nrad, nang_order, npts, E_1e, E_2e, E_scf, err_1e, err_2e, err_total.

Three panels plot |error| vs the TOTAL number of grid points (log-log) for the
one-electron energy (kinetic + nuclear attraction), the two-electron energy
(Coulomb + exact exchange) and the total. A curve that sits lower-and-left
reaches a given accuracy with fewer points (better accuracy-per-point);
reporting the components separately exposes how the near-nucleus one-electron
term and the RI two-electron term converge -- and can cancel non-monotonely in
the total.

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

# (error column, axis label) for the three energy components.
COMPONENTS = [
    ("err_1e", r"one-electron  $|E_{1e}-E_{1e}^{\mathrm{ref}}|$  (Ha)"),
    ("err_2e", r"two-electron  $|E_{2e}-E_{2e}^{\mathrm{ref}}|$  (Ha)"),
    ("err_total", r"total  $|E-E^{\mathrm{ref}}|$  (Ha)"),
]


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
                "err_1e": max(float(row["err_1e"]), ERR_FLOOR),
                "err_2e": max(float(row["err_2e"]), ERR_FLOOR),
                "err_total": max(float(row["err_total"]), ERR_FLOOR),
            }
        )
    if not rows:
        raise SystemExit(f"No data rows found in {csv_path}")
    return rows, parse_reference(comment_lines)


def parse_reference(comment_lines):
    text = " ".join(l.lstrip("#").strip() for l in comment_lines)

    def num(key):
        m = re.search(re.escape(key) + r"=(-?\d+\.\d+(?:[eE][-+]?\d+)?)", text)
        return float(m.group(1)) if m else None

    def integer(key):
        m = re.search(re.escape(key) + r"=(\d+)", text)
        return int(m.group(1)) if m else None

    return {
        "E_1e_ref": num("E_1e_ref"),
        "E_2e_ref": num("E_2e_ref"),
        "E_scf_ref": num("E_scf_ref"),
        "nrad_ref": integer("nrad_ref"),
        "nang_ref": integer("nang_order_ref"),
        "npts_ref": integer("npts_ref"),
    }


def reference_caption(meta):
    where = ""
    if meta.get("nrad_ref") and meta.get("nang_ref"):
        where = f"nrad={meta['nrad_ref']}, nang_order={meta['nang_ref']}"
        if meta.get("npts_ref"):
            where += f", {meta['npts_ref']:,} pts"
    parts = []
    if meta.get("E_scf_ref") is not None:
        parts.append(f"E_scf = {meta['E_scf_ref']:.10f} Ha")
    if meta.get("E_1e_ref") is not None:
        parts.append(f"E_1e = {meta['E_1e_ref']:.6f}")
    if meta.get("E_2e_ref") is not None:
        parts.append(f"E_2e = {meta['E_2e_ref']:.6f}")
    return (
        f"error = |E - E_ref| on a fine uniform unpruned reference grid ({where}).   "
        + ";  ".join(parts) + "."
    )


def series(rows, combo, col):
    pts = sorted((r for r in rows if r["combo"] == combo), key=lambda r: r["npts"])
    return np.array([r["npts"] for r in pts]), np.array([r[col] for r in pts])


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

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (col, ylab) in zip(axes, COMPONENTS):
        for combo in combos:
            x, y = series(rows, combo, col)
            ax.loglog(x, y, label=combo, markersize=7, linewidth=1.9, **STYLE[combo])
        ax.set_xlabel("total grid points")
        ax.set_ylabel(ylab)
        ax.grid(True, which="both", ls=":", alpha=0.5)
    axes[0].legend(title="pruning x radial sizing", fontsize=9, loc="lower left")

    fig.suptitle(
        "Adaptive grids on water: unrestricted Hartree-Fock convergence\n"
        "one- and two-electron energy components",
        fontsize=14,
    )
    fig.text(0.5, 0.01, reference_caption(meta), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))

    out = csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
