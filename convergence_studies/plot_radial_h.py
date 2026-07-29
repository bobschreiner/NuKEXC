#!/usr/bin/env python3
"""
Plot the radial-scheme convergence study produced by
convergence_studies/convergence_radial_h.cxx: Becke vs Treutler-Ahlrichs M3 vs TA-M4 on a
single hydrogen atom (core-Hamiltonian ground state).

The CSV starts with '#'-prefixed provenance lines (documenting the reference
energy E_ref) followed by the columns: scheme, nrad, npts, gs_energy,
abs_error.

What E_ref is
-------------
E_ref is the mean of the three schemes' ground-state energies on the finest
grid (nrad_ref) -- a shared self-reference isolating radial-grid error. For the
H atom in the QZ4P basis this coincides with the exact 1s energy (-0.5 Ha) to
~1e-13 Ha, so here error = |E - E_ref| is effectively the true grid error.

The figure (saved next to the CSV) overlays |error| vs nrad for the three
schemes on a log-log axis, with faint slope guides for the two TA mappings
(~nrad^-4 for M3, ~nrad^-6 for M4) whose convergence order is theoretically
clean. Becke gets no guide -- asymptotically it also behaves ~nrad^-4 but has
an erratic pre-asymptotic region at coarse nrad.

Accessibility: the curves are distinguished by line style + marker shape + a
direct end-of-line label, so the plot is readable in grayscale or with
color-vision deficiency.

Usage:
    python convergence_studies/plot_radial_h.py [convergence_radial_h.csv]
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
    "Becke": {"linestyle": "-", "marker": "o", "color": "#000000"},
    "TA-M3": {"linestyle": "--", "marker": "s", "color": "#E69F00"},
    "TA-M4": {"linestyle": "-.", "marker": "^", "color": "#56B4E9"},
}
PLOT_ORDER = ["Becke", "TA-M3", "TA-M4"]
# Slope guides only where the convergence order is theoretically clean.
GUIDE_SLOPE = {"TA-M3": -4, "TA-M4": -6}


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
                "nrad": int(row["nrad"]),
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

    return {"E_ref": num("E_ref"), "spread": num("spread"), "exact": num("exact 1s")}


def reference_caption(meta):
    if meta.get("E_ref") is None:
        return "E_ref provenance not found in CSV header."
    cap = (
        f"error = |E - E_ref|.   E_ref = {meta['E_ref']:.12f} Ha "
        f"= mean of Becke, TA-M3 & TA-M4 on the finest grid (shared self-reference)."
    )
    if meta.get("exact") is not None:
        cap += (
            f"\nFor H in QZ4P this matches the exact 1s energy "
            f"({meta['exact']:.1f} Ha), so error is effectively the true grid error."
        )
    return cap


def series(rows, scheme):
    pts = sorted((r for r in rows if r["scheme"] == scheme), key=lambda r: r["nrad"])
    return np.array([r["nrad"] for r in pts]), np.array([r["err"] for r in pts])


def add_slope_guide(ax, x, y, slope, anchor_idx=1):
    """Faint reference line of the given log-log slope through one data point."""
    x0, y0 = x[anchor_idx], y[anchor_idx]
    xs = np.array([x[0], x[-1]], dtype=float)
    ys = y0 * (xs / x0) ** slope
    ax.plot(xs, ys, ls=":", color="0.6", lw=1.2, zorder=1)
    ax.annotate(
        f"~nrad$^{{{slope}}}$",
        xy=(xs[-1], ys[-1]),
        xytext=(xs[-1] * 0.55, ys[-1] * 3),
        color="0.4",
        fontsize=8,
    )


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("convergence_radial_h.csv")
    if not csv_path.exists():
        raise SystemExit(
            f"CSV not found: {csv_path}\n"
            "Run the study first (build/convergence_studies/convergence_radial_h), then pass "
            "the path to convergence_radial_h.csv."
        )

    rows, meta = load(csv_path)
    schemes = [s for s in PLOT_ORDER if any(r["scheme"] == s for r in rows)]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for scheme in schemes:
        x, y = series(rows, scheme)
        ax.loglog(x, y, label=scheme, markersize=7, linewidth=1.9, **STYLE[scheme])
        ax.annotate(
            scheme, xy=(x[-1], y[-1]), xytext=(x[-1] * 1.1, y[-1]),
            va="center", fontsize=9, annotation_clip=False,
        )
        if scheme in GUIDE_SLOPE:
            add_slope_guide(ax, x, y, GUIDE_SLOPE[scheme])

    ax.set_xlabel("radial points  (nrad)")
    ax.set_ylabel("|E - E_ref|  (Ha)")
    ax.set_title("Radial-scheme convergence: Becke vs TA-M3 vs TA-M4\n"
                 "single H atom, core-Hamiltonian ground state")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(title="radial scheme", fontsize=9, loc="upper right")
    ax.margins(x=0.15)

    fig.text(0.5, 0.01, reference_caption(meta), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    print("Reference: " + reference_caption(meta).replace("\n", " "))


if __name__ == "__main__":
    main()
