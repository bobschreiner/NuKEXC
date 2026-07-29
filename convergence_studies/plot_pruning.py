#!/usr/bin/env python3
"""
Plot the angular pruning-scheme comparison produced by
convergence_studies/convergence_pruning.cxx: Unpruned vs Treutler vs Robust on water
(unrestricted Hartree-Fock), with per_element held OFF so only the pruning
differs.

The CSV starts with '#'-prefixed provenance lines (documenting the reference
energies) followed by the columns:
scheme, nrad, nang_order, npts, E_1e, E_2e, E_scf, err_1e, err_2e, err_total.

Three panels plot |error| vs the TOTAL number of grid points (log-log) for the
one-electron energy (kinetic + nuclear attraction), the two-electron energy
(Coulomb + exact exchange) and the total. Reporting the components separately
exposes how the near-nucleus one-electron term and the RI two-electron term
converge -- and can cancel non-monotonely in the total.

Accessibility: curves are distinguished by line style + marker shape (color is
only a redundant cue), so the plot is readable in grayscale or with color-vision
deficiency.

Usage:
    python convergence_studies/plot_pruning.py [convergence_pruning.csv]
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

# (error column, axis label) for the individual energy terms + the total,
# laid out 2x3: [kinetic, nuclear-attraction, total ; Coulomb, exchange, blank].
COMPONENTS = [
    ("err_kin", r"kinetic  $|E_{\mathrm{kin}}-E_{\mathrm{kin}}^{\mathrm{ref}}|$  (Ha)"),
    ("err_ne", r"nuclear attr.  $|E_{\mathrm{ne}}-E_{\mathrm{ne}}^{\mathrm{ref}}|$  (Ha)"),
    ("err_total", r"total  $|E-E^{\mathrm{ref}}|$  (Ha)"),
    ("err_J", r"Coulomb  $|E_J-E_J^{\mathrm{ref}}|$  (Ha)"),
    ("err_K", r"exchange  $|E_K-E_K^{\mathrm{ref}}|$  (Ha)"),
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
                "scheme": row["scheme"],
                "npts": int(row["npts"]),
                "err_kin": max(float(row["err_kin"]), ERR_FLOOR),
                "err_ne": max(float(row["err_ne"]), ERR_FLOOR),
                "err_J": max(float(row["err_J"]), ERR_FLOOR),
                "err_K": max(float(row["err_K"]), ERR_FLOOR),
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
        "E_kin_ref": num("E_kin_ref"),
        "E_ne_ref": num("E_ne_ref"),
        "E_J_ref": num("E_J_ref"),
        "E_K_ref": num("E_K_ref"),
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
    for key, lbl in (("E_kin_ref", "E_kin"), ("E_ne_ref", "E_ne"),
                     ("E_J_ref", "E_J"), ("E_K_ref", "E_K")):
        if meta.get(key) is not None:
            parts.append(f"{lbl} = {meta[key]:.6f}")
    return (
        f"error = |E - E_ref| on a fine uniform unpruned reference grid ({where}).   "
        + ";  ".join(parts) + "."
    )


def series(rows, scheme, col):
    pts = sorted((r for r in rows if r["scheme"] == scheme), key=lambda r: r["npts"])
    return np.array([r["npts"] for r in pts]), np.array([r[col] for r in pts])


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("convergence_pruning.csv")
    if not csv_path.exists():
        raise SystemExit(
            f"CSV not found: {csv_path}\n"
            "Run the study first (build/convergence_studies/convergence_pruning), then pass "
            "the path to convergence_pruning.csv."
        )

    rows, meta = load(csv_path)
    schemes = [s for s in PLOT_ORDER if any(r["scheme"] == s for r in rows)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    flat = list(axes.flatten())
    for ax, (col, ylab) in zip(flat, COMPONENTS):
        for scheme in schemes:
            x, y = series(rows, scheme, col)
            ax.loglog(x, y, label=scheme, markersize=7, linewidth=1.9, **STYLE[scheme])
        ax.set_xlabel("total grid points")
        ax.set_ylabel(ylab)
        ax.grid(True, which="both", ls=":", alpha=0.5)
    flat[-1].axis("off")  # leave the 6th slot blank
    flat[0].legend(title="pruning scheme", fontsize=9, loc="lower left")

    fig.suptitle(
        "Angular pruning schemes on water: unrestricted Hartree-Fock convergence\n"
        "per-term decomposition (kinetic, nuclear attraction, Coulomb, exchange)",
        fontsize=14,
    )
    fig.text(0.5, 0.01, reference_caption(meta), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))

    out = csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
