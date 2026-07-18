#!/usr/bin/env python3
"""
Plot the radial x angular grid convergence study produced by
tests/convergence_2d.cxx, comparing the Becke and Treutler-Ahlrichs (TA)
radial quadrature schemes for the H2+ core-Hamiltonian ground-state energy.

The CSV starts with '#'-prefixed provenance lines (documenting how the
reference energy E_ref was produced) followed by the columns:
scheme, nrad, nang_order, npts, gs_energy, abs_error.

What E_ref is
-------------
E_ref is NOT an external or analytic value. To compare the two schemes fairly
it is a SINGLE shared reference: the mean of the Becke and TA ground-state
energies on the finest grid in the sweep (nrad_ref, nang_order_ref), both
chosen larger than any swept value. Every abs_error is |gs_energy - E_ref|, so
the study measures grid self-convergence, not agreement with the exact energy.
The exact grid parameters, both finest values, their difference and the mean
are read from the CSV header and echoed onto the figure.

Figure (2 x 2), saved next to the CSV:
  (top-left)  |error| vs nrad at the finest angular order  -> the head-to-head
              radial comparison of the two schemes (angular error negligible).
  (top-right) |error| vs nang_order at the finest nrad     -> confirms the
              angular behaviour is scheme-independent (shared Lebedev grid).
  (bottom)    per-scheme plateau families (|error| vs nrad, one curve per
              nang_order), Becke and TA side by side on a shared y-axis so the
              plateau structure can be compared directly.

Accessibility: curves are distinguished by line style + marker shape + a direct
label at the end of each line (not color alone), so the plots are readable in
grayscale or with color-vision deficiency.

Usage:
    python tests/plot_convergence_2d.py [convergence_2d.csv]
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe; writes PNGs
import matplotlib.pyplot as plt


# A tiny floor so exact-zero / underflowed errors still plot on a log axis.
ERR_FLOOR = 1e-16

# Redundant styling channels: each series gets a distinct (style, marker) pair
# so it stays distinguishable without relying on color. Colors are the
# colorblind-safe Okabe-Ito set, used only as an extra cue.
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P"]
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]


def style_for(i):
    n = len(LINE_STYLES)
    return {
        "linestyle": LINE_STYLES[i % n],
        "marker": MARKERS[i % len(MARKERS)],
        "color": OKABE_ITO[i % len(OKABE_ITO)],
        "markersize": 6,
        "linewidth": 1.8,
    }


def load(csv_path):
    """Return (rows, meta) where meta documents the reference energy."""
    comment_lines = []
    data_lines = []
    with open(csv_path, newline="") as fh:
        for line in fh:
            (comment_lines if line.lstrip().startswith("#") else data_lines).append(
                line
            )

    rows = []
    for row in csv.DictReader(data_lines):
        rows.append(
            {
                "scheme": row["scheme"],
                "nrad": int(row["nrad"]),
                "nang": int(row["nang_order"]),
                "npts": int(row["npts"]),
                "energy": float(row["gs_energy"]),
                "err": max(float(row["abs_error"]), ERR_FLOOR),
            }
        )
    if not rows:
        raise SystemExit(f"No data rows found in {csv_path}")

    return rows, parse_reference(comment_lines)


def _num(text, key):
    m = re.search(key + r"=(-?\d+\.\d+(?:[eE][-+]?\d+)?)", text)
    return float(m.group(1)) if m else None


def _int(text, key):
    m = re.search(key + r"=(\d+)", text)
    return int(m.group(1)) if m else None


def parse_reference(comment_lines):
    """Pull the E_ref provenance out of the '#' header lines."""
    text = " ".join(l.lstrip("#").strip() for l in comment_lines)
    return {
        "nrad_ref": _int(text, "nrad_ref"),
        "nang_ref": _int(text, "nang_order_ref"),
        "E_ref": _num(text, "E_ref"),
        "E_ref_Becke": _num(text, "E_ref_Becke"),
        "E_ref_TA": _num(text, "E_ref_TA"),
        "diff": _num(text, r"\|diff\|"),
    }


def reference_caption(meta):
    if meta.get("E_ref") is None:
        return (
            "E_ref provenance not found in CSV header "
            "(regenerate convergence_2d.csv with the current convergence_2d)."
        )
    where = ""
    if meta.get("nrad_ref") and meta.get("nang_ref"):
        where = f" on the finest grid (nrad={meta['nrad_ref']}, nang_order={meta['nang_ref']})"
    agree = ""
    if meta.get("diff") is not None:
        agree = f"; the two schemes agree there to {meta['diff']:.1e} Ha"
    # Two centered lines so the caption never overflows the figure width.
    return (
        f"error = |E - E_ref|.   E_ref = {meta['E_ref']:.12f} Ha "
        f"= mean of Becke & TA{where}.\n"
        f"Shared self-reference (not an analytic value){agree}."
    )


def usort(values):
    return sorted(set(values))


def label_line_end(ax, x_last, y_last, text, x_is_log):
    """Put a direct label just past the last point of a curve."""
    dx = x_last * 1.08 if x_is_log else x_last + 0.6
    ax.annotate(
        text,
        xy=(x_last, y_last),
        xytext=(dx, y_last),
        va="center",
        fontsize=8,
        annotation_clip=False,
    )


def rows_where(rows, **conds):
    return [r for r in rows if all(r[k] == v for k, v in conds.items())]


def plot_scheme_vs_nrad(rows, schemes, nang, ax):
    """Head-to-head: |error| vs nrad at a fixed angular order, per scheme."""
    for i, scheme in enumerate(schemes):
        pts = sorted(rows_where(rows, scheme=scheme, nang=nang), key=lambda r: r["nrad"])
        xs = [r["nrad"] for r in pts]
        ys = [r["err"] for r in pts]
        ax.loglog(xs, ys, label=scheme, **style_for(i))
        label_line_end(ax, xs[-1], ys[-1], scheme, x_is_log=True)
    ax.set_xlabel("radial points  (nrad)")
    ax.set_ylabel("|E - E_ref|  (Ha)")
    ax.set_title(f"Radial convergence at nang_order={nang}\n(Becke vs TA)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(title="radial scheme", fontsize=8, loc="lower left")
    ax.margins(x=0.18)


def plot_scheme_vs_nang(rows, schemes, nrad, ax):
    """|error| vs angular order at a fixed nrad, per scheme."""
    for i, scheme in enumerate(schemes):
        pts = sorted(rows_where(rows, scheme=scheme, nrad=nrad), key=lambda r: r["nang"])
        xs = [r["nang"] for r in pts]
        ys = [r["err"] for r in pts]
        ax.semilogy(xs, ys, label=scheme, **style_for(i))
        label_line_end(ax, xs[-1], ys[-1], scheme, x_is_log=False)
    ax.set_xlabel("angular order  (nang_order)")
    ax.set_ylabel("|E - E_ref|  (Ha)")
    ax.set_title(f"Angular convergence at nrad={nrad}\n(shared Lebedev grid)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(title="radial scheme", fontsize=8, loc="lower left")
    ax.margins(x=0.18)


def plot_plateau_family(rows, scheme, nangs, ax, ylim):
    """|error| vs nrad, one curve per angular order, for a single scheme."""
    for j, nang in enumerate(nangs):
        pts = sorted(rows_where(rows, scheme=scheme, nang=nang), key=lambda r: r["nrad"])
        xs = [r["nrad"] for r in pts]
        ys = [r["err"] for r in pts]
        ax.loglog(xs, ys, label=f"nang={nang}", **style_for(j))
        label_line_end(ax, xs[-1], ys[-1], f"{nang}", x_is_log=True)
    ax.set_xlabel("radial points  (nrad)")
    ax.set_ylabel("|E - E_ref|  (Ha)")
    ax.set_title(f"{scheme}: radial plateaus\n(floor set by angular order)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_ylim(ylim)
    ax.legend(title="angular order", fontsize=7, loc="lower left", ncol=2)
    ax.margins(x=0.18)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("convergence_2d.csv")
    if not csv_path.exists():
        raise SystemExit(
            f"CSV not found: {csv_path}\n"
            "Run the sweep first (e.g. build/tests/convergence_2d), then pass "
            "the path to the generated convergence_2d.csv."
        )

    rows, meta = load(csv_path)
    schemes = usort(r["scheme"] for r in rows)
    nrads = usort(r["nrad"] for r in rows)
    nangs = usort(r["nang"] for r in rows)
    nang_hi, nrad_hi = max(nangs), max(nrads)

    # Shared y-limits for the two plateau panels so they compare directly.
    errs = [r["err"] for r in rows]
    ylim = (10 ** np.floor(np.log10(min(errs))), 10 ** np.ceil(np.log10(max(errs))))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    plot_scheme_vs_nrad(rows, schemes, nang_hi, axes[0, 0])
    plot_scheme_vs_nang(rows, schemes, nrad_hi, axes[0, 1])
    for ax, scheme in zip(axes[1], schemes):
        plot_plateau_family(rows, scheme, nangs, ax, ylim)

    fig.suptitle(
        "H2+ core-Hamiltonian convergence: Becke vs Treutler-Ahlrichs radial schemes",
        fontsize=14,
    )
    fig.text(0.5, 0.008, reference_caption(meta), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))

    out = csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    print("Reference: " + reference_caption(meta))


if __name__ == "__main__":
    main()
