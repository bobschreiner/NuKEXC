#!/usr/bin/env python3
"""
Plot the W4-11 B3LYP benchmark produced by tests/benchmark_w411.py.

Reads benchmark_w411_reactions.csv and benchmark_w411_species.csv and writes
three title-less vector PDFs into latex/Visualisations/ (all provenance belongs
in the LaTeX caption, matching the convergence figures):

  benchmark_w411_violin.pdf     error distribution of the 140 total atomization
                                energies, one violin per basis set
  benchmark_w411_scaling.pdf    total runtime vs problem size (basis functions
                                and quadrature points), log-log with fitted
                                power laws
  benchmark_w411_breakdown.pdf  where the time goes, per basis set: absolute
                                seconds and share of total

Accessibility: the Okabe-Ito colourblind-safe palette throughout, and in the
scaling panels curves are distinguished by marker shape and line style as well
as colour, so the figures survive grayscale printing.

Usage:
    python tests/plot_w411.py [--csv-prefix benchmark_w411] [--out DIR]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.55,
        "lines.linewidth": 1.9,
        "lines.markersize": 6,
        "legend.fontsize": 10,
        "figure.dpi": 120,
    }
)

# Okabe-Ito colourblind-safe palette.
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}
BASIS_ORDER = ["DZ", "TZP", "TZ2P", "QZ4P"]
BASIS_STYLE = {
    "DZ": {"color": OI["black"], "marker": "o", "linestyle": "-"},
    "TZP": {"color": OI["orange"], "marker": "s", "linestyle": "--"},
    "TZ2P": {"color": OI["sky"], "marker": "^", "linestyle": "-."},
    "QZ4P": {"color": OI["green"], "marker": "D", "linestyle": ":"},
}

# Collapse the driver's fine-grained timing sections into readable categories.
# Order fixes the stacking order (startup first, then the SCF work).
CATEGORIES = [
    ("Grid construction", OI["sky"], ["startup:Build molecular grid"]),
    ("Basis / setup", OI["grey"], [
        "startup:Load basis sets", "startup:Read XYZ",
        "startup:Resolve/init XC functionals", "startup:Nuclear repulsion energy",
        "startup:GWH initial guess", "startup:Copy core matrices to host (arma)",
    ]),
    ("Collocation fill", OI["yellow"], [
        "startup:Fill primary basis collocation",
        "startup:Fill primary basis gradient collocation",
        "startup:Fill auxiliary basis collocation + potential",
    ]),
    ("Core Hamiltonian", OI["purple"], [
        "startup:Core Hamiltonian (T + V_ne + S)",
        "startup:Orthogonalization matrix X = S^-1/2",
    ]),
    ("RI aux. overlap", OI["orange"], [
        "startup:Auxiliary Coulomb overlap (dense GEMM)",
        "startup:Auxiliary Coulomb overlap (sparse)",
        "startup:Auxiliary Coulomb overlap (tiled)",
        "startup:Half-inverse of auxiliary overlap",
    ]),
    ("SCF: Coulomb $J$", OI["blue"], ["fock:Coulomb (J) build"]),
    ("SCF: XC functional", OI["green"], [
        "fock:Exchange functional (Vx)", "fock:Correlation functional (Vc)",
    ]),
    ("SCF: exact exchange $K$", OI["vermillion"], [
        "fock:Exact exchange (hybrid) build", "fock:Exact exchange (K) build",
    ]),
    ("SCF: assembly", OI["black"], [
        "fock:Build density matrices", "fock:Energy assembly",
        "fock:Fock assembly + orthogonalization",
        "fock:Host->device transfer of orbitals",
    ]),
]


def load_csv(path):
    """Read a CSV whose leading '#' lines carry provenance, not data."""
    with open(path) as fh:
        rows = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(rows))


def fnum(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def plot_violin(reactions, out):
    """Error distribution of the 140 TAEs, one violin per basis set."""
    present = [b for b in BASIS_ORDER
               if any(r["basis"] == b for r in reactions)]
    data = [[fnum(r, "error") for r in reactions if r["basis"] == b]
            for b in present]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.axhline(0.0, color=OI["black"], lw=1.2, zorder=1)
    # +/-1 kcal/mol is the conventional "chemical accuracy" band.
    ax.axhspan(-1.0, 1.0, color=OI["green"], alpha=0.13, zorder=0,
               label="chemical accuracy ($\\pm$1 kcal/mol)")

    parts = ax.violinplot(data, showextrema=False, widths=0.78)
    for body, b in zip(parts["bodies"], present):
        body.set_facecolor(BASIS_STYLE[b]["color"])
        body.set_alpha(0.32)
        body.set_edgecolor(BASIS_STYLE[b]["color"])
        body.set_linewidth(1.4)

    rng = np.random.default_rng(0)  # fixed seed: reproducible jitter
    for i, (b, vals) in enumerate(zip(present, data), start=1):
        v = np.asarray(vals)
        jitter = rng.uniform(-0.10, 0.10, size=v.size)
        ax.plot(i + jitter, v, linestyle="none", marker=".", markersize=4.5,
                color=BASIS_STYLE[b]["color"], alpha=0.75, zorder=3)
        med = float(np.median(v))
        ax.plot([i - 0.29, i + 0.29], [med, med], color=BASIS_STYLE[b]["color"],
                lw=2.0, zorder=4)

    # Summary statistics go above the violins, where nothing else competes for
    # space (the distributions all hang below zero).
    for i, (b, vals) in enumerate(zip(present, data), start=1):
        v = np.asarray(vals)
        ax.annotate("MSD %.1f\nMAD %.1f" % (v.mean(), np.abs(v).mean()),
                    xy=(i, 0.985), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=8.5,
                    color=BASIS_STYLE[b]["color"])

    # Name the largest outliers per basis so the reader can see which chemistry
    # is hard, rather than just how wide the distribution is.
    for i, b in enumerate(present, start=1):
        rows = [r for r in reactions if r["basis"] == b]
        for r in sorted(rows, key=lambda r: -abs(fnum(r, "error")))[:2]:
            ax.annotate(r["reaction"], xy=(i + 0.11, fnum(r, "error")),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=BASIS_STYLE[b]["color"], va="center")

    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(present)
    ax.set_xlabel("Slater-type basis set")
    ax.set_ylabel(r"TAE error  $\mathrm{TAE}_{\mathrm{calc}}-"
                  r"\mathrm{TAE}_{\mathrm{ref}}$  (kcal/mol)")
    # Headroom for the MSD/MAD row above the violins.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))
    ax.legend(loc="lower left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out / "benchmark_w411_violin.pdf")
    plt.close(fig)

    print("Violin summary (kcal/mol):")
    print("  %-6s %5s %8s %8s %8s %8s" % ("basis", "n", "MSD", "MAD", "RMSD", "max"))
    for b, vals in zip(present, data):
        v = np.asarray(vals)
        print("  %-6s %5d %8.2f %8.2f %8.2f %8.2f"
              % (b, v.size, v.mean(), np.abs(v).mean(),
                 np.sqrt((v ** 2).mean()), np.abs(v).max()))


def plot_scaling(species, out):
    """Runtime against problem size, as total cost and as per-Fock-build cost.

    Both panels use the number of primary basis functions as the abscissa. The
    quadrature-point count is a poor alternative here: with uniform radial
    sizing it is just natoms * nrad * nang, so it is independent of the basis
    set and takes only as many distinct values as there are molecule sizes.
    The right panel divides out the SCF iteration count, isolating the cost of
    one Fock build (the kernel scaling) from how many iterations each molecule
    happened to need.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    panels = (
        (axes[0], lambda r: fnum(r, "t_total"), "total wall time (s)"),
        (axes[1],
         lambda r: (fnum(r, "t_fock") / fnum(r, "n_fock", 1.0)
                    if fnum(r, "n_fock") > 0 else 0.0),
         "wall time per Fock build (s)"),
    )
    for ax, ykey, ylabel in panels:
        allx, ally = [], []
        for b in BASIS_ORDER:
            rows = [r for r in species if r["basis"] == b]
            if not rows:
                continue
            x = np.array([fnum(r, "nbf") for r in rows])
            y = np.array([ykey(r) for r in rows])
            keep = (x > 0) & (y > 0)
            x, y = x[keep], y[keep]
            if x.size == 0:
                continue
            st = BASIS_STYLE[b]
            label = b
            # Fit a power law t ~ N^p over the points actually measured.
            if x.size > 2 and np.ptp(np.log10(x)) > 0.15:
                p = np.polyfit(np.log10(x), np.log10(y), 1)[0]
                label = "%s  ($p=%.1f$)" % (b, p)
            ax.plot(x, y, linestyle="none", marker=st["marker"],
                    color=st["color"], alpha=0.75, markersize=5, label=label)
            allx.append(x)
            ally.append(y)
        # Reference slopes anchored at the cloud's centre, as a visual guide for
        # how the measured exponents compare with N^2 / N^3 / N^4.
        if allx:
            xs, ys = np.concatenate(allx), np.concatenate(ally)
            gx = np.array([xs.min(), xs.max()])
            x0, y0 = np.median(xs), np.median(ys)
            for expo, style in ((2, (0, (6, 3))), (3, (0, (3, 2))),
                                (4, (0, (1, 2)))):
                ax.plot(gx, y0 * (gx / x0) ** expo, color=OI["grey"],
                        linestyle=style, lw=1.1, zorder=0)
                ax.annotate("$N^%d$" % expo, xy=(gx[-1], y0 * (gx[-1] / x0) ** expo),
                            fontsize=8, color=OI["grey"], ha="right", va="bottom")
            ax.set_ylim(0.5 * ys.min(), 20 * ys.max())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"primary basis functions $N_{\mathrm{bf}}$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out / "benchmark_w411_scaling.pdf")
    plt.close(fig)


def plot_breakdown(species, out):
    """Where the time goes, per basis set: absolute seconds and share."""
    present = [b for b in BASIS_ORDER if any(r["basis"] == b for r in species)]
    # Mean seconds per molecule, so bases with equal species counts compare
    # directly even if a few runs are missing.
    means = {}
    for b in present:
        rows = [r for r in species if r["basis"] == b]
        means[b] = [
            float(np.mean([sum(fnum(r, k) for k in keys) for r in rows]))
            for _, _, keys in CATEGORIES
        ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    xs = np.arange(len(present))
    for ax, normalise in ((axes[0], False), (axes[1], True)):
        bottoms = np.zeros(len(present))
        for ci, (label, colour, _) in enumerate(CATEGORIES):
            vals = np.array([means[b][ci] for b in present])
            if normalise:
                totals = np.array([sum(means[b]) for b in present])
                vals = 100.0 * vals / np.where(totals > 0, totals, 1.0)
            ax.bar(xs, vals, bottom=bottoms, width=0.62, color=colour,
                   edgecolor="white", linewidth=0.6,
                   label=label if not normalise else None)
            bottoms += vals
        ax.set_xticks(xs)
        ax.set_xticklabels(present)
        ax.set_xlabel("Slater-type basis set")
        ax.grid(True, axis="y", linestyle=":", alpha=0.55)
        ax.set_axisbelow(True)
        if not normalise:
            # Absolute totals on top of each stack, and the exact-exchange share
            # inside it -- that one segment dominates and is the whole point.
            for xi, b in zip(xs, present):
                total = sum(means[b])
                ax.annotate("%.1f s" % total, xy=(xi, total),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("mean wall time per molecule (s)")
    axes[0].set_ylim(0, 1.13 * max(sum(means[b]) for b in present))
    axes[1].set_ylabel("share of accounted time (%)")
    axes[1].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    # Legend outside the axes: nine categories will not fit inside a bar chart.
    fig.legend(handles[::-1], labels[::-1], loc="center right",
               framealpha=0.92, fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(out / "benchmark_w411_breakdown.pdf")
    plt.close(fig)

    print("\nTime breakdown (mean seconds per molecule):")
    hdr = "  %-26s" + "%10s" * len(present)
    print(hdr % tuple(["category"] + present))
    for ci, (label, _, _) in enumerate(CATEGORIES):
        plain = label.replace("$", "").replace("\\", "")
        print(("  %-26s" + "%10.3f" * len(present))
              % tuple([plain] + [means[b][ci] for b in present]))
    print(("  %-26s" + "%10.3f" * len(present))
          % tuple(["TOTAL (accounted)"] + [sum(means[b]) for b in present]))


def write_table(reactions, species, path):
    """Emit the summary statistics as a bare tabular for \\input.

    Only the tabular is written -- no surrounding table environment -- so the
    caption and label stay with the prose in main.tex. (The appendix
    convergence fragments in latex/Data/tables/ each carry their own float;
    nesting one of those inside a table environment would be an error.)
    """
    present = [b for b in BASIS_ORDER
               if any(r["basis"] == b for r in reactions)]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\small",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"  \toprule",
        r"  Basis & $N_\mathrm{rx}$ & $\overline{N_\mathrm{bf}}$ & MSD & MAD "
        r"& RMSD & max & $\overline{t}$ \\",
        r"   &  &  & (kcal/mol) & (kcal/mol) & (kcal/mol) & (kcal/mol) "
        r"& (s) \\",
        r"  \midrule",
    ]
    for b in present:
        v = np.array([fnum(r, "error") for r in reactions if r["basis"] == b])
        srows = [r for r in species if r["basis"] == b]
        nbf = np.mean([fnum(r, "nbf") for r in srows]) if srows else 0.0
        tmean = np.mean([fnum(r, "t_total") for r in srows]) if srows else 0.0
        lines.append(
            "  %s & %d & %.0f & %.2f & %.2f & %.2f & %.2f & %.1f \\\\"
            % (b, v.size, nbf, v.mean(), np.abs(v).mean(),
               np.sqrt((v ** 2).mean()), np.abs(v).max(), tmean)
        )
    lines += [r"  \bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def main():
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-prefix", default="benchmark_w411")
    ap.add_argument("--out", default=str(repo / "latex/Visualisations"))
    ap.add_argument("--table", default=str(repo / "latex/Data/tables/w411_summary.tex"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reactions = load_csv("%s_reactions.csv" % args.csv_prefix)
    species = load_csv("%s_species.csv" % args.csv_prefix)
    if not reactions or not species:
        raise SystemExit("Empty input CSVs -- run tests/benchmark_w411.py first")

    plot_violin(reactions, out)
    plot_scaling(species, out)
    plot_breakdown(species, out)
    write_table(reactions, species, Path(args.table))
    print("\nWrote 3 PDFs into %s" % out)
    print("Wrote summary table %s" % args.table)


if __name__ == "__main__":
    main()
