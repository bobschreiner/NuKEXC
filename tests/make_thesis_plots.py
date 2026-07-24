#!/usr/bin/env python3
"""Generate title-less convergence figures for the thesis.

Reads the five convergence CSVs in this directory and writes vector PDFs into
latex/Visualisations/. No titles are drawn on any figure (all provenance goes
into the LaTeX caption). Run from the directory holding the CSVs.
"""
import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Read the convergence CSVs from the current working directory (run this script
# from wherever the tests wrote their *.csv), and write PDFs into the thesis
# Visualisations folder. Override either with NUKEXC_CSV_DIR / NUKEXC_FIG_DIR.
HERE = Path(os.environ.get("NUKEXC_CSV_DIR", Path.cwd()))
OUT = Path(os.environ.get("NUKEXC_FIG_DIR",
                          Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)

ERR_FLOOR = 1e-16
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
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load(path):
    """Return list-of-dict data rows (skips '#' comment lines)."""
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def style(i):
    return dict(marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)])


def err(row, key="abs_error"):
    return max(float(row[key]), ERR_FLOOR)


def series(rows, group_key, x_key, order=None, y_key="abs_error"):
    """Yield (label, xs, ys) grouped by group_key, x sorted ascending."""
    groups = {}
    for r in rows:
        groups.setdefault(r[group_key], []).append(r)
    labels = order or sorted(groups)
    for lab in labels:
        if lab not in groups:
            continue
        pts = sorted(groups[lab], key=lambda r: float(r[x_key]))
        xs = [float(r[x_key]) for r in pts]
        ys = [err(r, y_key) for r in pts]
        yield lab, xs, ys


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ── 1. radial_h : single H atom, three radial schemes ────────────────────────
def plot_radial_h():
    rows = load(HERE / "convergence_radial_h.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for i, (lab, xs, ys) in enumerate(
        series(rows, "scheme", "nrad", order=["Becke", "TA-M3", "TA-M4"])
    ):
        ax.loglog(xs, ys, label=lab, **style(i))
    ax.set_xlabel(r"radial points $n_\mathrm{rad}$")
    ax.set_ylabel(r"$|E - E_\mathrm{ref}|$  (Ha)")
    ax.grid(True, which="both")
    ax.legend(title="radial scheme")
    save(fig, "convergence_radial_h.pdf")


# ── 2. h2plus : overlap S_AB vs exact, radial + angular sweeps ───────────────
def plot_h2plus():
    rows = load(HERE / "convergence_h2plus.csv")
    rad = [r for r in rows if r["sweep"] == "radial"]
    ang = [r for r in rows if r["sweep"] == "angular"]
    fig, (axr, axa) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    pts = sorted(rad, key=lambda r: float(r["param"]))
    axr.loglog(
        [float(r["param"]) for r in pts], [err(r) for r in pts], **style(0)
    )
    axr.set_xlabel(r"radial points $n_\mathrm{rad}$")
    axr.set_ylabel(r"$|S_{AB} - S_\mathrm{exact}|$")
    axr.grid(True, which="both")

    pts = sorted(ang, key=lambda r: float(r["param"]))
    axa.semilogy(
        [float(r["param"]) for r in pts], [err(r) for r in pts], **style(1)
    )
    axa.set_xlabel(r"angular order $L$")
    axa.set_ylabel(r"$|S_{AB} - S_\mathrm{exact}|$")
    axa.grid(True, which="both")
    save(fig, "convergence_h2plus.pdf")


# ── 3. 2d : Becke vs TA, radial/angular convergence + plateau coupling ───────
def plot_2d():
    rows = load(HERE / "convergence_2d.csv")
    schemes = ["Becke", "TA"]
    nangs = sorted({int(r["nang_order"]) for r in rows})
    nrads = sorted({int(r["nrad"]) for r in rows})
    nang_hi, nrad_hi = max(nangs), max(nrads)
    all_err = [err(r) for r in rows]
    ylim = (10 ** math.floor(math.log10(min(all_err))),
            10 ** math.ceil(math.log10(max(all_err))))

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9))

    # top-left: radial convergence at the finest angular order
    for i, s in enumerate(schemes):
        pts = sorted(
            [r for r in rows if r["scheme"] == s and int(r["nang_order"]) == nang_hi],
            key=lambda r: int(r["nrad"]),
        )
        ax[0, 0].loglog([int(r["nrad"]) for r in pts], [err(r) for r in pts],
                        label=s, **style(i))
    ax[0, 0].set_xlabel(r"radial points $n_\mathrm{rad}$")
    ax[0, 0].set_ylabel(r"$|E - E_\mathrm{ref}|$  (Ha)")
    ax[0, 0].grid(True, which="both")
    ax[0, 0].legend(title=f"radial scheme  (angular order $L={nang_hi}$)")

    # top-right: angular convergence at the finest radial grid
    for i, s in enumerate(schemes):
        pts = sorted(
            [r for r in rows if r["scheme"] == s and int(r["nrad"]) == nrad_hi],
            key=lambda r: int(r["nang_order"]),
        )
        ax[0, 1].semilogy([int(r["nang_order"]) for r in pts], [err(r) for r in pts],
                          label=s, **style(i))
    ax[0, 1].set_xlabel(r"angular order $L$")
    ax[0, 1].set_ylabel(r"$|E - E_\mathrm{ref}|$  (Ha)")
    ax[0, 1].grid(True, which="both")
    ax[0, 1].legend(title=f"radial scheme  ($n_\\mathrm{{rad}}={nrad_hi}$)")

    # bottom row: per-scheme plateau family (one curve per angular order)
    for col, s in enumerate(schemes):
        axp = ax[1, col]
        for j, nang in enumerate(nangs):
            pts = sorted(
                [r for r in rows if r["scheme"] == s and int(r["nang_order"]) == nang],
                key=lambda r: int(r["nrad"]),
            )
            if not pts:
                continue
            axp.loglog([int(r["nrad"]) for r in pts], [err(r) for r in pts],
                       label=f"{nang}", **style(j))
        axp.set_xlabel(r"radial points $n_\mathrm{rad}$")
        axp.set_ylabel(r"$|E - E_\mathrm{ref}|$  (Ha)  [" + s + "]")
        axp.grid(True, which="both")
        axp.set_ylim(ylim)
        axp.legend(title="angular order $L$", fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    save(fig, "convergence_2d.pdf")


# ── 4. pruning : water, three pruning schemes, accuracy per point ────────────
def plot_pruning():
    rows = load(HERE / "convergence_pruning.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for i, (lab, xs, ys) in enumerate(
        series(rows, "scheme", "npts", order=["Unpruned", "Treutler", "Robust"])
    ):
        ax.loglog(xs, ys, label=lab, **style(i))
    ax.set_xlabel("total grid points")
    ax.set_ylabel(r"$|E - E_\mathrm{ref}|$  (Ha)")
    ax.grid(True, which="both")
    ax.legend(title="pruning scheme")
    save(fig, "convergence_pruning.pdf")


# ── 5. adaptive : water, pruning x radial-sizing, accuracy per point ─────────
def plot_adaptive():
    rows = load(HERE / "convergence_adaptive.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for i, (lab, xs, ys) in enumerate(
        series(rows, "combo", "npts",
               order=["uniform", "pruned", "per-element", "both"])
    ):
        ax.loglog(xs, ys, label=lab, **style(i))
    ax.set_xlabel("total grid points")
    ax.set_ylabel(r"$|E - E_\mathrm{ref}|$  (Ha)")
    ax.grid(True, which="both")
    ax.legend(title=r"pruning $\times$ radial sizing")
    save(fig, "convergence_adaptive.pdf")


if __name__ == "__main__":
    # Each figure regenerates only if its CSV is present in the CSV dir, so you
    # can re-run just the water sweeps (pruning/adaptive) after a GPU run and
    # leave the integration figures (radial_h/h2plus/2d) untouched.
    jobs = {
        "convergence_radial_h.csv": plot_radial_h,
        "convergence_h2plus.csv": plot_h2plus,
        "convergence_2d.csv": plot_2d,
        "convergence_pruning.csv": plot_pruning,
        "convergence_adaptive.csv": plot_adaptive,
    }
    for name, fn in jobs.items():
        if (HERE / name).exists():
            fn()
        else:
            print(f"skip {name} (not found in {HERE})")
    print("done")
