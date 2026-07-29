#!/usr/bin/env python3
"""Regenerate every convergence figure and appendix data table of the thesis.

Reads the raw study data from   latex/Data/*.csv       (written by the C++
convergence tests in convergence_studies/), and writes

  * title-free vector figures  ->  latex/Visualisations/*.pdf
  * booktabs LaTeX data tables ->  latex/Data/tables/*.tex   (input by the
                                    appendix of main.tex)

Usage:  python3 generate_figures.py          (from anywhere; paths are derived
                                              from this file's location)

To refresh a study, re-run its test (convergence_studies/convergence_*.cxx), copy the
emitted convergence_*.csv into latex/Data/ and re-run this script. The two
water CSVs currently in Data/ were transcribed from the hi-res GPU run logs
and therefore carry only the per-term |errors|; a re-run of the tests
replaces them with full-precision files including the raw energies (this
script accepts both formats).
"""

import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "Data"
FIGS = HERE / "Visualisations"
TABLES = DATA / "tables"
FIGS.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

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


# ── shared helpers ───────────────────────────────────────────────────────────
def load(path):
    """Return (rows, header_comment_text) of a '#'-commented CSV."""
    comments, data_lines = [], []
    with open(path) as fh:
        for line in fh:
            (comments if line.lstrip().startswith("#") else data_lines).append(line)
    return list(csv.DictReader(data_lines)), " ".join(
        l.lstrip("#").strip() for l in comments
    )


def style(i):
    return dict(marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)])


def err(row, key):
    return max(float(row[key]), ERR_FLOOR)


def series(rows, group_key, group, x_key, y_key):
    pts = sorted(
        (r for r in rows if r[group_key] == group), key=lambda r: float(r[x_key])
    )
    return [float(r[x_key]) for r in pts], [err(r, y_key) for r in pts]


def save_fig(fig, name):
    p = FIGS / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("figure ->", p)


def write_table(name, content):
    p = TABLES / f"{name}.tex"
    p.write_text(content)
    print("table  ->", p)


def sci(x):
    """1.658e-03 -> $1.66\\times10^{-3}$ (3 significant figures)."""
    x = float(x)
    if x == 0:
        return "$0$"
    e = math.floor(math.log10(abs(x)))
    m = x / 10**e
    return f"${m:.2f}\\times10^{{{e}}}$"


def meta(text, key, integer=False):
    m = re.search(re.escape(key) + r"=(-?[\d.]+(?:[eE][-+]?\d+)?)", text)
    if not m:
        return None
    return int(m.group(1)) if integer else float(m.group(1))


# ── 1. radial_h : single H atom, three radial schemes ────────────────────────
def radial_h():
    rows, _ = load(DATA / "convergence_radial_h.csv")
    order = ["Becke", "TA-M3", "TA-M4"]

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for i, s in enumerate(order):
        xs, ys = series(rows, "scheme", s, "nrad", "abs_error")
        ax.loglog(xs, ys, label=s, **style(i))
    ax.set_xlabel(r"radial points $n_\mathrm{rad}$")
    ax.set_ylabel(r"$|E - E_\mathrm{ref}|$  (Ha)")
    ax.grid(True, which="both")
    ax.legend(title="radial scheme")
    save_fig(fig, "convergence_radial_h.pdf")

    # table: rows nrad/npts, one error column per scheme
    nrads = sorted({int(r["nrad"]) for r in rows})
    lines = [
        "\\begin{table}[H]",
        "  \\centering\\small",
        "  \\begin{tabular}{@{}rr" + "l" * len(order) + "@{}}",
        "    \\toprule",
        "    $n_\\mathrm{rad}$ & $N_\\mathrm{quad}$ & "
        + " & ".join(f"$|E-E_\\mathrm{{ref}}|$ {s}" for s in order)
        + " \\\\",
        "    \\midrule",
    ]
    for n in nrads:
        by = {r["scheme"]: r for r in rows if int(r["nrad"]) == n}
        npts = by[order[0]]["npts"]
        lines.append(
            f"    {n} & {int(npts):,} & "
            + " & ".join(sci(by[s]["abs_error"]) for s in order)
            + " \\\\"
        )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption{Radial-grid convergence data of the hydrogen-atom study "
        "(Figure~\\ref{fig:conv_radial_h}). Absolute error of the "
        "core-Hamiltonian lowest-MO energy per radial scheme.}",
        "  \\label{tab:data_radial_h}",
        "\\end{table}",
    ]
    write_table("radial_h", "\n".join(lines) + "\n")


# ── 2. h2plus : overlap S_AB vs the exact analytic value ─────────────────────
def h2plus():
    rows, _ = load(DATA / "convergence_h2plus.csv")
    rad = [r for r in rows if r["sweep"] == "radial"]
    ang = [r for r in rows if r["sweep"] == "angular"]

    fig, (axr, axa) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    pts = sorted(rad, key=lambda r: float(r["param"]))
    axr.loglog(
        [float(r["param"]) for r in pts], [err(r, "abs_error") for r in pts], **style(0)
    )
    axr.set_xlabel(r"radial points $n_\mathrm{rad}$")
    axr.set_ylabel(r"$|S_{AB} - S_\mathrm{exact}|$")
    axr.grid(True, which="both")
    pts = sorted(ang, key=lambda r: float(r["param"]))
    axa.semilogy(
        [float(r["param"]) for r in pts], [err(r, "abs_error") for r in pts], **style(1)
    )
    axa.set_xlabel(r"angular order $L$")
    axa.set_ylabel(r"$|S_{AB} - S_\mathrm{exact}|$")
    axa.grid(True, which="both")
    save_fig(fig, "convergence_h2plus.pdf")

    # table: the two sweeps stacked, separated by a midrule
    lines = [
        "\\begin{table}[H]",
        "  \\centering\\small",
        "  \\begin{tabular}{@{}lrrll@{}}",
        "    \\toprule",
        "    sweep & param.\\ & $N_\\mathrm{quad}$ & $S_{AB}$ & "
        "$|S_{AB}-S_\\mathrm{exact}|$ \\\\",
        "    \\midrule",
    ]
    for block, label in ((rad, "radial ($n_\\mathrm{rad}$)"), (ang, "angular ($L$)")):
        for r in sorted(block, key=lambda r: float(r["param"])):
            lines.append(
                f"    {label} & {r['param']} & {int(r['npts']):,} & "
                f"{float(r['S_AB']):.12f} & " + sci(r["abs_error"]) + " \\\\"
            )
        if label.startswith("radial"):
            lines.append("    \\midrule")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption{Convergence data of the H$_2^+$ overlap-integral study "
        "(Figure~\\ref{fig:conv_h2plus}); "
        "$S_\\mathrm{exact}=\\frac{7}{3}e^{-1}=0.858385362733$.}",
        "  \\label{tab:data_h2plus}",
        "\\end{table}",
    ]
    write_table("h2plus", "\n".join(lines) + "\n")


# ── 3. 2d : Becke vs TA on H2+, full radial x angular sweep ──────────────────
def conv2d():
    rows, _ = load(DATA / "convergence_2d.csv")
    schemes = ["Becke", "M4"]
    nangs = sorted({int(r["nang_order"]) for r in rows})
    nrads = sorted({int(r["nrad"]) for r in rows})
    nang_hi, nrad_hi = max(nangs), max(nrads)
    all_err = [err(r, "abs_error") for r in rows]
    ylim = (
        10 ** math.floor(math.log10(min(all_err))),
        10 ** math.ceil(math.log10(max(all_err))),
    )

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9))
    for i, s in enumerate(schemes):
        pts = sorted(
            (r for r in rows if r["scheme"] == s and int(r["nang_order"]) == nang_hi),
            key=lambda r: int(r["nrad"]),
        )
        ax[0, 0].loglog(
            [int(r["nrad"]) for r in pts],
            [err(r, "abs_error") for r in pts],
            label=s,
            **style(i),
        )
        pts = sorted(
            (r for r in rows if r["scheme"] == s and int(r["nrad"]) == nrad_hi),
            key=lambda r: int(r["nang_order"]),
        )
        ax[0, 1].semilogy(
            [int(r["nang_order"]) for r in pts],
            [err(r, "abs_error") for r in pts],
            label=s,
            **style(i),
        )
    ax[0, 0].set_xlabel(r"radial points $n_\mathrm{rad}$")
    ax[0, 0].set_ylabel(r"$|E - E_\mathrm{ref}|$  (Ha)")
    ax[0, 0].legend(title=f"radial scheme  (angular order $L={nang_hi}$)")
    ax[0, 1].set_xlabel(r"angular order $L$")
    ax[0, 1].set_ylabel(r"$|E - E_\mathrm{ref}|$  (Ha)")
    ax[0, 1].legend(title=f"radial scheme  ($n_\\mathrm{{rad}}={nrad_hi}$)")
    for a in (ax[0, 0], ax[0, 1]):
        a.grid(True, which="both")
    for col, s in enumerate(schemes):
        axp = ax[1, col]
        for j, nang in enumerate(nangs):
            pts = sorted(
                (r for r in rows if r["scheme"] == s and int(r["nang_order"]) == nang),
                key=lambda r: int(r["nrad"]),
            )
            if not pts:
                continue
            axp.loglog(
                [int(r["nrad"]) for r in pts],
                [err(r, "abs_error") for r in pts],
                label=f"{nang}",
                **style(j),
            )
        axp.set_xlabel(r"radial points $n_\mathrm{rad}$")
        axp.set_ylabel(r"$|E - E_\mathrm{ref}|$  (Ha)  [" + s + "]")
        axp.grid(True, which="both")
        axp.set_ylim(ylim)
        axp.legend(title="angular order $L$", fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    save_fig(fig, "convergence_2d.pdf")

    # tables: one error matrix (nrad x nang) per scheme
    for s in schemes:
        lines = [
            "\\begin{table}[H]",
            "  \\centering\\small",
            "  \\begin{tabular}{@{}r" + "l" * len(nangs) + "@{}}",
            "    \\toprule",
            "    $n_\\mathrm{rad}$ \\textbackslash\\ $L$ & "
            + " & ".join(str(n) for n in nangs)
            + " \\\\",
            "    \\midrule",
        ]
        for nrad in nrads:
            cells = []
            for nang in nangs:
                hit = [
                    r
                    for r in rows
                    if r["scheme"] == s
                    and int(r["nrad"]) == nrad
                    and int(r["nang_order"]) == nang
                ]
                cells.append(sci(hit[0]["abs_error"]) if hit else "--")
            lines.append(f"    {nrad} & " + " & ".join(cells) + " \\\\")
        lines += [
            "    \\bottomrule",
            "  \\end{tabular}",
            f"  \\caption{{H$_2^+$ two-dimensional grid-convergence data, "
            f"{s} radial scheme (Figure~\\ref{{fig:conv_2d}}): "
            "$|E-E_\\mathrm{ref}|$ in Ha for every combination of "
            "$n_\\mathrm{rad}$ and angular order $L$.}",
            f"  \\label{{tab:data_2d_{s.lower().replace('-', '_')}}}",
            "\\end{table}",
        ]
        write_table(f"2d_{s.lower().replace('-', '_')}", "\n".join(lines) + "\n")


# ── 4./5. water SCF studies : total energy and per-term decomposition ────────
# The total is shown on its own (it is the headline quantity, and its
# non-monotonicity is an error-cancellation artefact of the four terms below);
# the individual terms get a separate 2x2 matrix.
TOTAL_PANEL = ("err_total", r"total  $|E-E^\mathrm{ref}|$  (Ha)")
TERM_PANELS = [
    ("err_kin", r"kinetic  $|E_\mathrm{kin}-E_\mathrm{kin}^\mathrm{ref}|$  (Ha)"),
    ("err_ne", r"nuclear attr.  $|E_\mathrm{ne}-E_\mathrm{ne}^\mathrm{ref}|$  (Ha)"),
    ("err_J", r"Coulomb  $|E_J-E_J^\mathrm{ref}|$  (Ha)"),
    ("err_K", r"exchange  $|E_K-E_K^\mathrm{ref}|$  (Ha)"),
]
TERM_HEADS = [
    "$|\\Delta E_\\mathrm{kin}|$",
    "$|\\Delta E_\\mathrm{ne}|$",
    "$|\\Delta E_J|$",
    "$|\\Delta E_K|$",
    "$|\\Delta E_\\mathrm{tot}|$",
]
TERM_COLS = ["err_kin", "err_ne", "err_J", "err_K", "err_total"]


def water_total_figure(rows, group_key, order, out_name, legend_title):
    """Single panel: error of the total SCF energy vs total grid points."""
    col, ylab = TOTAL_PANEL
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    for i, g in enumerate(order):
        xs, ys = series(rows, group_key, g, "npts", col)
        if xs:
            ax.loglog(xs, ys, label=g, **style(i))
    ax.set_xlabel("total grid points")
    ax.set_ylabel(ylab)
    ax.grid(True, which="both")
    ax.legend(title=legend_title)
    save_fig(fig, out_name)


def water_terms_figure(rows, group_key, order, out_name, legend_title):
    """2x2 matrix: kinetic / nuclear attraction ; Coulomb / exchange."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    for ax, (col, ylab) in zip(axes.flatten(), TERM_PANELS):
        for i, g in enumerate(order):
            xs, ys = series(rows, group_key, g, "npts", col)
            if xs:
                ax.loglog(xs, ys, label=g, **style(i))
        ax.set_xlabel("total grid points")
        ax.set_ylabel(ylab)
        ax.grid(True, which="both")
    axes[0, 0].legend(title=legend_title)
    fig.tight_layout()
    save_fig(fig, out_name)


def water_table(rows, header_text, group_key, order, name, study, figrefs):
    ref = {
        k: meta(header_text, k)
        for k in ("E_kin_ref", "E_ne_ref", "E_J_ref", "E_K_ref", "E_scf_ref")
    }
    nrad_ref = meta(header_text, "nrad_ref", integer=True)
    nang_ref = meta(header_text, "nang_order_ref", integer=True)
    npts_ref = meta(header_text, "npts_ref", integer=True)

    lines = [
        "\\begin{table}[H]",
        "  \\centering\\small",
        "  \\begin{tabular}{@{}rrr" + "l" * 5 + "@{}}",
        "    \\toprule",
        "    $n_\\mathrm{rad}$ & $L$ & $N_\\mathrm{quad}$ & "
        + " & ".join(TERM_HEADS)
        + " \\\\",
    ]
    for g in order:
        pts = sorted(
            (r for r in rows if r[group_key] == g), key=lambda r: int(r["npts"])
        )
        if not pts:
            continue
        lines.append("    \\midrule")
        lines.append(f"    \\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{g}}}}} \\\\")
        for r in pts:
            lines.append(
                f"    {r['nrad']} & {r['nang_order']} & {int(r['npts']):,} & "
                + " & ".join(sci(r[c]) for c in TERM_COLS)
                + " \\\\"
            )
    caption = (
        f"Per-term convergence data of the water {study} study "
        f"({figrefs}): absolute errors of the kinetic, "
        "nuclear-attraction, Coulomb, exchange and total unrestricted "
        "Hartree--Fock energies. Reference: uniform unpruned grid with "
        f"$n_\\mathrm{{rad}}={nrad_ref}$, $L={nang_ref}$ "
        f"({npts_ref:,} points), "
        f"$E_\\mathrm{{kin}}={ref['E_kin_ref']:.10f}$, "
        f"$E_\\mathrm{{ne}}={ref['E_ne_ref']:.10f}$, "
        f"$E_J={ref['E_J_ref']:.10f}$, "
        f"$E_K={ref['E_K_ref']:.10f}$, "
        f"$E_\\mathrm{{scf}}={ref['E_scf_ref']:.10f}$~Ha."
    )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{tab:data_{name}}}",
        "\\end{table}",
    ]
    write_table(name, "\n".join(lines) + "\n")


def pruning():
    rows, header = load(DATA / "convergence_pruning.csv")
    order = ["Unpruned", "Treutler", "Robust"]
    water_total_figure(rows, "scheme", order,
                       "convergence_pruning_total.pdf", "pruning scheme")
    water_terms_figure(rows, "scheme", order,
                       "convergence_pruning_terms.pdf", "pruning scheme")
    water_table(rows, header, "scheme", order, "pruning", "angular-pruning",
                "Figures~\\ref{fig:conv_pruning_total} "
                "and~\\ref{fig:conv_pruning_terms}")


if __name__ == "__main__":
    # The adaptive-grid study (Data/convergence_adaptive.csv, produced by
    # convergence_studies/convergence_adaptive.cxx) is deliberately not rendered: it was cut
    # from the thesis. The raw data is kept, the figures and table are not.
    jobs = {
        "convergence_radial_h.csv": radial_h,
        "convergence_h2plus.csv": h2plus,
        "convergence_2d.csv": conv2d,
        "convergence_pruning.csv": pruning,
    }
    for name, fn in jobs.items():
        if (DATA / name).exists():
            fn()
        else:
            print(f"skip {name} (not found in {DATA})")
    print("done")
