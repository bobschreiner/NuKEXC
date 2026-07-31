# Convergence studies

Grid-convergence sweeps: each `convergence_*.cxx` varies one quadrature knob,
writes a `convergence_*.csv` next to the binary, and a Python script here turns
that CSV into a figure.

The binaries resolve their inputs relative to the working directory
(`input/zorabasis/QZ4P`, ...), and CMake copies `input/` into the build folder,
so **run them from `build/convergence_studies/`**.

| Study | Binary | Plot script | Figure |
|---|---|---|---|
| Radial schemes on a H atom (Becke / TA-M3 / TA-M4) | `convergence_radial_h` | `plot_radial_h.py` | `convergence_radial_h.pdf` |
| H₂⁺ overlap integral, radial + angular sweeps | `convergence_h2plus` | — (see `make_thesis_plots.py`) | `convergence_h2plus.pdf` |
| Becke vs TA, full radial × angular sweep | `convergence_2d` | `plot_convergence_2d.py` | `convergence_2d.pdf` |
| Pruning schemes on water (Unpruned / Treutler / Robust) | `convergence_pruning` | `plot_pruning.py` | `convergence_pruning_*.pdf` |
| Pruning × per-element radial sizing on water | `convergence_adaptive` | `plot_adaptive.py` | `convergence_adaptive.pdf` |
| Core-Hamiltonian and integration regression sweeps | `convergence_coreH`, `convergence_integration` | — | — |

`convergence_h2plus` and `convergence_coreH` assert on their results and are
registered with `ctest`; the rest are drivers you run by hand.

## Reproducing the figures

Two paths, depending on whether you want figures from a fresh run or the
archived thesis data.

**From a fresh run** — each plot script takes the CSV and writes a PNG beside
it (titles and provenance included, for reading on screen):

```bash
cd build/convergence_studies && ./convergence_radial_h
python ../../convergence_studies/plot_radial_h.py convergence_radial_h.csv
```

`make_thesis_plots.py` does all five at once, emitting title-less vector PDFs
instead (provenance belongs in the LaTeX caption). It skips any study whose CSV
is absent, so you can refresh just the sweeps you re-ran:

```bash
cd build/convergence_studies
NUKEXC_FIG_DIR=../../latex/Visualisations \
    python ../../convergence_studies/make_thesis_plots.py
```

**For the thesis** — `latex/generate_figures.py` is the canonical pass. It
works off the archived CSVs in `latex/Data/` and writes both the figures and
the appendix data tables:

```bash
python latex/generate_figures.py
```

To refresh a study there, copy its `convergence_*.csv` from the build folder
into `latex/Data/` and re-run that script.

## Layout note

`scf_driver.hpp` lives here because only `convergence_adaptive` and
`convergence_pruning` use it. The other shared driver helpers
(`standards.*`, `test_io.*`, `standalone_helpers.hpp`) stay in `tests/` and are
compiled in from there — see `CMakeLists.txt`.
