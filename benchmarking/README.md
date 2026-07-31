# Benchmarking

Performance and accuracy benchmarks. Nothing here is registered with `ctest` —
these are drivers you run by hand.

The binaries resolve their inputs relative to the working directory
(`input/zorabasis/TZP`, ...), and CMake copies `input/` into the build folder,
so **run them from `build/benchmarking/`**.

| Binary | What it measures |
|---|---|
| `benchmark_coreH` | Core-Hamiltonian assembly (T + V_ne + S) |
| `benchmark_cutoff` | Effect of the collocation screening cutoff |
| `benchmark_partition` | Becke partitioning weights |

```bash
cd build/benchmarking
./benchmark_coreH
```

End-to-end SCF wall time with a per-section breakdown comes from the
`standalone` driver instead — it lives in `tests/` (see the layout note below):

```bash
cd build/tests
./standalone
```

Profiling with the Kokkos simple kernel timer:

```bash
export KOKKOS_TOOLS_LIBS=/path/to/kokkos-tools/build/profiling/simple-kernel-timer/libkp_kernel_timer.so
./standalone
kp_reader *.dat
```

## The W4-11 sweep

`benchmark_w411.py` runs `build/tests/standalone` over the 140 W4-11 total
atomization energies for a sweep of ADF Slater-type basis sets, parsing both
energies and timings out of its stdout. `plot_w411.py` turns the two CSVs into
the three thesis figures plus the summary table.

Both scripts derive every default path from the repository root, so run them
from anywhere:

```bash
python benchmarking/benchmark_w411.py --jobs 8      # writes benchmark_w411_{species,reactions}.csv
python benchmarking/plot_w411.py                    # -> latex/Visualisations/, latex/Data/tables/
```

The sweep caches one JSON per (basis, species) run under `--cache`, keyed by a
hash of every parameter that affects the physics, so it is resumable and can
never silently reuse results from a different parameter set. Interrupt it and
re-run, and only the missing points are recomputed.

`plot_w411.py` reads `benchmark_w411_*.csv` from the **current directory** —
run it from wherever the sweep wrote them (`--csv-prefix` overrides the name).
It writes:

| Figure | Content |
|---|---|
| `benchmark_w411_violin.pdf` | TAE error distribution, one violin per basis set |
| `benchmark_w411_scaling.pdf` | Runtime vs problem size, log-log with fitted power laws |
| `benchmark_w411_breakdown.pdf` | Where the time goes, per basis set |
| `latex/Data/tables/w411_summary.tex` | MSD / MAD / RMSD / max summary tabular |

Only fully converged runs are plotted; dropped reactions are listed on stdout.

## Layout note

The shared driver helpers (`standards.*`, `test_io.*`) stay in `tests/` and are
compiled in from there — see `CMakeLists.txt`. `standalone` also stays in
`tests/`, since it is a general-purpose SCF driver rather than a benchmark;
that is why `benchmark_w411.py` defaults to `build/tests/standalone`.
