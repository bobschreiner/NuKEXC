#!/usr/bin/env python3
"""
Run the standalone SCF driver over the W4-11 total-atomization-energy set for a
sweep of ADF Slater-type basis sets, and collect both energies and timings.

W4-11 (Karton, Daon & Martin, Chem. Phys. Lett. 510, 165-178 (2011)) provides
140 zero-point-exclusive, non-relativistic, clamped-nuclei total atomization
energies -- which is exactly what a plain SCF total energy difference gives, so
no thermochemical corrections are needed. The dataset lives in input/W4-11 in
Grimme/GMTKN layout: one directory per species holding struc.xyz (directly
readable by Nukexc::read_xyz), plus optional .CHRG and .UHF files giving the
charge and the number of unpaired electrons. The reference energies and reaction
stoichiometries are in the top-level '.res' file.

For each (basis, species) pair this runs the standalone binary once, parses its
stdout, and caches one JSON file per run under --cache so the sweep is
resumable: interrupt it and re-run, and only the missing points are recomputed.
Two CSVs are written at the end:

  <out>_species.csv    one row per (basis, species): energy, sizes, timings
  <out>_reactions.csv  one row per (basis, reaction): TAE, reference, error

Note on the SCF threshold: every species is run at the same --conv-thr. The
open-shell atoms with a degenerate p shell used to need a looser gate, but that
plateau was a symptom of the GWH initial guess (see below); with --guess=core
they converge in a handful of iterations at the tight threshold. Runs that still
fail to converge are kept, with converged=0, so they can be counted rather than
silently dropped.

Note on serial execution: the sweep runs one calculation at a time, on purpose.
Each standalone process spawns Kokkos threads across every core, so running
several concurrently oversubscribes them and changes the order of the
floating-point reductions that build rho and grad(rho). The energies still agree
to ~1e-9, but a marginally-stable SCF takes a different trajectory and can flip
between converged and not: O/TZP came out at 20 Fock builds (converged) serially
and 240 (unconverged) with four concurrent jobs, from byte-identical inputs. A
reproducible exclusion list requires serial execution.

Note on linear dependence: --lin-dep defaults to the standalone value of 1e-6.
It was previously raised to 1e-4 because H2/QZ4P came out at +12.8 Ha instead of
-1.17, which looked like the canonical orthogonalisation failing to remove
near-null eigenvectors (QZ4P places 21 functions on every hydrogen). That
diagnosis was wrong: the wildly wrong energy came from the GWH initial guess
populating those directions, not from the threshold retaining them. Raising the
threshold masked it by deleting the directions the guess was abusing. The
overlap spectrum is in fact grid-independent -- the H2/QZ4P energy is flat from
1e-6 to 1e-4 (2.7e-7 Ha spread) and only moves at 1e-3 (1.5e-5) and 1e-2
(1.4e-3), so 1e-4 and below are noise while larger values truncate real
variational space. The flag is kept so the sensitivity can be re-checked; if it
is changed it must be applied uniformly to every basis set, since a per-basis
threshold would vary the variational space and make the cross-basis comparison
meaningless.

Note on the initial guess: the SCF stalls traced to standalone's GWH guess,
which estimates Fock off-diagonals from the diagonal of h_core and the overlap.
With near-linearly-dependent Slater sets (QZ4P places 21 functions on every
hydrogen) that estimate populates near-null directions, giving a starting
density with a POSITIVE one-electron energy (tr(D h_core) = +7.8 Ha for
H2/QZ4P, where it should be -2.6). Robust functionals climb out; LYP-containing
ones such as B3LYP do not, and whether they recover depends on the grid via
h_core's quadrature-computed V_ne -- which is why refining the grid appeared to
make results worse. --guess=core sidesteps it: H2/QZ4P B3LYP then converges in
5 iterations to the same -1.1739 Ha at nrad=75 and nrad=100. The two-phase
occupation freezing that used to paper over these stalls has been removed; it
masked the guess problem and could converge to a non-Aufbau configuration,
silently recording a wrong energy (O/TZP came out 0.045 Ha high).

Provenance: every parameter that affects the physics is archived in the CSV
headers of both output files, and the cache filename carries a hash of those
parameters -- rerunning with different parameters therefore recomputes instead
of silently reusing stale entries. (Corollary: old caches from before this
scheme no longer match and will be recomputed.)

Usage:
    python benchmarking/benchmark_w411.py                      # full sweep, 4 bases
    python benchmarking/benchmark_w411.py --species h2o o h    # just these
    python benchmarking/benchmark_w411.py --bases DZ QZ4P
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HARTREE_TO_KCAL = 627.5094740631

# Timing lines look like "  <indent><name>: <seconds> s", where the total
# indentation is 2 + 2*depth spaces (see TimingRegistry::report).
TIMING_RE = re.compile(r"^(\s+)(\S.*?)\s*:\s+([0-9]+\.[0-9]+) s\s*$")
BLOCK_TITLES = ("Startup Timing Breakdown", "Fock Build #", "Cumulative Fock Build")


def parse_res(res_path):
    """Parse the GMTKN '.res' file into a list of reactions.

    Each line has the form
        tmer2_GMTKN <species...> x <coefficients...> $w <reference kcal/mol>
    so that  sum_i coeff_i * E_i  is the total atomization energy.
    """
    reactions = []
    for raw in res_path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("tmer2_GMTKN"):
            continue
        tok = line.split()[1:]
        if "x" not in tok or "$w" not in tok:
            continue
        xi, wi = tok.index("x"), tok.index("$w")
        species = tok[:xi]
        coeffs = [float(c) for c in tok[xi + 1 : wi]]
        if len(species) != len(coeffs):
            continue
        reactions.append(
            {
                "species": species,
                "coeffs": coeffs,
                "ref_kcal": float(tok[wi + 1]),
                # A readable label: the species carrying the negative coefficient
                # is the molecule being atomized.
                "name": next((s for s, c in zip(species, coeffs) if c < 0), species[0]),
            }
        )
    return reactions


def species_spec(w411_dir, name):
    """Charge and multiplicity for one species directory."""

    def read_int(fname, default=0):
        p = w411_dir / name / fname
        try:
            return int(p.read_text().split()[0])
        except (OSError, IndexError, ValueError):
            return default

    charge = read_int(".CHRG")
    unpaired = read_int(".UHF")
    return charge, unpaired + 1, unpaired


def parse_timing_blocks(text):
    """Extract the named timing blocks as {block: [(name, depth, seconds)]}."""
    blocks, current = {}, None
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(t) for t in BLOCK_TITLES):
            current = stripped
            blocks.setdefault(current, [])
            continue
        if current is None:
            continue
        m = TIMING_RE.match(line)
        if not m:
            # A blank line or the '---' rule ends the block.
            if stripped == "" or stripped.startswith("---"):
                continue
            continue
        indent, name, secs = m.group(1), m.group(2), float(m.group(3))
        depth = max(0, (len(indent) - 2) // 2)
        blocks[current].append((name, depth, secs))
    return blocks


def top_level(entries):
    """Sum depth-0 entries by name, skipping the trailing total row."""
    out = {}
    for name, depth, secs in entries:
        if depth != 0 or name.startswith("Total (top level)"):
            continue
        out[name] = out.get(name, 0.0) + secs
    return out


def parse_run(text):
    """Pull energy, problem size and timing categories out of standalone stdout."""
    res = {}

    def grab(pattern, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    res["nbf"] = grab(r"Basis functions \(primary\):\s*(\d+)", int)
    res["nbf_aux"] = grab(r"Basis functions \(aux\)\s*:\s*(\d+)", int)
    res["nquad"] = grab(r"Quadrature points\s*:\s*(\d+)", int)
    res["nelec"] = grab(r"Number of Electrons:\s*(\d+)", int)

    # Prefer the solver's fully-converged energy (10 decimals). The run only
    # counts as converged if the solver ended in a "Converged" line, i.e. the
    # last one appears after the last "Iteration" line. Otherwise fall back to
    # the last energy the Fock builder printed and flag the run.
    #
    # Take the LOWEST converged energy, not the last: the SCF is variational,
    # so if more than one solver run reports convergence the lower energy is
    # the physical one. Taking the last silently accepted a non-Aufbau state
    # 0.045 Ha above the ground state for O/TZP.
    conv = re.findall(r"Converged to energy\s+(-?\d+\.\d+)", text)
    last_conv = text.rfind("Converged to energy")
    last_iter = text.rfind("\nIteration ")
    if conv and last_conv > last_iter:
        res["energy"] = min(float(c) for c in conv)
        res["converged"] = 1
    else:
        tot = re.findall(r"Total energy\s*:\s*(-?\d+\.\d+)", text)
        res["energy"] = float(tot[-1]) if tot else None
        res["converged"] = 0

    res["n_fock"] = len(re.findall(r"Fock Build #\d+ Timing Breakdown", text))
    res["t_scf"] = grab(r"SCF wall time\s*:\s*([0-9.]+) s")
    res["t_total"] = grab(r"Total program wall time:\s*([0-9.]+) s")

    blocks = parse_timing_blocks(text)

    startup = top_level(blocks.get("Startup Timing Breakdown", []))
    res["t_startup"] = sum(startup.values())
    res["startup_parts"] = startup

    # Sum every Fock build's top-level sections, so the SCF categories reflect
    # the whole run rather than one representative iteration.
    fock = {}
    for title, entries in blocks.items():
        if not title.startswith("Fock Build #"):
            continue
        for name, secs in top_level(entries).items():
            fock[name] = fock.get(name, 0.0) + secs
    res["fock_parts"] = fock
    res["t_fock"] = sum(fock.values())
    return res


def run_one(binary, xyz, basis_dir, charge, mult, conv_thr, args):
    cmd = [
        str(binary),
        f"--xyz={xyz}",
        f"--basis={basis_dir}",
        "--method=dft",
        f"--alg={args.alg}",
        f"--xfunc={args.xfunc}",
        f"--nrad={args.nrad}",
        f"--nang={args.nang}",
        f"--pruning={args.pruning}",
        f"--charge={charge}",
        f"--multiplicity={mult}",
        f"--conv-thr={conv_thr}",
        f"--lin-dep={args.lin_dep}",
        f"--guess={args.guess}",
    ]
    # Only forward non-default extras, so an older standalone binary without
    # these flags keeps working for the runs that don't need them.
    if args.radial is not None:
        cmd.append(f"--radial={args.radial}")
    if args.dens_thr is not None:
        cmd.append(f"--dens-thr={args.dens_thr}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            cwd=args.workdir,
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "cmd": " ".join(cmd)}, ""
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        return {
            "error": "exit %d: %s" % (proc.returncode, " | ".join(tail)),
            "cmd": " ".join(cmd),
        }, proc.stdout
    parsed = parse_run(proc.stdout)
    parsed["cmd"] = " ".join(cmd)
    if parsed.get("energy") is None:
        parsed["error"] = "no energy parsed"
    elif parsed["energy"] > 0.0:
        # Every W4-11 species is a bound neutral system, so a positive total
        # energy is never a converged answer -- it is the linear-dependence
        # blow-up described in the module docstring. Fail loudly instead of
        # letting it through as a merely "unconverged" number.
        parsed["error"] = (
            "positive total energy %.4f Ha -- raise --lin-dep" % parsed["energy"]
        )
    return parsed, proc.stdout


def main():
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bin", default=str(repo / "build/tests/standalone"))
    ap.add_argument("--w411-dir", default=str(repo / "input/W4-11"))
    ap.add_argument("--basis-root", default=str(repo / "input/zorabasis_cholesky"))
    ap.add_argument("--bases", nargs="+", default=["DZ", "TZP", "TZ2P", "QZ4P"])
    ap.add_argument(
        "--species",
        nargs="+",
        default=None,
        help="restrict to these species (default: all in .res)",
    )

    ap.add_argument("--alg", default="dense")
    ap.add_argument("--xfunc", default="gga_xc_b3lyp3")
    ap.add_argument("--nrad", type=int, default=100)
    ap.add_argument("--nang", type=int, default=35)
    ap.add_argument("--conv-thr", default="1e-7")
    ap.add_argument("--pruning", default="robust")
    ap.add_argument(
        "--guess",
        default="core",
        help="initial guess forwarded to standalone: core (default) or gwh. "
        "gwh stalls on near-linearly-dependent Slater sets (see docstring)",
    )
    ap.add_argument(
        "--lin-dep",
        default="1e-6",
        help="linear-dependence threshold (see module docstring); values "
        "above ~1e-4 truncate real variational space",
    )
    ap.add_argument(
        "--radial",
        default=None,
        help="per-element radial sizing (uniform/pyscf); omitted = binary default",
    )
    ap.add_argument(
        "--dens-thr",
        default=None,
        help="libxc density threshold passthrough; omitted = binary default",
    )
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--cache", default=str(repo / "w411_cache"))
    ap.add_argument("--out", default="benchmark_w411")
    ap.add_argument("--workdir", default=str(repo))
    ap.add_argument(
        "--keep-stdout", action="store_true", help="also cache raw stdout for each run"
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    w411 = Path(args.w411_dir)
    res_file = w411 / ".res"
    if not res_file.is_file():
        sys.exit("No .res file in %s" % w411)

    reactions = parse_res(res_file)
    needed = sorted({s for r in reactions for s in r["species"]})
    if args.species:
        keep = set(args.species)
        reactions = [r for r in reactions if set(r["species"]) <= keep]
        needed = sorted(keep & set(needed))

    missing = [s for s in needed if not (w411 / s / "struc.xyz").is_file()]
    if missing:
        sys.exit("Missing struc.xyz for: %s" % ", ".join(missing))

    print(
        "%d reactions, %d species, %d bases -> %d runs"
        % (len(reactions), len(needed), len(args.bases), len(needed) * len(args.bases))
    )
    if args.dry_run:
        return

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    binary = Path(args.bin).resolve()
    if not binary.is_file():
        sys.exit("standalone binary not found: %s" % binary)

    # Everything that affects the physics of a single run, in one dict. It is
    # archived in the CSV headers and hashed into the cache filenames, so a
    # parameter change can never silently reuse stale cached results.
    run_params = {
        "method": "dft",
        "alg": args.alg,
        "xfunc": args.xfunc,
        "nrad": args.nrad,
        "nang": args.nang,
        "pruning": args.pruning,
        "radial": args.radial if args.radial is not None else "(binary default)",
        "conv_thr": args.conv_thr,
        "guess": args.guess,
        "lin_dep": args.lin_dep,
        "dens_thr": args.dens_thr if args.dens_thr is not None else "(binary default)",
    }
    params_hash = hashlib.md5(
        json.dumps(run_params, sort_keys=True).encode()
    ).hexdigest()[:10]

    tasks = []
    for basis in args.bases:
        # The Cholesky sets live in <root>/<BASIS>.cholesky, the plain ones in
        # <root>/<BASIS>. Accept either so --basis-root can point at both.
        basis_dir = Path(args.basis_root) / ("%s.cholesky" % basis)
        if not basis_dir.is_dir():
            basis_dir = Path(args.basis_root) / basis
        if not basis_dir.is_dir():
            sys.exit("No such basis dir: %s" % basis_dir)
        for name in needed:
            charge, mult, _unpaired = species_spec(w411, name)
            tasks.append((basis, name, basis_dir, charge, mult, args.conv_thr))

    def execute(task):
        basis, name, basis_dir, charge, mult, thr = task
        jf = cache / ("%s__%s__%s.json" % (basis, name, params_hash))
        if jf.is_file():
            try:
                return basis, name, json.loads(jf.read_text()), True
            except json.JSONDecodeError:
                pass  # corrupt cache entry -> recompute
        parsed, stdout = run_one(
            binary,
            w411 / name / "struc.xyz",
            basis_dir,
            charge,
            mult,
            thr,
            args,
        )
        parsed.update(
            {
                "basis": basis,
                "species": name,
                "charge": charge,
                "multiplicity": mult,
                "conv_thr": thr,
                "params": run_params,
            }
        )
        jf.write_text(json.dumps(parsed, indent=1))
        if args.keep_stdout and stdout:
            (cache / ("%s__%s__%s.log" % (basis, name, params_hash))).write_text(stdout)
        return basis, name, parsed, False

    results = {}
    total = len(tasks)
    # Serial on purpose -- see the note on serial execution in the module
    # docstring. Concurrency changes the Kokkos reduction order and makes the
    # converged/unconverged verdict irreproducible.
    for i, task in enumerate(tasks, 1):
        basis, name, parsed, cached = execute(task)
        results[(basis, name)] = parsed
        flag = "cached" if cached else "ran"
        err = parsed.get("error")
        status = (
            "ERROR: %s" % err
            if err
            else "E=%.6f%s"
            % (parsed["energy"], "" if parsed.get("converged") else " (unconv)")
        )
        print(
            "[%4d/%4d] %-5s %-12s %-6s %s" % (i, total, basis, name, flag, status),
            flush=True,
        )

    # ---- species CSV -------------------------------------------------------
    startup_keys, fock_keys = set(), set()
    for r in results.values():
        startup_keys |= set(r.get("startup_parts") or {})
        fock_keys |= set(r.get("fock_parts") or {})
    startup_keys = sorted(startup_keys)
    fock_keys = sorted(fock_keys)

    # Shared provenance header: every parameter, plus enough context to rerun.
    provenance = (
        ["# NuKEXC W4-11 sweep"]
        + ["# param %s = %s" % (k, v) for k, v in sorted(run_params.items())]
        + [
            "# param bases = %s" % " ".join(args.bases),
            "# params_hash = %s" % params_hash,
            "# binary = %s" % binary,
            "# dataset = %s" % w411,
            "# basis_root = %s" % args.basis_root,
            "# date_utc = %s"
            % datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "# cmdline = %s" % " ".join(sys.argv),
        ]
    )

    sp_path = Path("%s_species.csv" % args.out)
    with sp_path.open("w") as f:
        for line in provenance:
            f.write(line + "\n")
        f.write(
            "# the conv_thr column is the per-species value actually "
            "used (open-shell species differ).\n"
        )
        f.write(
            "# Startup/Fock timing columns are seconds, summed over the "
            "whole run (all Fock builds).\n"
        )
        cols = (
            [
                "basis",
                "species",
                "charge",
                "multiplicity",
                "conv_thr",
                "converged",
                "nbf",
                "nbf_aux",
                "nquad",
                "nelec",
                "energy",
                "n_fock",
                "t_startup",
                "t_fock",
                "t_scf",
                "t_total",
            ]
            + ["startup:%s" % k for k in startup_keys]
            + ["fock:%s" % k for k in fock_keys]
        )
        f.write(",".join(cols) + "\n")
        for basis, name in sorted(results):
            r = results[(basis, name)]
            if r.get("error"):
                continue
            sp, fp = r.get("startup_parts") or {}, r.get("fock_parts") or {}
            row = [
                basis,
                name,
                r["charge"],
                r["multiplicity"],
                r.get("conv_thr", ""),
                r["converged"],
                r["nbf"],
                r["nbf_aux"],
                r["nquad"],
                r["nelec"],
                "%.10f" % r["energy"],
                r["n_fock"],
                "%.4f" % r["t_startup"],
                "%.4f" % r["t_fock"],
                "%.4f" % (r["t_scf"] or 0.0),
                "%.4f" % (r["t_total"] or 0.0),
            ]
            row += ["%.4f" % sp.get(k, 0.0) for k in startup_keys]
            row += ["%.4f" % fp.get(k, 0.0) for k in fock_keys]
            f.write(",".join(str(c) for c in row) + "\n")

    # ---- reaction CSV ------------------------------------------------------
    rx_path = Path("%s_reactions.csv" % args.out)
    n_ok = 0
    with rx_path.open("w") as f:
        for line in provenance:
            f.write(line + "\n")
        f.write(
            "# W4-11 total atomization energies (kcal/mol). Reference: "
            "zero-point-exclusive, non-relativistic, clamped-nuclei\n"
        )
        f.write(
            "# TAEs from Karton, Daon & Martin, Chem. Phys. Lett. 510, "
            "165-178 (2011).\n"
        )
        f.write("basis,reaction,tae_calc,tae_ref,error,all_converged\n")
        for basis in args.bases:
            for rx in reactions:
                energies, ok, allconv = [], True, True
                for s, c in zip(rx["species"], rx["coeffs"]):
                    r = results.get((basis, s))
                    if not r or r.get("error") or r.get("energy") is None:
                        ok = False
                        break
                    energies.append(c * r["energy"])
                    allconv &= bool(r.get("converged"))
                if not ok:
                    continue
                tae = sum(energies) * HARTREE_TO_KCAL
                f.write(
                    "%s,%s,%.4f,%.4f,%.4f,%d\n"
                    % (
                        basis,
                        rx["name"],
                        tae,
                        rx["ref_kcal"],
                        tae - rx["ref_kcal"],
                        int(allconv),
                    )
                )
                n_ok += 1

    failed = [(b, s) for (b, s), r in results.items() if r.get("error")]
    unconv = [
        (b, s)
        for (b, s), r in results.items()
        if not r.get("error") and not r.get("converged")
    ]
    print(
        "\nWrote %s (%d species rows) and %s (%d reaction rows)"
        % (sp_path, len(results) - len(failed), rx_path, n_ok)
    )
    if unconv:
        print(
            "SCF not converged (%d): %s"
            % (len(unconv), ", ".join("%s/%s" % t for t in sorted(unconv)[:20]))
        )
    if failed:
        print("Failed runs (%d):" % len(failed))
        for b, s in sorted(failed):
            print("   %-5s %-12s %s" % (b, s, results[(b, s)]["error"]))


if __name__ == "__main__":
    main()
