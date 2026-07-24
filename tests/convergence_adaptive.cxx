/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (C) 2026 Bob Schreiner
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

/*
 * Accuracy-per-point study of the adaptive-grid options on a full molecule
 * (water), for the core-Hamiltonian occupied-orbital energy sum.
 *
 * make_flat_grid exposes two orthogonal, independently toggleable knobs:
 *   * pruning     -- angular adaptivity (Unpruned vs Robust): fewer Lebedev
 *                    points on the inner/outer radial shells.
 *   * radial_sizing -- radial adaptivity: RadialSizing::PySCF gives heavier
 *                      atoms (here O) more radial points than light ones (H),
 *                      following the GauXC/PySCF per-period pattern.
 * That gives four combinations, swept here over grid levels that grow the
 * radial count and Lebedev order together (a fixed angular order would cap every
 * curve at the same angular floor and hide the differences):
 *   uniform      (Unpruned, radial=Uniform)
 *   pruned       (Robust,   radial=Uniform)
 *   per-element  (Unpruned, radial=PySCF)
 *   both         (Robust,   radial=PySCF)
 *
 * (Robust pruning is used rather than Treutler: its middle-shell order tracks
 * the base order instead of being fixed at 11, so it converges to the reference
 * rather than plateauing -- see tests/convergence_pruning.cxx.)
 *
 * For each combination and base nrad we record the TOTAL number of grid points
 * actually used and the error of the observable. Plotting error vs total points
 * (tests/plot_adaptive.py) shows accuracy-per-point: the adaptive schemes aim to
 * reach a given accuracy with fewer points than the uniform grid.
 *
 * Observable: sum of the lowest nocc = Z_total/2 core-Hamiltonian MO energies
 * (a grid-sensitive scalar spanning the O 1s core -- radial-stressing -- and the
 * valence orbitals -- angular-stressing).
 *
 * Reference: a fine UNIFORM UNPRUNED grid (nrad_ref, nang_ref) larger than any
 * swept grid; all four schemes converge to it, so error = |E - E_ref| is a fair
 * common yardstick (grid self-convergence, not an analytic value).
 */

#include <Kokkos_Core.hpp>

#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/diagonalizer.hpp>
#include <nukexc/grid.hpp> // make_flat_grid, TA_M4
#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp> // load_adf_basis

#include "scf_driver.hpp" // run_uhf_scf_energy
#include "standards.hpp"  // make_water

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace Nukexc;

using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;
using PruningScheme = IntegratorXX::PruningScheme;

static constexpr double WEIGHT_THRESHOLD = 1e-30;

struct Combo {
  const char *label;
  PruningScheme pruning;
  RadialSizing radial_sizing;
};

// A grid "level": both axes grow together so every combination can converge
// (a fixed angular order would cap all curves at the same angular floor and
// hide the accuracy-per-point differences).
struct Level {
  size_t nrad;
  size_t nang_order;
};

// Converged unrestricted-HF total energy on the grid built for the given grid
// knobs; also returns the total number of grid points actually used.
static double scf_energy(ExecSpace &space, const Molecule &mol,
                         const STOBasisSet &basis, const STOBasisSet &basis_aux,
                         size_t nrad, size_t nang_order, PruningScheme pruning,
                         RadialSizing radial_sizing, size_t &npts_out) {
  auto grid = make_flat_grid<ta_type, ll_type>(mol, nrad, nang_order,
                                               WEIGHT_THRESHOLD, TA_M4, pruning,
                                               radial_sizing);
  npts_out = grid.quad_points.extent(0);
  return run_uhf_scf_energy(space, mol, basis, basis_aux, grid);
}

int main() {
  Kokkos::initialize();
  int status = 0;
  {
    // Resolution. Default is a coarse, CPU-friendly prototype sweep; set the
    // environment variable NUKEXC_HIRES=1 for the fine (GPU) production sweep.
    // NOTE: Lebedev-Laikov order 25 has negative weights and is avoided (grid
    // construction now throws on negative weights).
    const bool hires = std::getenv("NUKEXC_HIRES") != nullptr;
    const std::vector<Level> levels =
        hires ? std::vector<Level>{{40, 17},  {60, 21},  {90, 28},
                                   {130, 29}, {200, 35}, {300, 41}}
              : std::vector<Level>{{30, 15}, {40, 17}, {60, 21}};
    const size_t nrad_ref = hires ? 600 : 90;
    const size_t nang_ref = hires ? 47 : 28;

    ExecSpace space;
    auto mol = make_water();
    auto basis = load_adf_basis(mol, "input/zorabasis/QZ4P");
    auto basis_aux = load_adf_basis(mol, "input/zorabasis/QZ4P", 1e-10, true);

    // Reference: fine uniform unpruned grid (finer than every swept level).
    size_t npts_ref = 0;
    const double e_ref =
        scf_energy(space, mol, basis, basis_aux, nrad_ref, nang_ref,
                   PruningScheme::Unpruned, RadialSizing::Uniform, npts_ref);
    std::cout << std::fixed << std::setprecision(10)
              << "Molecule: water (Z_total=" << mol.Z_total << ")\n"
              << "Resolution: " << (hires ? "HI-RES (GPU)" : "prototype (CPU)")
              << "\nReference (UHF, uniform unpruned, nrad=" << nrad_ref
              << ", nang_order=" << nang_ref << ", npts=" << npts_ref
              << "): E_scf = " << e_ref << " Ha\n";

    const std::vector<Combo> combos = {
        {"uniform", PruningScheme::Unpruned, RadialSizing::Uniform},
        {"pruned", PruningScheme::Robust, RadialSizing::Uniform},
        {"per-element", PruningScheme::Unpruned, RadialSizing::PySCF},
        {"both", PruningScheme::Robust, RadialSizing::PySCF},
    };

    const std::string csv_path = "convergence_adaptive.csv";
    std::ofstream csv(csv_path);
    csv << std::setprecision(15);
    csv << "# Adaptive-grid accuracy-per-point study on water (unrestricted HF "
           "total energy)\n";
    csv << "# radial scheme: TA-M4 ; angular: Lebedev-Laikov ; basis: QZ4P "
           "(+QZ4P fit)\n";
    csv << "# knobs: pruning (Unpruned/Robust) x radial sizing "
           "(Uniform/PySCF per-period)\n";
    csv << "# grid levels sweep (nrad, nang_order) together so curves converge\n";
    csv << "# observable: converged unrestricted Hartree-Fock total energy "
           "(Ha)\n";
    csv << "# resolution: " << (hires ? "hi-res (GPU)" : "prototype (CPU)")
        << "\n";
    csv << "# E_ref = fine uniform unpruned grid (shared self-reference, NOT "
           "analytic):\n";
    csv << "#   nrad_ref=" << nrad_ref << ", nang_order_ref=" << nang_ref
        << ", npts_ref=" << npts_ref << ", E_ref=" << e_ref << " Ha\n";
    csv << "# abs_error = |E_scf - E_ref| ; npts = total grid points used\n";
    csv << "combo,nrad,nang_order,npts,E_scf,abs_error\n";

    const int w = 14;
    for (const auto &c : combos) {
      std::cout << "\n=== " << c.label << " (pruning="
                << (c.pruning == PruningScheme::Unpruned ? "Unpruned"
                                                         : "Robust")
                << ", radial="
                << (c.radial_sizing == RadialSizing::Uniform ? "Uniform"
                                                             : "PySCF")
                << ") ===\n";
      std::cout << std::setw(w) << std::left << "nrad" << std::setw(w)
                << std::left << "nang" << std::setw(w) << std::right << "npts"
                << std::setw(20) << std::right << "E_scf (Ha)" << std::setw(w)
                << std::right << "|error|"
                << "\n";
      std::cout << std::string(w * 5 + 6, '-') << "\n";

      for (const auto &lv : levels) {
        size_t npts = 0;
        const double e =
            scf_energy(space, mol, basis, basis_aux, lv.nrad, lv.nang_order,
                       c.pruning, c.radial_sizing, npts);
        const double err = std::abs(e - e_ref);

        csv << c.label << ',' << lv.nrad << ',' << lv.nang_order << ',' << npts
            << ',' << e << ',' << err << '\n';

        std::cout << std::setw(w) << std::left << lv.nrad << std::setw(w)
                  << std::left << lv.nang_order << std::setw(w) << std::right
                  << npts << std::setw(20) << std::right << std::fixed
                  << std::setprecision(8) << e << std::setw(w) << std::right
                  << std::scientific << std::setprecision(3) << err << "\n";
      }
    }

    csv.close();
    std::cout << "\nWrote " << (combos.size() * levels.size())
              << " rows to " << std::filesystem::absolute(csv_path).string()
              << "\nPlot with: python tests/plot_adaptive.py "
              << std::filesystem::absolute(csv_path).string() << "\n";
  }
  Kokkos::finalize();
  return status;
}
