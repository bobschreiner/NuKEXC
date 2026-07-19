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

#include "standards.hpp" // make_water

#include <cmath>
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

// Sum of the lowest nocc core-Hamiltonian MO energies for the given grid knobs;
// also returns the total number of grid points actually used.
static double band_sum(const STOBasisSet &basis, const Molecule &mol,
                       size_t nrad, size_t nang_order, PruningScheme pruning,
                       RadialSizing radial_sizing, int nocc, size_t &npts_out) {
  auto grid = make_flat_grid<ta_type, ll_type>(mol, nrad, nang_order,
                                               WEIGHT_THRESHOLD, TA_M4, pruning,
                                               radial_sizing);
  npts_out = grid.quad_points.extent(0);

  CoreHamiltonianResult coreH = compute_core_hamiltonian(basis, grid);

  Diagonalizer diagonalizer(basis.nbf());
  auto X = diagonalizer.compute_transformation(coreH.overlap);
  const int K = X.extent(1);

  DeviceView2DLeft mo_coeff("mo coeff", basis.nbf(), K);
  DeviceView1D mo_energies("mo energies", K);
  diagonalizer.solve(coreH.hamiltonian, mo_coeff, mo_energies);

  auto mo_energies_h = Kokkos::create_mirror_view(mo_energies);
  Kokkos::deep_copy(mo_energies_h, mo_energies);

  double s = 0.0;
  for (int i = 0; i < nocc && i < K; ++i)
    s += mo_energies_h(i);
  return s;
}

int main() {
  Kokkos::initialize();
  int status = 0;
  {
    const std::vector<Level> levels = {{20, 11}, {30, 15}, {40, 17}, {60, 21},
                                       {80, 23}, {120, 29}};
    const size_t nrad_ref = 220;
    const size_t nang_ref = 35;

    auto mol = make_water();
    auto basis = load_adf_basis(mol, "input/zorabasis/QZ4P");
    const int nocc = mol.Z_total / 2; // neutral closed-shell occupation

    // Reference: fine uniform unpruned grid (finer than every level in both
    // radial and angular).
    size_t npts_ref = 0;
    const double e_ref =
        band_sum(basis, mol, nrad_ref, nang_ref, PruningScheme::Unpruned,
                 RadialSizing::Uniform, nocc, npts_ref);
    std::cout << std::fixed << std::setprecision(10)
              << "Molecule: water (Z_total=" << mol.Z_total
              << ", nocc=" << nocc << ")\n"
              << "Reference (uniform unpruned, nrad=" << nrad_ref
              << ", nang_order=" << nang_ref << ", npts=" << npts_ref
              << "): band sum = " << e_ref << " Ha\n";

    const std::vector<Combo> combos = {
        {"uniform", PruningScheme::Unpruned, RadialSizing::Uniform},
        {"pruned", PruningScheme::Robust, RadialSizing::Uniform},
        {"per-element", PruningScheme::Unpruned, RadialSizing::PySCF},
        {"both", PruningScheme::Robust, RadialSizing::PySCF},
    };

    const std::string csv_path = "convergence_adaptive.csv";
    std::ofstream csv(csv_path);
    csv << std::setprecision(15);
    csv << "# Adaptive-grid accuracy-per-point study on water (core-Hamiltonian "
           "occupied-orbital energy sum)\n";
    csv << "# radial scheme: TA-M4 ; angular: Lebedev-Laikov ; basis: QZ4P\n";
    csv << "# knobs: pruning (Unpruned/Robust) x radial sizing "
           "(Uniform/PySCF per-period)\n";
    csv << "# grid levels sweep (nrad, nang_order) together so curves converge\n";
    csv << "# observable: sum of lowest nocc=" << nocc << " core-H MO energies\n";
    csv << "# E_ref = fine uniform unpruned grid (shared self-reference, NOT "
           "analytic):\n";
    csv << "#   nrad_ref=" << nrad_ref << ", nang_order_ref=" << nang_ref
        << ", npts_ref=" << npts_ref << ", E_ref=" << e_ref << " Ha\n";
    csv << "# abs_error = |band_sum - E_ref| ; npts = total grid points used\n";
    csv << "combo,nrad,nang_order,npts,band_sum,abs_error\n";

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
                << std::setw(20) << std::right << "band_sum (Ha)" << std::setw(w)
                << std::right << "|error|"
                << "\n";
      std::cout << std::string(w * 5 + 6, '-') << "\n";

      for (const auto &lv : levels) {
        size_t npts = 0;
        const double e = band_sum(basis, mol, lv.nrad, lv.nang_order, c.pruning,
                                  c.radial_sizing, nocc, npts);
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
