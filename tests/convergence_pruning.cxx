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
 * Angular pruning-scheme comparison on water (core-Hamiltonian occupied-orbital
 * energy sum), holding radial sizing at Uniform so only the pruning differs.
 *
 * The three schemes (IntegratorXX::PruningScheme) partition each atom's radial
 * shells into regions with different Lebedev orders:
 *
 *   Unpruned : full order on every shell.
 *   Treutler : [0,r/3) order 7, [r/3,r/2) order 11, [r/2,rsz) full.
 *              The low/medium orders are FIXED (7, 11) regardless of the base.
 *   Robust   : [0,r/4) order 7, [r/4,r/2) order (base-6), [r/2,rsz) full.
 *              The medium order tracks the base order, so it refines with it.
 *
 * Because Treutler's inner/middle shells are stuck at orders 7/11, its error
 * plateaus once the base order passes them; Robust's middle region grows with
 * the base, so it keeps converging (closer to Unpruned) while still saving
 * points versus the full grid.
 *
 * Each grid "level" grows nrad and the Lebedev order together so every scheme
 * can converge. We record the TOTAL points used and the error; plotting error
 * vs total points (tests/plot_pruning.py) shows accuracy-per-point.
 *
 * Reference: a fine UNIFORM UNPRUNED grid (nrad_ref, nang_ref) finer than any
 * swept level; error = |E - E_ref| (grid self-convergence, not analytic).
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

struct Scheme {
  const char *label;
  PruningScheme pruning;
};

// A grid "level": both axes grow together so every scheme can converge.
struct Level {
  size_t nrad;
  size_t nang_order;
};

// Converged unrestricted-HF total energy on the grid built for the given
// pruning scheme; also returns the total number of grid points actually used.
static double scf_energy(ExecSpace &space, const Molecule &mol,
                         const STOBasisSet &basis, const STOBasisSet &basis_aux,
                         size_t nrad, size_t nang_order, PruningScheme pruning,
                         size_t &npts_out) {
  auto grid =
      make_flat_grid<ta_type, ll_type>(mol, nrad, nang_order, WEIGHT_THRESHOLD,
                                       TA_M4, pruning, RadialSizing::Uniform);
  npts_out = grid.quad_points.extent(0);
  return run_uhf_scf_energy(space, mol, basis, basis_aux, grid);
}

int main() {
  Kokkos::initialize();
  int status = 0;
  {
    // Resolution. Default is a coarse, CPU-friendly prototype sweep; set the
    // environment variable NUKEXC_HIRES=1 for the fine (GPU) production sweep.
    const bool hires = std::getenv("NUKEXC_HIRES") != nullptr;
    const std::vector<Level> levels =
        hires ? std::vector<Level>{{40, 17},  {60, 21},  {90, 28}, {130, 30},
                                   {200, 35}, {300, 41}, {600, 45}}
              : std::vector<Level>{{30, 15}, {40, 17}, {60, 21}};
    const size_t nrad_ref = hires ? 1000 : 90;
    const size_t nang_ref = hires ? 50 : 28;

    ExecSpace space;
    auto mol = make_water();
    auto basis = load_adf_basis(mol, "input/zorabasis/QZ4P");
    auto basis_aux = load_adf_basis(mol, "input/zorabasis/QZ4P", 1e-10, true);

    // Reference: fine uniform unpruned grid (finer than every swept level).
    size_t npts_ref = 0;
    const double e_ref =
        scf_energy(space, mol, basis, basis_aux, nrad_ref, nang_ref,
                   PruningScheme::Unpruned, npts_ref);
    std::cout << std::fixed << std::setprecision(10)
              << "Molecule: water (Z_total=" << mol.Z_total << ")\n"
              << "Resolution: " << (hires ? "HI-RES (GPU)" : "prototype (CPU)")
              << "\nReference (UHF, uniform unpruned, nrad=" << nrad_ref
              << ", nang_order=" << nang_ref << ", npts=" << npts_ref
              << "): E_scf = " << e_ref << " Ha\n";

    const std::vector<Scheme> schemes = {
        {"Unpruned", PruningScheme::Unpruned},
        {"Treutler", PruningScheme::Treutler},
        {"Robust", PruningScheme::Robust},
    };

    const std::string csv_path = "convergence_pruning.csv";
    std::ofstream csv(csv_path);
    csv << std::setprecision(15);
    csv << "# Angular pruning-scheme comparison on water (unrestricted HF "
           "total energy)\n";
    csv << "# radial scheme: TA-M4 ; angular: Lebedev-Laikov ; basis: QZ4P "
           "(+QZ4P fit) ; radial sizing=Uniform\n";
    csv << "# schemes: Unpruned, Treutler (fixed 7/11), Robust (7 / base-6)\n";
    csv << "# grid levels sweep (nrad, nang_order) together so curves "
           "converge\n";
    csv << "# observable: converged unrestricted Hartree-Fock total energy "
           "(Ha)\n";
    csv << "# resolution: " << (hires ? "hi-res (GPU)" : "prototype (CPU)")
        << "\n";
    csv << "# E_ref = fine uniform unpruned grid (shared self-reference, NOT "
           "analytic):\n";
    csv << "#   nrad_ref=" << nrad_ref << ", nang_order_ref=" << nang_ref
        << ", npts_ref=" << npts_ref << ", E_ref=" << e_ref << " Ha\n";
    csv << "# abs_error = |E_scf - E_ref| ; npts = total grid points used\n";
    csv << "scheme,nrad,nang_order,npts,E_scf,abs_error\n";

    const int w = 14;
    for (const auto &s : schemes) {
      std::cout << "\n=== " << s.label << " ===\n";
      std::cout << std::setw(w) << std::left << "nrad" << std::setw(w)
                << std::left << "nang" << std::setw(w) << std::right << "npts"
                << std::setw(20) << std::right << "E_scf (Ha)" << std::setw(w)
                << std::right << "|error|"
                << "\n";
      std::cout << std::string(w * 5 + 6, '-') << "\n";

      for (const auto &lv : levels) {
        size_t npts = 0;
        const double e = scf_energy(space, mol, basis, basis_aux, lv.nrad,
                                    lv.nang_order, s.pruning, npts);
        const double err = std::abs(e - e_ref);

        csv << s.label << ',' << lv.nrad << ',' << lv.nang_order << ',' << npts
            << ',' << e << ',' << err << '\n';

        std::cout << std::setw(w) << std::left << lv.nrad << std::setw(w)
                  << std::left << lv.nang_order << std::setw(w) << std::right
                  << npts << std::setw(20) << std::right << std::fixed
                  << std::setprecision(8) << e << std::setw(w) << std::right
                  << std::scientific << std::setprecision(3) << err << "\n";
      }
    }

    csv.close();
    std::cout << "\nWrote " << (schemes.size() * levels.size()) << " rows to "
              << std::filesystem::absolute(csv_path).string()
              << "\nPlot with: python tests/plot_pruning.py "
              << std::filesystem::absolute(csv_path).string() << "\n";
  }
  Kokkos::finalize();
  return status;
}
