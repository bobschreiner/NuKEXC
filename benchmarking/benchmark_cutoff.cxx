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
 *
 */

#include <Kokkos_Core.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <iomanip>
#include <iostream>
#include <nukexc/grid.hpp>

#include <nukexc/molecule.hpp>
#include <nukexc/stobasis.hpp>

#include "test_io.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

struct Config {
  std::string xyz_file = "input/water.xyz";
  std::string basis_dir = "input/zorabasis/QZ4P";
  int nrad = 30;
  int nang = 10;
  double tol_start = 1e-6;
  double tol_end = 1e-15;
};

Config parse_args(int argc, char *argv[]) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    ArgParser p{argv[i]};

    if (p.arg == "--help" || p.arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "  --xyz=<file>        XYZ input file          (default: "
                << cfg.xyz_file << ")\n"
                << "  --basis=<dir>       Basis set directory     (default: "
                << cfg.basis_dir << ")\n"
                << "  --nrad=<int>        Radial grid points      (default: "
                << cfg.nrad << ")\n"
                << "  --nang=<int>        Angular grid points     (default: "
                << cfg.nang << ")\n"
                << "  --tol-start=<float> Starting cutoff tol     (default: "
                << cfg.tol_start << ")\n"
                << "  --tol-end=<float>   Ending cutoff tol       (default: "
                << cfg.tol_end << ")\n";
      std::exit(0);

    } else if (!p.string_opt("--xyz=", cfg.xyz_file) &&
               !p.string_opt("--basis=", cfg.basis_dir) &&
               !p.int_opt("--nrad=", cfg.nrad) &&
               !p.int_opt("--nang=", cfg.nang) &&
               !p.double_opt("--tol-start=", cfg.tol_start) &&
               !p.double_opt("--tol-end=", cfg.tol_end)) {
      throw std::runtime_error("Unknown argument: " + p.arg + " (try --help)");
    }
  }

  if (cfg.tol_start <= cfg.tol_end)
    throw std::runtime_error("--tol-start must be greater than --tol-end");
  if (cfg.nrad <= 0 || cfg.nang <= 0)
    throw std::runtime_error("--nrad and --nang must be positive");

  return cfg;
}

int main(int argc, char *argv[]) {

  using namespace Nukexc;
  Config cfg;
  try {
    cfg = parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  Kokkos::initialize();
  {
    using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
    using ll_type = IntegratorXX::LebedevLaikov<double>;

    Molecule mol;
    read_xyz(cfg.xyz_file, mol);
    FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad, cfg.nang);
    int G = grid.quad_points.extent(0);

    print_config_box("Benchmark Configuration",
                     {
                         {"XYZ file", cfg.xyz_file},
                         {"Basis directory", cfg.basis_dir},
                         {"Radial points", cfg_val(cfg.nrad)},
                         {"Angular points", cfg_val(cfg.nang)},
                         {"Tolerance start", cfg_val(cfg.tol_start)},
                         {"Tolerance end", cfg_val(cfg.tol_end)},
                         {"Grid points", cfg_val(G)},
                     });

    std::cout << "┌─────────────────┬───────────────┬───────────────┐\n";
    std::cout << "│   Tolerance     │  Basis Fns    │  % Outside    │\n";
    std::cout << "├─────────────────┼───────────────┼───────────────┤\n";

    for (double tol = cfg.tol_start; tol > cfg.tol_end; tol /= 10) {
      STOBasisSet basis = load_adf_basis(mol, cfg.basis_dir, tol);
      int N = basis.nbf();

      Kokkos::View<int64_t> counter("counter");
      Kokkos::parallel_for(
          "Count points outside cutoff",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, G}),
          KOKKOS_LAMBDA(const int i, const int g) {
            double r = dist(basis.O(i), grid.quad_points(g));
            if (r > basis.cutoff_radii(i))
              Kokkos::atomic_fetch_add(&counter(), int64_t(1));
          });
      Kokkos::fence();

      auto count_h = Kokkos::create_mirror_view(counter);
      Kokkos::deep_copy(count_h, counter);
      double percent = (double)count_h() / (double)(N * G) * 100;
      std::cout << "│ " << std::setw(15) << std::scientific
                << std::setprecision(2) << tol << " │ " << std::setw(13)
                << std::fixed << N << " │ " << std::setw(12)
                << std::setprecision(4) << percent << "% │\n";
      std::cout << std::flush;
    }
    std::cout << "└─────────────────┴───────────────┴───────────────┘\n";
  }
  Kokkos::finalize();
  return 0;
};
