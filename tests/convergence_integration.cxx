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

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <catch2/matchers/catch_matchers.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include <catch2/catch_all.hpp>

#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include <nukexc/grid.hpp>
#include <nukexc/integration.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp>

#include "standards.hpp"

using namespace NuKEXC;

using bk_type = IntegratorXX::Becke<double, double>;
using mk_type = IntegratorXX::MuraKnowles<double, double>;
using mhl_type = IntegratorXX::MurrayHandyLaming<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;

using ah_type = IntegratorXX::AhrensBeylkin<double>;
using de_type = IntegratorXX::Delley<double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;
using wo_type = IntegratorXX::Womersley<double>;

// ── Config
// ────────────────────────────────────────────────────────────────────

struct Config {
  std::string basis_dir = "input/zorabasis/TZP";
  int n_max = 7;
  int m_max = 7;
  double screening_tol = 1e-6;
  int max_points_per_box = 64;
  int screening = 1;
};

Config parse_args(int argc, char *argv[]) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    auto parse_string = [&](const std::string &prefix, std::string &out) {
      if (arg.rfind(prefix, 0) == 0) {
        out = arg.substr(prefix.size());
        return true;
      }
      return false;
    };
    auto parse_int = [&](const std::string &prefix, int &out) {
      if (arg.rfind(prefix, 0) == 0) {
        out = std::stoi(arg.substr(prefix.size()));
        return true;
      }
      return false;
    };
    auto parse_double = [&](const std::string &prefix, double &out) {
      if (arg.rfind(prefix, 0) == 0) {
        out = std::stod(arg.substr(prefix.size()));
        return true;
      }
      return false;
    };

    if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"
          << "  --basis=<dir>          Basis set directory       (default: "
          << cfg.basis_dir << ")\n"
          << "  --n_max=<int>           Radial grid points        (default: "
          << cfg.n_max << ")\n"
          << "  --m_max=<int>           Angular grid points       (default: "
          << cfg.m_max << ")\n"
          << "  --tol=<float>          Screening tolerance       (default: "
          << cfg.screening_tol << ")\n"
          << "  --box-size=<int>       Max points per box        (default: "
          << cfg.max_points_per_box << ")\n"
          << "  --screening=<int>       Turn on/off screening        (default: "
          << cfg.screening << ")\n";

      std::exit(0);
    } else if (!parse_string("--basis=", cfg.basis_dir) &&
               !parse_int("--n_max=", cfg.n_max) &&
               !parse_int("--m_max=", cfg.m_max) &&
               !parse_double("--tol=", cfg.screening_tol) &&
               !parse_int("--box-size=", cfg.max_points_per_box) &&
               !parse_int("--screening=", cfg.screening)) {
      throw std::runtime_error("Unknown argument: " + arg + " (try --help)");
    }
  }
  if (cfg.n_max <= 0 || cfg.m_max <= 0)
    throw std::runtime_error("--n_max and --m_max must be positive");
  return cfg;
}
template <typename radial_type, typename angular_type>
CoreHamiltonianResult compute_reference(const Config cfg, size_t nrad,
                                        size_t ang_order) {
  Molecule mol = make_benzene();
  STOBasisSet stobasis = load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);

  FlatGrid grid =
      make_flat_grid<radial_type, angular_type>(mol, nrad, ang_order);

  CoreHamiltonianResult core;
  if (cfg.screening) {
    auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
    NeighborList nl;
    build_neighbor_list(stobasis, bb, cfg.max_points_per_box,
                        grid.quad_points.extent(0), nl);
    core = compute_core_hamiltonian_screened(stobasis, grid, nl);
  } else {
    core = compute_core_hamiltonian(stobasis, grid);
  }
  return core;
}

template <typename radial_type, typename angular_type, typename REC>
void convergence_analysis(const Config &cfg,
                          const CoreHamiltonianResult &ref_core,
                          const size_t &nrad, const size_t &ang_order,
                          REC recorder) {
  using angular_traits = IntegratorXX::quadrature_traits<angular_type>;

  Molecule mol = make_benzene();
  STOBasisSet stobasis = load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);

  FlatGrid grid =
      make_flat_grid<radial_type, angular_type>(mol, nrad, ang_order);

  int N = stobasis.nbf();

  CoreHamiltonianResult core = compute_core_hamiltonian(stobasis, grid);

  double max_error_overlap_diag = 0.0;
  double max_error_overlap = 0.0;
  double max_error_hamiltontian = 0.0;
  double max_error_kinetic = 0.0;
  double max_error_nuclear = 0.0;

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(int i, int j, double &lmax) {
        double err = Kokkos::abs(ref_core.kinetic(i, j) - core.kinetic(i, j));
        if (err > lmax)
          lmax = err;
      },
      Kokkos::Max<double>(max_error_kinetic));

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(int i, int j, double &lmax) {
        double err = Kokkos::abs(ref_core.nuclear(i, j) - core.nuclear(i, j));
        if (err > lmax)
          lmax = err;
      },
      Kokkos::Max<double>(max_error_nuclear));

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(int i, int j, double &lmax) {
        double err =
            Kokkos::abs(ref_core.hamiltonian(i, j) - core.hamiltonian(i, j));
        if (err > lmax)
          lmax = err;
      },
      Kokkos::Max<double>(max_error_hamiltontian));

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(int i, int j, double &lmax) {
        double err = Kokkos::abs(ref_core.overlap(i, j) - core.overlap(i, j));
        if (err > lmax)
          lmax = err;
      },
      Kokkos::Max<double>(max_error_overlap));

  Kokkos::parallel_reduce(
      N,
      KOKKOS_LAMBDA(int i, double &lmax) {
        double err = Kokkos::abs(1.0 - core.overlap(i, i));
        if (err > lmax)
          lmax = err;
      },
      Kokkos::Max<double>(max_error_overlap_diag));

  size_t dynamic_nang = angular_traits::npts_by_algebraic_order(
      angular_traits::next_algebraic_order(ang_order));

  // Pass dynamic_nang to the recorder instead of raw ang_order
  recorder(nrad, dynamic_nang,
           grid.quad_points.extent(0) / grid.atom_centers.extent(0),
           grid.quad_points.extent(0), max_error_overlap_diag,

           max_error_overlap, max_error_kinetic, max_error_nuclear,
           max_error_hamiltontian);
}

auto repeat(const std::string &s, int n) {
  std::string r;
  for (int i = 0; i < n; ++i)
    r += s;
  return r;
}
void print_config(const Config &cfg) {
  int width = std::max(cfg.basis_dir.size(), size_t(20));
  std::string h = repeat("─", width + 2);

  std::cout << "\n";
  std::cout << "┌───────────────────────" << h << "┐\n";
  std::cout << "│       Integral Convergence Config." << repeat(" ", width - 10)
            << "│\n";
  std::cout << "├───────────────────────" << h << "┤\n";
  std::cout << "│ Basis directory      │ " << std::setw(width) << cfg.basis_dir
            << " │\n";
  std::cout << "│ Radial points        │ " << std::setw(width)
            << std::pow(2, cfg.n_max) << " │\n";
  std::cout << "│ Angular order        │ " << std::setw(width) << cfg.m_max * 10
            << " │\n";
  std::cout << "│ Screening tolerance  │ " << std::setw(width)
            << cfg.screening_tol << " │\n";
  std::cout << "│ Max points per box   │ " << std::setw(width)
            << cfg.max_points_per_box << " │\n";
  std::cout << "│ Screening            │ " << std::setw(width) << cfg.screening
            << " │\n";
  std::cout << "└──────────────────────┴" << h << "┘\n\n";
  std::cout << std::flush;
}
int main(int argc, char *argv[]) {
  Config cfg;
  try {
    cfg = parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  Kokkos::initialize();
  {

    print_config(cfg);
    using namespace IntegratorXX;
    using radial_type = ta_type;
    using angular_type = ll_type;

    std::vector<double> errors_overlap_diag;
    std::vector<double> errors_overlap;
    std::vector<double> errors_hamiltonian;
    std::vector<double> errors_kinetic;
    std::vector<double> errors_nuclear;
    std::vector<size_t> rad_npts;
    std::vector<size_t> ang_npts;
    std::vector<size_t> atom_npts;
    std::vector<size_t> total_npts;

    auto recorder = [&](size_t rad, size_t ang, size_t atom, size_t total,
                        double error_diag, double error_overlap,
                        double error_kinetic, double error_nuclear,
                        double error_hamiltonian) {
      rad_npts.push_back(rad);
      ang_npts.push_back(ang); // Stores actual point count (e.g., 50, 110...)
      atom_npts.push_back(atom);
      total_npts.push_back(total);
      errors_overlap_diag.push_back(error_diag);
      errors_overlap.push_back(error_overlap);
      errors_kinetic.push_back(error_kinetic);
      errors_nuclear.push_back(error_nuclear);
      errors_hamiltonian.push_back(error_hamiltonian);

      std::cout << std::setw(10) << rad << std::setw(10) << ang << std::setw(15)
                << atom << std::setw(15) << total << std::setw(20) << error_diag
                << std::setw(20) << error_overlap << std::setw(20)
                << error_kinetic << std::setw(20) << error_nuclear
                << std::setw(20) << error_hamiltonian << std::endl;
    };

    int n_max = cfg.n_max;
    int m_max = cfg.m_max;
    int nrad_max = std::pow(2, n_max);
    int nang_order_max = m_max * 10;

    CoreHamiltonianResult ref_core =
        compute_reference<radial_type, angular_type>(cfg, nrad_max,
                                                     nang_order_max);

    std::cout << "\n--- Final Convergence for benzene ---\n";
    std::cout << std::setw(10) << "rad_pts" << std::setw(10) << "ang_pts"
              << std::setw(15) << "pts_per_atom" << std::setw(15) << "pts_total"
              << std::setw(20) << "err_overlap_diag" << std::setw(20)
              << "err_overlap" << std::setw(20) << "err_kinetic"
              << std::setw(20) << "err_nuclear" << std::setw(20)
              << "err_hamiltonian" << std::endl;

    for (unsigned n = 1; n < n_max; ++n) {
      for (unsigned m = 1; m < m_max; ++m) {
        size_t nrad = std::pow(2, n);
        size_t ang_deg = m * 10;
        convergence_analysis<radial_type, angular_type>(cfg, ref_core, nrad,
                                                        ang_deg, recorder);
      }
    }
  }

  Kokkos::finalize();
  return 0;
}
