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
 */

#include <Kokkos_Core.hpp>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>

#include <nukexc/diagonalizer.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/integration.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/stobasis.hpp>

#include "standards.hpp"

using namespace NuKEXC;

// ── Config
// ────────────────────────────────────────────────────────────────────

struct Config {
  std::string basis_dir = "input/zorabasis/TZP";
  int nrad = 100;
  int nang = 50;
  double screening_tol = 1e-6;
  int max_points_per_box = 64;
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
          << "  --nrad=<int>           Radial grid points        (default: "
          << cfg.nrad << ")\n"
          << "  --nang=<int>           Angular grid points       (default: "
          << cfg.nang << ")\n"
          << "  --tol=<float>          Screening tolerance       (default: "
          << cfg.screening_tol << ")\n"
          << "  --box-size=<int>       Max points per box        (default: "
          << cfg.max_points_per_box << ")\n";
      std::exit(0);
    } else if (!parse_string("--basis=", cfg.basis_dir) &&
               !parse_int("--nrad=", cfg.nrad) &&
               !parse_int("--nang=", cfg.nang) &&
               !parse_double("--tol=", cfg.screening_tol) &&
               !parse_int("--box-size=", cfg.max_points_per_box)) {
      throw std::runtime_error("Unknown argument: " + arg + " (try --help)");
    }
  }
  if (cfg.nrad <= 0 || cfg.nang <= 0)
    throw std::runtime_error("--nrad and --nang must be positive");
  return cfg;
}

// ── Helpers
// ───────────────────────────────────────────────────────────────────

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
  std::cout << "│    SCF Benchmark Configuration" << repeat(" ", width - 6)
            << "│\n";
  std::cout << "├───────────────────────" << h << "┤\n";
  std::cout << "│ Basis directory      │ " << std::setw(width) << cfg.basis_dir
            << " │\n";
  std::cout << "│ Radial points        │ " << std::setw(width) << cfg.nrad
            << " │\n";
  std::cout << "│ Angular order        │ " << std::setw(width) << cfg.nang
            << " │\n";
  std::cout << "│ Screening tolerance  │ " << std::setw(width)
            << cfg.screening_tol << " │\n";
  std::cout << "│ Max points per box   │ " << std::setw(width)
            << cfg.max_points_per_box << " │\n";
  std::cout << "└──────────────────────┴" << h << "┘\n\n";
  std::cout << std::flush;
}

struct BenchmarkResult {
  std::string molecule;
  int nbf;
  int grid_points;
  double t_grid;
  double t_neighbors; // 0 if not applicable
  double t_hamiltonian;
  double t_diag;
  double t_total;
};

void print_results(const std::vector<BenchmarkResult> &results, bool screened) {
  const int w = 14;
  const int wm = 12;

  auto hline_top = [&](const std::string &left, const std::string &mid,
                       const std::string &right, const std::string &fill) {
    std::cout << left;
    std::cout << repeat(fill, wm + 2) << mid;
    std::cout << repeat(fill, 6 + 2) << mid;
    std::cout << repeat(fill, 8 + 2) << mid;
    if (screened)
      std::cout << repeat(fill, w + 2) << mid;
    std::cout << repeat(fill, w + 2) << mid;
    std::cout << repeat(fill, w + 2) << mid;
    std::cout << repeat(fill, w + 2) << right << "\n";
  };

  hline_top("┌", "┬", "┐", "─");
  std::cout << "│ " << std::setw(wm) << std::left << "Molecule"
            << " │ " << std::setw(6) << "N"
            << " │ " << std::setw(8) << "Grid pts";
  if (screened)
    std::cout << " │ " << std::setw(w) << "Neighbors (s)";
  std::cout << " │ " << std::setw(w) << "H_el (s)"
            << " │ " << std::setw(w) << "Diag (s)"
            << " │ " << std::setw(w) << "Total (s)"
            << " │\n";
  hline_top("├", "┼", "┤", "─");

  for (auto &r : results) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "│ " << std::setw(wm) << std::left << r.molecule << " │ "
              << std::setw(6) << r.nbf << " │ " << std::setw(8)
              << r.grid_points;
    if (screened)
      std::cout << " │ " << std::setw(w) << r.t_neighbors;
    std::cout << " │ " << std::setw(w) << r.t_hamiltonian << " │ "
              << std::setw(w) << r.t_diag << " │ " << std::setw(w) << r.t_total
              << " │\n";
  }
  hline_top("└", "┴", "┘", "─");
  std::cout << std::flush;
}

// ── Molecules
// ─────────────────────────────────────────────────────────────────

std::vector<std::pair<std::string, Molecule>> make_molecules() {
  std::vector<std::pair<std::string, Molecule>> mol_list;
  mol_list.push_back({"water", make_water()});
  mol_list.push_back({"benzene", make_benzene()});
  mol_list.push_back({"taxol", make_taxol()});
  return mol_list;
}

// ── Benchmarks
// ────────────────────────────────────────────────────────────────

void run_benchmark_fused(const Config &cfg) {
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  std::cout << "\n── Fused Core Hamiltonian ──\n";

  std::vector<BenchmarkResult> results;
  for (auto &[name, mol] : make_molecules()) {
    STOBasisSet basis = load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);

    Kokkos::Timer total_timer, grid_timer, hamiltonian_timer, diag_timer;
    total_timer.reset();

    grid_timer.reset();
    FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad, cfg.nang);
    double t_grid = grid_timer.seconds();

    hamiltonian_timer.reset();
    auto hcore = compute_core_hamiltonian(basis, grid);
    double t_hamiltonian = hamiltonian_timer.seconds();

    const int N = hcore.overlap.extent(0);
    DeviceView2DLeft mo_coeff("mo_coeff", N, N);
    DeviceView1D mo_energies("mo_energies", N);

    diag_timer.reset();
    Diagonalizer diag(N);
    diag.compute_transformation(hcore.overlap);
    diag.solve(hcore.hamiltonian, mo_coeff, mo_energies);
    double t_diag = diag_timer.seconds();

    results.push_back({name, N, (int)grid.quad_points.extent(0), t_grid, 0.0,
                       t_hamiltonian, t_diag, total_timer.seconds()});
  }
  print_results(results, false);
}

void run_benchmark_screened(const Config &cfg) {
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  std::cout << "\n── Screened Core Hamiltonian ──\n";

  std::vector<BenchmarkResult> results;
  for (auto &[name, mol] : make_molecules()) {
    STOBasisSet basis = load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);

    Kokkos::Timer total_timer, grid_timer, nl_timer, hamiltonian_timer,
        diag_timer;
    total_timer.reset();

    grid_timer.reset();
    FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad, cfg.nang);
    double t_grid = grid_timer.seconds();

    nl_timer.reset();
    auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
    NeighborList nl;
    build_neighbor_list(basis, bb, cfg.max_points_per_box,
                        grid.quad_points.extent(0), nl);
    double t_neighbors = nl_timer.seconds();

    hamiltonian_timer.reset();
    auto hcore = compute_core_hamiltonian_screened_scratch(basis, grid, nl);

    double t_hamiltonian = hamiltonian_timer.seconds();

    const int N = hcore.overlap.extent(0);
    DeviceView2DLeft mo_coeff("mo_coeff", N, N);
    DeviceView1D mo_energies("mo_energies", N);

    diag_timer.reset();
    Diagonalizer diag(N);
    diag.compute_transformation(hcore.overlap);
    diag.solve(hcore.hamiltonian, mo_coeff, mo_energies);
    double t_diag = diag_timer.seconds();

    results.push_back({name, N, (int)grid.quad_points.extent(0), t_grid,
                       t_neighbors, t_hamiltonian, t_diag,
                       total_timer.seconds()});
  }
  print_results(results, true);
}

void run_benchmark_screened_and_tiled(const Config &cfg) {
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  std::cout << "\n── Screened Core Hamiltonian ──\n";

  std::vector<BenchmarkResult> results;
  for (auto &[name, mol] : make_molecules()) {
    STOBasisSet basis = load_adf_basis(mol, cfg.basis_dir, cfg.screening_tol);

    Kokkos::Timer total_timer, grid_timer, nl_timer, hamiltonian_timer,
        diag_timer;
    total_timer.reset();

    grid_timer.reset();
    FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, cfg.nrad, cfg.nang);
    double t_grid = grid_timer.seconds();

    nl_timer.reset();
    auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
    NeighborList nl;
    build_neighbor_list(basis, bb, cfg.max_points_per_box,
                        grid.quad_points.extent(0), nl);
    double t_neighbors = nl_timer.seconds();

    hamiltonian_timer.reset();
#if defined(KOKKOS_ENABLE_HIP)
    auto hcore =
        compute_core_hamiltonian_screened_and_tiled<8>(basis, grid, nl);
#else
    auto hcore =
        compute_core_hamiltonian_screened_and_tiled<8>(basis, grid, nl);
#endif

    double t_hamiltonian = hamiltonian_timer.seconds();

    const int N = hcore.overlap.extent(0);
    DeviceView2DLeft mo_coeff("mo_coeff", N, N);
    DeviceView1D mo_energies("mo_energies", N);

    diag_timer.reset();
    Diagonalizer diag(N);
    diag.compute_transformation(hcore.overlap);
    diag.solve(hcore.hamiltonian, mo_coeff, mo_energies);
    double t_diag = diag_timer.seconds();

    results.push_back({name, N, (int)grid.quad_points.extent(0), t_grid,
                       t_neighbors, t_hamiltonian, t_diag,
                       total_timer.seconds()});
  }
  print_results(results, true);
}

// ── Main
// ──────────────────────────────────────────────────────────────────────

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
    run_benchmark_fused(cfg);
    run_benchmark_screened(cfg);
    // run_benchmark_screened_and_tiled(cfg);
    std::cout << "\n";
  }
  Kokkos::finalize();
  return 0;
}
