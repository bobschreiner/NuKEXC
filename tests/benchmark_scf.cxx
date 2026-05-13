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

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>
#include <iomanip>
#include <iostream>

#include <catch2/catch_all.hpp>

#include <nukexc/diagonalizer.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/integration.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/partitioning.hpp>

#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include "nukexc/octree.hpp"
#include "nukexc/stobasis.hpp"
#include "standards.hpp"
#include <Kokkos_Core.hpp>

using namespace NuKEXC;

void print_timing_table(
    const std::vector<std::string> &molecules,
    const std::vector<std::unordered_map<std::string, double>> &timings) {
  // Header setup
  const int w = 15; // column width

  // print header
  std::cout << std::string(w * 7, '-') << std::endl;
  std::cout << std::left << std::setw(w) << "Molecule" << std::setw(w)
            << "Grid (s)" << std::setw(w) << "Overlap (s)" << std::setw(w)
            << "Kinetic (s)" << std::setw(w) << "Nuclear (s)" << std::setw(w)
            << "Diag (s)" << std::setw(w) << "Total (s)" << std::endl;
  std::cout << std::string(w * 7, '-') << std::endl;

  // Row data
  for (int i = 0; i < molecules.size(); ++i) {
    std::string mol_name = molecules[i];
    auto mol_timing = timings[i];
    std::cout << std::fixed << std::setprecision(4) << std::left << std::setw(w)
              << mol_name << std::setw(w) << mol_timing.at("grid")
              << std::setw(w) << mol_timing.at("overlap") << std::setw(w)
              << mol_timing.at("kinetic") << std::setw(w)
              << mol_timing.at("nuclear") << std::setw(w)
              << mol_timing.at("diag") << std::setw(w) << mol_timing.at("total")
              << std::endl;
  }
  std::cout << std::string(w * 7, '-') << std::endl;
}

void print_timing_table_fused(
    const std::vector<std::string> &molecules,
    const std::vector<std::unordered_map<std::string, double>> &timings) {
  const int w = 15;
  std::cout << std::string(w * 7, '-') << std::endl;
  std::cout << std::left << std::setw(w) << "Molecule" << std::setw(w)
            << "Grid (s)" << std::setw(w) << "H_el (s)" << std::setw(w)
            << "Diag (s)" << std::setw(w) << "Total (s)" << std::endl;
  std::cout << std::string(w * 7, '-') << std::endl;

  for (int i = 0; i < molecules.size(); ++i) {
    auto &t = timings[i];
    std::cout << std::fixed << std::setprecision(4) << std::left << std::setw(w)
              << molecules[i] << std::setw(w) << t.at("grid") << std::setw(w)
              << t.at("hamiltonian") << std::setw(w) << t.at("diag")
              << std::setw(w) << t.at("total") << std::endl;
  }

  std::cout << std::string(w * 7, '-') << std::endl;
}

void print_timing_table_screened(
    const std::vector<std::string> &molecules,
    const std::vector<std::unordered_map<std::string, double>> &timings) {
  const int w = 15;
  std::cout << std::string(w * 7, '-') << std::endl;
  std::cout << std::left << std::setw(w) << "Molecule" << std::setw(w)
            << "Grid (s)" << std::setw(w) << "Neighbors (s)" << std::setw(w)
            << "H_el (s)" << std::setw(w) << "Diag (s)" << std::setw(w)
            << "Total (s)" << std::endl;
  std::cout << std::string(w * 7, '-') << std::endl;

  for (int i = 0; i < molecules.size(); ++i) {
    auto &t = timings[i];
    std::cout << std::fixed << std::setprecision(4) << std::left << std::setw(w)
              << molecules[i] << std::setw(w) << t.at("grid") << std::setw(w)
              << t.at("neighborlist") << std::setw(w) << t.at("hamiltonian")
              << std::setw(w) << t.at("diag") << std::setw(w) << t.at("total")
              << std::endl;
  }

  std::cout << std::string(w * 7, '-') << std::endl;
}
TEST_CASE("Benchmark Core Hamiltonian Separate Kernels",
          "[benchmark_scf][separate kernels]") {

  using namespace IntegratorXX;

  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;

  using ll_type = IntegratorXX::LebedevLaikov<double>;

  std::vector<std::string> molecule_names;
  std::vector<Molecule> molecules;

  molecules.push_back(make_water());
  molecules.push_back(make_benzene());

  molecule_names.push_back("water");
  molecule_names.push_back("benzene");

#ifdef KOKKOS_ENABLE_HIP
  molecule_names.push_back("taxol");

  molecules.push_back(make_taxol());
#endif

  static std::vector<std::unordered_map<std::string, double>> timings;
  for (int mol_ind = 0; mol_ind < molecule_names.size(); ++mol_ind) {
    SECTION(

        molecule_names[mol_ind]) {
      std::unordered_map<std::string, double> mol_timing;
      // Generate molecule
      Molecule mol = molecules[mol_ind];
      int natoms = mol.natoms;
      STOBasisSet basis = load_adf_basis(mol);

      Kokkos::Timer grid_construction_timer, overlap_integral_timer,
          kinetic_integral_timer, nuclear_potential_timer, diag_timer,
          total_timer;
      total_timer.reset();

      // Generate the grid
      grid_construction_timer.reset();
      FlatGrid quadrature_grid = make_flat_grid<ta_type, ll_type>(mol);
      mol_timing["grid"] = grid_construction_timer.seconds();

      // Compute all integrals
      overlap_integral_timer.reset();
      auto S = overlap_integral(basis, quadrature_grid.quad_points,
                                quadrature_grid.weights);
      mol_timing["overlap"] = overlap_integral_timer.seconds();

      kinetic_integral_timer.reset();
      auto T = kinetic_integral(basis, quadrature_grid.quad_points,
                                quadrature_grid.weights);
      mol_timing["kinetic"] = kinetic_integral_timer.seconds();

      nuclear_potential_timer.reset();
      auto V = nuclear_potential_integral(
          basis, quadrature_grid.quad_points, quadrature_grid.weights,
          quadrature_grid.atom_centers, quadrature_grid.Z);
      mol_timing["nuclear"] = nuclear_potential_timer.seconds();

      const int N = S.extent(0);
      DeviceView2DLeft F("Fock Matrix", N, N);
      DeviceView2DLeft mo_coeff("Molecular Orbital Coefficients", N, N);
      DeviceView1D mo_energies("Molecular Orbital Energies", N);
      // Add all contribution to create the Hamiltonian
      Kokkos::parallel_for(
          "Create Fock Matrix",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
          KOKKOS_LAMBDA(const int &i, const int &j) {
            F(i, j) = T(i, j) + V(i, j);
          });

      // Diagonalize Hamiltonian
      diag_timer.reset();
      Diagonalizer diagonalizer(N);
      diagonalizer.compute_transformation(S);
      diagonalizer.solve(F, mo_coeff, mo_energies);
      mol_timing["diag"] = diag_timer.seconds();
      mol_timing["total"] = total_timer.seconds();

      timings.push_back(mol_timing);
    }
  }
  // Print only after all sections have been executed
  if (timings.size() == molecule_names.size()) {
    print_timing_table(molecule_names, timings);
    timings.clear(); // Clear so it's fresh if the test is run again in the same
                     // session
  }
}

TEST_CASE("Benchmark Fused Core Hamiltonian Fused", "[benchmark_scf][fused]") {

  using namespace IntegratorXX;
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  std::vector<std::string> molecule_names;
  std::vector<Molecule> molecules;

  molecules.push_back(make_water());
  molecules.push_back(make_benzene());
  molecule_names.push_back("water");
  molecule_names.push_back("benzene");

#ifdef KOKKOS_ENABLE_HIP
  molecule_names.push_back("taxol");
  molecules.push_back(make_taxol());
#endif

  static std::vector<std::unordered_map<std::string, double>> timings;

  for (int mol_ind = 0; mol_ind < molecule_names.size(); ++mol_ind) {
    SECTION(molecule_names[mol_ind]) {
      std::unordered_map<std::string, double> mol_timing;

      Molecule mol = molecules[mol_ind];
      STOBasisSet basis = load_adf_basis(mol);

      Kokkos::Timer grid_timer, hamiltonian_timer, diag_timer, total_timer;
      total_timer.reset();

      // Generate the grid
      grid_timer.reset();
      FlatGrid quadrature_grid = make_flat_grid<ta_type, ll_type>(mol);
      mol_timing["grid"] = grid_timer.seconds();

      // Compute all integrals in a single fused pass
      hamiltonian_timer.reset();
      auto result = compute_core_hamiltonian(basis, quadrature_grid);
      double hamiltonian_time = hamiltonian_timer.seconds();

      // Report individual integral times as N/A — they are now fused.
      // The total hamiltonian time is what matters for benchmarking.
      mol_timing["overlap"] = 0.0; // fused
      mol_timing["kinetic"] = 0.0; // fused
      mol_timing["nuclear"] = 0.0; // fused
      mol_timing["hamiltonian"] = hamiltonian_time;

      const int N = result.overlap.extent(0);
      DeviceView2DLeft mo_coeff("Molecular Orbital Coefficients", N, N);
      DeviceView1D mo_energies("Molecular Orbital Energies", N);

      // Diagonalize — F is already T + V_n from the fused result
      diag_timer.reset();
      Diagonalizer diagonalizer(N);
      diagonalizer.compute_transformation(result.overlap);
      diagonalizer.solve(result.hamiltonian, mo_coeff, mo_energies);
      mol_timing["diag"] = diag_timer.seconds();
      mol_timing["total"] = total_timer.seconds();

      timings.push_back(mol_timing);
    }
  }

  if (timings.size() == molecule_names.size()) {
    print_timing_table_fused(molecule_names, timings);
    timings.clear();
  }
}

TEST_CASE("Benchmark Fused Core Hamiltonian Screened",
          "[benchmark_scf][screened]") {

  using namespace IntegratorXX;
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  std::vector<std::string> molecule_names;
  std::vector<Molecule> molecules;

  molecules.push_back(make_water());
  molecules.push_back(make_benzene());
  molecule_names.push_back("water");
  molecule_names.push_back("benzene");


  molecule_names.push_back("taxol");
  molecules.push_back(make_taxol());

  static std::vector<std::unordered_map<std::string, double>> timings;

  for (int mol_ind = 0; mol_ind < molecule_names.size(); ++mol_ind) {
    SECTION(molecule_names[mol_ind]) {
      std::unordered_map<std::string, double> mol_timing;

      Molecule mol = molecules[mol_ind];

      double screening_tol = 1e-6;
      STOBasisSet basis =
          load_adf_basis(mol, "input/zorabasis/TZP", screening_tol);

      Kokkos::Timer grid_timer, nl_timer, hamiltonian_timer, diag_timer,
          total_timer;
      total_timer.reset();

      // Generate the grid
      grid_timer.reset();
      FlatGrid quadrature_grid = make_flat_grid<ta_type, ll_type>(mol);
      mol_timing["grid"] = grid_timer.seconds();

      // Generate the grid
      nl_timer.reset();
      const int max_points_per_box = 64;
      auto bb = create_bounding_boxes(quadrature_grid, max_points_per_box);
      NeighborList nl;
      build_neighbor_list(basis, bb, max_points_per_box,
                          quadrature_grid.quad_points.extent(0), nl);

      mol_timing["neighborlist"] = nl_timer.seconds();

      // Compute all integrals in a single fused pass
      hamiltonian_timer.reset();
      auto result =
          compute_core_hamiltonian_screened(basis, quadrature_grid, nl);
      double hamiltonian_time = hamiltonian_timer.seconds();

      // Report individual integral times as N/A — they are now fused.
      // The total hamiltonian time is what matters for benchmarking.
      mol_timing["overlap"] = 0.0; // fused
      mol_timing["kinetic"] = 0.0; // fused
      mol_timing["nuclear"] = 0.0; // fused
      mol_timing["hamiltonian"] = hamiltonian_time;

      const int N = result.overlap.extent(0);
      DeviceView2DLeft mo_coeff("Molecular Orbital Coefficients", N, N);
      DeviceView1D mo_energies("Molecular Orbital Energies", N);

      // Diagonalize — F is already T + V_n from the fused result
      diag_timer.reset();
      Diagonalizer diagonalizer(N);
      diagonalizer.compute_transformation(result.overlap);
      diagonalizer.solve(result.hamiltonian, mo_coeff, mo_energies);
      mol_timing["diag"] = diag_timer.seconds();
      mol_timing["total"] = total_timer.seconds();

      timings.push_back(mol_timing);
    }
  }

  if (timings.size() == molecule_names.size()) {
    print_timing_table_screened(molecule_names, timings);
    timings.clear();
  }
}

int main() {

  Kokkos::initialize();
  int result = Catch::Session().run();
  Kokkos::finalize();

  return result;
}
