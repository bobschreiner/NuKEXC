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
// #include <iostream>
#include <catch2/matchers/catch_matchers.hpp>
#include <string_view>

#include <catch2/catch_all.hpp>

#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include <nukexc/grid.hpp>
#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp>

#include "nukexc/nukexc_config.hpp"
#include "nukexc/octree.hpp"
#include "standards.hpp"

using namespace Nukexc;

using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

TEST_CASE("H20", "[h20_weights]") {

  // Generate water
  Molecule mol = make_water();
  STOBasisSet stobasis = load_adf_basis(mol);
  FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol);

  DeviceView2DLeft S =
      overlap_integral(stobasis, grid.quad_points, grid.weights);

  auto S_h = Kokkos::create_mirror_view(S);
  Kokkos::deep_copy(S_h, S);

  for (unsigned i = 0; i < S_h.extent(0); ++i) {
    REQUIRE_THAT(S_h(i, i), Catch::Matchers::WithinRel(1.0, 1e-5));
  }
}

TEST_CASE("Compute core Hamiltonian with screening",
          "[h2o][screening][coreH]") {
  using radial_type = ta_type;
  using angular_type = ll_type;

  Molecule mol = make_water();
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", 1e-10);
  FlatGrid grid = make_flat_grid<radial_type, angular_type>(mol, 50, 30);
  FlatGrid grid_ref = make_flat_grid<radial_type, angular_type>(mol, 50, 30);

  // Unscreened reference
  CoreHamiltonianResult Hcore_ref = compute_core_hamiltonian(basis, grid_ref);

  // Create bounding boxes for screening
  const int total_points = grid.quad_points.extent(0);
  const int max_points_per_box = 512;

  Kokkos::View<Box *, ExecSpace> bounding_boxes =
      create_bounding_boxes(grid, max_points_per_box);

  // Create NeighborList, by screening bounding boxes
  NeighborList nl;
  build_neighbor_list(basis, bounding_boxes, max_points_per_box, total_points,
                      nl);

  // Compute screened Hamiltonian
  CoreHamiltonianResult Hcore_scr =
      compute_core_hamiltonian_screened_scratch(basis, grid, nl);

  int N = basis.nbf();
  auto S_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_ref.overlap);
  auto S_scr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_scr.overlap);
  auto H_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_ref.hamiltonian);
  auto H_scr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_scr.hamiltonian);

  const double tol = 1e-6;
  for (int i = 0; i < N; ++i) {
    // Diagonal of overlap should be ~1 (orthonormal basis)
    REQUIRE_THAT(S_scr(i, i), Catch::Matchers::WithinRel(1.0, 1e-5));
    for (int j = 0; j < N; ++j) {
      REQUIRE_THAT(S_scr(i, j), Catch::Matchers::WithinAbs(S_ref(i, j), tol));
      REQUIRE_THAT(H_scr(i, j), Catch::Matchers::WithinAbs(H_ref(i, j), tol));
    }
  }
}

TEST_CASE("Compute core Hamiltonian with screening (basis on the fly)",
          "[h2o][screening][on the fly basis][coreH]") {
  using radial_type = ta_type;
  using angular_type = ll_type;

  Molecule mol = make_water();
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", 1e-10);
  FlatGrid grid = make_flat_grid<radial_type, angular_type>(mol, 50, 30);
  FlatGrid grid_ref = make_flat_grid<radial_type, angular_type>(mol, 50, 30);

  // Unscreened reference
  CoreHamiltonianResult Hcore_ref = compute_core_hamiltonian(basis, grid_ref);

  // Create bounding boxes for screening
  const int total_points = grid.quad_points.extent(0);
  const int max_points_per_box = 512;

  Kokkos::View<Box *, ExecSpace> bounding_boxes =
      create_bounding_boxes(grid, max_points_per_box);

  // Create NeighborList, by screening bounding boxes
  NeighborList nl;
  build_neighbor_list(basis, bounding_boxes, max_points_per_box, total_points,
                      nl);

  // Compute screened Hamiltonian
  CoreHamiltonianResult Hcore_scr =
      compute_core_hamiltonian_screened(basis, grid, nl);

  int N = basis.nbf();
  auto S_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_ref.overlap);
  auto S_scr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_scr.overlap);
  auto H_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_ref.hamiltonian);
  auto H_scr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_scr.hamiltonian);

  const double tol = 1e-6;
  for (int i = 0; i < N; ++i) {
    // Diagonal of overlap should be ~1 (orthonormal basis)
    REQUIRE_THAT(S_scr(i, i), Catch::Matchers::WithinRel(1.0, 1e-5));
    for (int j = 0; j < N; ++j) {
      REQUIRE_THAT(S_scr(i, j), Catch::Matchers::WithinAbs(S_ref(i, j), tol));
      REQUIRE_THAT(H_scr(i, j), Catch::Matchers::WithinAbs(H_ref(i, j), tol));
    }
  }
}

TEST_CASE("Compute core Hamiltonian with screening (basis tiled)",
          "[h2o][screening][tiled][coreH]") {
  using radial_type = ta_type;
  using angular_type = ll_type;

  Molecule mol = make_water();
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", 1e-10);
  FlatGrid grid = make_flat_grid<radial_type, angular_type>(mol, 50, 30);
  FlatGrid grid_ref = make_flat_grid<radial_type, angular_type>(mol, 50, 30);

  // Unscreened reference
  CoreHamiltonianResult Hcore_ref = compute_core_hamiltonian(basis, grid_ref);

  // Create bounding boxes for screening
  const int total_points = grid.quad_points.extent(0);
  const int max_points_per_box = 8;

  Kokkos::View<Box *, ExecSpace> bounding_boxes =
      create_bounding_boxes(grid, max_points_per_box);

  // Create NeighborList, by screening bounding boxes
  NeighborList nl;
  build_neighbor_list(basis, bounding_boxes, max_points_per_box, total_points,
                      nl);

  // Compute screened Hamiltonian
  CoreHamiltonianResult Hcore_scr =
      compute_core_hamiltonian_screened_tiled(basis, grid, nl);

  int N = basis.nbf();
  auto S_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_ref.overlap);
  auto S_scr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_scr.overlap);
  auto T_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_ref.kinetic);
  auto T_scr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_scr.kinetic);

  auto V_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_ref.nuclear);
  auto V_scr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_scr.nuclear);

  auto H_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_ref.hamiltonian);
  auto H_scr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_scr.hamiltonian);

  const double tol = 1e-6;
  for (int i = 0; i < N; ++i) {
    // Diagonal of overlap should be ~1 (orthonormal basis)
    REQUIRE_THAT(S_scr(i, i), Catch::Matchers::WithinRel(1.0, 1e-5));
    for (int j = 0; j < N; ++j) {
      REQUIRE_THAT(S_scr(i, j), Catch::Matchers::WithinAbs(S_ref(i, j), tol));
      REQUIRE_THAT(T_scr(i, j), Catch::Matchers::WithinAbs(T_ref(i, j), tol));
      REQUIRE_THAT(V_scr(i, j), Catch::Matchers::WithinAbs(V_ref(i, j), tol));
      REQUIRE_THAT(H_scr(i, j), Catch::Matchers::WithinAbs(H_ref(i, j), tol));
    }
  }
}

TEST_CASE("Compute core Hamiltonian with screening (basis sparse)",
          "[h2o][screening][sparse][coreH]") {
  using radial_type = ta_type;
  using angular_type = ll_type;

  Molecule mol = make_water();
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", 1e-10);
  FlatGrid grid = make_flat_grid<radial_type, angular_type>(mol, 50, 30);
  FlatGrid grid_ref = make_flat_grid<radial_type, angular_type>(mol, 50, 30);

  // Unscreened reference
  CoreHamiltonianResult Hcore_ref = compute_core_hamiltonian(basis, grid_ref);

  // Create bounding boxes for screening
  const int total_points = grid.quad_points.extent(0);
  const int max_points_per_box = 8;

  Kokkos::View<Box *, ExecSpace> bounding_boxes =
      create_bounding_boxes(grid, max_points_per_box);

  // Create NeighborList, by screening bounding boxes
  NeighborList nl;
  build_neighbor_list(basis, bounding_boxes, max_points_per_box, total_points,
                      nl);

  // Compute screened Hamiltonian
  CoreHamiltonianResult Hcore_scr =
      compute_core_hamiltonian_screened_sparse(basis, grid, nl);

  int N = basis.nbf();
  auto S_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_ref.overlap);
  auto S_scr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_scr.overlap);
  auto T_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_ref.kinetic);
  auto T_scr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_scr.kinetic);

  auto V_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_ref.nuclear);
  auto V_scr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_scr.nuclear);

  auto H_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_ref.hamiltonian);
  auto H_scr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                   Hcore_scr.hamiltonian);

  const double tol = 1e-6;
  for (int i = 0; i < N; ++i) {
    // Diagonal of overlap should be ~1 (orthonormal basis)
    REQUIRE_THAT(S_scr(i, i), Catch::Matchers::WithinRel(1.0, 1e-5));
    for (int j = 0; j < N; ++j) {
      REQUIRE_THAT(S_scr(i, j), Catch::Matchers::WithinAbs(S_ref(i, j), tol));
      REQUIRE_THAT(T_scr(i, j), Catch::Matchers::WithinAbs(T_ref(i, j), tol));
      REQUIRE_THAT(V_scr(i, j), Catch::Matchers::WithinAbs(V_ref(i, j), tol));
      REQUIRE_THAT(H_scr(i, j), Catch::Matchers::WithinAbs(H_ref(i, j), tol));
    }
  }
}
int main() {

  Kokkos::initialize();
  int result = Catch::Session().run();
  Kokkos::finalize();

  return result;
}
