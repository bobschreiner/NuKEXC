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

#include "nukexc/grid.hpp"
#include <Kokkos_Core.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <impl/Kokkos_CheckUsage.hpp>
#include <iostream>

#include <nukexc/molecule.hpp>
#include <nukexc/stobasis.hpp>

#include <catch2/catch_all.hpp>
#include <catch2/catch_assertion_info.hpp>

using namespace NuKEXC;

TEST_CASE("H2O_thakkar", "[h20_thakkar]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = load_thakkar_basis(mol, "input/k99light/neutral");

  // On CPU we can print out the basis set
  // Copy to host device
  auto n_h = Kokkos::create_mirror_view(basis.n);
  auto l_h = Kokkos::create_mirror_view(basis.l);
  auto m_h = Kokkos::create_mirror_view(basis.m);
  auto norm_h = Kokkos::create_mirror_view(basis.norm);
  auto zeta_h = Kokkos::create_mirror_view(basis.zeta);
  auto O_h = Kokkos::create_mirror_view(basis.O);
  auto cutoff_h = Kokkos::create_mirror_view(basis.cutoff_radii);

  Kokkos::deep_copy(n_h, basis.n);
  Kokkos::deep_copy(l_h, basis.l);
  Kokkos::deep_copy(m_h, basis.m);
  Kokkos::deep_copy(zeta_h, basis.zeta);
  Kokkos::deep_copy(norm_h, basis.norm);
  Kokkos::deep_copy(O_h, basis.O);
  Kokkos::deep_copy(cutoff_h, basis.cutoff_radii);

  std::cout << "Thakkar Basis" << std::endl;
  for (int i = 0; i < basis.nbf(); ++i) {
    std::cout << "Basis function " << i << std::endl;
    std::cout << "n " << n_h(i) << std::endl;
    std::cout << "l " << l_h(i) << std::endl;
    std::cout << "m " << m_h(i) << std::endl;
    std::cout << "zeta " << zeta_h(i) << std::endl;
    std::cout << "norm " << norm_h(i) << std::endl;
    std::cout << "cutoff " << cutoff_h(i) << std::endl;
    std::cout << "O_h " << O_h(i, 0) << " " << O_h(i, 1) << " " << O_h(i, 2)
              << " " << std::endl
              << std::endl;
  }
  std::cout << "--------------------------------------------------"
            << std::endl;
};

TEST_CASE("H2O_adf_regular", "[h20][adf]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = load_adf_basis(mol);

  // On CPU we can print out the basis set
  // Copy to host device
  auto n_h = Kokkos::create_mirror_view(basis.n);
  auto l_h = Kokkos::create_mirror_view(basis.l);
  auto m_h = Kokkos::create_mirror_view(basis.m);
  auto norm_h = Kokkos::create_mirror_view(basis.norm);
  auto zeta_h = Kokkos::create_mirror_view(basis.zeta);
  auto O_h = Kokkos::create_mirror_view(basis.O);
  auto cutoff_h = Kokkos::create_mirror_view(basis.cutoff_radii);

  Kokkos::deep_copy(n_h, basis.n);
  Kokkos::deep_copy(l_h, basis.l);
  Kokkos::deep_copy(m_h, basis.m);
  Kokkos::deep_copy(zeta_h, basis.zeta);
  Kokkos::deep_copy(norm_h, basis.norm);
  Kokkos::deep_copy(O_h, basis.O);
  Kokkos::deep_copy(cutoff_h, basis.cutoff_radii);

  std::cout << "ADF TZP" << std::endl;
  for (int i = 0; i < basis.nbf(); ++i) {
    std::cout << "Basis function " << i << std::endl;
    std::cout << "n " << n_h(i) << std::endl;
    std::cout << "l " << l_h(i) << std::endl;
    std::cout << "m " << m_h(i) << std::endl;
    std::cout << "zeta " << zeta_h(i) << std::endl;
    std::cout << "norm " << norm_h(i) << std::endl;
    std::cout << "cutoff " << cutoff_h(i) << std::endl;
    std::cout << "O_h " << O_h(i, 0) << " " << O_h(i, 1) << " " << O_h(i, 2)
              << " " << std::endl
              << std::endl;
  }
  std::cout << "--------------------------------------------------"
            << std::endl;
};

TEST_CASE("H2O_adf_QZ4P", "[h20][adf]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P");

  // On CPU we can print out the basis set
  // Copy to host device
  auto n_h = Kokkos::create_mirror_view(basis.n);
  auto l_h = Kokkos::create_mirror_view(basis.l);
  auto m_h = Kokkos::create_mirror_view(basis.m);
  auto norm_h = Kokkos::create_mirror_view(basis.norm);
  auto zeta_h = Kokkos::create_mirror_view(basis.zeta);
  auto O_h = Kokkos::create_mirror_view(basis.O);
  auto cutoff_h = Kokkos::create_mirror_view(basis.cutoff_radii);

  Kokkos::deep_copy(n_h, basis.n);
  Kokkos::deep_copy(l_h, basis.l);
  Kokkos::deep_copy(m_h, basis.m);
  Kokkos::deep_copy(zeta_h, basis.zeta);
  Kokkos::deep_copy(norm_h, basis.norm);
  Kokkos::deep_copy(O_h, basis.O);
  Kokkos::deep_copy(cutoff_h, basis.cutoff_radii);

  std::cout << "ADF QZ4P" << std::endl;

  for (int i = 0; i < basis.nbf(); ++i) {
    std::cout << "Basis function " << i << std::endl;
    std::cout << "n " << n_h(i) << std::endl;
    std::cout << "l " << l_h(i) << std::endl;
    std::cout << "m " << m_h(i) << std::endl;
    std::cout << "zeta " << zeta_h(i) << std::endl;
    std::cout << "norm " << norm_h(i) << std::endl;
    std::cout << "cutoff " << cutoff_h(i) << std::endl;
    std::cout << "O_h " << O_h(i, 0) << " " << O_h(i, 1) << " " << O_h(i, 2)
              << " " << std::endl
              << std::endl;
  }
  std::cout << "--------------------------------------------------"
            << std::endl;
};

TEST_CASE("Basis Cutoff", "[h20][cutoff]") {

  using bk_type = IntegratorXX::Becke<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  double cutoff_tol = 1e-10;
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", cutoff_tol);

  FlatGrid grid = make_flat_grid<bk_type, ll_type>(mol);

  int N = basis.nbf();
  int G = grid.quad_points.extent(0);

  Kokkos::View<double **> grid_values("Values", N, G);
  Kokkos::View<double **> grid_r("Radii", N, G);

  Kokkos::parallel_for(
      "Evaluate basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, G}),
      KOKKOS_LAMBDA(const int i, const int g) {
        double x = basis.O(i, 0) - grid.quad_points(g, 0);
        double y = basis.O(i, 1) - grid.quad_points(g, 1);
        double z = basis.O(i, 2) - grid.quad_points(g, 2);
        double r = Kokkos::sqrt(x * x + y * y + z * z);
        grid_values(i, g) =
            basis_eval(basis, i, grid.quad_points(g, 0), grid.quad_points(g, 1),
                       grid.quad_points(g, 2));
        grid_r(i, g) = r;
      });

  auto grid_values_h = Kokkos::create_mirror_view(grid_values);
  auto grid_r_h = Kokkos::create_mirror_view(grid_r);
  auto cutoff_radii_h = Kokkos::create_mirror_view(basis.cutoff_radii);

  Kokkos::deep_copy(grid_values_h, grid_values);
  Kokkos::deep_copy(grid_r_h, grid_r);
  Kokkos::deep_copy(cutoff_radii_h, basis.cutoff_radii);

  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int g = 0; g < G; ++g) {
      if (cutoff_radii_h(i) < grid_r_h(i, g)) {
        REQUIRE(grid_values_h(i, g) < cutoff_tol);
      }
    }
  }
};

int main() {
  Kokkos::initialize();
  {
    int result = Catch::Session().run();
  }
  Kokkos::finalize();
  return 0;
}
