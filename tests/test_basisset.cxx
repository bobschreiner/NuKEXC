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

#include <iostream>

#include <nukexc/molecule.hpp>
#include <nukexc/stobasis.hpp>

#include <catch2/catch_all.hpp>
#include <catch2/catch_assertion_info.hpp>

using namespace NuKEXC;

TEST_CASE("h20_STO", "[h20_sto]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = load_sto_basis(mol, "input/k99light/neutral");

  // On CPU we can print out the basis set
  // Copy to host device
  auto n_h = Kokkos::create_mirror_view(basis.n_);
  auto l_h = Kokkos::create_mirror_view(basis.l_);
  auto m_h = Kokkos::create_mirror_view(basis.m_);
  auto norm_h = Kokkos::create_mirror_view(basis.norm_);
  auto zeta_h = Kokkos::create_mirror_view(basis.zeta_);
  auto O_h = Kokkos::create_mirror_view(basis.O_);

  Kokkos::deep_copy(n_h, basis.n_);
  Kokkos::deep_copy(l_h, basis.l_);
  Kokkos::deep_copy(m_h, basis.m_);
  Kokkos::deep_copy(zeta_h, basis.zeta_);
  Kokkos::deep_copy(norm_h, basis.norm_);
  Kokkos::deep_copy(O_h, basis.O_);

  for (int i = 0; i < basis.nbf(); ++i) {
    std::cout << "Basis function " << i << std::endl;
    std::cout << "n " << n_h(i) << std::endl;
    std::cout << "l " << l_h(i) << std::endl;
    std::cout << "m " << m_h(i) << std::endl;
    std::cout << "zeta " << zeta_h(i) << std::endl;
    std::cout << "coeff " << norm_h(i) << std::endl;
    std::cout << "O_h " << O_h(i, 0) << " " << O_h(i, 1) << " " << O_h(i, 2)
              << " " << std::endl
              << std::endl;
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
