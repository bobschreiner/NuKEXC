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

#include <catch2/catch_all.hpp>
#include <catch2/catch_assertion_info.hpp>

TEST_CASE("H20", "[h20_molecule]") {
  Nukexc::Molecule mol;
  Nukexc::read_xyz("input/water.xyz", mol);
  std::cout << "Loaded " << mol.natoms << " atoms." << std::endl;
  REQUIRE(mol.natoms == 3);
};

int main() {
  Kokkos::initialize();
  {
    int result = Catch::Session().run();
  }
  Kokkos::finalize();
  return 0;
}
