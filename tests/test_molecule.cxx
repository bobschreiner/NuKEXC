/*
 *    NuKEXC Numerical Kokkos Enhanced Exchange Correlation Integrator
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

#include "../src/molecule.hpp"
#include "../src/molecule_read.hpp"

int main() {
  Kokkos::initialize();
  {
    try {
      NuKEXC::Molecule mol(3);
      NuKEXC::read_xyz("input/water.xyz", mol);
      std::cout << "Loaded " << mol.natoms << " atoms." << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Error: " << e.what() << std::endl;
    }
  }
  Kokkos::finalize();
  return 0;
}
