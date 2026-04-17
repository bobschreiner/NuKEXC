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

#include "atomic_properties.hpp"
#include "molecule.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace NuKEXC {

void read_xyz(const std::string &filename, Molecule &mol) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  unsigned natoms;
  if (!(file >> natoms))
    return;

  // Skip the comment line
  std::string dummy;
  std::getline(file, dummy); // consume newline after natoms
  std::getline(file, dummy); // consume comment line

  std::vector<std::vector<double>> centers_v;
  std::vector<unsigned> Z_v;

  std::string symbol;
  double x, y, z;
  while (file >> symbol >> x >> y >> z) {
    Z_v.push_back(detail::get_atomic_number(symbol));
    centers_v.push_back({x * detail::ang_to_bohr, y * detail::ang_to_bohr,
                         z * detail::ang_to_bohr});
  }

  Molecule mol_tmp(centers_v, Z_v);
  mol.atom_centers = mol_tmp.atom_centers; // shallow copies
  mol.Z = mol_tmp.Z;                       // shallow copies
  mol.natoms = mol_tmp.natoms;
}

} // namespace NuKEXC
