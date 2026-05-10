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

#pragma once

#include "atomic_properties.hpp"
#include "nukexc_config.hpp"
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#
namespace NuKEXC {

struct Molecule {

  Kokkos::View<Point *, Kokkos::HostSpace>
      atom_centers; // Atom centers in cartesian coordinates (bohr)
  Kokkos::View<unsigned *, Kokkos::HostSpace> Z; // atomic numbers
  unsigned natoms; // number of atoms in the molecule
  std::set<unsigned>
      element_list; // contains a list of all elements present in the list

  /**
   * @ brief Default constructor
   */
  Molecule() = default;

  /**
   * @ brief Constructs Molecule from std::vector
   */
  Molecule(const std::vector<std::vector<double>> &atom_centers_v,
           const std::vector<unsigned> &Z_v) {

    // Initialize datastructures
    natoms = Z_v.size();
    atom_centers =
        Kokkos::View<Point *, Kokkos::HostSpace>("Atom centers", natoms);
    Z = Kokkos::View<unsigned *, Kokkos::HostSpace>("Atomic numbers ", natoms);

    element_list = std::set<unsigned>(Z_v.begin(), Z_v.end());

    // Fill Kokkos::View with data
    for (size_t i = 0; i < natoms; ++i) {
      atom_centers(i)[0] = atom_centers_v[i][0];
      atom_centers(i)[1] = atom_centers_v[i][1];
      atom_centers(i)[2] = atom_centers_v[i][2];
      Z(i) = Z_v[i];
    }
  };
}; // struct Molecule

/**
 * @ brief fills Molecule from file
 */
inline void read_xyz(const std::string &filename, Molecule &mol) {
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

/**
 * @ brief checks if two molecules are the same
 */
inline bool operator==(const Molecule &m1, const Molecule &m2) {
  if (m1.natoms != m2.natoms)
    return false;
  for (unsigned int i = 0; i < m1.natoms; ++i) {
    if (m1.Z(i) != m2.Z(i))
      return false;
    for (unsigned int j = 0; j < 3; ++j) {
      if (m1.atom_centers(i)[j] != m2.atom_centers(i)[3])
        return false;
    }
  }
  return true;
};
} // namespace NuKEXC
