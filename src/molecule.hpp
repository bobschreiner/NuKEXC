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

#include "kokkos_config.hpp"
#include <vector>

namespace NuKEXC {

struct Molecule {

  Kokkos::View<double *[3]>
      atom_centers;           // Atom centers in cartesian coordinates (bohr)
  Kokkos::View<unsigned *> Z; // atomic numbers
  unsigned natoms;            // number of atoms in the molecule

  /**
   * @ brief Default constructor
   */
  Molecule() = default;

  /**
   * @ brief Default constructor
   */
  Molecule(unsigned int natoms_) {
    // Initialize datastructures
    natoms = natoms_;
    atom_centers = Kokkos::View<double *[3]>("Atom centers", natoms);
    Z = Kokkos::View<unsigned *>("Atomic numbers ", natoms);
  }

  /**
   * @ brief Constructs Molecule from std::vector
   */
  Molecule(const std::vector<std::vector<double>> &atom_centers_v,
           const std::vector<unsigned> &Z_v) {

    // Initialize datastructures
    natoms = Z_v.size();
    atom_centers = Kokkos::View<double *[3]>("Atom centers", natoms);
    Z = Kokkos::View<unsigned *>("Atomic numbers ", natoms);

    // Fill Kokkos::View with data
    for (size_t i = 0; i < natoms; ++i) {
      atom_centers(i, 0) = atom_centers_v[i][0];
      atom_centers(i, 1) = atom_centers_v[i][1];
      atom_centers(i, 2) = atom_centers_v[i][2];
      Z(i) = Z_v[i];
    }
  };
}; // struct Molecule

/**
 * @ brief checks if two molecules are the same
 */
inline bool operator==(const Molecule &m1, const Molecule &m2) {
  if (m1.natoms != m2.natoms)
    return false;
  for (unsigned int i = 0; i < m1.natoms; ++i) {
    if (m1.Z(i) != m2.Z(i))
      return false;
    for (unsigned int j = 0; i < 3; ++j) {
      if (m1.atom_centers(i, j) != m1.atom_centers(i, j))
        return false;
    }
  }
  return true;
};
} // namespace NuKEXC
