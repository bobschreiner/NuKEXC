/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Bob Schreiner
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 *
 *    Redistribution and use in source and binary forms, with or without
 *    modification, are permitted provided that the following conditions are
 *    met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 *    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "atomic_properties.hpp"
#include "nukexc_config.hpp"
#include <Kokkos_Core.hpp>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace Nukexc {

struct Molecule {

  Kokkos::View<Point *, ExecSpace>
      atom_centers; // Atom centers in cartesian coordinates (bohr)
  Kokkos::View<unsigned *, ExecSpace> Z; // atomic numbers
  unsigned natoms;                       // number of atoms in the molecule
  std::set<unsigned>
      element_list; // contains a list of all elements present in the list
  unsigned Z_total;

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
    atom_centers = Kokkos::View<Point *, ExecSpace>("Atom centers", natoms);
    Z = Kokkos::View<unsigned *, ExecSpace>("Atomic numbers", natoms);
    element_list = std::set<unsigned>(Z_v.begin(), Z_v.end());
    Z_total = 0;

    auto Z_h = Kokkos::create_mirror_view(Z);
    auto atom_centers_h = Kokkos::create_mirror_view(atom_centers);

    // Fill Kokkos::View with data
    for (size_t i = 0; i < natoms; ++i) {
      atom_centers_h(i)[0] = atom_centers_v[i][0];
      atom_centers_h(i)[1] = atom_centers_v[i][1];
      atom_centers_h(i)[2] = atom_centers_v[i][2];
      Z_h(i) = Z_v[i];
      Z_total += Z_v[i];
    }
    Kokkos::deep_copy(Z, Z_h);
    Kokkos::deep_copy(atom_centers, atom_centers_h);
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
  mol = mol_tmp;
}

} // namespace Nukexc
