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

#include "molecule.hpp"
#include "nukexc_utils.hpp"

namespace Nukexc {

double compute_nuclear_repulsion(Molecule mol) {
  const int natoms = mol.natoms;
  double E_nuc = 0.0;
  Kokkos::parallel_reduce(
      "Nuclear repulsion",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                        {natoms, natoms}),
      KOKKOS_LAMBDA(const int A, const int B, double &update) {
        if (B <= A)
          return;
        const double R = dist(mol.atom_centers(A), mol.atom_centers(B));
        update +=
            static_cast<double>(mol.Z(A)) * static_cast<double>(mol.Z(B)) / R;
      },
      E_nuc);

  return E_nuc;
};
}; // namespace Nukexc
