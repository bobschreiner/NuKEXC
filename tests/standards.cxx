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

#include "standards.hpp"
#include "../src/molecule.hpp"

namespace NuKEXC {

Molecule make_water() {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  return mol;
}
Molecule make_benzene() {
  Molecule mol;
  read_xyz("input/benzene.xyz", mol);
  return mol;
}
Molecule make_taxol() {
  Molecule mol;
  read_xyz("input/taxol.xyz", mol);
  return mol;
}
Molecule make_ubiquitin() {
  Molecule mol;
  read_xyz("input/ubiquitin.xyz", mol);
  return mol;
}

void make_water(Molecule &mol) { read_xyz("input/water.xyz", mol); }
void make_benzene(Molecule &mol) { read_xyz("input/benzene.xyz", mol); }
void make_taxol(Molecule &mol) { read_xyz("input/taxol.xyz", mol); }
void make_ubiquitin(Molecule &mol) { read_xyz("input/ubiquitin.xyz", mol); }

} // namespace NuKEXC
