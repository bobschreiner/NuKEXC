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

#pragma once

#include "atomic_properties.hpp"
#include "kokkos_config.hpp"
#include "molecule.hpp"
#include "nukexc_utils.hpp"
#include "spherical_harmonics.hpp"

namespace NuKEXC {

struct STOBasisSet {

  Kokkos::View<int *> n_;
  Kokkos::View<int *> l_;
  Kokkos::View<int *> m_;
  Kokkos::View<double *> coeff_;
  Kokkos::View<double *> alpha_;
  Kokkos::View<double *[3]> O_;

  size_t nbf() const { return coeff_.extent(0); };
};

void create_sto_basis_from_molecule(Molecule &mol) {
  for (unsigned element : mol.element_list) {
  }
};

// TODO: Test this code
void evaluate_sto_basis_shells_on_collocation_points(
    const STOBasisSet &basis_set,
    const Kokkos::View<double **> &collocation_points,
    Kokkos::View<double **> &collocation_values) {

  size_t col_points = collocation_points.extent(0);
  size_t nbasis_functions = basis_set.nbf();

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> md_policy(
      {0, 0}, {nbasis_functions, col_points});
  Kokkos::parallel_for(
      "Compute collocation of shells", md_policy,
      KOKKOS_LAMBDA(const int &i, const int &j) {
        const int n_val = basis_set.n_(i);
        const int l_val = basis_set.l_(i);
        const int m_val = basis_set.m_(i);
        const int coeff = basis_set.coeff_(i);
        const int a = basis_set.alpha_(i);

        // radial part of the shell
        // radial_part = R_nl(r) = r^(n-1) * C_nl * exp(-⍺ * r))
        double r = utils::rad_dist(
            Kokkos::subview(basis_set.O_, i, Kokkos::ALL()),
            Kokkos::subview(collocation_points, j, Kokkos::ALL()));

        double radial_part = coeff * Kokkos::exp(-a * r);

        radial_part *= Kokkos::pow(r, n_val - 1);

        // Angular part of the shell
        // https://en.wikipedia.org/wiki/Spherical_harmonics
        double x = collocation_points(j, 0);
        double y = collocation_points(j, 1);
        double z = collocation_points(j, 2);

        double angular_part =
            NuKEXC::detail::real_spherical_harmonic_cart(l_val, m_val, x, y, z);

        collocation_values(i, j) = radial_part * angular_part;
      });
}

} // namespace NuKEXC
