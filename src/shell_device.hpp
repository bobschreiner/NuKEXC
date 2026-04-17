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

#include "kokkos_config.hpp"
#include "nukexc_utils.hpp"
#include "spherical_harmonics.hpp"

namespace NuKEXC {

struct STOShellDevice {

  Kokkos::View<int *> n_;
  Kokkos::View<int *> l_;
  Kokkos::View<int *> m_;
  Kokkos::View<double *> coeff_;
  Kokkos::View<double *> alpha_;
  Kokkos::View<double *[3]> O_;

  size_t nbf() const { return coeff_.extent(0); };
};

void evaluate_sto_basis_shells_on_collocation_points(
    const STOShellDevice &device,
    const Kokkos::View<double **> &collocation_points,
    Kokkos::View<double **> &collocation_values) {

  size_t col_points = collocation_points.extent(0);
  size_t nbasis_functions = device.nbf();

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> md_policy(
      {0, 0}, {nbasis_functions, col_points});
  Kokkos::parallel_for(
      "Compute collocation of shells", md_policy,
      KOKKOS_LAMBDA(const int &i, const int &j) {
        const int n_val = device.n_(i);
        const int l_val = device.l_(i);
        const int coeff = device.coeff_(i);
        const int a = device.alpha_(i);

        // radial part of the shell
        // radial_part = R_nl(r) = r^(n-1) * C_nl * exp(-⍺ * r))
        double r = utils::rad_dist(
            Kokkos::subview(device.O_, i, Kokkos::ALL()),
            Kokkos::subview(collocation_points, j, Kokkos::ALL()));

        double radial_part = coeff * Kokkos::exp(-a * r);

        radial_part *= Kokkos::pow(r, n_val - 1);

        // Angular part of the shell
        // Angular part  = Y_lm = (-1)^m * sqrt{[(2l+1)/4π] * [(l-m)! / (l+m)!]}
        // P_ml(cos(θ)) * (cos(m*ϕ) + i*sin(m*ϕ))
        for (int m_val = -l_val; m_val < l_val + 1; ++m_val) {
          double x = collocation_points(j, 0);
          double y = collocation_points(j, 1);
          double z = collocation_points(j, 2);

          double angular_part = NuKEXC::detail::real_spherical_harmonic_cart(
              l_val, m_val, x, y, z);

          collocation_values(i, j);
        }
      });
}

} // namespace NuKEXC
