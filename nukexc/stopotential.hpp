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

#include "grid.hpp"
#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include "stobasis.hpp"

namespace Nukexc {

KOKKOS_INLINE_FUNCTION
double I_tilde(const int n, const int l, const double r, const double zeta) {
  int a = n + l + 2;
  int b = n - l + 1;

  if (r < 1e-12) {
    // lower_gamma(a, 0)/r^(2l+1) -> 0 in the limit; only the upper_gamma term
    // survives
    double fact_b_minus_1 = 1.0;
    for (int i = 2; i < b; ++i)
      fact_b_minus_1 *= i;
    return fact_b_minus_1 / Kokkos::pow(zeta, b);
  }

  double zr = zeta * r;
  return lower_gamma(a, zr) /
             (Kokkos::pow(zeta, a) * Kokkos::pow(r, 2 * l + 1)) +
         upper_gamma(b, zr) / Kokkos::pow(zeta, b);
}

KOKKOS_INLINE_FUNCTION
double C_prefactor(const int n, const int l, const double zeta) {
  return 4 * M_PI * Kokkos::pow(2 * zeta, n + 0.5) /
         (Kokkos::sqrt(factorial(2 * n)) * (2 * l + 1));
}

// Potential — just three multiplications
KOKKOS_INLINE_FUNCTION
double sto_potential(const int n, const int l, const int m, const double x,
                     const double y, const double z, const double r,
                     const double zeta) {
  double val;
  real_solid_harmonic_cart_precomputed(l, m, x, y, z, val);
  double C = C_prefactor(n, l, zeta);
  double I = I_tilde(n, l, r, zeta);

  return C * val * I;
}

void sto_potential_collocation(const ExecSpace space, const STOBasisSet basis,
                               const FlatGrid grid,
                               DeviceView2DLeft potential_collocation) {

  const int N_bf = basis.nbf();
  const int N_quad = grid.quad_points.extent(0);

  Kokkos::parallel_for(
      "Compute potentials",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(space, {0, 0}, {N_bf, N_quad}),
      KOKKOS_LAMBDA(const int i, const int g) {
        const int n = basis.n(i);
        const int l = basis.l(i);
        const int m = basis.m(i);
        const double zeta = basis.zeta(i);
        const double x = grid.quad_points(g)[0] - basis.O(i)[0];
        const double y = grid.quad_points(g)[1] - basis.O(i)[1];
        const double z = grid.quad_points(g)[2] - basis.O(i)[2];
        const double r = dist(grid.quad_points(g), basis.O(i)) + epsilon_shift;
        potential_collocation(i, g) = sto_potential(n, l, m, x, y, z, r, zeta);
      });
}

void sto_potential_collocation_scaled(const ExecSpace space,
                                      const STOBasisSet basis,
                                      const FlatGrid grid,
                                      DeviceView2DLeft potential_collocation) {

  const int N_bf = basis.nbf();
  const int N_quad = grid.quad_points.extent(0);

  Kokkos::parallel_for(
      "Compute potentials",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(space, {0, 0}, {N_bf, N_quad}),
      KOKKOS_LAMBDA(const int i, const int g) {
        const int n = basis.n(i);
        const int l = basis.l(i);
        const int m = basis.m(i);
        const double zeta = basis.zeta(i);
        const double x = grid.quad_points(g)[0] - basis.O(i)[0];
        const double y = grid.quad_points(g)[1] - basis.O(i)[1];
        const double z = grid.quad_points(g)[2] - basis.O(i)[2];
        const double r = dist(grid.quad_points(g), basis.O(i)) + epsilon_shift;
        potential_collocation(i, g) = sto_potential(n, l, m, x, y, z, r, zeta) * grid.weights(g);
      });
}
} // namespace Nukexc
