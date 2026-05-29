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

#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include "stobasis.hpp"

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace NuKEXC {
KOKKOS_INLINE_FUNCTION
double I_tilde(const int n, const int l, const double r, const double zeta) {
  int a = n + l + 2;
  int b = n - l + 1;
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
  return C_prefactor(n, l, zeta) * val * I_tilde(n, l, r, zeta);
}

} // namespace NuKEXC
