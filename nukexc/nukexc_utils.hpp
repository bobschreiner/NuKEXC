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

#include <Kokkos_Core.hpp>

namespace NuKEXC {

KOKKOS_INLINE_FUNCTION
double rad_dist(const Kokkos::View<double *, Kokkos::LayoutStride> &a,
                const Kokkos::View<double *, Kokkos::LayoutStride> &b) {
  double dist = 0;
  for (int i = 0; i < a.extent(0); ++i) {
    dist += std::pow(a(i) - b(i), 2);
  }
  dist = std::sqrt(dist);
  return dist;
}

KOKKOS_INLINE_FUNCTION
double double_factorial(int n) {
  if (n <= 0)
    return 1.0;

  double result = 1.0;
  for (int i = n; i > 0; i -= 2) {
    result *= i;
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
double factorial(int n) {

  if (n <= 0)
    return 1.0;

  double result = 1.0;
  for (int i = 1; i <= n; ++i) {
    result *= i;
  }
  return result;
}
KOKKOS_INLINE_FUNCTION
long int binomial(int n, int k) {
  long int result = factorial(n) / (factorial(k) * factorial(n - k));
  return result;
}

KOKKOS_INLINE_FUNCTION
double log_factorial_ratio(int top, int bottom) {
  // Computes log(top! / bottom!) = log(bottom+1) + ... + log(top)
  // Assumes top >= bottom >= 0
  double result = 0.0;
  for (int i = bottom + 1; i <= top; ++i)
    result += Kokkos::log((double)i);
  return result;
}

KOKKOS_INLINE_FUNCTION
double safe_pow(double base, int exp) {
  if (exp == 0)
    return 1.0; // always 1 regardless of base
  if (base == 0.0)
    return 0.0; // 0^positive = 0
  return Kokkos::pow(base, (double)exp);
}
} // namespace NuKEXC
