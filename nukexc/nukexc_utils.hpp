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

#include "nukexc/nukexc_config.hpp"
#include <Kokkos_Core.hpp>

namespace Nukexc {

KOKKOS_INLINE_FUNCTION
double rad_dist(const Kokkos::View<double *, Kokkos::LayoutStride> &a,
                const Kokkos::View<double *, Kokkos::LayoutStride> &b) {
  double dist = 0;
  for (int i = 0; i < a.extent(0); ++i) {
    dist += Kokkos::pow(a(i) - b(i), 2);
  }
  dist = std::sqrt(dist);
  return dist;
}

KOKKOS_INLINE_FUNCTION
double dist(const Point a, const Point b) {
  double dist = 0;
  dist += Kokkos::pow(a[0] - b[0], 2);
  dist += Kokkos::pow(a[1] - b[1], 2);
  dist += Kokkos::pow(a[2] - b[2], 2);
  return Kokkos::sqrt(dist);
}

KOKKOS_INLINE_FUNCTION
double dist_squared(const Point a, const Point b) {
  double dist = 0;
  dist += Kokkos::pow(a[0] - b[0], 2);
  dist += Kokkos::pow(a[1] - b[1], 2);
  dist += Kokkos::pow(a[2] - b[2], 2);
  return dist;
}

KOKKOS_INLINE_FUNCTION
double double_factorial(const int n) {
  if (n <= 0)
    return 1.0;

  double result = 1.0;
  for (int i = n; i > 0; i -= 2) {
    result *= i;
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
double factorial(const int n) {

  if (n <= 0)
    return 1.0;

  double result = 1.0;
  for (int i = 1; i <= n; ++i) {
    result *= i;
  }
  return result;
}
KOKKOS_INLINE_FUNCTION
long int binomial(const int n, const int k) {
  long int result = factorial(n) / (factorial(k) * factorial(n - k));
  return result;
}

KOKKOS_INLINE_FUNCTION
double log_factorial_ratio(const int top, const int bottom) {
  // Computes log(top! / bottom!) = log(bottom+1) + ... + log(top)
  // Assumes top >= bottom >= 0
  double result = 0.0;
  for (int i = bottom + 1; i <= top; ++i)
    result += Kokkos::log((double)i);
  return result;
}

KOKKOS_INLINE_FUNCTION
double safe_pow(const double base, const int exp) {
  if (exp == 0)
    return 1.0; // always 1 regardless of base
  if (base == 0.0)
    return 0.0; // 0^positive = 0
  return Kokkos::pow(base, (double)exp);
}

KOKKOS_INLINE_FUNCTION
double int_pow(const double r, const int k) {

  double result = 0.;
  switch (k) {
  case 0:
    result = 1.;
    break;
  case 1:
    result = r;
    break;
  case 2:
    result = r * r;
    break;
  case 3:
    result = r * r * r;
    break;
  case 4:
    result = r * r * r * r;
    break;
  case 5:
    result = r * r * r * r * r;
    break;
  case 6:
    result = r * r * r * r * r * r;
    break;
  case 7:
    result = r * r * r * r * r * r * r;
    break;
  default:
    result = Kokkos::pow(r, (double)k);
    break;
  }
  return result;
}
KOKKOS_INLINE_FUNCTION
double lower_gamma(const int n, const double x) {
  double result = 0.0;
  for (int k = 0; k < n; ++k) {
    result += int_pow(x, k) / factorial(k);
  }
  result *= Kokkos::exp(-x);
  result = 1.0 - result;
  result *= factorial(n - 1);
  return result;
}

KOKKOS_INLINE_FUNCTION
double upper_gamma(const int n, const double x) {
  double result = 0.0;
  for (int k = 0; k < n; ++k) {
    result += int_pow(x, k) / factorial(k);
  }
  result *= Kokkos::exp(-x);
  result *= factorial(n - 1);
  return result;
}

} // namespace Nukexc
