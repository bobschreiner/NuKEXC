#pragma once

#include "kokkos_config.hpp"

namespace NuKEXC {
namespace detail {

KOKKOS_INLINE_FUNCTION
long int double_factorial(long int n) {
  if (n == 0 || n == -1)
    return 1;

  else
    return double_factorial(n - 2) * n;
}

KOKKOS_INLINE_FUNCTION
double assoc_legendre(const int l, const int m, const double x) {

  assert(m >= 0 && "m must be non-negative");
  assert(l >= m && "l must be larger than m");

  // Start on the diagonal: P(m,m) = (-1)^m *(2*m -1)!!  * (1-x^2)^(m/2)
  int loc_l = m;
  double polynomial = Kokkos::pow(-1., m) * double_factorial(2 * m - 1) *
                      (Kokkos::pow(1. - (x * x), m/2.));

  if (loc_l == l)
    return polynomial;
  double prev_polynomial = polynomial;
  polynomial *= x * (2 * loc_l + 1);
  ++loc_l;

  if (loc_l == l)
    return polynomial;

  // use recurrence relation to increment local l until we hit l
  double next_polynomial;
  while (loc_l != l) {
    next_polynomial =
        ((2 * loc_l + 1) * x * polynomial - (loc_l + m) * prev_polynomial) /
        (loc_l - m + 1.);
    prev_polynomial = polynomial;
    polynomial = next_polynomial;
    ++loc_l;
  }
  return polynomial;
}

KOKKOS_INLINE_FUNCTION
double real_spherical_harmonic(const int l, const int m, const double theta,
                               const double phi) {

  double sin_cos_term;
  double sqrt2 = Kokkos::sqrt(2.);
  int abs_m = Kokkos::abs(m);

  if (m < 0) {
    sin_cos_term = sqrt2 * Kokkos::sin(abs_m * phi);
  } else if (m == 0) {
    sin_cos_term = 1.;
  } else {
    sin_cos_term = sqrt2 * Kokkos::cos(abs_m * phi);
  }

  double pre_factor = Kokkos::sqrt(
      ((2 * l + 1) / (4 * M_PI)) *
      (Kokkos::tgamma(l - abs_m + 1) / Kokkos::tgamma(l + abs_m + 1)));

  double phase = Kokkos::pow(-1., m);

  return phase * pre_factor * sin_cos_term *
         assoc_legendre(l, abs_m, Kokkos::cos(theta));
}

KOKKOS_INLINE_FUNCTION
double real_spherical_harmonic_cart(const int l, const int m, const double x,
                                    const double y, const double z) {

  const double r = Kokkos::sqrt(x * x + y * y + z * z);
  const double theta = Kokkos::acos(z / r);
  const double phi = Kokkos::atan2(y, x);

  return real_spherical_harmonic(l, m, theta, phi);
}
} // namespace detail
} // namespace NuKEXC
