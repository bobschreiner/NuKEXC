#pragma once

#include "kokkos_config.hpp"

namespace NuKEXC {
namespace detail {

KOKKOS_INLINE_FUNCTION
long int double_factorial(long int n) {
  if (n == 0 || n == 1)
    return 1;

  else if (n == 2)
    return 2;

  else
    return double_factorial(n - 2) * n;
}

KOKKOS_INLINE_FUNCTION void compute_prefactors_for_spherical_harmonics(
    const int l_max, Kokkos::View<double **> &pre_factors) {

  for (int l = 0; l < l_max + 1; ++l) {
    pre_factors(l, 0) = Kokkos::sqrt((2. * l + 1.) / (4 * M_PI));
    for (int m = 1; m < l + 1; ++m) {
      pre_factors(l, m) = Kokkos::sqrt(
          ((2 * l + 1) / (4 * M_PI)) *
          (Kokkos::tgammal(l - m + 1) / Kokkos::tgammal(l + m + 1)));
    }
  }
}

KOKKOS_INLINE_FUNCTION
double assoc_legendre(const int l, const int m, const double x) {

  assert(m >= 0 && "m must be non-negative");
  assert(l >= m && "l must be larger than m");

  // Start on the diagonal: P(m,m) = (-1)^m *(2*m -1)!!  * (1-x^2)^(m/2)
  int loc_l = m;
  int phase = Kokkos::pow(-1, m);
  double polynomial = phase * double_factorial(2 * m - 1) *
                      Kokkos::sqrt(Kokkos::pow(1. - (x * x), m));

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
        (l - m + 1.);
    prev_polynomial = polynomial;
    polynomial = next_polynomial;
    ++loc_l;
  }
  return polynomial;
}

KOKKOS_INLINE_FUNCTION
double real_spherical_harmonic(const int l, const int m, const double theta,
                               const double phi, double harmonic_pre_factor) {

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

  double harmonic = harmonic_pre_factor * sin_cos_term *
                    assoc_legendre(l, abs_m, Kokkos::cos(theta));
  return harmonic;
}

KOKKOS_INLINE_FUNCTION
double real_spherical_harmonic_cart(const int l, const int m, const double x,
                                    const double y, const double z,
                                    double harmonic_pre_factor) {

  const double r = Kokkos::sqrt(x * x + y * y + z * z);
  const double theta = Kokkos::acos(z / r);
  const double phi = Kokkos::atan2(y, x);

  return real_spherical_harmonic(l, m, theta, phi, harmonic_pre_factor);
}
} // namespace detail
} // namespace NuKEXC
