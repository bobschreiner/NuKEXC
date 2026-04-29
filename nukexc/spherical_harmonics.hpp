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

/*
 * The code in this file implements spherical real harmonics as presented on the
 * following Wikipedia page
 *
 *  https://en.wikipedia.org/wiki/Spherical_harmonics
 */

#pragma once
#include "kokkos_config.hpp"
#include "nukexc_utils.hpp"
#include <iostream>

namespace NuKEXC {

KOKKOS_INLINE_FUNCTION
double poly_A(double x, double y, unsigned m) {
  double result = 0;
  for (int p = 0; p < m + 1; ++p) {
    int cos_val = ((m - p) % 2 == 0) ? (((m - p) / 2) % 2 == 0 ? 1 : -1) : 0;
    result +=
        binomial(m, p) * Kokkos::pow(x, p) * Kokkos::pow(y, m - p) * cos_val;
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
double poly_B(double x, double y, unsigned m) {

  double result = 0;
  for (int p = 0; p < m + 1; ++p) {
    int sin_val =
        ((m - p) % 2 != 0) ? (((m - p - 1) / 2) % 2 == 0 ? 1 : -1) : 0;
    result +=
        binomial(m, p) * Kokkos::pow(x, p) * Kokkos::pow(y, m - p) * sin_val;
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
double poly_P(double r, double z, int l, unsigned m) {
  double result = 0;
  if (m == 0) {
    for (int k = 0; k < (l / 2) + 1; ++k) {

      double m1_pow_k = (k % 2 == 0) ? 1.0 : -1.0;
      result += m1_pow_k * Kokkos::pow(2., -l) * binomial(l, k) *
                binomial(2 * l - 2 * k, l) * Kokkos::pow(r, 2 * k) *
                Kokkos::pow(z, l - 2 * k);
    }
    return result;
  }

  // implicit flooring of (l-m)/2 inside the for loop
  for (int k = 0; k < ((l - m) / 2) + 1; ++k) {

    double m1_pow_k = (k % 2 == 0) ? 1.0 : -1.0;
    result +=
        m1_pow_k * Kokkos::pow(2., -l) * binomial(l, k) *
        binomial(2 * l - 2 * k, l) *
        (((double)factorial((l - 2 * k)) / (double)factorial(l - 2 * k - m))) *
        Kokkos::pow(r, 2 * k) * Kokkos::pow(z, l - 2 * k - m);
  }

  double pre_factor =
      Kokkos::sqrt((double)factorial(l - m) / (double)factorial(l + m));
  return pre_factor * result;
}

KOKKOS_INLINE_FUNCTION
double assoc_legendre(const int l, const int m, const double x) {
  assert(m >= 0 && "m must be non-negative");
  assert(l >= m && "l must be larger than m");

  // Start on the diagonal: P(m,m) = (-1)^m *(2*m -1)!!  * (1-x^2)^(m/2)
  int loc_l = m;

  double polynomial =
      double_factorial(2 * m - 1) * (Kokkos::pow(1. - (x * x), m / 2.));

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

/*
 * @brief computes real spherical harmonics from their spherical representation
 */
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

  double pre_factor =
      Kokkos::sqrt(((2. * l + 1.) / (4 * M_PI)) *
                   ((double)factorial(l - abs_m) / factorial(l + abs_m)));

  return pre_factor * sin_cos_term *
         assoc_legendre(l, abs_m, Kokkos::cos(theta));
}

/*
 * @brief computes real spherical harmonics from their spherical representation
 */
KOKKOS_INLINE_FUNCTION
double real_spherical_harmonic_sph_from_cart(const int l, const int m,
                                             const double x, const double y,
                                             const double z) {
  const double r = Kokkos::sqrt(x * x + y * y + z * z);
  const double theta = Kokkos::acos(z / r);
  const double phi = Kokkos::atan2(y, x);

  return real_spherical_harmonic(l, m, theta, phi);
}

/*
 * @brief computes real spherical harmonics from their cartesian representation
 */
KOKKOS_INLINE_FUNCTION
double real_spherical_harmonic_cart(const int l, const int m, const double x,
                                    const double y, const double z) {

  const double r2 = x * x + y * y + z * z;
  if (r2 < 1e-18)
    return (l == 0) ? 0.282094791773878 : 0.0; // 1./sqrt(4*M_PI)
  const double r = Kokkos::sqrt(r2);

  unsigned abs_m = Kokkos::abs(m);

  if (m == 0) {
    return Kokkos::sqrt(((2. * l + 1.) / (4. * M_PI))) *
           poly_P(r, z, l, abs_m) / Kokkos::pow(r, l);
  }
  double result =
      Kokkos::sqrt(((2. * l + 1.) / (2. * M_PI))) * poly_P(r, z, l, abs_m);

  if (m > 0) {
    result *= poly_A(x, y, abs_m);

  } else {
    result *= poly_B(x, y, abs_m);
  }

  result /= Kokkos::pow(r, l);
  return result;
}

/*
 * @brief computes real solid harmonics from their cartesian representation
 */
KOKKOS_INLINE_FUNCTION
double real_solid_harmonic_cart(const int l, const int m, const double x,
                                const double y, const double z) {

  const double r2 = x * x + y * y + z * z;
  if (r2 < 1e-18)
    return (l == 0) ? 0.282094791773878 : 0.0; // 1./sqrt(4*M_PI)
  const double r = Kokkos::sqrt(r2);

  unsigned abs_m = Kokkos::abs(m);

  if (m == 0) {
    return Kokkos::sqrt(((2. * l + 1.) / (4. * M_PI))) * poly_P(r, z, l, abs_m);
  }
  double result =
      Kokkos::sqrt(((2. * l + 1.) / (2. * M_PI))) * poly_P(r, z, l, abs_m);

  result *= m > 0 ? poly_A(x, y, abs_m) : poly_B(x, y, abs_m);

  return result;
}

KOKKOS_INLINE_FUNCTION
void grad_poly_P(double r, double z, int l, int m, double &dP_dr,
                 double &dP_dz) {
  dP_dr = 0.0;
  dP_dz = 0.0;

  unsigned abs_m = Kokkos::abs(m);
  double pre = Kokkos::pow(2.0, -l);

  // The polynomial index logic
  for (int k = 0; k <= (l - abs_m) / 2; ++k) {
    double m1_pow_k = (k % 2 == 0) ? 1.0 : -1.0;
    double coeff = m1_pow_k * pre * binomial(l, k) * binomial(2 * l - 2 * k, l);

    // Handle n-m scaling if using standard normalization
    if (abs_m > 0) {
      coeff *=
          (double)factorial(l - 2 * k) / (double)factorial(l - 2 * k - abs_m);
    }

    int p_r = 2 * k;
    int p_z = l - 2 * k - abs_m;

    // dP/dr part
    if (p_r > 0)
      dP_dr += coeff * p_r * Kokkos::pow(r, p_r - 1) * Kokkos::pow(z, p_z);

    // dP/dz part
    if (p_z > 0)
      dP_dz += coeff * Kokkos::pow(r, p_r) * p_z * Kokkos::pow(z, p_z - 1);
  }

  if (abs_m > 0) {
    double f = Kokkos::sqrt((double)factorial(l - abs_m) /
                            (double)factorial(l + abs_m));
    dP_dr *= f;
    dP_dz *= f;
  }
}

KOKKOS_INLINE_FUNCTION
void grad_poly_A(double x, double y, unsigned m, double &dx, double &dy) {
  dx = 0.;
  dy = 0.;
  for (int p = 0; p < m + 1; ++p) {
    int cos_val = ((m - p) % 2 == 0) ? (((m - p) / 2) % 2 == 0 ? 1 : -1) : 0;
    if (p > 0)
      dx += p * binomial(m, p) * Kokkos::pow(x, p - 1) * Kokkos::pow(y, m - p) *
            cos_val;
    if (m - p > 0)
      dy += (m - p) * binomial(m, p) * Kokkos::pow(x, p) *
            Kokkos::pow(y, m - p - 1) * cos_val;
  }
}

KOKKOS_INLINE_FUNCTION
void grad_poly_B(double x, double y, unsigned m, double &dx, double &dy) {
  dx = 0.;
  dy = 0.;
  for (int p = 0; p < m + 1; ++p) {
    int sin_val =
        ((m - p) % 2 != 0) ? (((m - p - 1) / 2) % 2 == 0 ? 1 : -1) : 0;
    if (p > 0)
      dx += p * binomial(m, p) * Kokkos::pow(x, p - 1) * Kokkos::pow(y, m - p) *
            sin_val;
    if (m - p > 0)
      dy += (m - p) * binomial(m, p) * Kokkos::pow(x, p) *
            Kokkos::pow(y, m - p - 1) * sin_val;
  }
}

/*
 * @brief computes the gradient of real solid harmonics from their cartesian
 * representation
 */
KOKKOS_INLINE_FUNCTION
void grad_real_solid_harmonic_cart(const int l, const int m, const double x,
                                   const double y, const double z, double &dx,
                                   double &dy, double &dz) {
  const double r = Kokkos::sqrt(x * x + y * y + z * z) + epsilon_shift;
  const int abs_m = Kokkos::abs(m);

  // Get Polynomial values and their internal derivatives
  double p_val = poly_P(r, z, l, abs_m);
  double dP_dr;
  double dP_dz;
  grad_poly_P(r, z, l, abs_m, dP_dr, dP_dz);

  // Angular parts (A for m>0, B for m<0, 1.0 for m=0)
  double ang = 1.0;
  double dAng_dx = 0.0;
  double dAng_dy = 0.0;

  if (m > 0) {
    ang = poly_A(x, y, abs_m);
    grad_poly_A(x, y, abs_m, dAng_dx, dAng_dy);
  } else if (m < 0) {
    ang = poly_B(x, y, abs_m);
    grad_poly_B(x, y, abs_m, dAng_dx, dAng_dy);
  }

  // Normalization prefactor
  double norm = Kokkos::sqrt((2. * l + 1.) / ((m == 0 ? 4. : 2.) * M_PI));

  // Combine using Product Rule and Chain Rule
  // dx = norm * ( (dP/dr * x/r) * ang + p_val * dAng/dx )
  double pr_x = dP_dr * (x / r);
  double pr_y = dP_dr * (y / r);
  double pr_z = dP_dr * (z / r);

  dx = norm * (pr_x * ang + p_val * dAng_dx);
  dy = norm * (pr_y * ang + p_val * dAng_dy);
  dz = norm * ((pr_z + dP_dz) * ang);
}

} // namespace NuKEXC
