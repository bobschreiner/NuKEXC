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
#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include <iostream>

namespace Nukexc {

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
      double log_ratio = log_factorial_ratio(l - 2 * k, l - 2 * k - abs_m);
      coeff *= Kokkos::exp(log_ratio);
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
    double log_ratio = -log_factorial_ratio(l + abs_m, l - abs_m);
    double f = Kokkos::exp(0.5 * log_ratio);
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
      dx +=
          p * binomial(m, p) * Kokkos::pow(x, p - 1) * pow(y, m - p) * cos_val;
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

KOKKOS_INLINE_FUNCTION
void real_solid_harmonic_cart_precomputed(const int l, const int m,
                                          const double x, const double y,
                                          const double z, double &val) {

  // Initialize outputs upfront to zero out registers and clean up branch code
  val = 0.0;

  const int idx = l * l + l + m;
  switch (idx) {
  case 0: // l=0, m=0
    val = 0.28209479177387814;
    break;
  case 1: // l=1, m=-1
    val = 0.48860251190291992 * y;
    break;
  case 2: // l=1, m=0
    val = 0.48860251190291992 * z;
    break;
  case 3: // l=1, m=1
    val = 0.48860251190291992 * x;
    break;
  case 4: // l=2, m=-2
    val = 1.0925484305920791 * x * y;
    break;
  case 5: // l=2, m=-1
    val = 1.0925484305920791 * y * z;
    break;
  case 6: // l=2, m=0
    val = -0.31539156525252001 * x * x - 0.31539156525252001 * y * y +
          0.63078313050504001 * z * z;
    break;
  case 7: // l=2, m=1
    val = 1.0925484305920791 * x * z;
    break;
  case 8: // l=2, m=2
    val = 0.54627421529603954 * x * x - 0.54627421529603954 * y * y;
    break;
  case 9: // l=3, m=-3
    val = 1.7701307697799305 * x * x * y - 0.59004358992664351 * y * y * y;
    break;
  case 10: // l=3, m=-2
    val = 2.8906114426405541 * x * y * z;
    break;
  case 11: // l=3, m=-1
    val = -0.45704579946446574 * x * x * y - 0.45704579946446574 * y * y * y +
          1.8281831978578629 * y * z * z;
    break;
  case 12: // l=3, m=0
    val = -1.1195289977703462 * x * x * z - 1.1195289977703462 * y * y * z +
          0.74635266518023078 * z * z * z;
    break;
  case 13: // l=3, m=1
    val = -0.45704579946446574 * x * x * x - 0.45704579946446574 * x * y * y +
          1.8281831978578629 * x * z * z;
    break;
  case 14: // l=3, m=2
    val = 1.445305721320277 * x * x * z - 1.445305721320277 * y * y * z;
    break;
  case 15: // l=3, m=3
    val = 0.59004358992664351 * x * x * x - 1.7701307697799305 * x * y * y;
    break;
  case 16: // l=4, m=-4
    val =
        2.5033429417967045 * x * x * x * y - 2.5033429417967045 * x * y * y * y;
    break;
  case 17: // l=4, m=-3
    val =
        5.3103923093397916 * x * x * y * z - 1.7701307697799305 * y * y * y * z;
    break;
  case 18: // l=4, m=-2
    val = -0.94617469575756002 * x * x * x * y -
          0.94617469575756002 * x * y * y * y +
          5.6770481745453601 * x * y * z * z;
    break;
  case 19: // l=4, m=-1
    val = -2.0071396306718675 * x * x * y * z -
          2.0071396306718675 * y * y * y * z +
          2.6761861742291567 * y * z * z * z;
    break;
  case 20: // l=4, m=0
    val = 0.31735664074561291 * x * x * x * x +
          0.63471328149122582 * x * x * y * y -
          2.5388531259649033 * x * x * z * z +
          0.31735664074561291 * y * y * y * y -
          2.5388531259649033 * y * y * z * z +
          0.84628437532163443 * z * z * z * z;
    break;
  case 21: // l=4, m=1
    val = -2.0071396306718675 * x * x * x * z -
          2.0071396306718675 * x * y * y * z +
          2.6761861742291567 * x * z * z * z;
    break;
  case 22: // l=4, m=2
    val = -0.47308734787878001 * x * x * x * x +
          2.8385240872726801 * x * x * z * z +
          0.47308734787878001 * y * y * y * y -
          2.8385240872726801 * y * y * z * z;
    break;
  case 23: // l=4, m=3
    val =
        1.7701307697799305 * x * x * x * z - 5.3103923093397916 * x * y * y * z;
    break;
  case 24: // l=4, m=4
    val = 0.62583573544917613 * x * x * x * x -
          3.7550144126950568 * x * x * y * y +
          0.62583573544917613 * y * y * y * y;
    break;
  case 25: // l=5, m=-5
    val = 3.2819102842008505 * x * x * x * x * y -
          6.563820568401701 * x * x * y * y * y +
          0.6563820568401701 * y * y * y * y * y;
    break;
  case 26: // l=5, m=-4
    val = 8.3026492595241651 * x * x * x * y * z -
          8.3026492595241651 * x * y * y * y * z;
    break;
  case 27: // l=5, m=-3
    val = -1.4677148983057512 * x * x * x * x * y -
          0.97847659887050078 * x * x * y * y * y +
          11.741719186446009 * x * x * y * z * z +
          0.48923829943525039 * y * y * y * y * y -
          3.9139063954820031 * y * y * y * z * z;
    break;
  case 28: // l=5, m=-2
    val = -4.7935367849733238 * x * x * x * y * z -
          4.7935367849733238 * x * y * y * y * z +
          9.5870735699466475 * x * y * z * z * z;
    break;
  case 29: // l=5, m=-1
    val = 0.45294665119569692 * x * x * x * x * y +
          0.90589330239139384 * x * x * y * y * y -
          5.4353598143483631 * x * x * y * z * z +
          0.45294665119569692 * y * y * y * y * y -
          5.4353598143483631 * y * y * y * z * z +
          3.6235732095655754 * y * z * z * z * z;
    break;
  case 30: // l=5, m=0
    val = 1.7542548368013539 * x * x * x * x * z +
          3.5085096736027079 * x * x * y * y * z -
          4.6780128981369439 * x * x * z * z * z +
          1.7542548368013539 * y * y * y * y * z -
          4.6780128981369439 * y * y * z * z * z +
          0.93560257962738877 * z * z * z * z * z;
    break;
  case 31: // l=5, m=1
    val = 0.45294665119569692 * x * x * x * x * x +
          0.90589330239139384 * x * x * x * y * y -
          5.4353598143483631 * x * x * x * z * z +
          0.45294665119569692 * x * y * y * y * y -
          5.4353598143483631 * x * y * y * z * z +
          3.6235732095655754 * x * z * z * z * z;
    break;
  case 32: // l=5, m=2
    val = -2.3967683924866619 * x * x * x * x * z +
          4.7935367849733238 * x * x * z * z * z +
          2.3967683924866619 * y * y * y * y * z -
          4.7935367849733238 * y * y * z * z * z;
    break;
  case 33: // l=5, m=3
    val = -0.48923829943525039 * x * x * x * x * x +
          0.97847659887050078 * x * x * x * y * y +
          3.9139063954820031 * x * x * x * z * z +
          1.4677148983057512 * x * y * y * y * y -
          11.741719186446009 * x * y * y * z * z;
    break;
  case 34: // l=5, m=4
    val = 2.0756623148810413 * x * x * x * x * z -
          12.453973889286248 * x * x * y * y * z +
          2.0756623148810413 * y * y * y * y * z;
    break;
  case 35: // l=5, m=5
    val = 0.6563820568401701 * x * x * x * x * x -
          6.563820568401701 * x * x * x * y * y +
          3.2819102842008505 * x * y * y * y * y;
    break;
  case 36: // l=6, m=-6
    val = 4.0991046311514859 * x * x * x * x * x * y -
          13.663682103838286 * x * x * x * y * y * y +
          4.0991046311514859 * x * y * y * y * y * y;
    break;
  case 37: // l=6, m=-5
    val = 11.83309581115876 * x * x * x * x * y * z -
          23.66619162231752 * x * x * y * y * y * z +
          2.366619162231752 * y * y * y * y * y * z;
    break;
  case 38: // l=6, m=-4
    val = -2.0182596029148966 * x * x * x * x * x * y +
          20.182596029148966 * x * x * x * y * z * z +
          2.0182596029148966 * x * y * y * y * y * y -
          20.182596029148966 * x * y * y * y * z * z;
    break;
  case 39: // l=6, m=-3
    val = -8.2908473356343115 * x * x * x * x * y * z -
          5.527231557089541 * x * x * y * y * y * z +
          22.108926228358164 * x * x * y * z * z * z +
          2.7636157785447705 * y * y * y * y * y * z -
          7.369642076119388 * y * y * y * z * z * z;
    break;
  case 40: // l=6, m=-2
    val = 0.9212052595149235 * x * x * x * x * x * y +
          1.842410519029847 * x * x * x * y * y * y -
          14.739284152238776 * x * x * x * y * z * z +
          0.9212052595149235 * x * y * y * y * y * y -
          14.739284152238776 * x * y * y * y * z * z +
          14.739284152238776 * x * y * z * z * z * z;
    break;
  case 41: // l=6, m=-1
    val = 2.9131068125936569 * x * x * x * x * y * z +
          5.8262136251873139 * x * x * y * y * y * z -
          11.652427250374628 * x * x * y * z * z * z +
          2.9131068125936569 * y * y * y * y * y * z -
          11.652427250374628 * y * y * y * z * z * z +
          4.6609709001498511 * y * z * z * z * z * z;
    break;
  case 42: // l=6, m=0
    val = -0.31784601133814213 * x * x * x * x * x * x -
          0.95353803401442639 * x * x * x * x * y * y +
          5.7212282040865583 * x * x * x * x * z * z -
          0.95353803401442639 * x * x * y * y * y * y +
          11.442456408173117 * x * x * y * y * z * z -
          7.6283042721154111 * x * x * z * z * z * z -
          0.31784601133814213 * y * y * y * y * y * y +
          5.7212282040865583 * y * y * y * y * z * z -
          7.6283042721154111 * y * y * z * z * z * z +
          1.0171072362820548 * z * z * z * z * z * z;
    break;
  case 43: // l=6, m=1
    val = 2.9131068125936569 * x * x * x * x * x * z +
          5.8262136251873139 * x * x * x * y * y * z -
          11.652427250374628 * x * x * x * z * z * z +
          2.9131068125936569 * x * y * y * y * y * z -
          11.652427250374628 * x * y * y * z * z * z +
          4.6609709001498511 * x * z * z * z * z * z;
    break;
  case 44: // l=6, m=2
    val = 0.46060262975746175 * x * x * x * x * x * x +
          0.46060262975746175 * x * x * x * x * y * y -
          7.369642076119388 * x * x * x * x * z * z -
          0.46060262975746175 * x * x * y * y * y * y +
          7.369642076119388 * x * x * z * z * z * z -
          0.46060262975746175 * y * y * y * y * y * y +
          7.369642076119388 * y * y * y * y * z * z -
          7.369642076119388 * y * y * z * z * z * z;
    break;
  case 45: // l=6, m=3
    val = -2.7636157785447705 * x * x * x * x * x * z +
          5.527231557089541 * x * x * x * y * y * z +
          7.369642076119388 * x * x * x * z * z * z +
          8.2908473356343115 * x * y * y * y * y * z -
          22.108926228358164 * x * y * y * z * z * z;
    break;
  case 46: // l=6, m=4
    val = -0.50456490072872416 * x * x * x * x * x * x +
          2.5228245036436208 * x * x * x * x * y * y +
          5.0456490072872416 * x * x * x * x * z * z +
          2.5228245036436208 * x * x * y * y * y * y -
          30.27389404372345 * x * x * y * y * z * z -
          0.50456490072872416 * y * y * y * y * y * y +
          5.0456490072872416 * y * y * y * y * z * z;
    break;
  case 47: // l=6, m=5
    val = 2.366619162231752 * x * x * x * x * x * z -
          23.66619162231752 * x * x * x * y * y * z +
          11.83309581115876 * x * y * y * y * y * z;
    break;
  case 48: // l=6, m=6
    val = 0.68318410519191432 * x * x * x * x * x * x -
          10.247761577878715 * x * x * x * x * y * y +
          10.247761577878715 * x * x * y * y * y * y -
          0.68318410519191432 * y * y * y * y * y * y;
    break;
  case 49: // l=7, m=-7
    val = 4.9501391276721732 * x * x * x * x * x * x * y -
          24.750695638360866 * x * x * x * x * y * y * y +
          14.85041738301652 * x * x * y * y * y * y * y -
          0.70716273252459618 * y * y * y * y * y * y * y;
    break;
  case 50: // l=7, m=-6
    val = 15.875763970811401 * x * x * x * x * x * y * z -
          52.919213236038004 * x * x * x * y * y * y * z +
          15.875763970811401 * x * y * y * y * y * y * z;
    break;
  case 51: // l=7, m=-5
    val = -2.5945778936013016 * x * x * x * x * x * x * y +
          2.5945778936013016 * x * x * x * x * y * y * y +
          31.134934723215619 * x * x * x * x * y * z * z +
          4.6702402084823429 * x * x * y * y * y * y * y -
          62.269869446431238 * x * x * y * y * y * z * z -
          0.51891557872026032 * y * y * y * y * y * y * y +
          6.2269869446431238 * y * y * y * y * y * z * z;
    break;
  case 52: // l=7, m=-4
    val = -12.453973889286248 * x * x * x * x * x * y * z +
          41.513246297620826 * x * x * x * y * z * z * z +
          12.453973889286248 * x * y * y * y * y * y * z -
          41.513246297620826 * x * y * y * y * z * z * z;
    break;
  case 53: // l=7, m=-3
    val = 1.4081304047606463 * x * x * x * x * x * x * y +
          2.3468840079344105 * x * x * x * x * y * y * y -
          28.162608095212926 * x * x * x * x * y * z * z +
          0.4693768015868821 * x * x * y * y * y * y * y -
          18.775072063475284 * x * x * y * y * y * z * z +
          37.550144126950568 * x * x * y * z * z * z * z -
          0.4693768015868821 * y * y * y * y * y * y * y +
          9.387536031737642 * y * y * y * y * y * z * z -
          12.516714708983523 * y * y * y * z * z * z * z;
    break;
  case 54: // l=7, m=-2
    val = 6.6379903866747395 * x * x * x * x * x * y * z +
          13.275980773349479 * x * x * x * y * y * y * z -
          35.402615395598611 * x * x * x * y * z * z * z +
          6.6379903866747395 * x * y * y * y * y * y * z -
          35.402615395598611 * x * y * y * y * z * z * z +
          21.241569237359166 * x * y * z * z * z * z * z;
    break;
  case 55: // l=7, m=-1
    val = -0.45165803791258657 * x * x * x * x * x * x * y -
          1.3549741137377597 * x * x * x * x * y * y * y +
          10.839792909902078 * x * x * x * x * y * z * z -
          1.3549741137377597 * x * x * y * y * y * y * y +
          21.679585819804155 * x * x * y * y * y * z * z -
          21.679585819804155 * x * x * y * z * z * z * z -
          0.45165803791258657 * y * y * y * y * y * y * y +
          10.839792909902078 * y * y * y * y * y * z * z -
          21.679585819804155 * y * y * y * z * z * z * z +
          5.7812228852811081 * y * z * z * z * z * z * z;
    break;
  case 56: // l=7, m=0
    val = -2.389949691920173 * x * x * x * x * x * x * z -
          7.1698490757605189 * x * x * x * x * y * y * z +
          14.339698151521038 * x * x * x * x * z * z * z -
          7.1698490757605189 * x * x * y * y * y * y * z +
          28.679396303042076 * x * x * y * y * z * z * z -
          11.47175852121683 * x * x * z * z * z * z * z -
          2.389949691920173 * y * y * y * y * y * y * z +
          14.339698151521038 * y * y * y * y * z * z * z -
          11.47175852121683 * y * y * z * z * z * z * z +
          1.0925484305920791 * z * z * z * z * z * z * z;
    break;
  case 57: // l=7, m=1
    val = -0.45165803791258657 * x * x * x * x * x * x * x -
          1.3549741137377597 * x * x * x * x * x * y * y +
          10.839792909902078 * x * x * x * x * x * z * z -
          1.3549741137377597 * x * x * x * y * y * y * y +
          21.679585819804155 * x * x * x * y * y * z * z -
          21.679585819804155 * x * x * x * z * z * z * z -
          0.45165803791258657 * x * y * y * y * y * y * y +
          10.839792909902078 * x * y * y * y * y * z * z -
          21.679585819804155 * x * y * y * z * z * z * z +
          5.7812228852811081 * x * z * z * z * z * z * z;
    break;
  case 58: // l=7, m=2
    val = 3.3189951933373697 * x * x * x * x * x * x * z +
          3.3189951933373697 * x * x * x * x * y * y * z -
          17.701307697799305 * x * x * x * x * z * z * z -
          3.3189951933373697 * x * x * y * y * y * y * z +
          10.620784618679583 * x * x * z * z * z * z * z -
          3.3189951933373697 * y * y * y * y * y * y * z +
          17.701307697799305 * y * y * y * y * z * z * z -
          10.620784618679583 * y * y * z * z * z * z * z;
    break;
  case 59: // l=7, m=3
    val = 0.4693768015868821 * x * x * x * x * x * x * x -
          0.4693768015868821 * x * x * x * x * x * y * y -
          9.387536031737642 * x * x * x * x * x * z * z -
          2.3468840079344105 * x * x * x * y * y * y * y +
          18.775072063475284 * x * x * x * y * y * z * z +
          12.516714708983523 * x * x * x * z * z * z * z -
          1.4081304047606463 * x * y * y * y * y * y * y +
          28.162608095212926 * x * y * y * y * y * z * z -
          37.550144126950568 * x * y * y * z * z * z * z;
    break;
  case 60: // l=7, m=4
    val = -3.1134934723215619 * x * x * x * x * x * x * z +
          15.56746736160781 * x * x * x * x * y * y * z +
          10.378311574405206 * x * x * x * x * z * z * z +
          15.56746736160781 * x * x * y * y * y * y * z -
          62.269869446431238 * x * x * y * y * z * z * z -
          3.1134934723215619 * y * y * y * y * y * y * z +
          10.378311574405206 * y * y * y * y * z * z * z;
    break;
  case 61: // l=7, m=5
    val = -0.51891557872026032 * x * x * x * x * x * x * x +
          4.6702402084823429 * x * x * x * x * x * y * y +
          6.2269869446431238 * x * x * x * x * x * z * z +
          2.5945778936013016 * x * x * x * y * y * y * y -
          62.269869446431238 * x * x * x * y * y * z * z -
          2.5945778936013016 * x * y * y * y * y * y * y +
          31.134934723215619 * x * y * y * y * y * z * z;
    break;
  case 62: // l=7, m=6
    val = 2.6459606618019002 * x * x * x * x * x * x * z -
          39.689409927028503 * x * x * x * x * y * y * z +
          39.689409927028503 * x * x * y * y * y * y * z -
          2.6459606618019002 * y * y * y * y * y * y * z;
    break;
  case 63: // l=7, m=7
    val = 0.70716273252459618 * x * x * x * x * x * x * x -
          14.85041738301652 * x * x * x * x * x * y * y +
          24.750695638360866 * x * x * x * y * y * y * y -
          4.9501391276721732 * x * y * y * y * y * y * y;
    break;
  default:
    break;
  }
}

KOKKOS_INLINE_FUNCTION
void real_solid_harmonic_cart_and_grad_precomputed(
    const int l, const int m, const double x, const double y, const double z,
    double &val, double &dx, double &dy, double &dz) {

  // Initialize outputs upfront to zero out registers and clean up branch code
  val = 0.0;
  dx = 0.0;
  dy = 0.0;
  dz = 0.0;

  const int idx = l * l + l + m;
  switch (idx) {
  case 0: // l=0, m=0
    val = 0.28209479177387814;
    break;
  case 1: // l=1, m=-1
    val = 0.48860251190291992 * y;
    dy = 0.48860251190291992;
    break;
  case 2: // l=1, m=0
    val = 0.48860251190291992 * z;
    dz = 0.48860251190291992;
    break;
  case 3: // l=1, m=1
    val = 0.48860251190291992 * x;
    dx = 0.48860251190291992;
    break;
  case 4: // l=2, m=-2
    val = 1.0925484305920791 * x * y;
    dx = 1.0925484305920791 * y;
    dy = 1.0925484305920791 * x;
    break;
  case 5: // l=2, m=-1
    val = 1.0925484305920791 * y * z;
    dy = 1.0925484305920791 * z;
    dz = 1.0925484305920791 * y;
    break;
  case 6: // l=2, m=0
    val = -0.31539156525252001 * x * x - 0.31539156525252001 * y * y +
          0.63078313050504001 * z * z;
    dx = -0.63078313050504001 * x;
    dy = -0.63078313050504001 * y;
    dz = 1.26156626101008 * z;
    break;
  case 7: // l=2, m=1
    val = 1.0925484305920791 * x * z;
    dx = 1.0925484305920791 * z;
    dz = 1.0925484305920791 * x;
    break;
  case 8: // l=2, m=2
    val = 0.54627421529603954 * x * x - 0.54627421529603954 * y * y;
    dx = 1.0925484305920791 * x;
    dy = -1.0925484305920791 * y;
    break;
  case 9: // l=3, m=-3
    val = 1.7701307697799305 * x * x * y - 0.59004358992664351 * y * y * y;
    dx = 3.5402615395598611 * x * y;
    dy = 1.7701307697799305 * x * x - 1.7701307697799305 * y * y;
    break;
  case 10: // l=3, m=-2
    val = 2.8906114426405541 * x * y * z;
    dx = 2.8906114426405541 * y * z;
    dy = 2.8906114426405541 * x * z;
    dz = 2.8906114426405541 * x * y;
    break;
  case 11: // l=3, m=-1
    val = -0.45704579946446574 * x * x * y - 0.45704579946446574 * y * y * y +
          1.8281831978578629 * y * z * z;
    dx = -0.91409159892893147 * x * y;
    dy = -0.45704579946446574 * x * x - 1.3711373983933972 * y * y +
         1.8281831978578629 * z * z;
    dz = 3.6563663957157259 * y * z;
    break;
  case 12: // l=3, m=0
    val = -1.1195289977703462 * x * x * z - 1.1195289977703462 * y * y * z +
          0.74635266518023078 * z * z * z;
    dx = -2.2390579955406924 * x * z;
    dy = -2.2390579955406924 * y * z;
    dz = -1.1195289977703462 * x * x - 1.1195289977703462 * y * y +
         2.2390579955406924 * z * z;
    break;
  case 13: // l=3, m=1
    val = -0.45704579946446574 * x * x * x - 0.45704579946446574 * x * y * y +
          1.8281831978578629 * x * z * z;
    dx = -1.3711373983933972 * x * x - 0.45704579946446574 * y * y +
         1.8281831978578629 * z * z;
    dy = -0.91409159892893147 * x * y;
    dz = 3.6563663957157259 * x * z;
    break;
  case 14: // l=3, m=2
    val = 1.445305721320277 * x * x * z - 1.445305721320277 * y * y * z;
    dx = 2.8906114426405541 * x * z;
    dy = -2.8906114426405541 * y * z;
    dz = 1.445305721320277 * x * x - 1.445305721320277 * y * y;
    break;
  case 15: // l=3, m=3
    val = 0.59004358992664351 * x * x * x - 1.7701307697799305 * x * y * y;
    dx = 1.7701307697799305 * x * x - 1.7701307697799305 * y * y;
    dy = -3.5402615395598611 * x * y;
    break;
  case 16: // l=4, m=-4
    val =
        2.5033429417967045 * x * x * x * y - 2.5033429417967045 * x * y * y * y;
    dx = 7.5100288253901136 * x * x * y - 2.5033429417967045 * y * y * y;
    dy = 2.5033429417967045 * x * x * x - 7.5100288253901136 * x * y * y;
    break;
  case 17: // l=4, m=-3
    val =
        5.3103923093397916 * x * x * y * z - 1.7701307697799305 * y * y * y * z;
    dx = 10.620784618679583 * x * y * z;
    dy = 5.3103923093397916 * x * x * z - 5.3103923093397916 * y * y * z;
    dz = 5.3103923093397916 * x * x * y - 1.7701307697799305 * y * y * y;
    break;
  case 18: // l=4, m=-2
    val = -0.94617469575756002 * x * x * x * y -
          0.94617469575756002 * x * y * y * y +
          5.6770481745453601 * x * y * z * z;
    dx = -2.8385240872726801 * x * x * y - 0.94617469575756002 * y * y * y +
         5.6770481745453601 * y * z * z;
    dy = -0.94617469575756002 * x * x * x - 2.8385240872726801 * x * y * y +
         5.6770481745453601 * x * z * z;
    dz = 11.35409634909072 * x * y * z;
    break;
  case 19: // l=4, m=-1
    val = -2.0071396306718675 * x * x * y * z -
          2.0071396306718675 * y * y * y * z +
          2.6761861742291567 * y * z * z * z;
    dx = -4.014279261343735 * x * y * z;
    dy = -2.0071396306718675 * x * x * z - 6.0214188920156025 * y * y * z +
         2.6761861742291567 * z * z * z;
    dz = -2.0071396306718675 * x * x * y - 2.0071396306718675 * y * y * y +
         8.02855852268747 * y * z * z;
    break;
  case 20: // l=4, m=0
    val = 0.31735664074561291 * x * x * x * x +
          0.63471328149122582 * x * x * y * y -
          2.5388531259649033 * x * x * z * z +
          0.31735664074561291 * y * y * y * y -
          2.5388531259649033 * y * y * z * z +
          0.84628437532163443 * z * z * z * z;
    dx = 1.2694265629824516 * x * x * x + 1.2694265629824516 * x * y * y -
         5.0777062519298066 * x * z * z;
    dy = 1.2694265629824516 * x * x * y + 1.2694265629824516 * y * y * y -
         5.0777062519298066 * y * z * z;
    dz = -5.0777062519298066 * x * x * z - 5.0777062519298066 * y * y * z +
         3.3851375012865377 * z * z * z;
    break;
  case 21: // l=4, m=1
    val = -2.0071396306718675 * x * x * x * z -
          2.0071396306718675 * x * y * y * z +
          2.6761861742291567 * x * z * z * z;
    dx = -6.0214188920156025 * x * x * z - 2.0071396306718675 * y * y * z +
         2.6761861742291567 * z * z * z;
    dy = -4.014279261343735 * x * y * z;
    dz = -2.0071396306718675 * x * x * x - 2.0071396306718675 * x * y * y +
         8.02855852268747 * x * z * z;
    break;
  case 22: // l=4, m=2
    val = -0.47308734787878001 * x * x * x * x +
          2.8385240872726801 * x * x * z * z +
          0.47308734787878001 * y * y * y * y -
          2.8385240872726801 * y * y * z * z;
    dx = -1.89234939151512 * x * x * x + 5.6770481745453601 * x * z * z;
    dy = 1.89234939151512 * y * y * y - 5.6770481745453601 * y * z * z;
    dz = 5.6770481745453601 * x * x * z - 5.6770481745453601 * y * y * z;
    break;
  case 23: // l=4, m=3
    val =
        1.7701307697799305 * x * x * x * z - 5.3103923093397916 * x * y * y * z;
    dx = 5.3103923093397916 * x * x * z - 5.3103923093397916 * y * y * z;
    dy = -10.620784618679583 * x * y * z;
    dz = 1.7701307697799305 * x * x * x - 5.3103923093397916 * x * y * y;
    break;
  case 24: // l=4, m=4
    val = 0.62583573544917613 * x * x * x * x -
          3.7550144126950568 * x * x * y * y +
          0.62583573544917613 * y * y * y * y;
    dx = 2.5033429417967045 * x * x * x - 7.5100288253901136 * x * y * y;
    dy = -7.5100288253901136 * x * x * y + 2.5033429417967045 * y * y * y;
    break;
  case 25: // l=5, m=-5
    val = 3.2819102842008505 * x * x * x * x * y -
          6.563820568401701 * x * x * y * y * y +
          0.6563820568401701 * y * y * y * y * y;
    dx =
        13.127641136803402 * x * x * x * y - 13.127641136803402 * x * y * y * y;
    dy = 3.2819102842008505 * x * x * x * x -
         19.691461705205103 * x * x * y * y +
         3.2819102842008505 * y * y * y * y;
    break;
  case 26: // l=5, m=-4
    val = 8.3026492595241651 * x * x * x * y * z -
          8.3026492595241651 * x * y * y * y * z;
    dx =
        24.907947778572495 * x * x * y * z - 8.3026492595241651 * y * y * y * z;
    dy =
        8.3026492595241651 * x * x * x * z - 24.907947778572495 * x * y * y * z;
    dz =
        8.3026492595241651 * x * x * x * y - 8.3026492595241651 * x * y * y * y;
    break;
  case 27: // l=5, m=-3
    val = -1.4677148983057512 * x * x * x * x * y -
          0.97847659887050078 * x * x * y * y * y +
          11.741719186446009 * x * x * y * z * z +
          0.48923829943525039 * y * y * y * y * y -
          3.9139063954820031 * y * y * y * z * z;
    dx = -5.8708595932230047 * x * x * x * y -
         1.9569531977410016 * x * y * y * y +
         23.483438372892019 * x * y * z * z;
    dy = -1.4677148983057512 * x * x * x * x -
         2.9354297966115023 * x * x * y * y +
         11.741719186446009 * x * x * z * z +
         2.4461914971762519 * y * y * y * y -
         11.741719186446009 * y * y * z * z;
    dz =
        23.483438372892019 * x * x * y * z - 7.8278127909640062 * y * y * y * z;
    break;
  case 28: // l=5, m=-2
    val = -4.7935367849733238 * x * x * x * y * z -
          4.7935367849733238 * x * y * y * y * z +
          9.5870735699466475 * x * y * z * z * z;
    dx = -14.380610354919971 * x * x * y * z -
         4.7935367849733238 * y * y * y * z +
         9.5870735699466475 * y * z * z * z;
    dy = -4.7935367849733238 * x * x * x * z -
         14.380610354919971 * x * y * y * z +
         9.5870735699466475 * x * z * z * z;
    dz = -4.7935367849733238 * x * x * x * y -
         4.7935367849733238 * x * y * y * y +
         28.761220709839943 * x * y * z * z;
    break;
  case 29: // l=5, m=-1
    val = 0.45294665119569692 * x * x * x * x * y +
          0.90589330239139384 * x * x * y * y * y -
          5.4353598143483631 * x * x * y * z * z +
          0.45294665119569692 * y * y * y * y * y -
          5.4353598143483631 * y * y * y * z * z +
          3.6235732095655754 * y * z * z * z * z;
    dx = 1.8117866047827877 * x * x * x * y +
         1.8117866047827877 * x * y * y * y -
         10.870719628696726 * x * y * z * z;
    dy = 0.45294665119569692 * x * x * x * x +
         2.7176799071741815 * x * x * y * y -
         5.4353598143483631 * x * x * z * z +
         2.2647332559784846 * y * y * y * y -
         16.306079443045089 * y * y * z * z +
         3.6235732095655754 * z * z * z * z;
    dz = -10.870719628696726 * x * x * y * z -
         10.870719628696726 * y * y * y * z +
         14.494292838262301 * y * z * z * z;
    break;
  case 30: // l=5, m=0
    val = 1.7542548368013539 * x * x * x * x * z +
          3.5085096736027079 * x * x * y * y * z -
          4.6780128981369439 * x * x * z * z * z +
          1.7542548368013539 * y * y * y * y * z -
          4.6780128981369439 * y * y * z * z * z +
          0.93560257962738877 * z * z * z * z * z;
    dx = 7.0170193472054158 * x * x * x * z +
         7.0170193472054158 * x * y * y * z -
         9.3560257962738877 * x * z * z * z;
    dy = 7.0170193472054158 * x * x * y * z +
         7.0170193472054158 * y * y * y * z -
         9.3560257962738877 * y * z * z * z;
    dz = 1.7542548368013539 * x * x * x * x +
         3.5085096736027079 * x * x * y * y -
         14.034038694410832 * x * x * z * z +
         1.7542548368013539 * y * y * y * y -
         14.034038694410832 * y * y * z * z +
         4.6780128981369439 * z * z * z * z;
    break;
  case 31: // l=5, m=1
    val = 0.45294665119569692 * x * x * x * x * x +
          0.90589330239139384 * x * x * x * y * y -
          5.4353598143483631 * x * x * x * z * z +
          0.45294665119569692 * x * y * y * y * y -
          5.4353598143483631 * x * y * y * z * z +
          3.6235732095655754 * x * z * z * z * z;
    dx = 2.2647332559784846 * x * x * x * x +
         2.7176799071741815 * x * x * y * y -
         16.306079443045089 * x * x * z * z +
         0.45294665119569692 * y * y * y * y -
         5.4353598143483631 * y * y * z * z +
         3.6235732095655754 * z * z * z * z;
    dy = 1.8117866047827877 * x * x * x * y +
         1.8117866047827877 * x * y * y * y -
         10.870719628696726 * x * y * z * z;
    dz = -10.870719628696726 * x * x * x * z -
         10.870719628696726 * x * y * y * z +
         14.494292838262301 * x * z * z * z;
    break;
  case 32: // l=5, m=2
    val = -2.3967683924866619 * x * x * x * x * z +
          4.7935367849733238 * x * x * z * z * z +
          2.3967683924866619 * y * y * y * y * z -
          4.7935367849733238 * y * y * z * z * z;
    dx = -9.5870735699466475 * x * x * x * z +
         9.5870735699466475 * x * z * z * z;
    dy =
        9.5870735699466475 * y * y * y * z - 9.5870735699466475 * y * z * z * z;
    dz = -2.3967683924866619 * x * x * x * x +
         14.380610354919971 * x * x * z * z +
         2.3967683924866619 * y * y * y * y -
         14.380610354919971 * y * y * z * z;
    break;
  case 33: // l=5, m=3
    val = -0.48923829943525039 * x * x * x * x * x +
          0.97847659887050078 * x * x * x * y * y +
          3.9139063954820031 * x * x * x * z * z +
          1.4677148983057512 * x * y * y * y * y -
          11.741719186446009 * x * y * y * z * z;
    dx = -2.4461914971762519 * x * x * x * x +
         2.9354297966115023 * x * x * y * y +
         11.741719186446009 * x * x * z * z +
         1.4677148983057512 * y * y * y * y -
         11.741719186446009 * y * y * z * z;
    dy = 1.9569531977410016 * x * x * x * y +
         5.8708595932230047 * x * y * y * y -
         23.483438372892019 * x * y * z * z;
    dz =
        7.8278127909640062 * x * x * x * z - 23.483438372892019 * x * y * y * z;
    break;
  case 34: // l=5, m=4
    val = 2.0756623148810413 * x * x * x * x * z -
          12.453973889286248 * x * x * y * y * z +
          2.0756623148810413 * y * y * y * y * z;
    dx =
        8.3026492595241651 * x * x * x * z - 24.907947778572495 * x * y * y * z;
    dy = -24.907947778572495 * x * x * y * z +
         8.3026492595241651 * y * y * y * z;
    dz = 2.0756623148810413 * x * x * x * x -
         12.453973889286248 * x * x * y * y +
         2.0756623148810413 * y * y * y * y;
    break;
  case 35: // l=5, m=5
    val = 0.6563820568401701 * x * x * x * x * x -
          6.563820568401701 * x * x * x * y * y +
          3.2819102842008505 * x * y * y * y * y;
    dx = 3.2819102842008505 * x * x * x * x -
         19.691461705205103 * x * x * y * y +
         3.2819102842008505 * y * y * y * y;
    dy = -13.127641136803402 * x * x * x * y +
         13.127641136803402 * x * y * y * y;
    break;
  case 36: // l=6, m=-6
    val = 4.0991046311514859 * x * x * x * x * x * y -
          13.663682103838286 * x * x * x * y * y * y +
          4.0991046311514859 * x * y * y * y * y * y;
    dx = 20.49552315575743 * x * x * x * x * y -
         40.991046311514859 * x * x * y * y * y +
         4.0991046311514859 * y * y * y * y * y;
    dy = 4.0991046311514859 * x * x * x * x * x -
         40.991046311514859 * x * x * x * y * y +
         20.49552315575743 * x * y * y * y * y;
    break;
  case 37: // l=6, m=-5
    val = 11.83309581115876 * x * x * x * x * y * z -
          23.66619162231752 * x * x * y * y * y * z +
          2.366619162231752 * y * y * y * y * y * z;
    dx = 47.332383244635041 * x * x * x * y * z -
         47.332383244635041 * x * y * y * y * z;
    dy = 11.83309581115876 * x * x * x * x * z -
         70.998574866952561 * x * x * y * y * z +
         11.83309581115876 * y * y * y * y * z;
    dz = 11.83309581115876 * x * x * x * x * y -
         23.66619162231752 * x * x * y * y * y +
         2.366619162231752 * y * y * y * y * y;
    break;
  case 38: // l=6, m=-4
    val = -2.0182596029148966 * x * x * x * x * x * y +
          20.182596029148966 * x * x * x * y * z * z +
          2.0182596029148966 * x * y * y * y * y * y -
          20.182596029148966 * x * y * y * y * z * z;
    dx = -10.091298014574483 * x * x * x * x * y +
         60.547788087446899 * x * x * y * z * z +
         2.0182596029148966 * y * y * y * y * y -
         20.182596029148966 * y * y * y * z * z;
    dy = -2.0182596029148966 * x * x * x * x * x +
         20.182596029148966 * x * x * x * z * z +
         10.091298014574483 * x * y * y * y * y -
         60.547788087446899 * x * y * y * z * z;
    dz = 40.365192058297933 * x * x * x * y * z -
         40.365192058297933 * x * y * y * y * z;
    break;
  case 39: // l=6, m=-3
    val = -8.2908473356343115 * x * x * x * x * y * z -
          5.527231557089541 * x * x * y * y * y * z +
          22.108926228358164 * x * x * y * z * z * z +
          2.7636157785447705 * y * y * y * y * y * z -
          7.369642076119388 * y * y * y * z * z * z;
    dx = -33.163389342537246 * x * x * x * y * z -
         11.054463114179082 * x * y * y * y * z +
         44.217852456716328 * x * y * z * z * z;
    dy = -8.2908473356343115 * x * x * x * x * z -
         16.581694671268623 * x * x * y * y * z +
         22.108926228358164 * x * x * z * z * z +
         13.818078892723852 * y * y * y * y * z -
         22.108926228358164 * y * y * z * z * z;
    dz = -8.2908473356343115 * x * x * x * x * y -
         5.527231557089541 * x * x * y * y * y +
         66.326778685074492 * x * x * y * z * z +
         2.7636157785447705 * y * y * y * y * y -
         22.108926228358164 * y * y * y * z * z;
    break;
  case 40: // l=6, m=-2
    val = 0.9212052595149235 * x * x * x * x * x * y +
          1.842410519029847 * x * x * x * y * y * y -
          14.739284152238776 * x * x * x * y * z * z +
          0.9212052595149235 * x * y * y * y * y * y -
          14.739284152238776 * x * y * y * y * z * z +
          14.739284152238776 * x * y * z * z * z * z;
    dx = 4.6060262975746175 * x * x * x * x * y +
         5.527231557089541 * x * x * y * y * y -
         44.217852456716328 * x * x * y * z * z +
         0.9212052595149235 * y * y * y * y * y -
         14.739284152238776 * y * y * y * z * z +
         14.739284152238776 * y * z * z * z * z;
    dy = 0.9212052595149235 * x * x * x * x * x +
         5.527231557089541 * x * x * x * y * y -
         14.739284152238776 * x * x * x * z * z +
         4.6060262975746175 * x * y * y * y * y -
         44.217852456716328 * x * y * y * z * z +
         14.739284152238776 * x * z * z * z * z;
    dz = -29.478568304477552 * x * x * x * y * z -
         29.478568304477552 * x * y * y * y * z +
         58.957136608955104 * x * y * z * z * z;
    break;
  case 41: // l=6, m=-1
    val = 2.9131068125936569 * x * x * x * x * y * z +
          5.8262136251873139 * x * x * y * y * y * z -
          11.652427250374628 * x * x * y * z * z * z +
          2.9131068125936569 * y * y * y * y * y * z -
          11.652427250374628 * y * y * y * z * z * z +
          4.6609709001498511 * y * z * z * z * z * z;
    dx = 11.652427250374628 * x * x * x * y * z +
         11.652427250374628 * x * y * y * y * z -
         23.304854500749256 * x * y * z * z * z;
    dy = 2.9131068125936569 * x * x * x * x * z +
         17.478640875561942 * x * x * y * y * z -
         11.652427250374628 * x * x * z * z * z +
         14.565534062968285 * y * y * y * y * z -
         34.957281751123883 * y * y * z * z * z +
         4.6609709001498511 * z * z * z * z * z;
    dz = 2.9131068125936569 * x * x * x * x * y +
         5.8262136251873139 * x * x * y * y * y -
         34.957281751123883 * x * x * y * z * z +
         2.9131068125936569 * y * y * y * y * y -
         34.957281751123883 * y * y * y * z * z +
         23.304854500749256 * y * z * z * z * z;
    break;
  case 42: // l=6, m=0
    val = -0.31784601133814213 * x * x * x * x * x * x -
          0.95353803401442639 * x * x * x * x * y * y +
          5.7212282040865583 * x * x * x * x * z * z -
          0.95353803401442639 * x * x * y * y * y * y +
          11.442456408173117 * x * x * y * y * z * z -
          7.6283042721154111 * x * x * z * z * z * z -
          0.31784601133814213 * y * y * y * y * y * y +
          5.7212282040865583 * y * y * y * y * z * z -
          7.6283042721154111 * y * y * z * z * z * z +
          1.0171072362820548 * z * z * z * z * z * z;
    dx = -1.9070760680288528 * x * x * x * x * x -
         3.8141521360577056 * x * x * x * y * y +
         22.884912816346233 * x * x * x * z * z -
         1.9070760680288528 * x * y * y * y * y +
         22.884912816346233 * x * y * y * z * z -
         15.256608544230822 * x * z * z * z * z;
    dy = -1.9070760680288528 * x * x * x * x * y -
         3.8141521360577056 * x * x * y * y * y +
         22.884912816346233 * x * x * y * z * z -
         1.9070760680288528 * y * y * y * y * y +
         22.884912816346233 * y * y * y * z * z -
         15.256608544230822 * y * z * z * z * z;
    dz = 11.442456408173117 * x * x * x * x * z +
         22.884912816346233 * x * x * y * y * z -
         30.513217088461644 * x * x * z * z * z +
         11.442456408173117 * y * y * y * y * z -
         30.513217088461644 * y * y * z * z * z +
         6.1026434176923289 * z * z * z * z * z;
    break;
  case 43: // l=6, m=1
    val = 2.9131068125936569 * x * x * x * x * x * z +
          5.8262136251873139 * x * x * x * y * y * z -
          11.652427250374628 * x * x * x * z * z * z +
          2.9131068125936569 * x * y * y * y * y * z -
          11.652427250374628 * x * y * y * z * z * z +
          4.6609709001498511 * x * z * z * z * z * z;
    dx = 14.565534062968285 * x * x * x * x * z +
         17.478640875561942 * x * x * y * y * z -
         34.957281751123883 * x * x * z * z * z +
         2.9131068125936569 * y * y * y * y * z -
         11.652427250374628 * y * y * z * z * z +
         4.6609709001498511 * z * z * z * z * z;
    dy = 11.652427250374628 * x * x * x * y * z +
         11.652427250374628 * x * y * y * y * z -
         23.304854500749256 * x * y * z * z * z;
    dz = 2.9131068125936569 * x * x * x * x * x +
         5.8262136251873139 * x * x * x * y * y -
         34.957281751123883 * x * x * x * z * z +
         2.9131068125936569 * x * y * y * y * y -
         34.957281751123883 * x * y * y * z * z +
         23.304854500749256 * x * z * z * z * z;
    break;
  case 44: // l=6, m=2
    val = 0.46060262975746175 * x * x * x * x * x * x +
          0.46060262975746175 * x * x * x * x * y * y -
          7.369642076119388 * x * x * x * x * z * z -
          0.46060262975746175 * x * x * y * y * y * y +
          7.369642076119388 * x * x * z * z * z * z -
          0.46060262975746175 * y * y * y * y * y * y +
          7.369642076119388 * y * y * y * y * z * z -
          7.369642076119388 * y * y * z * z * z * z;
    dx = 2.7636157785447705 * x * x * x * x * x +
         1.842410519029847 * x * x * x * y * y -
         29.478568304477552 * x * x * x * z * z -
         0.9212052595149235 * x * y * y * y * y +
         14.739284152238776 * x * z * z * z * z;
    dy = 0.9212052595149235 * x * x * x * x * y -
         1.842410519029847 * x * x * y * y * y -
         2.7636157785447705 * y * y * y * y * y +
         29.478568304477552 * y * y * y * z * z -
         14.739284152238776 * y * z * z * z * z;
    dz = -14.739284152238776 * x * x * x * x * z +
         29.478568304477552 * x * x * z * z * z +
         14.739284152238776 * y * y * y * y * z -
         29.478568304477552 * y * y * z * z * z;
    break;
  case 45: // l=6, m=3
    val = -2.7636157785447705 * x * x * x * x * x * z +
          5.527231557089541 * x * x * x * y * y * z +
          7.369642076119388 * x * x * x * z * z * z +
          8.2908473356343115 * x * y * y * y * y * z -
          22.108926228358164 * x * y * y * z * z * z;
    dx = -13.818078892723852 * x * x * x * x * z +
         16.581694671268623 * x * x * y * y * z +
         22.108926228358164 * x * x * z * z * z +
         8.2908473356343115 * y * y * y * y * z -
         22.108926228358164 * y * y * z * z * z;
    dy = 11.054463114179082 * x * x * x * y * z +
         33.163389342537246 * x * y * y * y * z -
         44.217852456716328 * x * y * z * z * z;
    dz = -2.7636157785447705 * x * x * x * x * x +
         5.527231557089541 * x * x * x * y * y +
         22.108926228358164 * x * x * x * z * z +
         8.2908473356343115 * x * y * y * y * y -
         66.326778685074492 * x * y * y * z * z;
    break;
  case 46: // l=6, m=4
    val = -0.50456490072872416 * x * x * x * x * x * x +
          2.5228245036436208 * x * x * x * x * y * y +
          5.0456490072872416 * x * x * x * x * z * z +
          2.5228245036436208 * x * x * y * y * y * y -
          30.27389404372345 * x * x * y * y * z * z -
          0.50456490072872416 * y * y * y * y * y * y +
          5.0456490072872416 * y * y * y * y * z * z;
    dx = -3.027389404372345 * x * x * x * x * x +
         10.091298014574483 * x * x * x * y * y +
         20.182596029148966 * x * x * x * z * z +
         5.0456490072872416 * x * y * y * y * y -
         60.547788087446899 * x * y * y * z * z;
    dy = 5.0456490072872416 * x * x * x * x * y +
         10.091298014574483 * x * x * y * y * y -
         60.547788087446899 * x * x * y * z * z -
         3.027389404372345 * y * y * y * y * y +
         20.182596029148966 * y * y * y * z * z;
    dz = 10.091298014574483 * x * x * x * x * z -
         60.547788087446899 * x * x * y * y * z +
         10.091298014574483 * y * y * y * y * z;
    break;
  case 47: // l=6, m=5
    val = 2.366619162231752 * x * x * x * x * x * z -
          23.66619162231752 * x * x * x * y * y * z +
          11.83309581115876 * x * y * y * y * y * z;
    dx = 11.83309581115876 * x * x * x * x * z -
         70.998574866952561 * x * x * y * y * z +
         11.83309581115876 * y * y * y * y * z;
    dy = -47.332383244635041 * x * x * x * y * z +
         47.332383244635041 * x * y * y * y * z;
    dz = 2.366619162231752 * x * x * x * x * x -
         23.66619162231752 * x * x * x * y * y +
         11.83309581115876 * x * y * y * y * y;
    break;
  case 48: // l=6, m=6
    val = 0.68318410519191432 * x * x * x * x * x * x -
          10.247761577878715 * x * x * x * x * y * y +
          10.247761577878715 * x * x * y * y * y * y -
          0.68318410519191432 * y * y * y * y * y * y;
    dx = 4.0991046311514859 * x * x * x * x * x -
         40.991046311514859 * x * x * x * y * y +
         20.49552315575743 * x * y * y * y * y;
    dy = -20.49552315575743 * x * x * x * x * y +
         40.991046311514859 * x * x * y * y * y -
         4.0991046311514859 * y * y * y * y * y;
    break;
  case 49: // l=7, m=-7
    val = 4.9501391276721732 * x * x * x * x * x * x * y -
          24.750695638360866 * x * x * x * x * y * y * y +
          14.85041738301652 * x * x * y * y * y * y * y -
          0.70716273252459618 * y * y * y * y * y * y * y;
    dx = 29.70083476603304 * x * x * x * x * x * y -
         99.002782553443465 * x * x * x * y * y * y +
         29.70083476603304 * x * y * y * y * y * y;
    dy = 4.9501391276721732 * x * x * x * x * x * x -
         74.252086915082599 * x * x * x * x * y * y +
         74.252086915082599 * x * x * y * y * y * y -
         4.9501391276721732 * y * y * y * y * y * y;
    break;
  case 50: // l=7, m=-6
    val = 15.875763970811401 * x * x * x * x * x * y * z -
          52.919213236038004 * x * x * x * y * y * y * z +
          15.875763970811401 * x * y * y * y * y * y * z;
    dx = 79.378819854057007 * x * x * x * x * y * z -
         158.75763970811401 * x * x * y * y * y * z +
         15.875763970811401 * y * y * y * y * y * z;
    dy = 15.875763970811401 * x * x * x * x * x * z -
         158.75763970811401 * x * x * x * y * y * z +
         79.378819854057007 * x * y * y * y * y * z;
    dz = 15.875763970811401 * x * x * x * x * x * y -
         52.919213236038004 * x * x * x * y * y * y +
         15.875763970811401 * x * y * y * y * y * y;
    break;
  case 51: // l=7, m=-5
    val = -2.5945778936013016 * x * x * x * x * x * x * y +
          2.5945778936013016 * x * x * x * x * y * y * y +
          31.134934723215619 * x * x * x * x * y * z * z +
          4.6702402084823429 * x * x * y * y * y * y * y -
          62.269869446431238 * x * x * y * y * y * z * z -
          0.51891557872026032 * y * y * y * y * y * y * y +
          6.2269869446431238 * y * y * y * y * y * z * z;
    dx = -15.56746736160781 * x * x * x * x * x * y +
         10.378311574405206 * x * x * x * y * y * y +
         124.53973889286248 * x * x * x * y * z * z +
         9.3404804169646858 * x * y * y * y * y * y -
         124.53973889286248 * x * y * y * y * z * z;
    dy = -2.5945778936013016 * x * x * x * x * x * x +
         7.7837336808039048 * x * x * x * x * y * y +
         31.134934723215619 * x * x * x * x * z * z +
         23.351201042411714 * x * x * y * y * y * y -
         186.80960833929372 * x * x * y * y * z * z -
         3.6324090510418222 * y * y * y * y * y * y +
         31.134934723215619 * y * y * y * y * z * z;
    dz = 62.269869446431238 * x * x * x * x * y * z -
         124.53973889286248 * x * x * y * y * y * z +
         12.453973889286248 * y * y * y * y * y * z;
    break;
  case 52: // l=7, m=-4
    val = -12.453973889286248 * x * x * x * x * x * y * z +
          41.513246297620826 * x * x * x * y * z * z * z +
          12.453973889286248 * x * y * y * y * y * y * z -
          41.513246297620826 * x * y * y * y * z * z * z;
    dx = -62.269869446431238 * x * x * x * x * y * z +
         124.53973889286248 * x * x * y * z * z * z +
         12.453973889286248 * y * y * y * y * y * z -
         41.513246297620826 * y * y * y * z * z * z;
    dy = -12.453973889286248 * x * x * x * x * x * z +
         41.513246297620826 * x * x * x * z * z * z +
         62.269869446431238 * x * y * y * y * y * z -
         124.53973889286248 * x * y * y * z * z * z;
    dz = -12.453973889286248 * x * x * x * x * x * y +
         124.53973889286248 * x * x * x * y * z * z +
         12.453973889286248 * x * y * y * y * y * y -
         124.53973889286248 * x * y * y * y * z * z;
    break;
  case 53: // l=7, m=-3
    val = 1.4081304047606463 * x * x * x * x * x * x * y +
          2.3468840079344105 * x * x * x * x * y * y * y -
          28.162608095212926 * x * x * x * x * y * z * z +
          0.4693768015868821 * x * x * y * y * y * y * y -
          18.775072063475284 * x * x * y * y * y * z * z +
          37.550144126950568 * x * x * y * z * z * z * z -
          0.4693768015868821 * y * y * y * y * y * y * y +
          9.387536031737642 * y * y * y * y * y * z * z -
          12.516714708983523 * y * y * y * z * z * z * z;
    dx = 8.4487824285638778 * x * x * x * x * x * y +
         9.387536031737642 * x * x * x * y * y * y -
         112.6504323808517 * x * x * x * y * z * z +
         0.9387536031737642 * x * y * y * y * y * y -
         37.550144126950568 * x * y * y * y * z * z +
         75.100288253901136 * x * y * z * z * z * z;
    dy = 1.4081304047606463 * x * x * x * x * x * x +
         7.0406520238032315 * x * x * x * x * y * y -
         28.162608095212926 * x * x * x * x * z * z +
         2.3468840079344105 * x * x * y * y * y * y -
         56.325216190425852 * x * x * y * y * z * z +
         37.550144126950568 * x * x * z * z * z * z -
         3.2856376111081747 * y * y * y * y * y * y +
         46.93768015868821 * y * y * y * y * z * z -
         37.550144126950568 * y * y * z * z * z * z;
    dz = -56.325216190425852 * x * x * x * x * y * z -
         37.550144126950568 * x * x * y * y * y * z +
         150.20057650780227 * x * x * y * z * z * z +
         18.775072063475284 * y * y * y * y * y * z -
         50.066858835934091 * y * y * y * z * z * z;
    break;
  case 54: // l=7, m=-2
    val = 6.6379903866747395 * x * x * x * x * x * y * z +
          13.275980773349479 * x * x * x * y * y * y * z -
          35.402615395598611 * x * x * x * y * z * z * z +
          6.6379903866747395 * x * y * y * y * y * y * z -
          35.402615395598611 * x * y * y * y * z * z * z +
          21.241569237359166 * x * y * z * z * z * z * z;
    dx = 33.189951933373697 * x * x * x * x * y * z +
         39.827942320048437 * x * x * y * y * y * z -
         106.20784618679583 * x * x * y * z * z * z +
         6.6379903866747395 * y * y * y * y * y * z -
         35.402615395598611 * y * y * y * z * z * z +
         21.241569237359166 * y * z * z * z * z * z;
    dy = 6.6379903866747395 * x * x * x * x * x * z +
         39.827942320048437 * x * x * x * y * y * z -
         35.402615395598611 * x * x * x * z * z * z +
         33.189951933373697 * x * y * y * y * y * z -
         106.20784618679583 * x * y * y * z * z * z +
         21.241569237359166 * x * z * z * z * z * z;
    dz = 6.6379903866747395 * x * x * x * x * x * y +
         13.275980773349479 * x * x * x * y * y * y -
         106.20784618679583 * x * x * x * y * z * z +
         6.6379903866747395 * x * y * y * y * y * y -
         106.20784618679583 * x * y * y * y * z * z +
         106.20784618679583 * x * y * z * z * z * z;
    break;
  case 55: // l=7, m=-1
    val = -0.45165803791258657 * x * x * x * x * x * x * y -
          1.3549741137377597 * x * x * x * x * y * y * y +
          10.839792909902078 * x * x * x * x * y * z * z -
          1.3549741137377597 * x * x * y * y * y * y * y +
          21.679585819804155 * x * x * y * y * y * z * z -
          21.679585819804155 * x * x * y * z * z * z * z -
          0.45165803791258657 * y * y * y * y * y * y * y +
          10.839792909902078 * y * y * y * y * y * z * z -
          21.679585819804155 * y * y * y * z * z * z * z +
          5.7812228852811081 * y * z * z * z * z * z * z;
    dx = -2.7099482274755194 * x * x * x * x * x * y -
         5.4198964549510389 * x * x * x * y * y * y +
         43.359171639608311 * x * x * x * y * z * z -
         2.7099482274755194 * x * y * y * y * y * y +
         43.359171639608311 * x * y * y * y * z * z -
         43.359171639608311 * x * y * z * z * z * z;
    dy = -0.45165803791258657 * x * x * x * x * x * x -
         4.0649223412132791 * x * x * x * x * y * y +
         10.839792909902078 * x * x * x * x * z * z -
         6.7748705686887986 * x * x * y * y * y * y +
         65.038757459412466 * x * x * y * y * z * z -
         21.679585819804155 * x * x * z * z * z * z -
         3.161606265388106 * y * y * y * y * y * y +
         54.198964549510389 * y * y * y * y * z * z -
         65.038757459412466 * y * y * z * z * z * z +
         5.7812228852811081 * z * z * z * z * z * z;
    dz = 21.679585819804155 * x * x * x * x * y * z +
         43.359171639608311 * x * x * y * y * y * z -
         86.718343279216622 * x * x * y * z * z * z +
         21.679585819804155 * y * y * y * y * y * z -
         86.718343279216622 * y * y * y * z * z * z +
         34.687337311686649 * y * z * z * z * z * z;
    break;
  case 56: // l=7, m=0
    val = -2.389949691920173 * x * x * x * x * x * x * z -
          7.1698490757605189 * x * x * x * x * y * y * z +
          14.339698151521038 * x * x * x * x * z * z * z -
          7.1698490757605189 * x * x * y * y * y * y * z +
          28.679396303042076 * x * x * y * y * z * z * z -
          11.47175852121683 * x * x * z * z * z * z * z -
          2.389949691920173 * y * y * y * y * y * y * z +
          14.339698151521038 * y * y * y * y * z * z * z -
          11.47175852121683 * y * y * z * z * z * z * z +
          1.0925484305920791 * z * z * z * z * z * z * z;
    dx = -14.339698151521038 * x * x * x * x * x * z -
         28.679396303042076 * x * x * x * y * y * z +
         57.358792606084151 * x * x * x * z * z * z -
         14.339698151521038 * x * y * y * y * y * z +
         57.358792606084151 * x * y * y * z * z * z -
         22.94351704243366 * x * z * z * z * z * z;
    dy = -14.339698151521038 * x * x * x * x * y * z -
         28.679396303042076 * x * x * y * y * y * z +
         57.358792606084151 * x * x * y * z * z * z -
         14.339698151521038 * y * y * y * y * y * z +
         57.358792606084151 * y * y * y * z * z * z -
         22.94351704243366 * y * z * z * z * z * z;
    dz = -2.389949691920173 * x * x * x * x * x * x -
         7.1698490757605189 * x * x * x * x * y * y +
         43.019094454563113 * x * x * x * x * z * z -
         7.1698490757605189 * x * x * y * y * y * y +
         86.038188909126227 * x * x * y * y * z * z -
         57.358792606084151 * x * x * z * z * z * z -
         2.389949691920173 * y * y * y * y * y * y +
         43.019094454563113 * y * y * y * y * z * z -
         57.358792606084151 * y * y * z * z * z * z +
         7.6478390141445535 * z * z * z * z * z * z;
    break;
  case 57: // l=7, m=1
    val = -0.45165803791258657 * x * x * x * x * x * x * x -
          1.3549741137377597 * x * x * x * x * x * y * y +
          10.839792909902078 * x * x * x * x * x * z * z -
          1.3549741137377597 * x * x * x * y * y * y * y +
          21.679585819804155 * x * x * x * y * y * z * z -
          21.679585819804155 * x * x * x * z * z * z * z -
          0.45165803791258657 * x * y * y * y * y * y * y +
          10.839792909902078 * x * y * y * y * y * z * z -
          21.679585819804155 * x * y * y * z * z * z * z +
          5.7812228852811081 * x * z * z * z * z * z * z;
    dx = -3.161606265388106 * x * x * x * x * x * x -
         6.7748705686887986 * x * x * x * x * y * y +
         54.198964549510389 * x * x * x * x * z * z -
         4.0649223412132791 * x * x * y * y * y * y +
         65.038757459412466 * x * x * y * y * z * z -
         65.038757459412466 * x * x * z * z * z * z -
         0.45165803791258657 * y * y * y * y * y * y +
         10.839792909902078 * y * y * y * y * z * z -
         21.679585819804155 * y * y * z * z * z * z +
         5.7812228852811081 * z * z * z * z * z * z;
    dy = -2.7099482274755194 * x * x * x * x * x * y -
         5.4198964549510389 * x * x * x * y * y * y +
         43.359171639608311 * x * x * x * y * z * z -
         2.7099482274755194 * x * y * y * y * y * y +
         43.359171639608311 * x * y * y * y * z * z -
         43.359171639608311 * x * y * z * z * z * z;
    dz = 21.679585819804155 * x * x * x * x * x * z +
         43.359171639608311 * x * x * x * y * y * z -
         86.718343279216622 * x * x * x * z * z * z +
         21.679585819804155 * x * y * y * y * y * z -
         86.718343279216622 * x * y * y * z * z * z +
         34.687337311686649 * x * z * z * z * z * z;
    break;
  case 58: // l=7, m=2
    val = 3.3189951933373697 * x * x * x * x * x * x * z +
          3.3189951933373697 * x * x * x * x * y * y * z -
          17.701307697799305 * x * x * x * x * z * z * z -
          3.3189951933373697 * x * x * y * y * y * y * z +
          10.620784618679583 * x * x * z * z * z * z * z -
          3.3189951933373697 * y * y * y * y * y * y * z +
          17.701307697799305 * y * y * y * y * z * z * z -
          10.620784618679583 * y * y * z * z * z * z * z;
    dx = 19.913971160024218 * x * x * x * x * x * z +
         13.275980773349479 * x * x * x * y * y * z -
         70.805230791197221 * x * x * x * z * z * z -
         6.6379903866747395 * x * y * y * y * y * z +
         21.241569237359166 * x * z * z * z * z * z;
    dy = 6.6379903866747395 * x * x * x * x * y * z -
         13.275980773349479 * x * x * y * y * y * z -
         19.913971160024218 * y * y * y * y * y * z +
         70.805230791197221 * y * y * y * z * z * z -
         21.241569237359166 * y * z * z * z * z * z;
    dz = 3.3189951933373697 * x * x * x * x * x * x +
         3.3189951933373697 * x * x * x * x * y * y -
         53.103923093397916 * x * x * x * x * z * z -
         3.3189951933373697 * x * x * y * y * y * y +
         53.103923093397916 * x * x * z * z * z * z -
         3.3189951933373697 * y * y * y * y * y * y +
         53.103923093397916 * y * y * y * y * z * z -
         53.103923093397916 * y * y * z * z * z * z;
    break;
  case 59: // l=7, m=3
    val = 0.4693768015868821 * x * x * x * x * x * x * x -
          0.4693768015868821 * x * x * x * x * x * y * y -
          9.387536031737642 * x * x * x * x * x * z * z -
          2.3468840079344105 * x * x * x * y * y * y * y +
          18.775072063475284 * x * x * x * y * y * z * z +
          12.516714708983523 * x * x * x * z * z * z * z -
          1.4081304047606463 * x * y * y * y * y * y * y +
          28.162608095212926 * x * y * y * y * y * z * z -
          37.550144126950568 * x * y * y * z * z * z * z;
    dx = 3.2856376111081747 * x * x * x * x * x * x -
         2.3468840079344105 * x * x * x * x * y * y -
         46.93768015868821 * x * x * x * x * z * z -
         7.0406520238032315 * x * x * y * y * y * y +
         56.325216190425852 * x * x * y * y * z * z +
         37.550144126950568 * x * x * z * z * z * z -
         1.4081304047606463 * y * y * y * y * y * y +
         28.162608095212926 * y * y * y * y * z * z -
         37.550144126950568 * y * y * z * z * z * z;
    dy = -0.9387536031737642 * x * x * x * x * x * y -
         9.387536031737642 * x * x * x * y * y * y +
         37.550144126950568 * x * x * x * y * z * z -
         8.4487824285638778 * x * y * y * y * y * y +
         112.6504323808517 * x * y * y * y * z * z -
         75.100288253901136 * x * y * z * z * z * z;
    dz = -18.775072063475284 * x * x * x * x * x * z +
         37.550144126950568 * x * x * x * y * y * z +
         50.066858835934091 * x * x * x * z * z * z +
         56.325216190425852 * x * y * y * y * y * z -
         150.20057650780227 * x * y * y * z * z * z;
    break;
  case 60: // l=7, m=4
    val = -3.1134934723215619 * x * x * x * x * x * x * z +
          15.56746736160781 * x * x * x * x * y * y * z +
          10.378311574405206 * x * x * x * x * z * z * z +
          15.56746736160781 * x * x * y * y * y * y * z -
          62.269869446431238 * x * x * y * y * z * z * z -
          3.1134934723215619 * y * y * y * y * y * y * z +
          10.378311574405206 * y * y * y * y * z * z * z;
    dx = -18.680960833929372 * x * x * x * x * x * z +
         62.269869446431238 * x * x * x * y * y * z +
         41.513246297620826 * x * x * x * z * z * z +
         31.134934723215619 * x * y * y * y * y * z -
         124.53973889286248 * x * y * y * z * z * z;
    dy = 31.134934723215619 * x * x * x * x * y * z +
         62.269869446431238 * x * x * y * y * y * z -
         124.53973889286248 * x * x * y * z * z * z -
         18.680960833929372 * y * y * y * y * y * z +
         41.513246297620826 * y * y * y * z * z * z;
    dz = -3.1134934723215619 * x * x * x * x * x * x +
         15.56746736160781 * x * x * x * x * y * y +
         31.134934723215619 * x * x * x * x * z * z +
         15.56746736160781 * x * x * y * y * y * y -
         186.80960833929372 * x * x * y * y * z * z -
         3.1134934723215619 * y * y * y * y * y * y +
         31.134934723215619 * y * y * y * y * z * z;
    break;
  case 61: // l=7, m=5
    val = -0.51891557872026032 * x * x * x * x * x * x * x +
          4.6702402084823429 * x * x * x * x * x * y * y +
          6.2269869446431238 * x * x * x * x * x * z * z +
          2.5945778936013016 * x * x * x * y * y * y * y -
          62.269869446431238 * x * x * x * y * y * z * z -
          2.5945778936013016 * x * y * y * y * y * y * y +
          31.134934723215619 * x * y * y * y * y * z * z;
    dx = -3.6324090510418222 * x * x * x * x * x * x +
         23.351201042411714 * x * x * x * x * y * y +
         31.134934723215619 * x * x * x * x * z * z +
         7.7837336808039048 * x * x * y * y * y * y -
         186.80960833929372 * x * x * y * y * z * z -
         2.5945778936013016 * y * y * y * y * y * y +
         31.134934723215619 * y * y * y * y * z * z;
    dy = 9.3404804169646858 * x * x * x * x * x * y +
         10.378311574405206 * x * x * x * y * y * y -
         124.53973889286248 * x * x * x * y * z * z -
         15.56746736160781 * x * y * y * y * y * y +
         124.53973889286248 * x * y * y * y * z * z;
    dz = 12.453973889286248 * x * x * x * x * x * z -
         124.53973889286248 * x * x * x * y * y * z +
         62.269869446431238 * x * y * y * y * y * z;
    break;
  case 62: // l=7, m=6
    val = 2.6459606618019002 * x * x * x * x * x * x * z -
          39.689409927028503 * x * x * x * x * y * y * z +
          39.689409927028503 * x * x * y * y * y * y * z -
          2.6459606618019002 * y * y * y * y * y * y * z;
    dx = 15.875763970811401 * x * x * x * x * x * z -
         158.75763970811401 * x * x * x * y * y * z +
         79.378819854057007 * x * y * y * y * y * z;
    dy = -79.378819854057007 * x * x * x * x * y * z +
         158.75763970811401 * x * x * y * y * y * z -
         15.875763970811401 * y * y * y * y * y * z;
    dz = 2.6459606618019002 * x * x * x * x * x * x -
         39.689409927028503 * x * x * x * x * y * y +
         39.689409927028503 * x * x * y * y * y * y -
         2.6459606618019002 * y * y * y * y * y * y;
    break;
  case 63: // l=7, m=7
    val = 0.70716273252459618 * x * x * x * x * x * x * x -
          14.85041738301652 * x * x * x * x * x * y * y +
          24.750695638360866 * x * x * x * y * y * y * y -
          4.9501391276721732 * x * y * y * y * y * y * y;
    dx = 4.9501391276721732 * x * x * x * x * x * x -
         74.252086915082599 * x * x * x * x * y * y +
         74.252086915082599 * x * x * y * y * y * y -
         4.9501391276721732 * y * y * y * y * y * y;
    dy = -29.70083476603304 * x * x * x * x * x * y +
         99.002782553443465 * x * x * x * y * y * y -
         29.70083476603304 * x * y * y * y * y * y;
    break;
  default:
    break;
  }
}
} // namespace Nukexc
