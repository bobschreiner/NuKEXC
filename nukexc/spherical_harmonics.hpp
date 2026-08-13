/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Bob Schreiner
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 *
 *    Redistribution and use in source and binary forms, with or without
 *    modification, are permitted provided that the following conditions are
 *    met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 *    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
  case 0: { // l=0, m=0
    val = 0.28209479177387814;
    break;
  }
  case 1: { // l=1, m=-1
    val = 0.48860251190291992*y;
    break;
  }
  case 2: { // l=1, m=0
    val = 0.48860251190291992*z;
    break;
  }
  case 3: { // l=1, m=1
    val = 0.48860251190291992*x;
    break;
  }
  case 4: { // l=2, m=-2
    val = 1.0925484305920791*x*y;
    break;
  }
  case 5: { // l=2, m=-1
    val = 1.0925484305920791*y*z;
    break;
  }
  case 6: { // l=2, m=0
    val = -(0.31539156525252001*(x*x) + 0.31539156525252001*(y*y) - 0.63078313050504001*z*z);
    break;
  }
  case 7: { // l=2, m=1
    val = 1.0925484305920791*x*z;
    break;
  }
  case 8: { // l=2, m=2
    val = 0.54627421529603954*(x*x) - 0.54627421529603954*y*y;
    break;
  }
  case 9: { // l=3, m=-3
    val = (1.7701307697799305*y)*(x*x) - 0.59004358992664351*y*y*y;
    break;
  }
  case 10: { // l=3, m=-2
    val = 2.8906114426405541*x*y*z;
    break;
  }
  case 11: { // l=3, m=-1
    val = -((0.45704579946446574*y)*(x*x) - 1.8281831978578629*y*z*z + 0.45704579946446574*(y*y*y));
    break;
  }
  case 12: { // l=3, m=0
    const double t0 = 1.1195289977703462*z;
    val = -(t0*(x*x) + t0*(y*y) - 0.74635266518023078*z*z*z);
    break;
  }
  case 13: { // l=3, m=1
    val = -((0.45704579946446574*x)*(y*y) - 1.8281831978578629*x*z*z + 0.45704579946446574*(x*x*x));
    break;
  }
  case 14: { // l=3, m=2
    const double t0 = 1.445305721320277*z;
    val = t0*(x*x) - t0*y*y;
    break;
  }
  case 15: { // l=3, m=3
    val = -1.7701307697799305*x*y*y + 0.59004358992664351*(x*x*x);
    break;
  }
  case 16: { // l=4, m=-4
    val = -2.5033429417967045*x*y*y*y + (2.5033429417967045*y)*(x*x*x);
    break;
  }
  case 17: { // l=4, m=-3
    val = -1.7701307697799305*z*y*y*y + (5.3103923093397916*y*z)*(x*x);
    break;
  }
  case 18: { // l=4, m=-2
    val = -((0.94617469575756002*x)*(y*y*y) - 5.6770481745453601*x*y*z*z + (0.94617469575756002*y)*(x*x*x));
    break;
  }
  case 19: { // l=4, m=-1
    const double t0 = 2.0071396306718675*z;
    val = -(t0*y*(x*x) + t0*(y*y*y) - 2.6761861742291567*y*z*z*z);
    break;
  }
  case 20: { // l=4, m=0
    const double t0 = x*x;
    const double t1 = y*y;
    const double t2 = z*z;
    val = 0.63471328149122582*t0*t1 - 2.5388531259649033*t0*t2 - 2.5388531259649033*t1*t2 + 0.31735664074561291*(x*x*x*x) + 0.31735664074561291*(y*y*y*y) + 0.84628437532163443*(z*z*z*z);
    break;
  }
  case 21: { // l=4, m=1
    const double t0 = 2.0071396306718675*z;
    val = -(t0*x*(y*y) + t0*(x*x*x) - 2.6761861742291567*x*z*z*z);
    break;
  }
  case 22: { // l=4, m=2
    const double t0 = z*z;
    val = 2.8385240872726801*t0*(x*x) - 2.8385240872726801*t0*y*y - 0.47308734787878001*x*x*x*x + 0.47308734787878001*(y*y*y*y);
    break;
  }
  case 23: { // l=4, m=3
    val = -5.3103923093397916*x*z*y*y + (1.7701307697799305*z)*(x*x*x);
    break;
  }
  case 24: { // l=4, m=4
    val = -3.7550144126950568*x*x*y*y + 0.62583573544917613*(x*x*x*x) + 0.62583573544917613*(y*y*y*y);
    break;
  }
  case 25: { // l=5, m=-5
    val = (3.2819102842008505*y)*(x*x*x*x) - 6.563820568401701*x*x*y*y*y + 0.6563820568401701*(y*y*y*y*y);
    break;
  }
  case 26: { // l=5, m=-4
    const double t0 = 8.3026492595241651*z;
    val = -t0*x*y*y*y + t0*y*(x*x*x);
    break;
  }
  case 27: { // l=5, m=-3
    const double t0 = x*x;
    const double t1 = y*y*y;
    const double t2 = z*z;
    val = -(0.97847659887050078*t0*t1 - 11.741719186446009*t0*t2*y + 3.9139063954820031*t1*t2 + (1.4677148983057512*y)*(x*x*x*x) - 0.48923829943525039*y*y*y*y*y);
    break;
  }
  case 28: { // l=5, m=-2
    const double t0 = 4.7935367849733238*z;
    val = -(t0*x*(y*y*y) + t0*y*(x*x*x) - 9.5870735699466475*x*y*z*z*z);
    break;
  }
  case 29: { // l=5, m=-1
    const double t0 = x*x;
    const double t1 = y*y*y;
    const double t2 = z*z;
    val = 0.90589330239139384*t0*t1 - 5.4353598143483631*t0*t2*y - 5.4353598143483631*t1*t2 + (0.45294665119569692*y)*(x*x*x*x) + (3.6235732095655754*y)*(z*z*z*z) + 0.45294665119569692*(y*y*y*y*y);
    break;
  }
  case 30: { // l=5, m=0
    const double t0 = 1.7542548368013539*z;
    const double t1 = x*x;
    const double t2 = z*z*z;
    const double t3 = y*y;
    val = t0*(x*x*x*x) + t0*(y*y*y*y) - 4.6780128981369439*t1*t2 + 3.5085096736027079*t1*t3*z - 4.6780128981369439*t2*t3 + 0.93560257962738877*(z*z*z*z*z);
    break;
  }
  case 31: { // l=5, m=1
    const double t0 = x*x*x;
    const double t1 = y*y;
    const double t2 = z*z;
    val = 0.90589330239139384*t0*t1 - 5.4353598143483631*t0*t2 - 5.4353598143483631*t1*t2*x + (0.45294665119569692*x)*(y*y*y*y) + (3.6235732095655754*x)*(z*z*z*z) + 0.45294665119569692*(x*x*x*x*x);
    break;
  }
  case 32: { // l=5, m=2
    const double t0 = 2.3967683924866619*z;
    const double t1 = z*z*z;
    val = -t0*x*x*x*x + t0*(y*y*y*y) + 4.7935367849733238*t1*(x*x) - 4.7935367849733238*t1*y*y;
    break;
  }
  case 33: { // l=5, m=3
    const double t0 = x*x*x;
    const double t1 = y*y;
    const double t2 = z*z;
    val = 0.97847659887050078*t0*t1 + 3.9139063954820031*t0*t2 - 11.741719186446009*t1*t2*x + (1.4677148983057512*x)*(y*y*y*y) - 0.48923829943525039*x*x*x*x*x;
    break;
  }
  case 34: { // l=5, m=4
    const double t0 = 2.0756623148810413*z;
    val = t0*(x*x*x*x) + t0*(y*y*y*y) - 12.453973889286248*z*(x*x)*(y*y);
    break;
  }
  case 35: { // l=5, m=5
    val = (3.2819102842008505*x)*(y*y*y*y) - 6.563820568401701*x*x*x*y*y + 0.6563820568401701*(x*x*x*x*x);
    break;
  }
  case 36: { // l=6, m=-6
    val = (4.0991046311514859*x)*(y*y*y*y*y) + (4.0991046311514859*y)*(x*x*x*x*x) - 13.663682103838286*x*x*x*y*y*y;
    break;
  }
  case 37: { // l=6, m=-5
    val = -23.66619162231752*z*(x*x)*(y*y*y) + (2.366619162231752*z)*(y*y*y*y*y) + (11.83309581115876*y*z)*(x*x*x*x);
    break;
  }
  case 38: { // l=6, m=-4
    const double t0 = z*z;
    val = -20.182596029148966*t0*x*y*y*y + 20.182596029148966*t0*y*(x*x*x) + (2.0182596029148966*x)*(y*y*y*y*y) - 2.0182596029148966*y*x*x*x*x*x;
    break;
  }
  case 39: { // l=6, m=-3
    const double t0 = y*y*y;
    const double t1 = z*z*z;
    const double t2 = x*x;
    val = -(7.369642076119388*t0*t1 + 5.527231557089541*t0*t2*z - 22.108926228358164*t1*t2*y - 2.7636157785447705*z*y*y*y*y*y + (8.2908473356343115*y*z)*(x*x*x*x));
    break;
  }
  case 40: { // l=6, m=-2
    const double t0 = 14.739284152238776*x;
    const double t1 = x*x*x;
    const double t2 = y*y*y;
    const double t3 = z*z;
    val = -t0*t2*t3 + t0*y*(z*z*z*z) + 1.842410519029847*t1*t2 - 14.739284152238776*t1*t3*y + (0.9212052595149235*x)*(y*y*y*y*y) + (0.9212052595149235*y)*(x*x*x*x*x);
    break;
  }
  case 41: { // l=6, m=-1
    const double t0 = 2.9131068125936569*z;
    const double t1 = y*y*y;
    const double t2 = z*z*z;
    const double t3 = x*x;
    val = t0*y*(x*x*x*x) + t0*(y*y*y*y*y) - 11.652427250374628*t1*t2 + 5.8262136251873139*t1*t3*z - 11.652427250374628*t2*t3*y + (4.6609709001498511*y)*(z*z*z*z*z);
    break;
  }
  case 42: { // l=6, m=0
    const double t0 = x*x;
    const double t1 = y*y*y*y;
    const double t2 = z*z*z*z;
    const double t3 = y*y;
    const double t4 = x*x*x*x;
    const double t5 = z*z;
    val = -(0.95353803401442639*t0*t1 + 7.6283042721154111*t0*t2 - 11.442456408173117*t0*t3*t5 - 5.7212282040865583*t1*t5 + 7.6283042721154111*t2*t3 + 0.95353803401442639*t3*t4 - 5.7212282040865583*t4*t5 + 0.31784601133814213*(x*x*x*x*x*x) + 0.31784601133814213*(y*y*y*y*y*y) - 1.0171072362820548*z*z*z*z*z*z);
    break;
  }
  case 43: { // l=6, m=1
    const double t0 = 2.9131068125936569*z;
    const double t1 = x*x*x;
    const double t2 = z*z*z;
    const double t3 = y*y;
    val = t0*x*(y*y*y*y) + t0*(x*x*x*x*x) - 11.652427250374628*t1*t2 + 5.8262136251873139*t1*t3*z - 11.652427250374628*t2*t3*x + (4.6609709001498511*x)*(z*z*z*z*z);
    break;
  }
  case 44: { // l=6, m=2
    const double t0 = x*x;
    const double t1 = y*y*y*y;
    const double t2 = z*z*z*z;
    const double t3 = y*y;
    const double t4 = x*x*x*x;
    const double t5 = z*z;
    val = -0.46060262975746175*t0*t1 + 7.369642076119388*t0*t2 + 7.369642076119388*t1*t5 - 7.369642076119388*t2*t3 + 0.46060262975746175*t3*t4 - 7.369642076119388*t4*t5 + 0.46060262975746175*(x*x*x*x*x*x) - 0.46060262975746175*y*y*y*y*y*y;
    break;
  }
  case 45: { // l=6, m=3
    const double t0 = x*x*x;
    const double t1 = z*z*z;
    const double t2 = y*y;
    val = 7.369642076119388*t0*t1 + 5.527231557089541*t0*t2*z - 22.108926228358164*t1*t2*x - 2.7636157785447705*z*x*x*x*x*x + (8.2908473356343115*x*z)*(y*y*y*y);
    break;
  }
  case 46: { // l=6, m=4
    const double t0 = x*x;
    const double t1 = y*y*y*y;
    const double t2 = x*x*x*x;
    const double t3 = y*y;
    const double t4 = z*z;
    val = 2.5228245036436208*t0*t1 - 30.27389404372345*t0*t3*t4 + 5.0456490072872416*t1*t4 + 2.5228245036436208*t2*t3 + 5.0456490072872416*t2*t4 - 0.50456490072872416*x*x*x*x*x*x - 0.50456490072872416*y*y*y*y*y*y;
    break;
  }
  case 47: { // l=6, m=5
    val = -23.66619162231752*z*(x*x*x)*(y*y) + (2.366619162231752*z)*(x*x*x*x*x) + (11.83309581115876*x*z)*(y*y*y*y);
    break;
  }
  case 48: { // l=6, m=6
    val = 10.247761577878715*(x*x)*(y*y*y*y) - 10.247761577878715*x*x*x*x*y*y + 0.68318410519191432*(x*x*x*x*x*x) - 0.68318410519191432*y*y*y*y*y*y;
    break;
  }
  case 49: { // l=7, m=-7
    val = (4.9501391276721732*y)*(x*x*x*x*x*x) + 14.85041738301652*(x*x)*(y*y*y*y*y) - 24.750695638360866*x*x*x*x*y*y*y - 0.70716273252459618*y*y*y*y*y*y*y;
    break;
  }
  case 50: { // l=7, m=-6
    const double t0 = 15.875763970811401*z;
    val = t0*x*(y*y*y*y*y) + t0*y*(x*x*x*x*x) - 52.919213236038004*z*(x*x*x)*(y*y*y);
    break;
  }
  case 51: { // l=7, m=-5
    const double t0 = x*x;
    const double t1 = y*y*y*y*y;
    const double t2 = x*x*x*x;
    const double t3 = y*y*y;
    const double t4 = z*z;
    val = 4.6702402084823429*t0*t1 - 62.269869446431238*t0*t3*t4 + 6.2269869446431238*t1*t4 + 2.5945778936013016*t2*t3 + 31.134934723215619*t2*t4*y - 2.5945778936013016*y*x*x*x*x*x*x - 0.51891557872026032*y*y*y*y*y*y*y;
    break;
  }
  case 52: { // l=7, m=-4
    const double t0 = 12.453973889286248*z;
    const double t1 = z*z*z;
    val = t0*x*(y*y*y*y*y) - t0*y*x*x*x*x*x - 41.513246297620826*t1*x*y*y*y + 41.513246297620826*t1*y*(x*x*x);
    break;
  }
  case 53: { // l=7, m=-3
    const double t0 = x*x;
    const double t1 = y*y*y*y*y;
    const double t2 = y*y*y;
    const double t3 = z*z*z*z;
    const double t4 = x*x*x*x;
    const double t5 = z*z;
    val = 0.4693768015868821*t0*t1 - 18.775072063475284*t0*t2*t5 + 37.550144126950568*t0*t3*y + 9.387536031737642*t1*t5 - 12.516714708983523*t2*t3 + 2.3468840079344105*t2*t4 - 28.162608095212926*t4*t5*y + (1.4081304047606463*y)*(x*x*x*x*x*x) - 0.4693768015868821*y*y*y*y*y*y*y;
    break;
  }
  case 54: { // l=7, m=-2
    const double t0 = 6.6379903866747395*z;
    const double t1 = y*y*y;
    const double t2 = z*z*z;
    const double t3 = x*x*x;
    val = t0*x*(y*y*y*y*y) + t0*y*(x*x*x*x*x) - 35.402615395598611*t1*t2*x + 13.275980773349479*t1*t3*z - 35.402615395598611*t2*t3*y + (21.241569237359166*x*y)*(z*z*z*z*z);
    break;
  }
  case 55: { // l=7, m=-1
    const double t0 = x*x;
    const double t1 = y*y*y*y*y;
    const double t2 = y*y*y;
    const double t3 = z*z*z*z;
    const double t4 = x*x*x*x;
    const double t5 = z*z;
    val = -(1.3549741137377597*t0*t1 - 21.679585819804155*t0*t2*t5 + 21.679585819804155*t0*t3*y - 10.839792909902078*t1*t5 + 21.679585819804155*t2*t3 + 1.3549741137377597*t2*t4 - 10.839792909902078*t4*t5*y + (0.45165803791258657*y)*(x*x*x*x*x*x) - 5.7812228852811081*y*z*z*z*z*z*z + 0.45165803791258657*(y*y*y*y*y*y*y));
    break;
  }
  case 56: { // l=7, m=0
    const double t0 = 2.389949691920173*z;
    const double t1 = x*x;
    const double t2 = z*z*z*z*z;
    const double t3 = y*y;
    const double t4 = x*x*x*x;
    const double t5 = z*z*z;
    const double t6 = y*y*y*y;
    const double t7 = 7.1698490757605189*z;
    val = -(t0*(x*x*x*x*x*x) + t0*(y*y*y*y*y*y) + 11.47175852121683*t1*t2 - 28.679396303042076*t1*t3*t5 + t1*t6*t7 + 11.47175852121683*t2*t3 + t3*t4*t7 - 14.339698151521038*t4*t5 - 14.339698151521038*t5*t6 - 1.0925484305920791*z*z*z*z*z*z*z);
    break;
  }
  case 57: { // l=7, m=1
    const double t0 = x*x*x;
    const double t1 = y*y*y*y;
    const double t2 = z*z*z*z;
    const double t3 = x*x*x*x*x;
    const double t4 = y*y;
    const double t5 = z*z;
    val = -(1.3549741137377597*t0*t1 + 21.679585819804155*t0*t2 - 21.679585819804155*t0*t4*t5 - 10.839792909902078*t1*t5*x + 21.679585819804155*t2*t4*x + 1.3549741137377597*t3*t4 - 10.839792909902078*t3*t5 + (0.45165803791258657*x)*(y*y*y*y*y*y) - 5.7812228852811081*x*z*z*z*z*z*z + 0.45165803791258657*(x*x*x*x*x*x*x));
    break;
  }
  case 58: { // l=7, m=2
    const double t0 = 3.3189951933373697*z;
    const double t1 = x*x;
    const double t2 = z*z*z*z*z;
    const double t3 = y*y;
    const double t4 = x*x*x*x;
    const double t5 = z*z*z;
    const double t6 = y*y*y*y;
    val = -t0*t1*t6 + t0*t3*t4 + t0*(x*x*x*x*x*x) - t0*y*y*y*y*y*y + 10.620784618679583*t1*t2 - 10.620784618679583*t2*t3 - 17.701307697799305*t4*t5 + 17.701307697799305*t5*t6;
    break;
  }
  case 59: { // l=7, m=3
    const double t0 = x*x*x;
    const double t1 = y*y*y*y;
    const double t2 = z*z*z*z;
    const double t3 = x*x*x*x*x;
    const double t4 = y*y;
    const double t5 = z*z;
    val = -(2.3468840079344105*t0*t1 - 12.516714708983523*t0*t2 - 18.775072063475284*t0*t4*t5 - 28.162608095212926*t1*t5*x + 37.550144126950568*t2*t4*x + 0.4693768015868821*t3*t4 + 9.387536031737642*t3*t5 + (1.4081304047606463*x)*(y*y*y*y*y*y) - 0.4693768015868821*x*x*x*x*x*x*x);
    break;
  }
  case 60: { // l=7, m=4
    const double t0 = 3.1134934723215619*z;
    const double t1 = x*x*x*x;
    const double t2 = z*z*z;
    const double t3 = y*y*y*y;
    const double t4 = 15.56746736160781*z;
    const double t5 = x*x;
    const double t6 = y*y;
    val = -t0*x*x*x*x*x*x - t0*y*y*y*y*y*y + 10.378311574405206*t1*t2 + t1*t4*t6 + 10.378311574405206*t2*t3 - 62.269869446431238*t2*t5*t6 + t3*t4*t5;
    break;
  }
  case 61: { // l=7, m=5
    const double t0 = x*x*x;
    const double t1 = y*y*y*y;
    const double t2 = x*x*x*x*x;
    const double t3 = y*y;
    const double t4 = z*z;
    val = 2.5945778936013016*t0*t1 - 62.269869446431238*t0*t3*t4 + 31.134934723215619*t1*t4*x + 4.6702402084823429*t2*t3 + 6.2269869446431238*t2*t4 - 2.5945778936013016*x*y*y*y*y*y*y - 0.51891557872026032*x*x*x*x*x*x*x;
    break;
  }
  case 62: { // l=7, m=6
    const double t0 = 2.6459606618019002*z;
    const double t1 = 39.689409927028503*z;
    val = t0*(x*x*x*x*x*x) - t0*y*y*y*y*y*y + t1*(x*x)*(y*y*y*y) - t1*x*x*x*x*y*y;
    break;
  }
  case 63: { // l=7, m=7
    val = -4.9501391276721732*x*y*y*y*y*y*y + 24.750695638360866*(x*x*x)*(y*y*y*y) - 14.85041738301652*x*x*x*x*x*y*y + 0.70716273252459618*(x*x*x*x*x*x*x);
    break;
  }
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
  case 0: { // l=0, m=0
    val = 0.28209479177387814;
    break;
  }
  case 1: { // l=1, m=-1
    val = 0.48860251190291992*y;
    dy = 0.48860251190291992;
    break;
  }
  case 2: { // l=1, m=0
    val = 0.48860251190291992*z;
    dz = 0.48860251190291992;
    break;
  }
  case 3: { // l=1, m=1
    val = 0.48860251190291992*x;
    dx = 0.48860251190291992;
    break;
  }
  case 4: { // l=2, m=-2
    const double t0 = 1.0925484305920791*y;
    val = t0*x;
    dx = t0;
    dy = 1.0925484305920791*x;
    break;
  }
  case 5: { // l=2, m=-1
    const double t0 = 1.0925484305920791*z;
    val = t0*y;
    dy = t0;
    dz = 1.0925484305920791*y;
    break;
  }
  case 6: { // l=2, m=0
    val = -(0.31539156525252001*(x*x) + 0.31539156525252001*(y*y) - 0.63078313050504001*z*z);
    dx = -0.63078313050504001*x;
    dy = -0.63078313050504001*y;
    dz = 1.26156626101008*z;
    break;
  }
  case 7: { // l=2, m=1
    const double t0 = 1.0925484305920791*z;
    val = t0*x;
    dx = t0;
    dz = 1.0925484305920791*x;
    break;
  }
  case 8: { // l=2, m=2
    val = 0.54627421529603954*(x*x) - 0.54627421529603954*y*y;
    dx = 1.0925484305920791*x;
    dy = -1.0925484305920791*y;
    break;
  }
  case 9: { // l=3, m=-3
    const double t0 = x*x;
    val = 1.7701307697799305*t0*y - 0.59004358992664351*y*y*y;
    dx = 3.5402615395598611*x*y;
    dy = 1.7701307697799305*t0 - 1.7701307697799305*y*y;
    break;
  }
  case 10: { // l=3, m=-2
    const double t0 = 2.8906114426405541*z;
    const double t1 = t0*y;
    val = t1*x;
    dx = t1;
    dy = t0*x;
    dz = 2.8906114426405541*x*y;
    break;
  }
  case 11: { // l=3, m=-1
    const double t0 = x*x;
    const double t1 = z*z;
    val = -(0.45704579946446574*t0*y - 1.8281831978578629*t1*y + 0.45704579946446574*(y*y*y));
    dx = -0.91409159892893147*x*y;
    dy = -(0.45704579946446574*t0 - 1.8281831978578629*t1 + 1.3711373983933972*(y*y));
    dz = 3.6563663957157259*y*z;
    break;
  }
  case 12: { // l=3, m=0
    const double t0 = 1.1195289977703462*z;
    const double t1 = x*x;
    const double t2 = y*y;
    const double t3 = 2.2390579955406924*z;
    val = -(t0*t1 + t0*t2 - 0.74635266518023078*z*z*z);
    dx = -t3*x;
    dy = -t3*y;
    dz = -(1.1195289977703462*t1 + 1.1195289977703462*t2 - 2.2390579955406924*z*z);
    break;
  }
  case 13: { // l=3, m=1
    const double t0 = y*y;
    const double t1 = z*z;
    val = -(0.45704579946446574*t0*x - 1.8281831978578629*t1*x + 0.45704579946446574*(x*x*x));
    dx = -(0.45704579946446574*t0 - 1.8281831978578629*t1 + 1.3711373983933972*(x*x));
    dy = -0.91409159892893147*x*y;
    dz = 3.6563663957157259*x*z;
    break;
  }
  case 14: { // l=3, m=2
    const double t0 = 1.445305721320277*z;
    const double t1 = x*x;
    const double t2 = y*y;
    const double t3 = 2.8906114426405541*z;
    val = t0*t1 - t0*t2;
    dx = t3*x;
    dy = -t3*y;
    dz = 1.445305721320277*t1 - 1.445305721320277*t2;
    break;
  }
  case 15: { // l=3, m=3
    const double t0 = y*y;
    val = -1.7701307697799305*t0*x + 0.59004358992664351*(x*x*x);
    dx = -1.7701307697799305*t0 + 1.7701307697799305*(x*x);
    dy = -3.5402615395598611*x*y;
    break;
  }
  case 16: { // l=4, m=-4
    const double t0 = y*y*y;
    const double t1 = x*x*x;
    val = -2.5033429417967045*t0*x + 2.5033429417967045*t1*y;
    dx = -2.5033429417967045*t0 + (7.5100288253901136*y)*(x*x);
    dy = 2.5033429417967045*t1 - 7.5100288253901136*x*y*y;
    break;
  }
  case 17: { // l=4, m=-3
    const double t0 = y*y*y;
    const double t1 = 5.3103923093397916*z;
    const double t2 = x*x;
    val = -1.7701307697799305*t0*z + t1*t2*y;
    dx = 10.620784618679583*x*y*z;
    dy = t1*t2 - t1*y*y;
    dz = -1.7701307697799305*t0 + 5.3103923093397916*t2*y;
    break;
  }
  case 18: { // l=4, m=-2
    const double t0 = y*y*y;
    const double t1 = x*x*x;
    const double t2 = z*z;
    val = -0.94617469575756002*t0*x - 0.94617469575756002*t1*y + 5.6770481745453601*t2*x*y;
    dx = -(0.94617469575756002*t0 - 5.6770481745453601*t2*y + (2.8385240872726801*y)*(x*x));
    dy = -(0.94617469575756002*t1 - 5.6770481745453601*t2*x + (2.8385240872726801*x)*(y*y));
    dz = 11.35409634909072*x*y*z;
    break;
  }
  case 19: { // l=4, m=-1
    const double t0 = z*z*z;
    const double t1 = 2.0071396306718675*z;
    const double t2 = y*y*y;
    const double t3 = x*x;
    val = 2.6761861742291567*t0*y - t1*t2 - t1*t3*y;
    dx = -4.014279261343735*x*y*z;
    dy = -(-2.6761861742291567*t0 + t1*t3 + (6.0214188920156025*z)*(y*y));
    dz = -(2.0071396306718675*t2 + 2.0071396306718675*t3*y - 8.02855852268747*y*z*z);
    break;
  }
  case 20: { // l=4, m=0
    const double t0 = x*x;
    const double t1 = y*y;
    const double t2 = z*z;
    const double t3 = 5.0777062519298066*z;
    val = 0.63471328149122582*t0*t1 - 2.5388531259649033*t0*t2 - 2.5388531259649033*t1*t2 + 0.31735664074561291*(x*x*x*x) + 0.31735664074561291*(y*y*y*y) + 0.84628437532163443*(z*z*z*z);
    dx = 1.2694265629824516*t1*x - 5.0777062519298066*t2*x + 1.2694265629824516*(x*x*x);
    dy = 1.2694265629824516*t0*y - 5.0777062519298066*t2*y + 1.2694265629824516*(y*y*y);
    dz = -(t0*t3 + t1*t3 - 3.3851375012865377*z*z*z);
    break;
  }
  case 21: { // l=4, m=1
    const double t0 = z*z*z;
    const double t1 = 2.0071396306718675*z;
    const double t2 = x*x*x;
    const double t3 = y*y;
    val = 2.6761861742291567*t0*x - t1*t2 - t1*t3*x;
    dx = -(-2.6761861742291567*t0 + t1*t3 + (6.0214188920156025*z)*(x*x));
    dy = -4.014279261343735*x*y*z;
    dz = -(2.0071396306718675*t2 + 2.0071396306718675*t3*x - 8.02855852268747*x*z*z);
    break;
  }
  case 22: { // l=4, m=2
    const double t0 = x*x;
    const double t1 = z*z;
    const double t2 = y*y;
    const double t3 = 5.6770481745453601*z;
    val = 2.8385240872726801*t0*t1 - 2.8385240872726801*t1*t2 - 0.47308734787878001*x*x*x*x + 0.47308734787878001*(y*y*y*y);
    dx = 5.6770481745453601*t1*x - 1.89234939151512*x*x*x;
    dy = -5.6770481745453601*t1*y + 1.89234939151512*(y*y*y);
    dz = t0*t3 - t2*t3;
    break;
  }
  case 23: { // l=4, m=3
    const double t0 = x*x*x;
    const double t1 = 5.3103923093397916*z;
    const double t2 = y*y;
    val = 1.7701307697799305*t0*z - t1*t2*x;
    dx = -t1*t2 + t1*(x*x);
    dy = -10.620784618679583*x*y*z;
    dz = 1.7701307697799305*t0 - 5.3103923093397916*t2*x;
    break;
  }
  case 24: { // l=4, m=4
    const double t0 = x*x;
    const double t1 = y*y;
    val = -3.7550144126950568*t0*t1 + 0.62583573544917613*(x*x*x*x) + 0.62583573544917613*(y*y*y*y);
    dx = -7.5100288253901136*t1*x + 2.5033429417967045*(x*x*x);
    dy = -7.5100288253901136*t0*y + 2.5033429417967045*(y*y*y);
    break;
  }
  case 25: { // l=5, m=-5
    const double t0 = x*x*x*x;
    const double t1 = x*x;
    const double t2 = y*y*y;
    val = 3.2819102842008505*t0*y - 6.563820568401701*t1*t2 + 0.6563820568401701*(y*y*y*y*y);
    dx = -13.127641136803402*t2*x + (13.127641136803402*y)*(x*x*x);
    dy = 3.2819102842008505*t0 - 19.691461705205103*t1*y*y + 3.2819102842008505*(y*y*y*y);
    break;
  }
  case 26: { // l=5, m=-4
    const double t0 = 8.3026492595241651*z;
    const double t1 = y*y*y;
    const double t2 = x*x*x;
    const double t3 = 24.907947778572495*z;
    val = -t0*t1*x + t0*t2*y;
    dx = -t0*t1 + t3*y*(x*x);
    dy = t0*t2 - t3*x*y*y;
    dz = -8.3026492595241651*t1*x + 8.3026492595241651*t2*y;
    break;
  }
  case 27: { // l=5, m=-3
    const double t0 = x*x*x*x;
    const double t1 = x*x;
    const double t2 = y*y*y;
    const double t3 = z*z;
    const double t4 = y*y;
    val = -(1.4677148983057512*t0*y + 0.97847659887050078*t1*t2 - 11.741719186446009*t1*t3*y + 3.9139063954820031*t2*t3 - 0.48923829943525039*y*y*y*y*y);
    dx = -(1.9569531977410016*t2*x - 23.483438372892019*t3*x*y + (5.8708595932230047*y)*(x*x*x));
    dy = -(1.4677148983057512*t0 - 11.741719186446009*t1*t3 + 2.9354297966115023*t1*t4 + 11.741719186446009*t3*t4 - 2.4461914971762519*y*y*y*y);
    dz = 23.483438372892019*t1*y*z - 7.8278127909640062*t2*z;
    break;
  }
  case 28: { // l=5, m=-2
    const double t0 = z*z*z;
    const double t1 = 4.7935367849733238*z;
    const double t2 = y*y*y;
    const double t3 = x*x*x;
    const double t4 = 14.380610354919971*z;
    val = 9.5870735699466475*t0*x*y - t1*t2*x - t1*t3*y;
    dx = -(-9.5870735699466475*t0*y + t1*t2 + t4*y*(x*x));
    dy = -(-9.5870735699466475*t0*x + t1*t3 + t4*x*(y*y));
    dz = -(4.7935367849733238*t2*x + 4.7935367849733238*t3*y - 28.761220709839943*x*y*z*z);
    break;
  }
  case 29: { // l=5, m=-1
    const double t0 = x*x*x*x;
    const double t1 = z*z*z*z;
    const double t2 = x*x;
    const double t3 = y*y*y;
    const double t4 = z*z;
    const double t5 = t2*t4;
    const double t6 = y*y;
    const double t7 = 10.870719628696726*z;
    val = 0.45294665119569692*t0*y + 3.6235732095655754*t1*y + 0.90589330239139384*t2*t3 - 5.4353598143483631*t3*t4 - 5.4353598143483631*t5*y + 0.45294665119569692*(y*y*y*y*y);
    dx = 1.8117866047827877*t3*x - 10.870719628696726*t4*x*y + (1.8117866047827877*y)*(x*x*x);
    dy = 0.45294665119569692*t0 + 3.6235732095655754*t1 + 2.7176799071741815*t2*t6 - 16.306079443045089*t4*t6 - 5.4353598143483631*t5 + 2.2647332559784846*(y*y*y*y);
    dz = -(t2*t7*y + t3*t7 - 14.494292838262301*y*z*z*z);
    break;
  }
  case 30: { // l=5, m=0
    const double t0 = 1.7542548368013539*z;
    const double t1 = x*x*x*x;
    const double t2 = y*y*y*y;
    const double t3 = x*x;
    const double t4 = z*z*z;
    const double t5 = y*y;
    const double t6 = t3*t5;
    const double t7 = 7.0170193472054158*z;
    const double t8 = z*z;
    val = t0*t1 + t0*t2 - 4.6780128981369439*t3*t4 - 4.6780128981369439*t4*t5 + 3.5085096736027079*t6*z + 0.93560257962738877*(z*z*z*z*z);
    dx = -9.3560257962738877*t4*x + t5*t7*x + t7*(x*x*x);
    dy = t3*t7*y - 9.3560257962738877*t4*y + t7*(y*y*y);
    dz = 1.7542548368013539*t1 + 1.7542548368013539*t2 - 14.034038694410832*t3*t8 - 14.034038694410832*t5*t8 + 3.5085096736027079*t6 + 4.6780128981369439*(z*z*z*z);
    break;
  }
  case 31: { // l=5, m=1
    const double t0 = y*y*y*y;
    const double t1 = z*z*z*z;
    const double t2 = x*x*x;
    const double t3 = y*y;
    const double t4 = z*z;
    const double t5 = t3*t4;
    const double t6 = x*x;
    const double t7 = 10.870719628696726*z;
    val = 0.45294665119569692*t0*x + 3.6235732095655754*t1*x + 0.90589330239139384*t2*t3 - 5.4353598143483631*t2*t4 - 5.4353598143483631*t5*x + 0.45294665119569692*(x*x*x*x*x);
    dx = 0.45294665119569692*t0 + 3.6235732095655754*t1 + 2.7176799071741815*t3*t6 - 16.306079443045089*t4*t6 - 5.4353598143483631*t5 + 2.2647332559784846*(x*x*x*x);
    dy = 1.8117866047827877*t2*y - 10.870719628696726*t4*x*y + (1.8117866047827877*x)*(y*y*y);
    dz = -(t2*t7 + t3*t7*x - 14.494292838262301*x*z*z*z);
    break;
  }
  case 32: { // l=5, m=2
    const double t0 = 2.3967683924866619*z;
    const double t1 = x*x*x*x;
    const double t2 = y*y*y*y;
    const double t3 = x*x;
    const double t4 = z*z*z;
    const double t5 = y*y;
    const double t6 = 9.5870735699466475*z;
    const double t7 = z*z;
    val = -t0*t1 + t0*t2 + 4.7935367849733238*t3*t4 - 4.7935367849733238*t4*t5;
    dx = 9.5870735699466475*t4*x - t6*x*x*x;
    dy = -9.5870735699466475*t4*y + t6*(y*y*y);
    dz = -2.3967683924866619*t1 + 2.3967683924866619*t2 + 14.380610354919971*t3*t7 - 14.380610354919971*t5*t7;
    break;
  }
  case 33: { // l=5, m=3
    const double t0 = y*y*y*y;
    const double t1 = x*x*x;
    const double t2 = y*y;
    const double t3 = z*z;
    const double t4 = t2*t3;
    const double t5 = x*x;
    const double t6 = 23.483438372892019*x;
    val = 1.4677148983057512*t0*x + 0.97847659887050078*t1*t2 + 3.9139063954820031*t1*t3 - 11.741719186446009*t4*x - 0.48923829943525039*x*x*x*x*x;
    dx = 1.4677148983057512*t0 + 2.9354297966115023*t2*t5 + 11.741719186446009*t3*t5 - 11.741719186446009*t4 - 2.4461914971762519*x*x*x*x;
    dy = 1.9569531977410016*t1*y - t3*t6*y + (5.8708595932230047*x)*(y*y*y);
    dz = 7.8278127909640062*t1*z - t2*t6*z;
    break;
  }
  case 34: { // l=5, m=4
    const double t0 = 2.0756623148810413*z;
    const double t1 = x*x*x*x;
    const double t2 = y*y*y*y;
    const double t3 = x*x;
    const double t4 = y*y;
    const double t5 = t3*t4;
    const double t6 = 8.3026492595241651*z;
    const double t7 = 24.907947778572495*z;
    val = t0*t1 + t0*t2 - 12.453973889286248*t5*z;
    dx = -t4*t7*x + t6*(x*x*x);
    dy = -t3*t7*y + t6*(y*y*y);
    dz = 2.0756623148810413*t1 + 2.0756623148810413*t2 - 12.453973889286248*t5;
    break;
  }
  case 35: { // l=5, m=5
    const double t0 = y*y*y*y;
    const double t1 = x*x*x;
    const double t2 = y*y;
    val = 3.2819102842008505*t0*x - 6.563820568401701*t1*t2 + 0.6563820568401701*(x*x*x*x*x);
    dx = 3.2819102842008505*t0 - 19.691461705205103*t2*x*x + 3.2819102842008505*(x*x*x*x);
    dy = -13.127641136803402*t1*y + (13.127641136803402*x)*(y*y*y);
    break;
  }
  case 36: { // l=6, m=-6
    const double t0 = y*y*y*y*y;
    const double t1 = x*x*x*x*x;
    const double t2 = x*x*x;
    const double t3 = y*y*y;
    val = 4.0991046311514859*t0*x + 4.0991046311514859*t1*y - 13.663682103838286*t2*t3;
    dx = 4.0991046311514859*t0 - 40.991046311514859*t3*x*x + (20.49552315575743*y)*(x*x*x*x);
    dy = 4.0991046311514859*t1 - 40.991046311514859*t2*y*y + (20.49552315575743*x)*(y*y*y*y);
    break;
  }
  case 37: { // l=6, m=-5
    const double t0 = y*y*y*y*y;
    const double t1 = 11.83309581115876*z;
    const double t2 = x*x*x*x;
    const double t3 = x*x;
    const double t4 = y*y*y;
    const double t5 = t3*t4;
    const double t6 = 47.332383244635041*z;
    val = 2.366619162231752*t0*z + t1*t2*y - 23.66619162231752*t5*z;
    dx = -t4*t6*x + t6*y*(x*x*x);
    dy = t1*t2 + t1*(y*y*y*y) - 70.998574866952561*t3*z*y*y;
    dz = 2.366619162231752*t0 + 11.83309581115876*t2*y - 23.66619162231752*t5;
    break;
  }
  case 38: { // l=6, m=-4
    const double t0 = y*y*y*y*y;
    const double t1 = x*x*x*x*x;
    const double t2 = y*y*y;
    const double t3 = z*z;
    const double t4 = t2*t3;
    const double t5 = x*x*x;
    const double t6 = t3*t5;
    const double t7 = 40.365192058297933*z;
    val = 2.0182596029148966*t0*x - 2.0182596029148966*t1*y - 20.182596029148966*t4*x + 20.182596029148966*t6*y;
    dx = 2.0182596029148966*t0 + 60.547788087446899*t3*y*(x*x) - 20.182596029148966*t4 - 10.091298014574483*y*x*x*x*x;
    dy = -2.0182596029148966*t1 - 60.547788087446899*t3*x*y*y + 20.182596029148966*t6 + (10.091298014574483*x)*(y*y*y*y);
    dz = -t2*t7*x + t5*t7*y;
    break;
  }
  case 39: { // l=6, m=-3
    const double t0 = y*y*y*y*y;
    const double t1 = 8.2908473356343115*z;
    const double t2 = x*x*x*x;
    const double t3 = y*y*y;
    const double t4 = z*z*z;
    const double t5 = x*x;
    const double t6 = t3*t5;
    const double t7 = y*y;
    const double t8 = z*z;
    val = 2.7636157785447705*t0*z - t1*t2*y - 7.369642076119388*t3*t4 + 22.108926228358164*t4*t5*y - 5.527231557089541*t6*z;
    dx = -(11.054463114179082*t3*x*z - 44.217852456716328*t4*x*y + (33.163389342537246*y*z)*(x*x*x));
    dy = -(t1*t2 - 22.108926228358164*t4*t5 + 22.108926228358164*t4*t7 + 16.581694671268623*t5*t7*z - 13.818078892723852*z*y*y*y*y);
    dz = 2.7636157785447705*t0 - 8.2908473356343115*t2*y - 22.108926228358164*t3*t8 + 66.326778685074492*t5*t8*y - 5.527231557089541*t6;
    break;
  }
  case 40: { // l=6, m=-2
    const double t0 = y*y*y*y*y;
    const double t1 = x*x*x*x*x;
    const double t2 = 14.739284152238776*x;
    const double t3 = z*z*z*z;
    const double t4 = x*x*x;
    const double t5 = y*y*y;
    const double t6 = z*z;
    const double t7 = t5*t6;
    const double t8 = 14.739284152238776*y;
    const double t9 = t4*t6;
    const double t10 = x*x;
    const double t11 = y*y;
    const double t12 = 29.478568304477552*z;
    val = 0.9212052595149235*t0*x + 0.9212052595149235*t1*y + t2*t3*y - t2*t7 + 1.842410519029847*t4*t5 - t8*t9;
    dx = 0.9212052595149235*t0 + 5.527231557089541*t10*t5 - 44.217852456716328*t10*t6*y + t3*t8 - 14.739284152238776*t7 + (4.6060262975746175*y)*(x*x*x*x);
    dy = 0.9212052595149235*t1 + 5.527231557089541*t11*t4 - 44.217852456716328*t11*t6*x + t2*t3 - 14.739284152238776*t9 + (4.6060262975746175*x)*(y*y*y*y);
    dz = -(t12*t4*y + t12*t5*x - 58.957136608955104*x*y*z*z*z);
    break;
  }
  case 41: { // l=6, m=-1
    const double t0 = z*z*z*z*z;
    const double t1 = 2.9131068125936569*z;
    const double t2 = y*y*y*y*y;
    const double t3 = x*x*x*x;
    const double t4 = y*y*y;
    const double t5 = z*z*z;
    const double t6 = 11.652427250374628*y;
    const double t7 = x*x;
    const double t8 = t5*t7;
    const double t9 = t4*t7;
    const double t10 = 23.304854500749256*y;
    const double t11 = y*y;
    const double t12 = z*z;
    val = 4.6609709001498511*t0*y + t1*t2 + t1*t3*y - 11.652427250374628*t4*t5 - t6*t8 + 5.8262136251873139*t9*z;
    dx = -t10*t5*x + 11.652427250374628*t4*x*z + t6*z*(x*x*x);
    dy = 4.6609709001498511*t0 + t1*t3 - 34.957281751123883*t11*t5 + 17.478640875561942*t11*t7*z - 11.652427250374628*t8 + (14.565534062968285*z)*(y*y*y*y);
    dz = t10*(z*z*z*z) - 34.957281751123883*t12*t4 - 34.957281751123883*t12*t7*y + 2.9131068125936569*t2 + 2.9131068125936569*t3*y + 5.8262136251873139*t9;
    break;
  }
  case 42: { // l=6, m=0
    const double t0 = x*x;
    const double t1 = y*y*y*y;
    const double t2 = z*z*z*z;
    const double t3 = y*y;
    const double t4 = x*x*x*x;
    const double t5 = z*z;
    const double t6 = x*x*x;
    const double t7 = y*y*y;
    const double t8 = 11.442456408173117*z;
    const double t9 = z*z*z;
    val = -(0.95353803401442639*t0*t1 + 7.6283042721154111*t0*t2 - 11.442456408173117*t0*t3*t5 - 5.7212282040865583*t1*t5 + 7.6283042721154111*t2*t3 + 0.95353803401442639*t3*t4 - 5.7212282040865583*t4*t5 + 0.31784601133814213*(x*x*x*x*x*x) + 0.31784601133814213*(y*y*y*y*y*y) - 1.0171072362820548*z*z*z*z*z*z);
    dx = -(1.9070760680288528*t1*x + 15.256608544230822*t2*x - 22.884912816346233*t3*t5*x + 3.8141521360577056*t3*t6 - 22.884912816346233*t5*t6 + 1.9070760680288528*(x*x*x*x*x));
    dy = -(-22.884912816346233*t0*t5*y + 3.8141521360577056*t0*t7 + 15.256608544230822*t2*y + 1.9070760680288528*t4*y - 22.884912816346233*t5*t7 + 1.9070760680288528*(y*y*y*y*y));
    dz = 22.884912816346233*t0*t3*z - 30.513217088461644*t0*t9 + t1*t8 - 30.513217088461644*t3*t9 + t4*t8 + 6.1026434176923289*(z*z*z*z*z);
    break;
  }
  case 43: { // l=6, m=1
    const double t0 = z*z*z*z*z;
    const double t1 = 2.9131068125936569*z;
    const double t2 = x*x*x*x*x;
    const double t3 = y*y*y*y;
    const double t4 = x*x*x;
    const double t5 = z*z*z;
    const double t6 = 11.652427250374628*x;
    const double t7 = y*y;
    const double t8 = t5*t7;
    const double t9 = t4*t7;
    const double t10 = x*x;
    const double t11 = 23.304854500749256*x;
    const double t12 = z*z;
    val = 4.6609709001498511*t0*x + t1*t2 + t1*t3*x - 11.652427250374628*t4*t5 - t6*t8 + 5.8262136251873139*t9*z;
    dx = 4.6609709001498511*t0 + t1*t3 - 34.957281751123883*t10*t5 + 17.478640875561942*t10*t7*z - 11.652427250374628*t8 + (14.565534062968285*z)*(x*x*x*x);
    dy = -t11*t5*y + 11.652427250374628*t4*y*z + t6*z*(y*y*y);
    dz = t11*(z*z*z*z) - 34.957281751123883*t12*t4 - 34.957281751123883*t12*t7*x + 2.9131068125936569*t2 + 2.9131068125936569*t3*x + 5.8262136251873139*t9;
    break;
  }
  case 44: { // l=6, m=2
    const double t0 = x*x;
    const double t1 = y*y*y*y;
    const double t2 = z*z*z*z;
    const double t3 = y*y;
    const double t4 = x*x*x*x;
    const double t5 = z*z;
    const double t6 = x*x*x;
    const double t7 = y*y*y;
    const double t8 = 14.739284152238776*z;
    const double t9 = z*z*z;
    val = -0.46060262975746175*t0*t1 + 7.369642076119388*t0*t2 + 7.369642076119388*t1*t5 - 7.369642076119388*t2*t3 + 0.46060262975746175*t3*t4 - 7.369642076119388*t4*t5 + 0.46060262975746175*(x*x*x*x*x*x) - 0.46060262975746175*y*y*y*y*y*y;
    dx = -0.9212052595149235*t1*x + 14.739284152238776*t2*x + 1.842410519029847*t3*t6 - 29.478568304477552*t5*t6 + 2.7636157785447705*(x*x*x*x*x);
    dy = -(1.842410519029847*t0*t7 + 14.739284152238776*t2*y - 0.9212052595149235*t4*y - 29.478568304477552*t5*t7 + 2.7636157785447705*(y*y*y*y*y));
    dz = 29.478568304477552*t0*t9 + t1*t8 - 29.478568304477552*t3*t9 - t4*t8;
    break;
  }
  case 45: { // l=6, m=3
    const double t0 = x*x*x*x*x;
    const double t1 = 8.2908473356343115*z;
    const double t2 = y*y*y*y;
    const double t3 = x*x*x;
    const double t4 = z*z*z;
    const double t5 = y*y;
    const double t6 = t4*t5;
    const double t7 = t3*t5;
    const double t8 = x*x;
    const double t9 = z*z;
    val = -2.7636157785447705*t0*z + t1*t2*x + 7.369642076119388*t3*t4 - 22.108926228358164*t6*x + 5.527231557089541*t7*z;
    dx = t1*t2 + 22.108926228358164*t4*t8 + 16.581694671268623*t5*t8*z - 22.108926228358164*t6 - 13.818078892723852*z*x*x*x*x;
    dy = 11.054463114179082*t3*y*z - 44.217852456716328*t4*x*y + (33.163389342537246*x*z)*(y*y*y);
    dz = -2.7636157785447705*t0 + 8.2908473356343115*t2*x + 22.108926228358164*t3*t9 - 66.326778685074492*t5*t9*x + 5.527231557089541*t7;
    break;
  }
  case 46: { // l=6, m=4
    const double t0 = x*x;
    const double t1 = y*y*y*y;
    const double t2 = x*x*x*x;
    const double t3 = y*y;
    const double t4 = z*z;
    const double t5 = x*x*x;
    const double t6 = y*y*y;
    const double t7 = 10.091298014574483*z;
    val = 2.5228245036436208*t0*t1 - 30.27389404372345*t0*t3*t4 + 5.0456490072872416*t1*t4 + 2.5228245036436208*t2*t3 + 5.0456490072872416*t2*t4 - 0.50456490072872416*x*x*x*x*x*x - 0.50456490072872416*y*y*y*y*y*y;
    dx = 5.0456490072872416*t1*x - 60.547788087446899*t3*t4*x + 10.091298014574483*t3*t5 + 20.182596029148966*t4*t5 - 3.027389404372345*x*x*x*x*x;
    dy = -60.547788087446899*t0*t4*y + 10.091298014574483*t0*t6 + 5.0456490072872416*t2*y + 20.182596029148966*t4*t6 - 3.027389404372345*y*y*y*y*y;
    dz = -60.547788087446899*t0*t3*z + t1*t7 + t2*t7;
    break;
  }
  case 47: { // l=6, m=5
    const double t0 = x*x*x*x*x;
    const double t1 = 11.83309581115876*z;
    const double t2 = y*y*y*y;
    const double t3 = x*x*x;
    const double t4 = y*y;
    const double t5 = t3*t4;
    const double t6 = 47.332383244635041*z;
    val = 2.366619162231752*t0*z + t1*t2*x - 23.66619162231752*t5*z;
    dx = t1*t2 + t1*(x*x*x*x) - 70.998574866952561*t4*z*x*x;
    dy = -t3*t6*y + t6*x*(y*y*y);
    dz = 2.366619162231752*t0 + 11.83309581115876*t2*x - 23.66619162231752*t5;
    break;
  }
  case 48: { // l=6, m=6
    const double t0 = x*x;
    const double t1 = y*y*y*y;
    const double t2 = x*x*x*x;
    const double t3 = y*y;
    val = 10.247761577878715*t0*t1 - 10.247761577878715*t2*t3 + 0.68318410519191432*(x*x*x*x*x*x) - 0.68318410519191432*y*y*y*y*y*y;
    dx = 20.49552315575743*t1*x - 40.991046311514859*t3*x*x*x + 4.0991046311514859*(x*x*x*x*x);
    dy = -(-40.991046311514859*t0*y*y*y + 20.49552315575743*t2*y + 4.0991046311514859*(y*y*y*y*y));
    break;
  }
  case 49: { // l=7, m=-7
    const double t0 = x*x*x*x*x*x;
    const double t1 = x*x;
    const double t2 = y*y*y*y*y;
    const double t3 = x*x*x*x;
    const double t4 = y*y*y;
    val = 4.9501391276721732*t0*y + 14.85041738301652*t1*t2 - 24.750695638360866*t3*t4 - 0.70716273252459618*y*y*y*y*y*y*y;
    dx = 29.70083476603304*t2*x - 99.002782553443465*t4*x*x*x + (29.70083476603304*y)*(x*x*x*x*x);
    dy = 4.9501391276721732*t0 + 74.252086915082599*t1*(y*y*y*y) - 74.252086915082599*t3*y*y - 4.9501391276721732*y*y*y*y*y*y;
    break;
  }
  case 50: { // l=7, m=-6
    const double t0 = 15.875763970811401*z;
    const double t1 = y*y*y*y*y;
    const double t2 = x*x*x*x*x;
    const double t3 = x*x*x;
    const double t4 = y*y*y;
    const double t5 = t3*t4;
    const double t6 = 79.378819854057007*z;
    const double t7 = 158.75763970811401*z;
    val = t0*t1*x + t0*t2*y - 52.919213236038004*t5*z;
    dx = t0*t1 - t4*t7*x*x + t6*y*(x*x*x*x);
    dy = t0*t2 - t3*t7*y*y + t6*x*(y*y*y*y);
    dz = 15.875763970811401*t1*x + 15.875763970811401*t2*y - 52.919213236038004*t5;
    break;
  }
  case 51: { // l=7, m=-5
    const double t0 = x*x*x*x*x*x;
    const double t1 = x*x;
    const double t2 = y*y*y*y*y;
    const double t3 = x*x*x*x;
    const double t4 = y*y*y;
    const double t5 = z*z;
    const double t6 = t3*t5;
    const double t7 = x*x*x;
    const double t8 = y*y*y*y;
    const double t9 = y*y;
    val = -2.5945778936013016*t0*y + 4.6702402084823429*t1*t2 - 62.269869446431238*t1*t4*t5 + 6.2269869446431238*t2*t5 + 2.5945778936013016*t3*t4 + 31.134934723215619*t6*y - 0.51891557872026032*y*y*y*y*y*y*y;
    dx = 9.3404804169646858*t2*x - 124.53973889286248*t4*t5*x + 10.378311574405206*t4*t7 + 124.53973889286248*t5*t7*y - 15.56746736160781*y*x*x*x*x*x;
    dy = -2.5945778936013016*t0 - 186.80960833929372*t1*t5*t9 + 23.351201042411714*t1*t8 + 7.7837336808039048*t3*t9 + 31.134934723215619*t5*t8 + 31.134934723215619*t6 - 3.6324090510418222*y*y*y*y*y*y;
    dz = -124.53973889286248*t1*t4*z + 12.453973889286248*t2*z + 62.269869446431238*t3*y*z;
    break;
  }
  case 52: { // l=7, m=-4
    const double t0 = 12.453973889286248*z;
    const double t1 = y*y*y*y*y;
    const double t2 = x*x*x*x*x;
    const double t3 = y*y*y;
    const double t4 = z*z*z;
    const double t5 = t3*t4;
    const double t6 = x*x*x;
    const double t7 = t4*t6;
    const double t8 = 62.269869446431238*z;
    const double t9 = 124.53973889286248*y;
    const double t10 = 124.53973889286248*x;
    const double t11 = z*z;
    val = t0*t1*x - t0*t2*y - 41.513246297620826*t5*x + 41.513246297620826*t7*y;
    dx = t0*t1 + t4*t9*(x*x) - 41.513246297620826*t5 - t8*y*x*x*x*x;
    dy = -t0*t2 - t10*t4*y*y + 41.513246297620826*t7 + t8*x*(y*y*y*y);
    dz = 12.453973889286248*t1*x - t10*t11*t3 + t11*t6*t9 - 12.453973889286248*t2*y;
    break;
  }
  case 53: { // l=7, m=-3
    const double t0 = x*x*x*x*x*x;
    const double t1 = x*x;
    const double t2 = y*y*y*y*y;
    const double t3 = y*y*y;
    const double t4 = z*z*z*z;
    const double t5 = x*x*x*x;
    const double t6 = z*z;
    const double t7 = t1*t4;
    const double t8 = t5*t6;
    const double t9 = x*x*x;
    const double t10 = y*y*y*y;
    const double t11 = y*y;
    const double t12 = z*z*z;
    val = 1.4081304047606463*t0*y + 0.4693768015868821*t1*t2 - 18.775072063475284*t1*t3*t6 + 9.387536031737642*t2*t6 - 12.516714708983523*t3*t4 + 2.3468840079344105*t3*t5 + 37.550144126950568*t7*y - 28.162608095212926*t8*y - 0.4693768015868821*y*y*y*y*y*y*y;
    dx = 0.9387536031737642*t2*x - 37.550144126950568*t3*t6*x + 9.387536031737642*t3*t9 + 75.100288253901136*t4*x*y - 112.6504323808517*t6*t9*y + (8.4487824285638778*y)*(x*x*x*x*x);
    dy = 1.4081304047606463*t0 + 2.3468840079344105*t1*t10 - 56.325216190425852*t1*t11*t6 + 46.93768015868821*t10*t6 - 37.550144126950568*t11*t4 + 7.0406520238032315*t11*t5 + 37.550144126950568*t7 - 28.162608095212926*t8 - 3.2856376111081747*y*y*y*y*y*y;
    dz = 150.20057650780227*t1*t12*y - 37.550144126950568*t1*t3*z - 50.066858835934091*t12*t3 + 18.775072063475284*t2*z - 56.325216190425852*t5*y*z;
    break;
  }
  case 54: { // l=7, m=-2
    const double t0 = 21.241569237359166*y;
    const double t1 = z*z*z*z*z;
    const double t2 = 6.6379903866747395*z;
    const double t3 = y*y*y*y*y;
    const double t4 = x*x*x*x*x;
    const double t5 = y*y*y;
    const double t6 = z*z*z;
    const double t7 = t5*t6;
    const double t8 = x*x*x;
    const double t9 = t6*t8;
    const double t10 = t5*t8;
    const double t11 = 33.189951933373697*z;
    const double t12 = 106.20784618679583*y;
    const double t13 = x*x;
    const double t14 = 39.827942320048437*z;
    const double t15 = 106.20784618679583*x;
    const double t16 = y*y;
    const double t17 = z*z;
    val = t0*t1*x + 13.275980773349479*t10*z + t2*t3*x + t2*t4*y - 35.402615395598611*t7*x - 35.402615395598611*t9*y;
    dx = t0*t1 + t11*y*(x*x*x*x) - t12*t13*t6 + t13*t14*t5 + t2*t3 - 35.402615395598611*t7;
    dy = 21.241569237359166*t1*x + t11*x*(y*y*y*y) + t14*t16*t8 - t15*t16*t6 + t2*t4 - 35.402615395598611*t9;
    dz = 13.275980773349479*t10 - t12*t17*t8 + t12*x*(z*z*z*z) - t15*t17*t5 + 6.6379903866747395*t3*x + 6.6379903866747395*t4*y;
    break;
  }
  case 55: { // l=7, m=-1
    const double t0 = x*x*x*x*x*x;
    const double t1 = z*z*z*z*z*z;
    const double t2 = x*x;
    const double t3 = y*y*y*y*y;
    const double t4 = y*y*y;
    const double t5 = z*z*z*z;
    const double t6 = x*x*x*x;
    const double t7 = z*z;
    const double t8 = 21.679585819804155*y;
    const double t9 = t2*t5;
    const double t10 = x*x*x;
    const double t11 = y*y*y*y;
    const double t12 = y*y;
    const double t13 = z*z*z;
    val = -(0.45165803791258657*t0*y - 5.7812228852811081*t1*y + 1.3549741137377597*t2*t3 - 21.679585819804155*t2*t4*t7 - 10.839792909902078*t3*t7 + 21.679585819804155*t4*t5 + 1.3549741137377597*t4*t6 - 10.839792909902078*t6*t7*y + t8*t9 + 0.45165803791258657*(y*y*y*y*y*y*y));
    dx = -(5.4198964549510389*t10*t4 - 43.359171639608311*t10*t7*y + 2.7099482274755194*t3*x - 43.359171639608311*t4*t7*x + 43.359171639608311*t5*x*y + (2.7099482274755194*y)*(x*x*x*x*x));
    dy = -(0.45165803791258657*t0 - 5.7812228852811081*t1 + 6.7748705686887986*t11*t2 - 54.198964549510389*t11*t7 - 65.038757459412466*t12*t2*t7 + 65.038757459412466*t12*t5 + 4.0649223412132791*t12*t6 - 10.839792909902078*t6*t7 + 21.679585819804155*t9 + 3.161606265388106*(y*y*y*y*y*y));
    dz = -86.718343279216622*t13*t2*y - 86.718343279216622*t13*t4 + 43.359171639608311*t2*t4*z + 21.679585819804155*t3*z + t6*t8*z + (34.687337311686649*y)*(z*z*z*z*z);
    break;
  }
  case 56: { // l=7, m=0
    const double t0 = 2.389949691920173*z;
    const double t1 = x*x*x*x*x*x;
    const double t2 = y*y*y*y*y*y;
    const double t3 = x*x;
    const double t4 = z*z*z*z*z;
    const double t5 = y*y;
    const double t6 = x*x*x*x;
    const double t7 = z*z*z;
    const double t8 = y*y*y*y;
    const double t9 = 7.1698490757605189*z;
    const double t10 = t3*t8;
    const double t11 = t5*t6;
    const double t12 = 14.339698151521038*z;
    const double t13 = x*x*x;
    const double t14 = 28.679396303042076*z;
    const double t15 = y*y*y;
    const double t16 = z*z*z*z;
    const double t17 = z*z;
    val = -(t0*t1 + t0*t2 + t10*t9 + t11*t9 + 11.47175852121683*t3*t4 - 28.679396303042076*t3*t5*t7 + 11.47175852121683*t4*t5 - 14.339698151521038*t6*t7 - 14.339698151521038*t7*t8 - 1.0925484305920791*z*z*z*z*z*z*z);
    dx = -(t12*t8*x + t12*(x*x*x*x*x) + t13*t14*t5 - 57.358792606084151*t13*t7 + 22.94351704243366*t4*x - 57.358792606084151*t5*t7*x);
    dy = -(t12*t6*y + t12*(y*y*y*y*y) + t14*t15*t3 - 57.358792606084151*t15*t7 - 57.358792606084151*t3*t7*y + 22.94351704243366*t4*y);
    dz = -(2.389949691920173*t1 + 7.1698490757605189*t10 + 7.1698490757605189*t11 + 57.358792606084151*t16*t3 + 57.358792606084151*t16*t5 - 86.038188909126227*t17*t3*t5 - 43.019094454563113*t17*t6 - 43.019094454563113*t17*t8 + 2.389949691920173*t2 - 7.6478390141445535*z*z*z*z*z*z);
    break;
  }
  case 57: { // l=7, m=1
    const double t0 = y*y*y*y*y*y;
    const double t1 = z*z*z*z*z*z;
    const double t2 = x*x*x;
    const double t3 = y*y*y*y;
    const double t4 = z*z*z*z;
    const double t5 = x*x*x*x*x;
    const double t6 = y*y;
    const double t7 = z*z;
    const double t8 = 21.679585819804155*x;
    const double t9 = t4*t6;
    const double t10 = x*x;
    const double t11 = x*x*x*x;
    const double t12 = y*y*y;
    const double t13 = z*z*z;
    val = -(0.45165803791258657*t0*x - 5.7812228852811081*t1*x + 1.3549741137377597*t2*t3 + 21.679585819804155*t2*t4 - 21.679585819804155*t2*t6*t7 - 10.839792909902078*t3*t7*x + 1.3549741137377597*t5*t6 - 10.839792909902078*t5*t7 + t8*t9 + 0.45165803791258657*(x*x*x*x*x*x*x));
    dx = -(0.45165803791258657*t0 - 5.7812228852811081*t1 + 4.0649223412132791*t10*t3 + 65.038757459412466*t10*t4 - 65.038757459412466*t10*t6*t7 + 6.7748705686887986*t11*t6 - 54.198964549510389*t11*t7 - 10.839792909902078*t3*t7 + 21.679585819804155*t9 + 3.161606265388106*(x*x*x*x*x*x));
    dy = -(5.4198964549510389*t12*t2 - 43.359171639608311*t12*t7*x - 43.359171639608311*t2*t7*y + 43.359171639608311*t4*x*y + 2.7099482274755194*t5*y + (2.7099482274755194*x)*(y*y*y*y*y));
    dz = -86.718343279216622*t13*t2 - 86.718343279216622*t13*t6*x + 43.359171639608311*t2*t6*z + t3*t8*z + 21.679585819804155*t5*z + (34.687337311686649*x)*(z*z*z*z*z);
    break;
  }
  case 58: { // l=7, m=2
    const double t0 = 3.3189951933373697*z;
    const double t1 = x*x*x*x*x*x;
    const double t2 = y*y*y*y*y*y;
    const double t3 = x*x;
    const double t4 = z*z*z*z*z;
    const double t5 = y*y;
    const double t6 = x*x*x*x;
    const double t7 = z*z*z;
    const double t8 = y*y*y*y;
    const double t9 = t3*t8;
    const double t10 = t5*t6;
    const double t11 = 19.913971160024218*z;
    const double t12 = x*x*x;
    const double t13 = 13.275980773349479*z;
    const double t14 = y*y*y;
    const double t15 = z*z*z*z;
    const double t16 = z*z;
    val = t0*t1 + t0*t10 - t0*t2 - t0*t9 + 10.620784618679583*t3*t4 - 10.620784618679583*t4*t5 - 17.701307697799305*t6*t7 + 17.701307697799305*t7*t8;
    dx = t11*(x*x*x*x*x) + t12*t13*t5 - 70.805230791197221*t12*t7 + 21.241569237359166*t4*x - 6.6379903866747395*t8*x*z;
    dy = -(t11*(y*y*y*y*y) + t13*t14*t3 - 70.805230791197221*t14*t7 + 21.241569237359166*t4*y - 6.6379903866747395*t6*y*z);
    dz = 3.3189951933373697*t1 + 3.3189951933373697*t10 + 53.103923093397916*t15*t3 - 53.103923093397916*t15*t5 - 53.103923093397916*t16*t6 + 53.103923093397916*t16*t8 - 3.3189951933373697*t2 - 3.3189951933373697*t9;
    break;
  }
  case 59: { // l=7, m=3
    const double t0 = y*y*y*y*y*y;
    const double t1 = x*x*x;
    const double t2 = y*y*y*y;
    const double t3 = z*z*z*z;
    const double t4 = x*x*x*x*x;
    const double t5 = y*y;
    const double t6 = z*z;
    const double t7 = t3*t5;
    const double t8 = x*x;
    const double t9 = x*x*x*x;
    const double t10 = y*y*y;
    const double t11 = z*z*z;
    val = -(1.4081304047606463*t0*x + 2.3468840079344105*t1*t2 - 12.516714708983523*t1*t3 - 18.775072063475284*t1*t5*t6 - 28.162608095212926*t2*t6*x + 0.4693768015868821*t4*t5 + 9.387536031737642*t4*t6 + 37.550144126950568*t7*x - 0.4693768015868821*x*x*x*x*x*x*x);
    dx = -(1.4081304047606463*t0 - 28.162608095212926*t2*t6 + 7.0406520238032315*t2*t8 - 37.550144126950568*t3*t8 - 56.325216190425852*t5*t6*t8 + 2.3468840079344105*t5*t9 + 46.93768015868821*t6*t9 + 37.550144126950568*t7 - 3.2856376111081747*x*x*x*x*x*x);
    dy = -(9.387536031737642*t1*t10 - 37.550144126950568*t1*t6*y - 112.6504323808517*t10*t6*x + 75.100288253901136*t3*x*y + 0.9387536031737642*t4*y + (8.4487824285638778*x)*(y*y*y*y*y));
    dz = 50.066858835934091*t1*t11 + 37.550144126950568*t1*t5*z - 150.20057650780227*t11*t5*x + 56.325216190425852*t2*x*z - 18.775072063475284*t4*z;
    break;
  }
  case 60: { // l=7, m=4
    const double t0 = 3.1134934723215619*z;
    const double t1 = x*x*x*x*x*x;
    const double t2 = y*y*y*y*y*y;
    const double t3 = x*x*x*x;
    const double t4 = z*z*z;
    const double t5 = y*y*y*y;
    const double t6 = 15.56746736160781*z;
    const double t7 = x*x;
    const double t8 = t5*t7;
    const double t9 = y*y;
    const double t10 = t3*t9;
    const double t11 = 18.680960833929372*z;
    const double t12 = 31.134934723215619*z;
    const double t13 = x*x*x;
    const double t14 = 62.269869446431238*z;
    const double t15 = y*y*y;
    const double t16 = z*z;
    val = -t0*t1 - t0*t2 + t10*t6 + 10.378311574405206*t3*t4 + 10.378311574405206*t4*t5 - 62.269869446431238*t4*t7*t9 + t6*t8;
    dx = -t11*x*x*x*x*x + t12*t5*x + t13*t14*t9 + 41.513246297620826*t13*t4 - 124.53973889286248*t4*t9*x;
    dy = -t11*y*y*y*y*y + t12*t3*y + t14*t15*t7 + 41.513246297620826*t15*t4 - 124.53973889286248*t4*t7*y;
    dz = -3.1134934723215619*t1 + 15.56746736160781*t10 + 31.134934723215619*t16*t3 + 31.134934723215619*t16*t5 - 186.80960833929372*t16*t7*t9 - 3.1134934723215619*t2 + 15.56746736160781*t8;
    break;
  }
  case 61: { // l=7, m=5
    const double t0 = y*y*y*y*y*y;
    const double t1 = x*x*x;
    const double t2 = y*y*y*y;
    const double t3 = x*x*x*x*x;
    const double t4 = y*y;
    const double t5 = z*z;
    const double t6 = t2*t5;
    const double t7 = x*x;
    const double t8 = x*x*x*x;
    const double t9 = y*y*y;
    val = -2.5945778936013016*t0*x + 2.5945778936013016*t1*t2 - 62.269869446431238*t1*t4*t5 + 4.6702402084823429*t3*t4 + 6.2269869446431238*t3*t5 + 31.134934723215619*t6*x - 0.51891557872026032*x*x*x*x*x*x*x;
    dx = -2.5945778936013016*t0 + 7.7837336808039048*t2*t7 - 186.80960833929372*t4*t5*t7 + 23.351201042411714*t4*t8 + 31.134934723215619*t5*t8 + 31.134934723215619*t6 - 3.6324090510418222*x*x*x*x*x*x;
    dy = -124.53973889286248*t1*t5*y + 10.378311574405206*t1*t9 + 9.3404804169646858*t3*y + 124.53973889286248*t5*t9*x - 15.56746736160781*x*y*y*y*y*y;
    dz = -124.53973889286248*t1*t4*z + 62.269869446431238*t2*x*z + 12.453973889286248*t3*z;
    break;
  }
  case 62: { // l=7, m=6
    const double t0 = 2.6459606618019002*z;
    const double t1 = x*x*x*x*x*x;
    const double t2 = y*y*y*y*y*y;
    const double t3 = 39.689409927028503*z;
    const double t4 = x*x;
    const double t5 = y*y*y*y;
    const double t6 = t4*t5;
    const double t7 = x*x*x*x;
    const double t8 = y*y;
    const double t9 = t7*t8;
    const double t10 = 15.875763970811401*z;
    const double t11 = 79.378819854057007*z;
    val = t0*t1 - t0*t2 + t3*t6 - t3*t9;
    dx = t10*(x*x*x*x*x) + t11*t5*x - 158.75763970811401*t8*z*x*x*x;
    dy = -(t10*(y*y*y*y*y) + t11*t7*y - 158.75763970811401*t4*z*y*y*y);
    dz = 2.6459606618019002*t1 - 2.6459606618019002*t2 + 39.689409927028503*t6 - 39.689409927028503*t9;
    break;
  }
  case 63: { // l=7, m=7
    const double t0 = y*y*y*y*y*y;
    const double t1 = x*x*x;
    const double t2 = y*y*y*y;
    const double t3 = x*x*x*x*x;
    const double t4 = y*y;
    val = -4.9501391276721732*t0*x + 24.750695638360866*t1*t2 - 14.85041738301652*t3*t4 + 0.70716273252459618*(x*x*x*x*x*x*x);
    dx = -4.9501391276721732*t0 + 74.252086915082599*t2*(x*x) - 74.252086915082599*t4*x*x*x*x + 4.9501391276721732*(x*x*x*x*x*x);
    dy = -(-99.002782553443465*t1*y*y*y + 29.70083476603304*t3*y + (29.70083476603304*x)*(y*y*y*y*y));
    break;
  }
  default:
    break;
  }
}

} // namespace Nukexc
