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

#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_gemm_impl.hpp>

#include <KokkosLapack_svd.hpp>
#include <impl/Kokkos_CheckUsage.hpp>

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
double upper_gamma(const int n, const double x) {
  if (n <= 0)
    return 0.0;

  // This iterative Horner-like approach preserves precision perfectly.
  double sum = 1.0;
  double term = 1.0;
  for (int k = 1; k < n; ++k) {
    term *= x / k;
    sum += term;
  }

  double fact = factorial(n - 1);
  return fact * Kokkos::exp(-x) * sum;
}

KOKKOS_INLINE_FUNCTION
double lower_gamma(const int n, const double x) {
  if (n <= 0)
    return 0.0;

  // For small/moderate x, use the stable ascending series:
  // gamma(n, x) = x^n * e^-x * \sum_{k=0}^\infty \frac{x^k}{n(n+1)...(n+k)}
  // This avoids the catastrophic "1.0 - result" cancellation entirely.
  if (x < n) {
    double sum = 1.0 / n;
    double term = 1.0 / n;

    // Loop to a safe convergence limit or a fixed precision bound
    for (int k = 1; k < 1000; ++k) {
      term *= x / (n + k);
      sum += term;
      if (term < 1e-16 * sum)
        break; // Converged to machine precision
    }

    double log_prefactor = n * Kokkos::log(x) - x;
    return Kokkos::exp(log_prefactor) * sum;

  } else {
    double Gamma_n = factorial(n - 1);
    return Gamma_n - upper_gamma(n, x);
  }
}

/*
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
*/

DeviceView2DLeft compute_half_inverse(const DeviceView2DLeft &overlap_matrix,
                                      const double lin_dep_threshold = 1e-5) {

  const int N = overlap_matrix.extent(0);
  DeviceView2DLeft Us("U", N, N);
  DeviceView2DLeft VTs("VTs", N, N);
  DeviceView1D sigma("sigma", N);
  DeviceView2DLeft A("matrix A", N, N);
  DeviceView1D D_N("Digaonal Preconditioner matrix", N);

  Kokkos::parallel_for(
      "Compute preconditioner matrix", N, KOKKOS_LAMBDA(const int i) {
        D_N(i) = Kokkos::sqrt(overlap_matrix(i, i));
      });

  Kokkos::parallel_for(
      "Apply preconditioner matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i, const int j) {
        A(i, j) = overlap_matrix(i, j) / (D_N(i) * D_N(j));
      });

  // SVD of S to handle potential singularity
  KokkosLapack::svd("S", "S", A, sigma, Us, VTs);

  Kokkos::parallel_for(
      "SwitchSigns", N, KOKKOS_LAMBDA(const int j) {
        double dot = 0.0;
        for (int k = 0; k < N; ++k) {
          dot += Us(k, j) * VTs(j, k);
        }
        if (dot < 0) {

          sigma(j) = -sigma(j);
          Kokkos::printf("Negative sigma %d : %f \n", j, sigma(j));
        }
      });

  auto sigma_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sigma);

  int K = 0;
  std::vector<int> kept;
  kept.reserve(N);
  for (int i = 0; i < N; ++i) {
    if (sigma_h(i) > lin_dep_threshold) {
      kept.push_back(i);
      ++K;
    }
  }

  // Upload the index map so the kernel can use it on device
  Kokkos::View<int *, Kokkos::HostSpace> kept_h(kept.data(), K);
  Kokkos::View<int *> kept_d("kept_d", K);
  Kokkos::deep_copy(kept_d, kept_h);

  DeviceView2DLeft X("Half inverse", N, K);

  Kokkos::parallel_for(
      "Invert overlap", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, K}),
      KOKKOS_LAMBDA(const int i, const int k) {
        const int src = kept_d(k);
        X(i, k) = Us(i, src) / Kokkos::sqrt(sigma(src));
        X(i, k) /= D_N(i);
      });

  return X;
}
} // namespace Nukexc
