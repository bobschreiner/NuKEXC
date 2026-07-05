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

// TODO Computation of the potential of sto's seems fine, but python and c++
// algorithms give different results

#include <Kokkos_Core.hpp>

#include <catch2/catch_all.hpp>

#include <cmath>
#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include "standards.hpp"
#include <nukexc/coulomb.hpp>

#include <map>
#include <vector>

#include <iostream>

using namespace Nukexc;
using namespace IntegratorXX;

using bk_type = IntegratorXX::Becke<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

TEST_CASE("Potential satisfies Poisson's equation", "[potential][poisson]") {
  using Nukexc::STOFunc;

  // ------------------------------------------------------------------
  // 1. Uniform Cartesian lattice
  // ------------------------------------------------------------------
  // zeta = 1.0 STOs decay as exp(-r); for n <= 3 essentially all of the
  // density/potential structure lives within a few Bohr of the origin.
  // L is chosen generously so boundary truncation doesn't contaminate
  // the interior points we actually test (2-cell buffer for the FD
  // stencil is trimmed off below).
  constexpr double L = 6.0;  // half-width of box, Bohr
  constexpr double h = 0.1;  // grid spacing, Bohr
  constexpr double PI = 3.14159265358979323846;
  const int N = static_cast<int>(2.0 * L / h) + 1;  // points per axis

  Kokkos::View<double ***, Layout, ExecSpace> V("V", N, N, N);
  Kokkos::View<double ***, Layout, ExecSpace> source("source", N, N, N);
  Kokkos::View<double ***, Layout, ExecSpace> laplacian("laplacian", N, N, N);

  const double zeta = 1.0;

  int idx = 0;
  for (int n = 1; n < 4; ++n) {
    for (int l = 0; l < n; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        Kokkos::printf("Testing Poisson relation for n=%d, l=%d, m=%d\n", n, l,
                        m);

        // Single-function basis so we reuse the exact normalization/harmonic
        // convention that sto_potential/sto_potential_pre are built from.
        auto basis =
            Nukexc::make_manual_basis({STOFunc{n, l, m, zeta, 0.0, 0.0, 0.0}});

        // ---- Fill potential (analytic) and source term (psi, signed) ----
        Kokkos::parallel_for(
            "Fill V and density",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0},
                                                                {N, N, N}),
            KOKKOS_LAMBDA(int i, int j, int k) {
              const double x = -L + i * h;
              const double y = -L + j * h;
              const double z = -L + k * h;
              const double r = Kokkos::sqrt(x * x + y * y + z * z);

              V(i, j, k) = sto_potential(n, l, m, x, y, z, r, zeta);

              double psi;
              Nukexc::basis_eval(basis, 0, x, y, z, psi);
              source(i, j, k) = psi;  // signed -- source is psi, not psi^2
            });

        // ---- 4th-order (5-point) finite-difference Laplacian ----
        Kokkos::parallel_for(
            "Laplacian",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                {2, 2, 2}, {N - 2, N - 2, N - 2}),
            KOKKOS_LAMBDA(int i, int j, int k) {
              const double d2x =
                  (-V(i - 2, j, k) + 16.0 * V(i - 1, j, k) -
                   30.0 * V(i, j, k) + 16.0 * V(i + 1, j, k) -
                   V(i + 2, j, k)) /
                  (12.0 * h * h);
              const double d2y =
                  (-V(i, j - 2, k) + 16.0 * V(i, j - 1, k) -
                   30.0 * V(i, j, k) + 16.0 * V(i, j + 1, k) -
                   V(i, j + 2, k)) /
                  (12.0 * h * h);
              const double d2z =
                  (-V(i, j, k - 2) + 16.0 * V(i, j, k - 1) -
                   30.0 * V(i, j, k) + 16.0 * V(i, j, k + 1) -
                   V(i, j, k + 2)) /
                  (12.0 * h * h);
              laplacian(i, j, k) = d2x + d2y + d2z;
            });

        // ---- Reduce to max error over interior points, split by ----
        // ---- whether |source| is numerically significant there.  ----
        // source (psi) is signed, so branch on |source|, not source itself.
        // STOs have a cusp at the nucleus (Kato's cusp condition): psi is
        // continuous but not differentiable at r=0, so a Cartesian FD
        // stencil has O(1) error AT that single point regardless of h.
        // Exclude a small ball around the nucleus -- standard practice for
        // real-space Poisson checks of cusped basis functions.
        constexpr double source_floor = 1e-4;
        constexpr double cusp_exclusion_radius = 3.0 * h;

        double max_abs_err = 0.0;
        Kokkos::parallel_reduce(
            "Max abs error (small |psi|)",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                {2, 2, 2}, {N - 2, N - 2, N - 2}),
            KOKKOS_LAMBDA(int i, int j, int k, double &local_max) {
              const double x = -L + i * h;
              const double y = -L + j * h;
              const double z = -L + k * h;
              if (x * x + y * y + z * z <
                  cusp_exclusion_radius * cusp_exclusion_radius)
                return;

              const double psi = source(i, j, k);
              if (Kokkos::abs(psi) < source_floor) {
                const double rhs = -4.0 * PI * psi;
                const double err = Kokkos::abs(laplacian(i, j, k) - rhs);
                if (err > local_max) local_max = err;
              }
            },
            Kokkos::Max<double>(max_abs_err));

        double max_rel_err = 0.0;
        Kokkos::parallel_reduce(
            "Max rel error (significant |psi|)",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                {2, 2, 2}, {N - 2, N - 2, N - 2}),
            KOKKOS_LAMBDA(int i, int j, int k, double &local_max) {
              const double x = -L + i * h;
              const double y = -L + j * h;
              const double z = -L + k * h;
              if (x * x + y * y + z * z <
                  cusp_exclusion_radius * cusp_exclusion_radius)
                return;

              const double psi = source(i, j, k);
              if (Kokkos::abs(psi) >= source_floor) {
                const double rhs = -4.0 * PI * psi;
                const double err =
                    Kokkos::abs((laplacian(i, j, k) - rhs) / rhs);
                if (err > local_max) local_max = err;
              }
            },
            Kokkos::Max<double>(max_rel_err));

        INFO("n=" << n << " l=" << l << " m=" << m
                  << "  max_abs_err=" << max_abs_err
                  << "  max_rel_err=" << max_rel_err);
        REQUIRE(max_abs_err < 1e-3);
        REQUIRE(max_rel_err < 5e-2);

        idx += 1;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////
int main() {
  Kokkos::initialize();
  int result = Catch::Session().run();
  Kokkos::finalize();

  return result;
}
