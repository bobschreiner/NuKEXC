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
#include <nukexc/poisson.hpp>

#include <map>
#include <vector>

#include <iostream>

using namespace NuKEXC;
using namespace IntegratorXX;

using bk_type = IntegratorXX::Becke<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

KOKKOS_INLINE_FUNCTION
double sto_potential_pre(const int idx, const double x, const double y,
                         const double z, const double r, const double zeta) {

  switch (idx) {
  case 0: // n=1, l=0, m=0
    return -7.0898154036220641 * pow(zeta, -0.5) * exp(-r * zeta) +
           14.179630807244128 * pow(zeta, -1.5) / r -
           14.179630807244128 * pow(zeta, -1.5) * exp(-r * zeta) / r;
  case 1: // n=2, l=0, m=0
    return -4.093306831785954 * r * sqrt(zeta) * exp(-r * zeta) -
           16.373227327143816 * pow(zeta, -0.5) * exp(-r * zeta) +
           24.559840990715724 * pow(zeta, -1.5) / r -
           24.559840990715724 * pow(zeta, -1.5) * exp(-r * zeta) / r;
  case 2: // n=2, l=1, m=-1
    return -7.0898154036220641 * y * sqrt(zeta) * exp(-r * zeta) -
           28.359261614488256 * y * pow(zeta, -0.5) * exp(-r * zeta) / r -
           56.718523228976513 * y * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 2) +
           56.718523228976513 * y * pow(zeta, -2.5) / pow(r, 3) -
           56.718523228976513 * y * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 3);
  case 3: // n=2, l=1, m=0
    return -7.0898154036220641 * z * sqrt(zeta) * exp(-r * zeta) -
           28.359261614488256 * z * pow(zeta, -0.5) * exp(-r * zeta) / r -
           56.718523228976513 * z * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 2) +
           56.718523228976513 * z * pow(zeta, -2.5) / pow(r, 3) -
           56.718523228976513 * z * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 3);
  case 4: // n=2, l=1, m=1
    return -7.0898154036220641 * x * sqrt(zeta) * exp(-r * zeta) -
           28.359261614488256 * x * pow(zeta, -0.5) * exp(-r * zeta) / r -
           56.718523228976513 * x * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 2) +
           56.718523228976513 * x * pow(zeta, -2.5) / pow(r, 3) -
           56.718523228976513 * x * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 3);
  case 5: // n=3, l=0, m=0
    return -1.494664324372781 * pow(r, 2) * pow(zeta, 1.5) * exp(-r * zeta) -
           8.9679859462366859 * r * sqrt(zeta) * exp(-r * zeta) -
           26.903957838710058 * pow(zeta, -0.5) * exp(-r * zeta) +
           35.871943784946744 * pow(zeta, -1.5) / r -
           35.871943784946744 * pow(zeta, -1.5) * exp(-r * zeta) / r;
  case 6: // n=3, l=1, m=-1
    return -2.5888345500742657 * r * y * pow(zeta, 1.5) * exp(-r * zeta) -
           15.533007300445594 * y * sqrt(zeta) * exp(-r * zeta) -
           51.776691001485313 * y * pow(zeta, -0.5) * exp(-r * zeta) / r -
           103.55338200297063 * y * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 2) +
           103.55338200297063 * y * pow(zeta, -2.5) / pow(r, 3) -
           103.55338200297063 * y * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 3);
  case 7: // n=3, l=1, m=0
    return -2.5888345500742657 * r * z * pow(zeta, 1.5) * exp(-r * zeta) -
           15.533007300445594 * z * sqrt(zeta) * exp(-r * zeta) -
           51.776691001485313 * z * pow(zeta, -0.5) * exp(-r * zeta) / r -
           103.55338200297063 * z * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 2) +
           103.55338200297063 * z * pow(zeta, -2.5) / pow(r, 3) -
           103.55338200297063 * z * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 3);
  case 8: // n=3, l=1, m=1
    return -2.5888345500742657 * r * x * pow(zeta, 1.5) * exp(-r * zeta) -
           15.533007300445594 * x * sqrt(zeta) * exp(-r * zeta) -
           51.776691001485313 * x * pow(zeta, -0.5) * exp(-r * zeta) / r -
           103.55338200297063 * x * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 2) +
           103.55338200297063 * x * pow(zeta, -2.5) / pow(r, 3) -
           103.55338200297063 * x * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 3);
  case 9: // n=3, l=2, m=-2
    return -5.7888100364661413 * x * y * pow(zeta, 1.5) * exp(-r * zeta) -
           34.732860218796848 * x * y * sqrt(zeta) * exp(-r * zeta) / r -
           138.93144087518739 * x * y * pow(zeta, -0.5) * exp(-r * zeta) /
               pow(r, 2) -
           416.79432262556217 * x * y * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 3) -
           833.58864525112434 * x * y * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 4) +
           833.58864525112434 * x * y * pow(zeta, -3.5) / pow(r, 5) -
           833.58864525112434 * x * y * pow(zeta, -3.5) * exp(-r * zeta) /
               pow(r, 5);
  case 10: // n=3, l=2, m=-1
    return -5.7888100364661413 * y * z * pow(zeta, 1.5) * exp(-r * zeta) -
           34.732860218796848 * y * z * sqrt(zeta) * exp(-r * zeta) / r -
           138.93144087518739 * y * z * pow(zeta, -0.5) * exp(-r * zeta) /
               pow(r, 2) -
           416.79432262556217 * y * z * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 3) -
           833.58864525112434 * y * z * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 4) +
           833.58864525112434 * y * z * pow(zeta, -3.5) / pow(r, 5) -
           833.58864525112434 * y * z * pow(zeta, -3.5) * exp(-r * zeta) /
               pow(r, 5);
  case 11: // n=3, l=2, m=0
    return 1.671085516420667 * pow(x, 2) * pow(zeta, 1.5) * exp(-r * zeta) +
           1.671085516420667 * pow(y, 2) * pow(zeta, 1.5) * exp(-r * zeta) -
           3.342171032841334 * pow(z, 2) * pow(zeta, 1.5) * exp(-r * zeta) +
           10.026513098524002 * pow(x, 2) * sqrt(zeta) * exp(-r * zeta) / r +
           10.026513098524002 * pow(y, 2) * sqrt(zeta) * exp(-r * zeta) / r -
           20.053026197048004 * pow(z, 2) * sqrt(zeta) * exp(-r * zeta) / r +
           40.106052394096008 * pow(x, 2) * pow(zeta, -0.5) * exp(-r * zeta) /
               pow(r, 2) +
           40.106052394096008 * pow(y, 2) * pow(zeta, -0.5) * exp(-r * zeta) /
               pow(r, 2) -
           80.212104788192016 * pow(z, 2) * pow(zeta, -0.5) * exp(-r * zeta) /
               pow(r, 2) +
           120.31815718228802 * pow(x, 2) * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 3) +
           120.31815718228802 * pow(y, 2) * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 3) -
           240.63631436457605 * pow(z, 2) * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 3) +
           240.63631436457605 * pow(x, 2) * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 4) +
           240.63631436457605 * pow(y, 2) * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 4) -
           481.2726287291521 * pow(z, 2) * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 4) -
           240.63631436457605 * pow(x, 2) * pow(zeta, -3.5) / pow(r, 5) +
           240.63631436457605 * pow(x, 2) * pow(zeta, -3.5) * exp(-r * zeta) /
               pow(r, 5) -
           240.63631436457605 * pow(y, 2) * pow(zeta, -3.5) / pow(r, 5) +
           240.63631436457605 * pow(y, 2) * pow(zeta, -3.5) * exp(-r * zeta) /
               pow(r, 5) +
           481.2726287291521 * pow(z, 2) * pow(zeta, -3.5) / pow(r, 5) -
           481.2726287291521 * pow(z, 2) * pow(zeta, -3.5) * exp(-r * zeta) /
               pow(r, 5);
  case 12: // n=3, l=2, m=1
    return -5.7888100364661413 * x * z * pow(zeta, 1.5) * exp(-r * zeta) -
           34.732860218796848 * x * z * sqrt(zeta) * exp(-r * zeta) / r -
           138.93144087518739 * x * z * pow(zeta, -0.5) * exp(-r * zeta) /
               pow(r, 2) -
           416.79432262556217 * x * z * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 3) -
           833.58864525112434 * x * z * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 4) +
           833.58864525112434 * x * z * pow(zeta, -3.5) / pow(r, 5) -
           833.58864525112434 * x * z * pow(zeta, -3.5) * exp(-r * zeta) /
               pow(r, 5);
  case 13: // n=3, l=2, m=2
    return -2.8944050182330706 * pow(x, 2) * pow(zeta, 1.5) * exp(-r * zeta) +
           2.8944050182330706 * pow(y, 2) * pow(zeta, 1.5) * exp(-r * zeta) -
           17.366430109398424 * pow(x, 2) * sqrt(zeta) * exp(-r * zeta) / r +
           17.366430109398424 * pow(y, 2) * sqrt(zeta) * exp(-r * zeta) / r -
           69.465720437593695 * pow(x, 2) * pow(zeta, -0.5) * exp(-r * zeta) /
               pow(r, 2) +
           69.465720437593695 * pow(y, 2) * pow(zeta, -0.5) * exp(-r * zeta) /
               pow(r, 2) -
           208.39716131278109 * pow(x, 2) * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 3) +
           208.39716131278109 * pow(y, 2) * pow(zeta, -1.5) * exp(-r * zeta) /
               pow(r, 3) -
           416.79432262556217 * pow(x, 2) * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 4) +
           416.79432262556217 * pow(y, 2) * pow(zeta, -2.5) * exp(-r * zeta) /
               pow(r, 4) +
           416.79432262556217 * pow(x, 2) * pow(zeta, -3.5) / pow(r, 5) -
           416.79432262556217 * pow(x, 2) * pow(zeta, -3.5) * exp(-r * zeta) /
               pow(r, 5) -
           416.79432262556217 * pow(y, 2) * pow(zeta, -3.5) / pow(r, 5) +
           416.79432262556217 * pow(y, 2) * pow(zeta, -3.5) * exp(-r * zeta) /
               pow(r, 5);
  default:
    return 0.0;
  }
}

TEST_CASE("Potential vs precomputed Potentia", "[potential]") {

  using radial_type = bk_type;
  using angular_type = ll_type;
  using angular_traits = quadrature_traits<angular_type>;

  using spherical_type = SphericalQuadrature<radial_type, angular_type>;

  size_t nrad = 10;
  size_t nang = angular_traits::npts_by_algebraic_order(
      angular_traits::next_algebraic_order(
          5)); // Smallest possible angular grid

  // Generate via runtime API
  auto rad_spec = radial_from_type<radial_type>();
  auto rad_traits = make_radial_traits(rad_spec, nrad, 1.0);
  UnprunedSphericalGridSpecification unp(
      rad_spec, *rad_traits, angular_from_type<angular_type>(), nang);

  auto sph = SphericalGridFactory::generate_grid(unp);

  const auto npts = sph->npts();

  Kokkos::View<Point *, Layout, ExecSpace> quadrature_points_device(
      "quadrature_points", npts);

  auto quadrature_points_h =
      Kokkos::create_mirror_view(quadrature_points_device);

  for (int i = 0; i < npts; ++i) {
    quadrature_points_h(i)[0] = sph->points()[i][0];
    quadrature_points_h(i)[1] = sph->points()[i][1];
    quadrature_points_h(i)[2] = sph->points()[i][2];
  }
  Kokkos::deep_copy(quadrature_points_device, quadrature_points_h);

  ///////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////
  Kokkos::View<double *> potential_pre("potential_pre", npts);
  auto potential_pre_h = Kokkos::create_mirror_view(potential_pre);

  Kokkos::View<double *> potential("potential_harmonics", npts);
  auto potential_h = Kokkos::create_mirror_view(potential);

  /*
   * Compare analytical spherical harmonics in cartesian coordinates to finite
   * difference solutions
   */
  int idx = 0;
  for (int n = 1; n < 4; ++n) {
    for (int l = 0; l < n; ++l) {
      for (int m = -l; m < l + 1; ++m) {

        Kokkos::printf("Testing n = %d , l = %d , m = %d \n", n, l, m);
        Kokkos::printf("idx = %d\n", idx);
        Kokkos::parallel_for(
            "For loop", npts, KOKKOS_LAMBDA(int i) {
              const double x = quadrature_points_device(i)[0];
              const double y = quadrature_points_device(i)[1];
              const double z = quadrature_points_device(i)[2];
              const double r = Kokkos::sqrt(x * x + y * y + z * z);
              const double zeta = 0.69420;

              potential(i) = sto_potential(n, l, m, x, y, z, r, zeta);
              potential_pre(i) = sto_potential_pre(idx, x, y, z, r, zeta);
            });

        Kokkos::deep_copy(potential_h, potential);
        Kokkos::deep_copy(potential_pre_h, potential_pre);

        for (int i = 0; i < npts; ++i) {
          if (potential_h(i) < 1e-5) {
            REQUIRE_THAT(potential_h(i),
                         Catch::Matchers::WithinAbs(potential_pre_h(i), 1e-5));
          } else {
            REQUIRE_THAT(potential_h(i), Catch::Matchers::WithinRel(
                                             potential_pre_h(i), 1e-3));
          }
        }
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
