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

#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include "../src/spherical_harmonics.hpp"
#include "standards.hpp"
#include <map>
#include <vector>

#include <iostream>

using namespace NuKEXC;
using namespace IntegratorXX;

using bk_type = IntegratorXX::Becke<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

TEST_CASE("Sph harmonicss", "[compute_spherical_harmonics]") {

  ////////////////////////////////////////////////////////////////////////
  /*
   * Setup the quadrature grid
   */

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

  Kokkos::View<double *[3], Layout, ExecSpace> quadrature_points_device(
      "quadrature_points", npts);

  auto quadrature_points_h =
      Kokkos::create_mirror_view(quadrature_points_device);

  for (int i = 0; i < npts; ++i) {
    quadrature_points_h(i, 0) = sph->points()[i][0];
    quadrature_points_h(i, 1) = sph->points()[i][1];
    quadrature_points_h(i, 2) = sph->points()[i][2];
  }
  Kokkos::deep_copy(quadrature_points_device, quadrature_points_h);

  ////////////////////////////////////////////////////////////////////////

  /*
   * Generate analytical solutions
   */

  auto ref_00 = [](double x, double y, double z) -> double {
    return 0.5 * (1. / std::sqrt(M_PI));
  };

  auto ref_1m1 = [](double x, double y, double z) {
    double r = std::sqrt(x * x + y * y + z * z);
    return (y / r) * std::sqrt(3. / (4. * M_PI));
  };
  auto ref_10 = [](double x, double y, double z) {
    double r = std::sqrt(x * x + y * y + z * z);
    return (z / r) * std::sqrt(3. / (4. * M_PI));
  };
  auto ref_11 = [](double x, double y, double z) {
    double r = std::sqrt(x * x + y * y + z * z);
    return (x / r) * std::sqrt(3. / (4. * M_PI));
  };
  auto ref_2m2 = [](double x, double y, double z) {
    double r = std::sqrt(x * x + y * y + z * z);
    return 0.5 * ((x * y) / (r * r)) * std::sqrt(15. / (M_PI));
  };
  auto ref_2m1 = [](double x, double y, double z) {
    double r = std::sqrt(x * x + y * y + z * z);
    return 0.5 * ((z * y) / (r * r)) * std::sqrt(15. / (M_PI));
  };

  auto ref_20 = [](double x, double y, double z) {
    double r = std::sqrt(x * x + y * y + z * z);
    return 0.25 * ((3 * z * z - r * r) / (r * r)) * std::sqrt(5. / (M_PI));
  };

  auto ref_21 = [](double x, double y, double z) {
    double r = std::sqrt(x * x + y * y + z * z);
    return 0.5 * ((x * z) / (r * r)) * std::sqrt(15. / (M_PI));
  };
  auto ref_22 = [](double x, double y, double z) {
    double r = std::sqrt(x * x + y * y + z * z);
    return 0.25 * ((x * x - y * y) / (r * r)) * std::sqrt(15. / (M_PI));
  };

  std::vector<std::map<int, std::function<double(double, double, double)>>>
      ref_function_vector;

  std::map<int, std::function<double(double, double, double)>> s_harmonics;
  std::map<int, std::function<double(double, double, double)>> p_harmonics;
  std::map<int, std::function<double(double, double, double)>> d_harmonics;

  s_harmonics[0] = ref_00;

  p_harmonics[0] = ref_10;
  p_harmonics[1] = ref_11;
  p_harmonics[-1] = ref_1m1;

  d_harmonics[0] = ref_20;
  d_harmonics[1] = ref_21;
  d_harmonics[2] = ref_22;
  d_harmonics[-1] = ref_2m1;
  d_harmonics[-2] = ref_2m2;

  ref_function_vector.push_back(s_harmonics);
  ref_function_vector.push_back(p_harmonics);
  ref_function_vector.push_back(d_harmonics);

  ///////////////////////////////////////////////////////////////////////////

  /*
   * Compute spherical harmonics in cartesian and in spherical coordinates
   */

  Kokkos::View<double *> harmonic_cart("harmonic", npts);
  auto harmonic_cart_h = Kokkos::create_mirror_view(harmonic_cart);

  Kokkos::View<double *> harmonic_sph("harmonic_sph", npts);
  auto harmonic_sph_h = Kokkos::create_mirror_view(harmonic_cart);

  // helper function to keep code concise
  auto check_harmonics =
      [&harmonic_cart_h, &npts, &quadrature_points_h](
          std::function<double(double, double, double)> ref_function) {
        for (int i = 0; i < npts; ++i) {
          double ref =
              ref_function(quadrature_points_h(i, 0), quadrature_points_h(i, 1),
                           quadrature_points_h(i, 2));

          REQUIRE_THAT(harmonic_cart_h(i),
                       Catch::Matchers::WithinAbs(ref, 1e-12));
        }
      };

  /*
   * Compare spherical harmonics in cartesian coordinates to analytical
   * solutions
   */

  for (int l = 0; l < 3; ++l) {
    for (int m = -l; m < l + 1; ++m) {

      std::cout << "Testing analytical l = " << l << " , m = " << m
                << std::endl;
      Kokkos::parallel_for(
          "For loop", npts, KOKKOS_LAMBDA(int i) {
            harmonic_cart(i) = NuKEXC::detail::real_spherical_harmonic_cart(
                l, m, quadrature_points_device(i, 0),
                quadrature_points_device(i, 1), quadrature_points_device(i, 2));
          });

      Kokkos::deep_copy(harmonic_cart_h, harmonic_cart);
      check_harmonics(ref_function_vector[l][m]);
    }
  }

  /*
   * Compare spherical harmonics in cartesian coordinates to analytical
   * solutions
   */

  for (int l = 0; l < 7; ++l) {
    for (int m = -l; m < l + 1; ++m) {

      std::cout << "Testing numerical l = " << l << " , m = " << m << std::endl;
      Kokkos::parallel_for(
          "For loop", npts, KOKKOS_LAMBDA(int i) {
            harmonic_sph(i) =
                NuKEXC::detail::real_spherical_harmonic_sph_from_cart(
                    l, m, quadrature_points_device(i, 0),
                    quadrature_points_device(i, 1),
                    quadrature_points_device(i, 2));

            harmonic_cart(i) = NuKEXC::detail::real_spherical_harmonic_cart(
                l, m, quadrature_points_device(i, 0),
                quadrature_points_device(i, 1), quadrature_points_device(i, 2));
          });
      Kokkos::deep_copy(harmonic_cart_h, harmonic_cart);
      Kokkos::deep_copy(harmonic_sph_h, harmonic_sph);
      for (int i = 0; i < npts; ++i) {
        REQUIRE_THAT(harmonic_cart_h(i),
                     Catch::Matchers::WithinAbs(harmonic_sph_h(i), 1e-12));
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
