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

using namespace NuKEXC;
using namespace IntegratorXX;

using bk_type = IntegratorXX::Becke<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

TEST_CASE("Sph harmonicss", "[compute_spherical_harmonics]") {

  ////////////////////////////////////////////////////////////////////////

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

  ///////////////////////////////////////////////////////////////////////////

  for (int i = 0; i < npts; ++i) {
    double ref = ref_00(quadrature_points_h(i, 0), quadrature_points_h(i, 1),
                        quadrature_points_h(i, 2));

    double harmonic = NuKEXC::detail::real_spherical_harmonic_cart(
        0, 0, quadrature_points_device(i, 0), quadrature_points_device(i, 1),
        quadrature_points_device(i, 2));
    REQUIRE_THAT(harmonic, Catch::Matchers::WithinAbs(ref, 1e-15));
  }
  for (int i = 0; i < npts; ++i) {
    double ref = ref_10(quadrature_points_h(i, 0), quadrature_points_h(i, 1),
                        quadrature_points_h(i, 2));

    double harmonic = NuKEXC::detail::real_spherical_harmonic_cart(
        1, 0, quadrature_points_device(i, 0), quadrature_points_device(i, 1),
        quadrature_points_device(i, 2));
    REQUIRE_THAT(harmonic, Catch::Matchers::WithinAbs(ref, 1e-15));
  }
  for (int i = 0; i < npts; ++i) {
    double ref = ref_11(quadrature_points_h(i, 0), quadrature_points_h(i, 1),
                        quadrature_points_h(i, 2));

    double harmonic = NuKEXC::detail::real_spherical_harmonic_cart(
        1, 1, quadrature_points_device(i, 0), quadrature_points_device(i, 1),
        quadrature_points_device(i, 2));
    REQUIRE_THAT(harmonic, Catch::Matchers::WithinAbs(ref, 1e-15));
  }
  for (int i = 0; i < npts; ++i) {
    double ref = ref_1m1(quadrature_points_h(i, 0), quadrature_points_h(i, 1),
                         quadrature_points_h(i, 2));

    double harmonic = NuKEXC::detail::real_spherical_harmonic_cart(
        1, -1, quadrature_points_device(i, 0), quadrature_points_device(i, 1),
        quadrature_points_device(i, 2));
    REQUIRE_THAT(harmonic, Catch::Matchers::WithinAbs(ref, 1e-15));
  }

  for (int i = 0; i < npts; ++i) {
    double ref = ref_2m2(quadrature_points_h(i, 0), quadrature_points_h(i, 1),
                         quadrature_points_h(i, 2));

    double harmonic = NuKEXC::detail::real_spherical_harmonic_cart(
        2, -2, quadrature_points_device(i, 0), quadrature_points_device(i, 1),
        quadrature_points_device(i, 2));
    REQUIRE_THAT(harmonic, Catch::Matchers::WithinAbs(ref, 1e-15));
  }
  for (int i = 0; i < npts; ++i) {
    double ref = ref_2m1(quadrature_points_h(i, 0), quadrature_points_h(i, 1),
                         quadrature_points_h(i, 2));

    double harmonic = NuKEXC::detail::real_spherical_harmonic_cart(
        2, -1, quadrature_points_device(i, 0), quadrature_points_device(i, 1),
        quadrature_points_device(i, 2));
    REQUIRE_THAT(harmonic, Catch::Matchers::WithinAbs(ref, 1e-15));
  }
  for (int i = 0; i < npts; ++i) {
    double ref = ref_20(quadrature_points_h(i, 0), quadrature_points_h(i, 1),
                        quadrature_points_h(i, 2));

    double harmonic = NuKEXC::detail::real_spherical_harmonic_cart(
        2, 0, quadrature_points_device(i, 0), quadrature_points_device(i, 1),
        quadrature_points_device(i, 2));
    REQUIRE_THAT(harmonic, Catch::Matchers::WithinAbs(ref, 1e-15));
  }
  for (int i = 0; i < npts; ++i) {
    double ref = ref_21(quadrature_points_h(i, 0), quadrature_points_h(i, 1),
                        quadrature_points_h(i, 2));

    double harmonic = NuKEXC::detail::real_spherical_harmonic_cart(
        2, 1, quadrature_points_device(i, 0), quadrature_points_device(i, 1),
        quadrature_points_device(i, 2));
    REQUIRE_THAT(harmonic, Catch::Matchers::WithinAbs(ref, 1e-15));
  }
  for (int i = 0; i < npts; ++i) {
    double ref = ref_22(quadrature_points_h(i, 0), quadrature_points_h(i, 1),
                        quadrature_points_h(i, 2));

    double harmonic = NuKEXC::detail::real_spherical_harmonic_cart(
        2, 2, quadrature_points_device(i, 0), quadrature_points_device(i, 1),
        quadrature_points_device(i, 2));
    REQUIRE_THAT(harmonic, Catch::Matchers::WithinAbs(ref, 1e-15));
  }
}

int main() {

  Kokkos::initialize();
  int result = Catch::Session().run();
  Kokkos::finalize();

  return result;
}
