#include <iostream>
#include <string_view>

#include <catch2/catch_all.hpp>
#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>

#include "../src/molecule.hpp"

#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include "../src/partitioning.hpp"
#include "standards.hpp"

using namespace NuKEXC;

// Source - https://stackoverflow.com/a/20170989
// Posted by Howard Hinnant, modified by community. See post 'Timeline' for
// change history Retrieved 2026-04-02, License - CC BY-SA 4.0

template <class T> constexpr std::string_view type_name() {
  using namespace std;
#ifdef __clang__
  string_view p = __PRETTY_FUNCTION__;
  return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
  string_view p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
  return string_view(p.data() + 36, p.size() - 36 - 1);
#else
  return string_view(p.data() + 49, p.find(';', 49) - 49);
#endif
#elif defined(_MSC_VER)
  string_view p = __FUNCSIG__;
  return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}

using bk_type = IntegratorXX::Becke<double, double>;
using mk_type = IntegratorXX::MuraKnowles<double, double>;
using mhl_type = IntegratorXX::MurrayHandyLaming<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;

using ah_type = IntegratorXX::AhrensBeylkin<double>;
using de_type = IntegratorXX::Delley<double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;
using wo_type = IntegratorXX::Womersley<double>;

TEST_CASE("H20", "[h20_weights]") {

  using namespace IntegratorXX;

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

  // Generate water
  Molecule mol = make_water();
  int natoms = mol.natoms();
  // Send all the data to Kokkos Views
  Kokkos::View<double *[3]> atom_centers("atom centers", natoms);
  Kokkos::View<double **[3]> quadrature_points("quadrature_points", natoms,
                                               npts);
  Kokkos::View<double **> weights("weights", natoms, npts);

  for (int i = 0; i < natoms; ++i) {
    atom_centers(i, 0) = mol[i].x;
    atom_centers(i, 1) = mol[i].y;
    atom_centers(i, 2) = mol[i].z;

    for (int j = 0; j < npts; ++j) {
      quadrature_points(i, j, 0) = atom_centers(j, 0) + sph->points()[j][0];
      quadrature_points(i, j, 1) = atom_centers(j, 1) + sph->points()[j][1];
      quadrature_points(i, j, 2) = atom_centers(j, 2) + sph->points()[j][2];

      // weights(i,j) = sph->weights()[i];
      weights(i, j) = 1.0;
    }
  }

  // Compute the adjusted weights
  exec_space stream;
  partition_becke(stream, atom_centers, quadrature_points, weights);

#if 0
  // Compute distance from atom centers
  std::cout << "Test is after becke" << std::endl;
  for (int i = 0; i < natoms; ++i) {
    std::cout << "Atom center : (" << atom_centers(i, 0) << " "
              << atom_centers(i, 1) << " " << atom_centers(i, 2) << ")"
              << std::endl;

    for (int j = 0; j < npts; ++j) {
      std::cout << "Quadrature point : (" << quadrature_points(i, j, 0) << " "
                << quadrature_points(i, j, 1) << " "
                << quadrature_points(i, j, 2) << ")" << std::endl;
      std::cout << "Weight : " << weights(i, j) << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "Weights have dimension " << weights.extent(0) << " "
            << weights.extent(1) << std::endl;
#endif
}

TEST_CASE("one-half", "[weights_one_half]") {

  int natoms = 2;
  int npts = 10 * 10;

  Kokkos::View<double *[3]> atom_centers("atom centers", natoms);
  Kokkos::View<double **[3]> quadrature_points("quadrature_points", natoms,
                                               npts);
  Kokkos::View<double **> weights("weights", natoms, npts);

  for (int i = 0; i < natoms; ++i) {
    atom_centers(i, 0) = (2 * i - 1);
    atom_centers(i, 1) = 0;
    atom_centers(i, 2) = 0;

    for (int j = 0; j < 10; ++j) {

      for (int k = 0; k < 10; ++k) {
        quadrature_points(i, j * 10 + k, 0) = 0;
        quadrature_points(i, j * 10 + k, 1) = j / 10.;
        quadrature_points(i, j * 10 + k, 2) = k / 10.;

        weights(i, j * 10 + k) = 1.0;
      }
    }
  }

  // Compute the adjusted weights
  exec_space stream;
  partition_becke(stream, atom_centers, quadrature_points, weights);

#if 0
  // Compute distance from atom centers
  std::cout << "Test is after becke" << std::endl;
  for (int i = 0; i < natoms; ++i) {
    std::cout << "Atom center : (" << atom_centers(i, 0) << " "
              << atom_centers(i, 1) << " " << atom_centers(i, 2) << ")"
              << std::endl;

    for (int j = 0; j < npts; ++j) {
      std::cout << "Quadrature point : (" << quadrature_points(i, j, 0) << " "
                << quadrature_points(i, j, 1) << " "
                << quadrature_points(i, j, 2) << ")" << std::endl;
      std::cout << "Weight : " << weights(i, j) << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "Weights have dimension " << weights.extent(0) << " "
            << weights.extent(1) << std::endl;
#endif
  for (int i = 0; i < natoms; ++i) {
    for (int j = 0; j < npts; ++j) {
      REQUIRE_THAT(weights(i, j), Catch::Matchers::WithinAbs(0.5, 1e-15));
    }
  }
}

TEST_CASE("SUM_TO_ONE", "[weights_sum_to_one]") {

  int natoms = 10;
  int npts = 10 * 10;

  Kokkos::View<double *[3]> atom_centers("atom centers", natoms);
  Kokkos::View<double **[3]> quadrature_points("quadrature_points", natoms,
                                               npts);
  Kokkos::View<double **> weights("weights", natoms, npts);

  // Place Atoms semi-randomly
  //
  for (int i = 0; i < natoms; ++i) {
    atom_centers(i, 0) = i;
    atom_centers(i, 1) = i % 2;
    atom_centers(i, 2) = i % 3;

    for (int j = 0; j < 10; ++j) {
      for (int k = 0; k < 10; ++k) {
        quadrature_points(i, j * 10 + k, 0) = 0;
        quadrature_points(i, j * 10 + k, 1) = j / 10.;
        quadrature_points(i, j * 10 + k, 2) = k / 10.;

        weights(i, j * 10 + k) = 1.0;
      }
    }
  }

  // Compute the adjusted weights
  exec_space stream;
  partition_becke(stream, atom_centers, quadrature_points, weights);

#if 1
  // Compute distance from atom centers
  std::cout << "Test is after becke" << std::endl;
  for (int i = 0; i < natoms; ++i) {
    std::cout << "Atom center : (" << atom_centers(i, 0) << " "
              << atom_centers(i, 1) << " " << atom_centers(i, 2) << ")"
              << std::endl;

    for (int j = 0; j < npts; ++j) {
      std::cout << "Quadrature point : (" << quadrature_points(i, j, 0) << " "
                << quadrature_points(i, j, 1) << " "
                << quadrature_points(i, j, 2) << ")" << std::endl;
      std::cout << "Weight : " << weights(i, j) << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "Weights have dimension " << weights.extent(0) << " "
            << weights.extent(1) << std::endl;
#endif
  for (int j = 0; j < npts; ++j) {

    double sum_weights = 0;
    for (int i = 0; i < natoms; ++i) {
      sum_weights += weights(i, j);
    }
    REQUIRE_THAT(sum_weights, Catch::Matchers::WithinAbs(1.0, 1e-10));
  }
}

int main() {

  Kokkos::initialize();
  int result = Catch::Session().run();
  Kokkos::finalize();

  return result;
}
