#include <Kokkos_Core.hpp>
// #include <iostream>
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

  // Create all the Kokkos Views on host device
  Kokkos::View<double *[3], Layout, ExecSpace> atom_centers_device(
      "atom centers", natoms);
  Kokkos::View<double **[3], Layout, ExecSpace> quadrature_points_device(
      "quadrature_points", natoms, npts);
  Kokkos::View<double **, Layout, ExecSpace> weights_device("weights", natoms,
                                                            npts);

  // Create all the Kokkos Mirror Views on Execution device
  auto atom_centers_h = Kokkos::create_mirror_view(atom_centers_device);
  auto quadrature_points_h =
      Kokkos::create_mirror_view(quadrature_points_device);
  auto weights_h = Kokkos::create_mirror_view(weights_device);

  for (int i = 0; i < natoms; ++i) {
    atom_centers_h(i, 0) = mol[i].x;
    atom_centers_h(i, 1) = mol[i].y;
    atom_centers_h(i, 2) = mol[i].z;

    for (int j = 0; j < npts; ++j) {
      quadrature_points_h(i, j, 0) = atom_centers_h(j, 0) + sph->points()[j][0];
      quadrature_points_h(i, j, 1) = atom_centers_h(j, 1) + sph->points()[j][1];
      quadrature_points_h(i, j, 2) = atom_centers_h(j, 2) + sph->points()[j][2];

      // weights(i,j) = sph->weights()[i];
      weights_h(i, j) = 1.0;
    }
  }

  // Copy the views from host device to the execution device
  Kokkos::deep_copy(atom_centers_device, atom_centers_h);
  Kokkos::deep_copy(quadrature_points_device, quadrature_points_h);
  Kokkos::deep_copy(weights_device, weights_h);

  // Compute the adjusted weights
  ExecSpace stream;
  partition_becke_team(stream, atom_centers_device, quadrature_points_device,
                      weights_device);
}

TEST_CASE("one-half", "[weights_one_half]") {

  int natoms = 2;
  int npts = 10 * 10;

  // Create all the Kokkos Views on host device
  Kokkos::View<double *[3], Layout, ExecSpace> atom_centers_device(
      "atom centers", natoms);
  Kokkos::View<double **[3], Layout, ExecSpace> quadrature_points_device(
      "quadrature_points", natoms, npts);
  Kokkos::View<double **, Layout, ExecSpace> weights_device("weights", natoms,
                                                            npts);

  // Create all the Kokkos Mirror Views on Execution device
  auto atom_centers_h = Kokkos::create_mirror_view(atom_centers_device);
  auto quadrature_points_h =
      Kokkos::create_mirror_view(quadrature_points_device);
  auto weights_h = Kokkos::create_mirror_view(weights_device);

  for (int i = 0; i < natoms; ++i) {
    atom_centers_h(i, 0) = (2 * i - 1);
    atom_centers_h(i, 1) = 0;
    atom_centers_h(i, 2) = 0;

    for (int j = 0; j < 10; ++j) {

      for (int k = 0; k < 10; ++k) {
        quadrature_points_h(i, j * 10 + k, 0) = 0;
        quadrature_points_h(i, j * 10 + k, 1) = j / 10.;
        quadrature_points_h(i, j * 10 + k, 2) = k / 10.;

        weights_h(i, j * 10 + k) = 1.0;
      }
    }
  }

  // Copy the views from host device to the execution device
  Kokkos::deep_copy(atom_centers_device, atom_centers_h);
  Kokkos::deep_copy(quadrature_points_device, quadrature_points_h);
  Kokkos::deep_copy(weights_device, weights_h);

  // Compute the adjusted weights
  ExecSpace stream;
  partition_becke_team(stream, atom_centers_device, quadrature_points_device,
                      weights_device);

  // Copy weights back to the host device
  Kokkos::deep_copy(weights_h, weights_device);

  for (int i = 0; i < natoms; ++i) {
    for (int j = 0; j < npts; ++j) {
      REQUIRE_THAT(weights_h(i, j), Catch::Matchers::WithinAbs(0.5, 1e-15));
    }
  }
}

TEST_CASE("SUM_TO_ONE", "[weights_sum_to_one]") {

  int natoms = 10;
  int npts = 10 * 10;

  // Create all the Kokkos Views on host device
  Kokkos::View<double *[3], Layout, ExecSpace> atom_centers_device(
      "atom centers", natoms);
  Kokkos::View<double **[3], Layout, ExecSpace> quadrature_points_device(
      "quadrature_points", natoms, npts);
  Kokkos::View<double **, Layout, ExecSpace> weights_device("weights", natoms,
                                                            npts);

  // Create all the Kokkos Mirror Views on Execution device
  auto atom_centers_h = Kokkos::create_mirror_view(atom_centers_device);
  auto quadrature_points_h =
      Kokkos::create_mirror_view(quadrature_points_device);
  auto weights_h = Kokkos::create_mirror_view(weights_device);

  // Place Atoms semi-randomly
  for (int i = 0; i < natoms; ++i) {
    atom_centers_h(i, 0) = i;
    atom_centers_h(i, 1) = i % 2;
    atom_centers_h(i, 2) = i % 3;

    for (int j = 0; j < 10; ++j) {
      for (int k = 0; k < 10; ++k) {
        quadrature_points_h(i, j * 10 + k, 0) = 0;
        quadrature_points_h(i, j * 10 + k, 1) = j / 10.;
        quadrature_points_h(i, j * 10 + k, 2) = k / 10.;

        weights_h(i, j * 10 + k) = 1.0;
      }
    }
  }

  // Copy the views from host device to the execution device
  Kokkos::deep_copy(atom_centers_device, atom_centers_h);
  Kokkos::deep_copy(quadrature_points_device, quadrature_points_h);
  Kokkos::deep_copy(weights_device, weights_h);

  // Compute the adjusted weights
  ExecSpace stream;
  partition_becke_team(stream, atom_centers_device, quadrature_points_device,
                      weights_device);

  // Copy weights back to the host device
  Kokkos::deep_copy(weights_h, weights_device);

  for (int j = 0; j < npts; ++j) {

    double sum_weights = 0;
    for (int i = 0; i < natoms; ++i) {
      sum_weights += weights_h(i, j);
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
