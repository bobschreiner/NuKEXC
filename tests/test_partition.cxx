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
// #include <iostream>
#include <catch2/catch_test_macros.hpp>
#include <string_view>

#include <catch2/catch_all.hpp>

#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include "standards.hpp"
#include <nukexc/molecule.hpp>
#include <nukexc/partitioning.hpp>

using namespace Nukexc;

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

  size_t nrad = 100;
  size_t nang = angular_traits::npts_by_algebraic_order(
      angular_traits::next_algebraic_order(
          12)); // Smallest possible angular grid

  // Generate via runtime API
  auto rad_spec = radial_from_type<radial_type>();
  auto rad_traits = make_radial_traits(rad_spec, nrad, 1.0);
  UnprunedSphericalGridSpecification unp(
      rad_spec, *rad_traits, angular_from_type<angular_type>(), nang);

  auto sph = SphericalGridFactory::generate_grid(unp);

  const auto npts = sph->npts();

  // Generate water
  Molecule mol = make_water();
  unsigned int natoms = mol.natoms;

  const size_t total = (size_t)natoms * npts;

  // Flat grid: one owner tag per point instead of a 2D (atom, point) shape.
  Kokkos::View<Point *> atom_centers_device("atom centers", natoms);
  Kokkos::View<Point *> quadrature_points_device("quadrature_points", total);
  Kokkos::View<int *> point_owner_device("point owner", total);
  Kokkos::View<double *> weights_device("weights", total);

  // Create all the Kokkos Mirror Views on Execution device
  auto atom_centers_h = Kokkos::create_mirror_view(atom_centers_device);
  auto quadrature_points_h =
      Kokkos::create_mirror_view(quadrature_points_device);
  auto point_owner_h = Kokkos::create_mirror_view(point_owner_device);
  auto weights_h = Kokkos::create_mirror_view(weights_device);

  Kokkos::deep_copy(atom_centers_h, mol.atom_centers);

  for (int i = 0; i < natoms; ++i) {
    for (int j = 0; j < npts; ++j) {
      const size_t g = (size_t)i * npts + j;
      quadrature_points_h(g)[0] = atom_centers_h(i)[0] + sph->points()[j][0];
      quadrature_points_h(g)[1] = atom_centers_h(i)[1] + sph->points()[j][1];
      quadrature_points_h(g)[2] = atom_centers_h(i)[2] + sph->points()[j][2];

      weights_h(g) = sph->weights()[i];
      point_owner_h(g) = i;
    }
  }

  // Copy the views from host device to the execution device
  Kokkos::deep_copy(atom_centers_device, atom_centers_h);
  Kokkos::deep_copy(quadrature_points_device, quadrature_points_h);
  Kokkos::deep_copy(point_owner_device, point_owner_h);
  Kokkos::deep_copy(weights_device, weights_h);

  // Compute the adjusted weights
  partition_becke_team(atom_centers_device, quadrature_points_device,
                       point_owner_device, weights_device);

  Kokkos::deep_copy(weights_h, weights_device);
  for (size_t g = 0; g < total; ++g) {
    REQUIRE_FALSE(Catch::isnan(weights_h(g)));
  }
}

TEST_CASE("one-half", "[weights_one_half]") {

  int natoms = 2;
  const int ngrid = 10;
  int npts = 10 * 10;
  const size_t total = (size_t)natoms * npts;

  // Flat grid: one owner tag per point instead of a 2D (atom, point) shape.
  Kokkos::View<Point *> atom_centers_device("atom centers", natoms);
  Kokkos::View<Point *> quadrature_points_device("quadrature_points", total);
  Kokkos::View<int *> point_owner_device("point owner", total);
  Kokkos::View<double *> weights_device("weights", total);

  // Create all the Kokkos Mirror Views on Execution device
  auto atom_centers_h = Kokkos::create_mirror_view(atom_centers_device);
  auto quadrature_points_h =
      Kokkos::create_mirror_view(quadrature_points_device);
  auto point_owner_h = Kokkos::create_mirror_view(point_owner_device);
  auto weights_h = Kokkos::create_mirror_view(weights_device);

  for (int i = 0; i < natoms; ++i) {
    atom_centers_h(i)[0] = (2 * i - 1);
    atom_centers_h(i)[1] = 0;
    atom_centers_h(i)[2] = 0;

    for (int j = 0; j < 10; ++j) {

      for (int k = 0; k < 10; ++k) {
        const size_t g = (size_t)i * npts + (j * ngrid + k);
        quadrature_points_h(g)[0] = 0;
        quadrature_points_h(g)[1] = j / 10.;
        quadrature_points_h(g)[2] = k / 10.;

        weights_h(g) = 1.0;
        point_owner_h(g) = i;
      }
    }
  }

  // Copy the views from host device to the execution device
  Kokkos::deep_copy(atom_centers_device, atom_centers_h);
  Kokkos::deep_copy(quadrature_points_device, quadrature_points_h);
  Kokkos::deep_copy(point_owner_device, point_owner_h);
  Kokkos::deep_copy(weights_device, weights_h);

  // Compute the adjusted weights
  partition_becke(atom_centers_device, quadrature_points_device,
                  point_owner_device, weights_device);

  // Copy weights back to the host device
  Kokkos::deep_copy(weights_h, weights_device);

  for (size_t g = 0; g < total; ++g) {
    REQUIRE_THAT(weights_h(g), Catch::Matchers::WithinAbs(0.5, 1e-15));
  }
}

TEST_CASE("SUM_TO_ONE", "[weights_sum_to_one]") {

  int natoms = 10;
  int npts = 10 * 10;
  const size_t total = (size_t)natoms * npts;

  // Flat grid: one owner tag per point instead of a 2D (atom, point) shape.
  Kokkos::View<Point *> atom_centers_device("atom centers", natoms);
  Kokkos::View<Point *> quadrature_points_device("quadrature_points", total);
  Kokkos::View<int *> point_owner_device("point owner", total);
  Kokkos::View<double *> weights_device("weights", total);

  // Create all the Kokkos Mirror Views on Execution device
  auto atom_centers_h = Kokkos::create_mirror_view(atom_centers_device);
  auto quadrature_points_h =
      Kokkos::create_mirror_view(quadrature_points_device);
  auto point_owner_h = Kokkos::create_mirror_view(point_owner_device);
  auto weights_h = Kokkos::create_mirror_view(weights_device);

  // Place Atoms semi-randomly. Every atom is given the same set of sample
  // points, so at each point the partial weights over all atoms must sum to 1.
  for (int i = 0; i < natoms; ++i) {
    atom_centers_h(i)[0] = i;
    atom_centers_h(i)[1] = i % 2;
    atom_centers_h(i)[2] = i % 3;

    for (int j = 0; j < 10; ++j) {

      for (int k = 0; k < 10; ++k) {
        const size_t g = (size_t)i * npts + (j * 10 + k);
        quadrature_points_h(g)[0] = 0;
        quadrature_points_h(g)[1] = j / 10.;
        quadrature_points_h(g)[2] = k / 10.;

        weights_h(g) = 1.0;
        point_owner_h(g) = i;
      }
    }
  }

  // Copy the views from host device to the execution device
  Kokkos::deep_copy(atom_centers_device, atom_centers_h);
  Kokkos::deep_copy(quadrature_points_device, quadrature_points_h);
  Kokkos::deep_copy(point_owner_device, point_owner_h);
  Kokkos::deep_copy(weights_device, weights_h);

  // Compute the adjusted weights
  partition_becke_team(atom_centers_device, quadrature_points_device,
                       point_owner_device, weights_device);

  // Copy weights back to the host device
  Kokkos::deep_copy(weights_h, weights_device);

  for (int j = 0; j < npts; ++j) {

    double sum_weights = 0;
    for (int i = 0; i < natoms; ++i) {
      sum_weights += weights_h((size_t)i * npts + j);
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
