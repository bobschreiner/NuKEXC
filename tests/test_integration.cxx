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
#include <catch2/matchers/catch_matchers.hpp>
#include <string_view>

#include <catch2/catch_all.hpp>

#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include <nukexc/grid.hpp>
#include <nukexc/integration.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp>

#include "nukexc/kokkos_config.hpp"
#include "nukexc/octree.hpp"
#include "standards.hpp"

using namespace NuKEXC;

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
          26)); // Smallest possible angular grid

  // Generate via runtime API
  auto rad_spec = radial_from_type<radial_type>();
  auto rad_traits = make_radial_traits(rad_spec, nrad, 1.0);
  UnprunedSphericalGridSpecification unp(
      rad_spec, *rad_traits, angular_from_type<angular_type>(), nang);

  auto sph = SphericalGridFactory::generate_grid(unp);

  const unsigned npts = sph->npts();

  // Generate water
  Molecule mol = make_water();
  unsigned int natoms = mol.natoms;
  STOBasisSet stobasis = load_adf_basis(mol);

  // Create all the Kokkos Views on host device
  Kokkos::View<double *[3]> atom_centers_device("atom centers", natoms);
  Kokkos::View<double **[3]> quadrature_points_device("quadrature_points",
                                                      natoms, npts);
  Kokkos::View<double **> weights_device("weights", natoms, npts);

  // Create all the Kokkos Mirror Views on Execution device
  auto atom_centers_h = Kokkos::create_mirror_view(atom_centers_device);
  auto quadrature_points_h =
      Kokkos::create_mirror_view(quadrature_points_device);
  auto weights_h = Kokkos::create_mirror_view(weights_device);

  Kokkos::deep_copy(atom_centers_h, mol.atom_centers);

  for (int i = 0; i < natoms; ++i) {
    for (int j = 0; j < npts; ++j) {
      quadrature_points_h(i, j, 0) = atom_centers_h(i, 0) + sph->points()[j][0];
      quadrature_points_h(i, j, 1) = atom_centers_h(i, 1) + sph->points()[j][1];
      quadrature_points_h(i, j, 2) = atom_centers_h(i, 2) + sph->points()[j][2];

      weights_h(i, j) = sph->weights()[j];
    }
  }

  // Copy the views from host device to the execution device
  Kokkos::deep_copy(atom_centers_device, atom_centers_h);
  Kokkos::deep_copy(quadrature_points_device, quadrature_points_h);
  Kokkos::deep_copy(weights_device, weights_h);

  // Compute the adjusted weights
  partition_becke_team(atom_centers_device, quadrature_points_device,
                       weights_device);

  // Flatten the weights and quadrature_points

  Kokkos::View<double *> weights_1d("Weights 1D", weights_device.extent(0) *
                                                      weights_device.extent(1));

  Kokkos::View<double *[3]> quad_points_1d(
      "Quadrature points 1D",
      quadrature_points_device.extent(0) * quadrature_points_device.extent(1));

  Kokkos::parallel_for(
      "FlattenViews",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {natoms, npts}),
      KOKKOS_LAMBDA(const int i, const int j) {
        // Calculate the logical 1D index
        int flat_idx = i * npts + j;

        weights_1d(flat_idx) = weights_device(i, j);

        quad_points_1d(flat_idx, 0) = quadrature_points_device(i, j, 0);
        quad_points_1d(flat_idx, 1) = quadrature_points_device(i, j, 1);
        quad_points_1d(flat_idx, 2) = quadrature_points_device(i, j, 2);
      });

  DeviceView2DLeft S = overlap_integral(stobasis, quad_points_1d, weights_1d);

  auto S_h = Kokkos::create_mirror_view(S);
  Kokkos::deep_copy(S_h, S);

  for (unsigned i = 0; i < S_h.extent(0); ++i) {
    REQUIRE_THAT(S_h(i, i), Catch::Matchers::WithinRel(1.0, 1e-5));
  }
}

TEST_CASE("H2O overlap w/ octree screening", "[h2o][screening][overlap]") {
  using namespace IntegratorXX;
  using radial_type = bk_type;
  using angular_type = ll_type;

  Molecule mol = make_water();
  STOBasisSet stobasis = load_adf_basis(mol);
  int nbf = stobasis.nbf();

  FlatGrid grid = make_flat_grid<radial_type, angular_type>(mol);
  // Octree coming soon
}
int main() {

  Kokkos::initialize();
  int result = Catch::Session().run();
  Kokkos::finalize();

  return result;
}
