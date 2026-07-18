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

#include <iomanip>
#include <iostream>

#include <catch2/catch_all.hpp>

#include <nukexc/molecule.hpp>
#include <nukexc/partitioning.hpp>

#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include "standards.hpp"
#include <Kokkos_Core.hpp>

using namespace Nukexc;

using bk_type = IntegratorXX::Becke<double, double>;
using mk_type = IntegratorXX::MuraKnowles<double, double>;
using mhl_type = IntegratorXX::MurrayHandyLaming<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;

using ah_type = IntegratorXX::AhrensBeylkin<double>;
using de_type = IntegratorXX::Delley<double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;
using wo_type = IntegratorXX::Womersley<double>;

TEST_CASE("Fuzzy cell partitioning", "[fuzzy_cells]") {

  using namespace IntegratorXX;

  using radial_type = bk_type;
  using angular_type = ll_type;
  using angular_traits = quadrature_traits<angular_type>;

  using spherical_type = SphericalQuadrature<radial_type, angular_type>;

  std::vector<std::string> molecule_names;
  std::vector<Molecule> molecules;

  molecules.push_back(make_water());
  molecules.push_back(make_benzene());
  molecules.push_back(make_taxol());

  molecule_names.push_back("water");
  molecule_names.push_back("benzene");
  molecule_names.push_back("taxol");

#ifdef KOKKOS_ENABLE_HIP
  molecules.push_back(make_ubiquitin());
  molecule_names.push_back("ubiquitin");
#endif

  size_t nrad = 120;
  size_t nang = angular_traits::npts_by_algebraic_order(
      angular_traits::next_algebraic_order(40));

  // Generate via runtime API
  auto rad_spec = radial_from_type<radial_type>();
  auto rad_traits = make_radial_traits(rad_spec, nrad, 1.0);
  UnprunedSphericalGridSpecification unp(
      rad_spec, *rad_traits, angular_from_type<angular_type>(), nang);

  auto sph = SphericalGridFactory::generate_grid(unp);

  const auto npts = sph->npts();

  for (int mol_ind = 0; mol_ind < molecule_names.size(); ++mol_ind) {
    SECTION(molecule_names[mol_ind]) {
      // Generate water
      Molecule mol = molecules[mol_ind];
      int natoms = mol.natoms;

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

          weights_h(g) = 1.0;
          point_owner_h(g) = i;
        }
      }

      // Copy the views from host device to the execution device
      Kokkos::deep_copy(atom_centers_device, atom_centers_h);
      Kokkos::deep_copy(quadrature_points_device, quadrature_points_h);
      Kokkos::deep_copy(point_owner_device, point_owner_h);
      Kokkos::deep_copy(weights_device, weights_h);

      // Compute the adjusted weights

      Kokkos::Timer timer;
      double time;
      partition_becke(atom_centers_device, quadrature_points_device,
                      point_owner_device, weights_device);
      time = timer.seconds();

      std::cout << std::left << std::setw(50) << "Partitioning "
                << std::setw(15) << molecule_names[mol_ind] << " took "
                << std::setw(15) << std::setprecision(10) << time << " seconds"
                << std::endl;
      timer.reset();

      partition_becke_team(atom_centers_device, quadrature_points_device,
                           point_owner_device, weights_device);
      time = timer.seconds();
      std::cout << std::left << std::setw(50)
                << "Partitioning using thread teams " << std::setw(15)
                << molecule_names[mol_ind] << " took " << std::setw(15)
                << std::setprecision(10) << time << " seconds" << std::endl;
    }
  }
}

int main() {

  Kokkos::initialize();
  int result = Catch::Session().run();
  Kokkos::finalize();

  return result;
}
