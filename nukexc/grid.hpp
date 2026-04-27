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
 */

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include <nukexc/integration.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp>

#include <cmath>
#include <vector>

//
// ============================================================
//  Flat integration grid
// ============================================================
//
// Builds a Becke+Lebedev spherical grid for `mol`, applies Becke
// partitioning, and returns flattened 1D arrays ready for the
// overlap/kinetic/potential integrals.  All the repetitive Kokkos
// boilerplate lives here so test cases stay readable.

namespace NuKEXC {
struct FlatGrid {
  Kokkos::View<double *[3]> quad_points;
  Kokkos::View<double *> weights;
  Kokkos::View<double *[3]> atom_centers;
  Kokkos::View<unsigned *> Z;
};

template <typename radial_type, typename angular_type>
FlatGrid make_flat_grid(const Molecule &mol, size_t nrad = 120,
                        size_t nang_order = 40) {
  using namespace IntegratorXX;
  using angular_traits = quadrature_traits<angular_type>;

  const size_t nang = angular_traits::npts_by_algebraic_order(
      angular_traits::next_algebraic_order(nang_order));
  const unsigned natoms = mol.natoms;

  auto rad_spec = radial_from_type<radial_type>();
  auto rad_traits = make_radial_traits(rad_spec, nrad, 1.0);
  UnprunedSphericalGridSpecification unp(
      rad_spec, *rad_traits, angular_from_type<angular_type>(), nang);
  auto sph = SphericalGridFactory::generate_grid(unp);

  const unsigned npts = sph->npts();

  Kokkos::View<double *[3]> ac_dev("atom centers", natoms);
  Kokkos::View<unsigned *> Z_dev("Z", natoms);
  Kokkos::View<double **[3]> qp_2d("quadrature points", natoms, npts);
  Kokkos::View<double **> wt_2d("weights", natoms, npts);

  auto ac_h = Kokkos::create_mirror_view(ac_dev);
  auto Z_h = Kokkos::create_mirror_view(Z_dev);
  auto qp_h = Kokkos::create_mirror_view(qp_2d);
  auto wt_h = Kokkos::create_mirror_view(wt_2d);

  Kokkos::deep_copy(ac_h, mol.atom_centers);
  Kokkos::deep_copy(Z_h, mol.Z);

  for (unsigned i = 0; i < natoms; ++i)
    for (unsigned j = 0; j < npts; ++j) {
      qp_h(i, j, 0) = ac_h(i, 0) + sph->points()[j][0];
      qp_h(i, j, 1) = ac_h(i, 1) + sph->points()[j][1];
      qp_h(i, j, 2) = ac_h(i, 2) + sph->points()[j][2];
      wt_h(i, j) = sph->weights()[j];
    }

  Kokkos::deep_copy(ac_dev, ac_h);
  Kokkos::deep_copy(Z_dev, Z_h);
  Kokkos::deep_copy(qp_2d, qp_h);
  Kokkos::deep_copy(wt_2d, wt_h);

  partition_becke_team(ac_dev, qp_2d, wt_2d);

  Kokkos::View<double *[3]> qp_1d("quad points 1D", natoms * npts);
  Kokkos::View<double *> wt_1d("weights 1D", natoms * npts);

  Kokkos::parallel_for(
      "FlattenViews",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
          {0, 0}, {(int)natoms, (int)npts}),
      KOKKOS_LAMBDA(const int i, const int j) {
        const int idx = i * npts + j;
        wt_1d(idx) = wt_2d(i, j);
        qp_1d(idx, 0) = qp_2d(i, j, 0);
        qp_1d(idx, 1) = qp_2d(i, j, 1);
        qp_1d(idx, 2) = qp_2d(i, j, 2);
      });

  return {qp_1d, wt_1d, ac_dev, Z_dev};
}
} // namespace NuKEXC
