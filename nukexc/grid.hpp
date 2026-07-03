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

#pragma once

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/radial/becke.hpp>
#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include "molecule.hpp"
#include "nukexc/atomic_properties.hpp"
#include "partitioning.hpp"
#include "stobasis.hpp"

#include <vector>

//
// ============================================================
//  Flat integration grid
// ============================================================
//
// Builds a spherical grid for `mol`, applies Becke
// partitioning, and returns flattened 1D arrays ready for the
// overlap/kinetic/potential integrals.  All the repetitive Kokkos
// boilerplate lives here so test cases stay readable.

namespace Nukexc {

const std::vector<double> BECKE_SLATER_RADII = {
    0.00,       // 0: Dummy
    0.35, 0.35, // 1: H (Becke modified), 2: He
    1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.35, // 3-10: Li to Ne
    1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 0.70, // 11-18: Na to Ar
    2.20, 1.80, 1.60, 1.45, 1.40, 1.35, 1.40, 1.35,
    1.35, 1.35,                                     // 19-28: K to Ni
    1.35, 1.35, 1.30, 1.25, 1.15, 1.10, 1.00, 0.90, // 29-36: Cu to Kr
    2.35, 2.00, 1.80, 1.55, 1.45, 1.45, 1.35, 1.35,
    1.40, 1.40,                                     // 37-46: Rb to Pd
    1.40, 1.45, 1.45, 1.45, 1.45, 1.40, 1.35, 1.20, // 47-54: Ag to Xe
    2.60, 2.15, 1.95, 1.85, 1.85, 1.85, 1.85, 1.85,
    1.80, 1.75, // 55-64: Cs to Gd
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.70,
    1.60, 1.45, // 65-74: Tb to W
    1.35, 1.35, 1.35, 1.35, 1.35, 1.40, 1.45, 1.50,
    1.50, 1.45, // 75-84: Re to Po
    1.40, 1.30, // 85-86: At, Rn
    2.80, 2.35, 2.15, 1.95, 1.80, 1.80, 1.75, 1.75,
    1.75, 1.75, // 87-96: Fr to Cm
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, // 97-106: Bk to Sg
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, // 107-116: Bh to Lv
    1.75, 1.75  // 117-118: Ts, Og
};

const std::vector<double> TA_XI = {
    0.00, // 0: Dummy
    0.8,  0.9, 1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9, 1.4, 1.3,
    1.3,  1.2, 1.1, 1.0, 1.0, 1.0, 1.5, 1.4, 1.3, 1.2, 1.2, 1.2,
    1.2,  1.2, 1.2, 1.1, 1.1, 1.1, 1.1, 1.0, 0.9, 0.9, 0.9, 0.9};
struct FlatGrid {

  Kokkos::View<Point *> quad_points;
  Kokkos::View<double *> weights;
  Kokkos::View<Point *> atom_centers;
  Kokkos::View<unsigned *> Z;
  unsigned nrad; // radial points per atom
  unsigned nang; // angular points per radial shell
};

template <typename radial_type, typename angular_type>
FlatGrid make_flat_grid(const Molecule &mol, const size_t nrad = 50,
                        const size_t nang_order = 30,
                        const double weight_threshold = 1e-30) {

  using namespace IntegratorXX;
  using angular_traits = quadrature_traits<angular_type>;

  const size_t nang = angular_traits::npts_by_algebraic_order(
      angular_traits::next_algebraic_order(nang_order));
  const unsigned natoms = mol.natoms;

  auto rad_spec = radial_from_type<radial_type>();

  const int npts = nrad * nang;

  Kokkos::View<Point *> ac_dev("atom centers", natoms);
  Kokkos::View<unsigned *> Z_dev("Z", natoms);
  Kokkos::View<Point **> qp_2d("quadrature points", natoms, npts);
  Kokkos::View<double **> wt_2d("weights", natoms, npts);

  auto ac_h = Kokkos::create_mirror_view(ac_dev);
  auto Z_h = Kokkos::create_mirror_view(Z_dev);
  auto qp_h = Kokkos::create_mirror_view(qp_2d);
  auto wt_h = Kokkos::create_mirror_view(wt_2d);

  Kokkos::deep_copy(ac_h, mol.atom_centers);
  Kokkos::deep_copy(Z_h, mol.Z);

  for (unsigned i = 0; i < natoms; ++i) {

    unsigned atomic_number = Z_h(i);

    // Fallback scale to 1.0 if Z is out of range of your vector
    double r_atomic = 1.0;

    if (typeid(IntegratorXX::TreutlerAhlrichs<double, double>).name() ==
        typeid(radial_type).name()) {
      /*
if (atomic_number < TA_XI.size()) {
  r_atomic = TA_XI[atomic_number];
}
*/

    } else if ((typeid(IntegratorXX::Becke<double, double>).name() ==
                typeid(radial_type).name())) {
      if (atomic_number < BECKE_SLATER_RADII.size()) {
        r_atomic =
            0.5 * BECKE_SLATER_RADII[atomic_number] * detail::ang_to_bohr;
      }
    }

    auto rad_traits = make_radial_traits(rad_spec, nrad, r_atomic);
    UnprunedSphericalGridSpecification unp(
        rad_spec, *rad_traits, angular_from_type<angular_type>(), nang);
    auto sph = SphericalGridFactory::generate_grid(unp);

    for (unsigned j = 0; j < npts; ++j) {
      qp_h(i, j)[0] = ac_h(i)[0] + sph->points()[j][0];
      qp_h(i, j)[1] = ac_h(i)[1] + sph->points()[j][1];
      qp_h(i, j)[2] = ac_h(i)[2] + sph->points()[j][2];
      wt_h(i, j) = sph->weights()[j];
    }
  }

  Kokkos::deep_copy(ac_dev, ac_h);
  Kokkos::deep_copy(Z_dev, Z_h);
  Kokkos::deep_copy(qp_2d, qp_h);
  Kokkos::deep_copy(wt_2d, wt_h);

  partition_becke_team(ac_dev, qp_2d, wt_2d);

  // Remove all weights below the weight threshold
  Kokkos::View<int *> w_counter("Counter", 1);
  Kokkos::parallel_for(
      "Count weights",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {natoms, npts}),
      KOKKOS_LAMBDA(const int iatoms, const int ipts) {
        if (wt_2d(iatoms, ipts) > weight_threshold) {
          Kokkos::atomic_add(&w_counter(0), 1);
        };
      });

  auto w_counter_h =
      Kokkos::create_mirror_view_and_copy(HostSpace{}, w_counter);

  // Initialize the Flatterend views with the correct sizes
  Kokkos::View<Point *> qp_1d("quad points 1D", w_counter_h(0));
  Kokkos::View<double *> wt_1d("weights 1D", w_counter_h(0));

  // Reset the counter
  Kokkos::deep_copy(w_counter, 0);

  Kokkos::parallel_for(
      "FlattenViews",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
          {0, 0, 0}, {(int)natoms, (int)nrad, (int)nang}),
      KOKKOS_LAMBDA(const int iatoms, const int iradial, const int iangular) {
        const int src = iradial * nang + iangular;

        if (wt_2d(iatoms, src) > weight_threshold) {
          int dest = Kokkos::atomic_fetch_add(&w_counter(0), 1);
          wt_1d(dest) = wt_2d(iatoms, src);
          qp_1d(dest) = qp_2d(iatoms, src);
        }
      });

  Kokkos::printf("Reduced weight count from %d to %d (%f %%) for a weight "
                 "threshold of %e\n",
                 npts * natoms, w_counter_h(0),
                 100 * (1.0 - w_counter_h(0) / double(npts * natoms)),
                 weight_threshold);

  return {qp_1d, wt_1d, ac_dev, Z_dev, (unsigned)nrad, (unsigned)nang};
}

} // namespace Nukexc
