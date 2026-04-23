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

#pragma once
#include "kokkos_config.hpp"
#include "molecule.hpp"
#include "partitioning.hpp"
#include "stobasis.hpp"

#include <KokkosBlas3_gemm.hpp>

namespace NuKEXC {

Kokkos::View<double **>
overlap_integral(STOBasisSet &basis,
                        Kokkos::View<double *[3]> quadrature_points,
                        Kokkos::View<double *> quadrature_weights) {

  size_t N = basis.nbf();
  size_t nquad_points = quadrature_points.extent(0);
  Kokkos::View<double **> collocation_points =
      evaluate_sto_basis_shells_on_collocation_points(basis, quadrature_points);

  // Can be replaced by Kokkos kernel later
  Kokkos::View<double **> overlap_matrix("Overlap matrix", N, N);
  Kokkos::View<double **> weighted_points("Overlap matrix", N, nquad_points);

  Kokkos::parallel_for(
      "Scale Points",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, nquad_points}),
      KOKKOS_LAMBDA(const int &i, const int &g) {
        weighted_points(i, g) =
            quadrature_weights(g) * collocation_points(i, g);
      });
  KokkosBlas::gemm("N", "T", 1.0, weighted_points, collocation_points, 0.0,
                   overlap_matrix);

  return overlap_matrix;
}
} // namespace NuKEXC
