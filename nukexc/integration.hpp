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

#include <KokkosBatched_Dot.hpp>
#include <KokkosBlas3_gemm.hpp>

namespace NuKEXC {

Kokkos::View<double **>
overlap_integral(STOBasisSet &basis,
                 Kokkos::View<double *[3]> quadrature_points,
                 Kokkos::View<double *> quadrature_weights) {

  size_t N = basis.nbf();
  size_t nquad_points = quadrature_points.extent(0);
  Kokkos::View<double **> collocation_points =
      evaluate_sto_basis_on_collocation_points(basis, quadrature_points);

  // Can be replaced by Kokkos kernel later
  Kokkos::View<double **> overlap_matrix("Overlap matrix", N, N);
  Kokkos::View<double **> weighted_points("Weighted points", N, nquad_points);

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

Kokkos::View<double **>
diag_overlap_integral(STOBasisSet &basis,
                      Kokkos::View<double *[3]> quadrature_points,
                      Kokkos::View<double *> quadrature_weights) {

  size_t N = basis.nbf();
  size_t nquad_points = quadrature_points.extent(0);
  Kokkos::View<double **> collocation_points =
      evaluate_sto_basis_on_collocation_points(basis, quadrature_points);

  // Can be replaced by Kokkos kernel later
  Kokkos::View<double **> overlap_matrix("Overlap matrix", N, N);

  Kokkos::parallel_for(
      "Compute diag{S}", N, KOKKOS_LAMBDA(const int &i) {
        for (int g = 0; g < nquad_points; ++g) {
          overlap_matrix(i,i) += quadrature_weights(g) * collocation_points(i, g) *
                         collocation_points(i, g);
        }
      });
  return overlap_matrix;
}
Kokkos::View<double **> nuclear_potential_integral(
    STOBasisSet &basis, Kokkos::View<double *[3]> quadrature_points,
    Kokkos::View<double *> quadrature_weights,
    Kokkos::View<double *[3]> atom_centers, Kokkos::View<unsigned *> Z) {

  size_t N = basis.nbf();
  size_t nquad_points = quadrature_points.extent(0);
  Kokkos::View<double **> collocation_points =
      evaluate_sto_basis_on_collocation_points(basis, quadrature_points);

  // Can be replaced by Kokkos kernel later
  Kokkos::View<double **> V_n("Nuclear potential matrix", N, N);
  Kokkos::View<double **> weighted_points("Weighted points", N, nquad_points);

  Kokkos::parallel_for(
      "Scale Points",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, nquad_points}),
      KOKKOS_LAMBDA(const int &i, const int &g) {
        for (unsigned int k = 0; k < atom_centers.extent(0); ++k) {
          double dx = quadrature_points(g, 0) - atom_centers(k, 0);
          double dy = quadrature_points(g, 1) - atom_centers(k, 1);
          double dz = quadrature_points(g, 2) - atom_centers(k, 2);
          double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) + 1e-15;

          weighted_points(i, g) -=
              (Z(k) / r) * quadrature_weights(g) * collocation_points(i, g);
        }
      });
  KokkosBlas::gemm("N", "T", 1.0, weighted_points, collocation_points, 0.0,
                   V_n);

  return V_n;
}
Kokkos::View<double **>
kinetic_integral(STOBasisSet &basis,
                 Kokkos::View<double *[3]> quadrature_points,
                 Kokkos::View<double *> quadrature_weights) {

  size_t nbasis = basis.nbf();
  size_t nquad_points = quadrature_points.extent(0);
  Kokkos::View<double **[3]> grad_collocation_points =
      evaluate_sto_basis_grad_on_collocation_points(basis, quadrature_points);

  // Can be replaced by Kokkos kernel later
  Kokkos::View<double **> kinetic_matrix("Overlap matrix", nbasis, nbasis);
  Kokkos::View<double **> Gx("Gradient in x direction", nbasis, nquad_points);
  Kokkos::View<double **> Gy("Gradient in y direction", nbasis, nquad_points);
  Kokkos::View<double **> Gz("Gradient in z direction", nbasis, nquad_points);

  Kokkos::parallel_for(
      "Weighted Grad Collocation",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nbasis, nquad_points}),

      KOKKOS_LAMBDA(const int &i, const int &g) {
        double weight_factor = Kokkos::sqrt(quadrature_weights(g));
        Gx(i, g) = grad_collocation_points(i, g, 0) * weight_factor;
        Gy(i, g) = grad_collocation_points(i, g, 1) * weight_factor;
        Gz(i, g) = grad_collocation_points(i, g, 2) * weight_factor;
      });

  // T = 0.5 * (Gx * Gx^T) + 0.0 * T
  KokkosBlas::gemm("N", "T", 0.5, Gx, Gx, 0.0, kinetic_matrix);

  // T = 0.5 * (Gy * Gy^T) + 1.0 * T (accumulate)
  KokkosBlas::gemm("N", "T", 0.5, Gy, Gy, 1.0, kinetic_matrix);

  // T = 0.5 * (Gz * Gz^T) + 1.0 * T (accumulate)
  KokkosBlas::gemm("N", "T", 0.5, Gz, Gz, 1.0, kinetic_matrix);

  return kinetic_matrix;
}
} // namespace NuKEXC
