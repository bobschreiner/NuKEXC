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

#include <KokkosBatched_Eigendecomposition_Decl.hpp>
#include <KokkosBlas3_gemm.hpp>

namespace NuKEXC {

void diagonalise(const Kokkos::View<double ***> &fock_matrix,
                 Kokkos::View<double ***> &mo_coeff,
                 Kokkos::View<double **> &mo_energies) {

  int batch_size = fock_matrix.extent(0);
  int n = fock_matrix.extent(1);

  //==================================================
  // Allocate matrices for Eigenvalu decomposition
  //==================================================

  using team_policy_type = Kokkos::TeamPolicy<ExecSpace>;
  team_policy_type policy_team(batch_size, Kokkos::AUTO, 32);

  Kokkos::View<double ***, Kokkos::LayoutRight> UL("UL", batch_size, n,
                                                   n); // Left eigenvectors
  Kokkos::View<double ***, Kokkos::LayoutRight> UR("UR", batch_size, n,
                                                   n); // Right eigenvectors

  Kokkos::View<double **, Kokkos::LayoutRight> er(
      "er", batch_size, n); // Real parts of eigenvalues
  Kokkos::View<double **, Kokkos::LayoutRight> ei(
      "ei", batch_size, n); // Imaginary parts of eigenvalues

  // Workspace (size = 2*n*n + 5*n)
  Kokkos::View<double **, Kokkos::LayoutRight> W("W", batch_size,
                                                 2 * n * n + 5 * n);
  //==================================================
  // Compute X = S^(-1/2) using Cholesky
  //==================================================

  //==================================================
  // Compute F in othogoanl Basis
  //==================================================

  //==================================================
  // Diagonalise F
  //==================================================
  Kokkos::parallel_for(
      "Diagonalise F", policy_team,
      KOKKOS_LAMBDA(const typename team_policy_type::member_type &member) {
        // Get batch index from team rank
        const int i = member.league_rank();

        // Extract batch slice
        auto A_i =
            Kokkos::subview(fock_matrix, i, Kokkos::ALL(), Kokkos::ALL());
        auto er_i = Kokkos::subview(er, i, Kokkos::ALL());
        auto ei_i = Kokkos::subview(ei, i, Kokkos::ALL());
        auto UR_i = Kokkos::subview(UR, i, Kokkos::ALL(), Kokkos::ALL());
        auto UL_i = Kokkos::subview(UL, i, Kokkos::ALL(), Kokkos::ALL());
        auto W_i = Kokkos::subview(W, i, Kokkos::ALL());

        // Perform eigendecomposition
        KokkosBatched::TeamVectorEigendecomposition<
            typename team_policy_type::member_type>::invoke(member, A_i, er_i,
                                                            ei_i, UL_i, UR_i,
                                                            W_i);
      });

  Kokkos::fence();

  //==================================================
  // Compute mo_coeff in original basis
  //==================================================
}

} // namespace NuKEXC
