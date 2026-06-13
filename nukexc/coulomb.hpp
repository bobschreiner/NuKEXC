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

#include "density.hpp"
#include "grid.hpp"
#include "nukexc_config.hpp"
#include "stobasis.hpp"
#include "stopotential.hpp"

#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas2_gemv_impl.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_gemm_impl.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace Nukexc {

DeviceView2DLeft compute_coulomb(const STOBasisSet basis,
                                 const STOBasisSet basis_aux,
                                 const FlatGrid grid,
                                 const DeviceView2DLeft mo_orbitals,
                                 const DeviceView1D mo_coeff,
                                 const double lin_dep_threshold = 1e-5) {

  ExecSpace space;
  const int N_bf = basis.nbf();
  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = grid.quad_points.extent(0);

  DeviceView2DLeft basis_collocation("Basis collocation", N_bf, N_quad);
  DeviceView2DLeft basis_aux_collocation("Auxillary basis collocation",
                                         N_bf_aux, N_quad);
  DeviceView2DLeft basis_collocation_scaled("Basis collocation Scaled", N_bf,
                                            N_quad);
  DeviceView1DLeft expansion_coeff("Expansion coeff", N_bf_aux);
  DeviceView1DLeft potential_on_grid("Expansion coeff scaled", N_quad);
  DeviceView2DLeft result("Coulomb matrix", N_bf, N_bf);
  DeviceView2DLeft aux_overlap("Aux overlap", N_bf_aux, N_bf_aux);
  DeviceView2DLeft aux_overlap_sym("Aux overlap sym", N_bf_aux, N_bf_aux);

  fill_collocation(space, basis, grid.quad_points, basis_collocation);
  fill_collocation(space, basis_aux, grid.quad_points, basis_aux_collocation);

  DeviceView1DLeft density =
      compute_density(basis_collocation, mo_orbitals, mo_coeff);
  DeviceView2DLeft potential_collocation =
      sto_potential_collocation(space, basis_aux, grid, basis_aux_collocation);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      "Scale potential", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        const double w_g = grid.weights(g);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf_aux),
            [=](const int i) { potential_collocation(i, g) *= w_g; });
      });

  // Compute expansion coefficients
  KokkosBlas::gemv(space, "N", 1.0, potential_collocation, density, 0.0,
                   expansion_coeff);

  // Compute (A|B)
  KokkosBlas::gemm(space, "N", "T", 1.0, basis_aux_collocation,
                   potential_collocation, 0.0, aux_overlap);

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf_aux, N_bf_aux}),
      KOKKOS_LAMBDA(int i, int j) {
        aux_overlap_sym(i, j) = 0.5 * (aux_overlap(i, j) + aux_overlap(j, i));
      });

  DeviceView2DLeft X = compute_half_inverse(aux_overlap_sym, lin_dep_threshold);
  // Invert (A|B) using svd
  // DeviceView2DLeft X = compute_half_inverse(aux_overlap, lin_dep_threshold);
  const int K = X.extent(1);
  DeviceView1DLeft scaling_factor("Scaling factor", K);

  // Apply (A|B)^{-1}
  KokkosBlas::gemv(space, "T", 1.0, X, expansion_coeff, 0.0, scaling_factor);
  KokkosBlas::gemv(space, "N", 1.0, X, scaling_factor, 0.0, expansion_coeff);

  // Compute potential on grid
  KokkosBlas::gemv(space, "T", 1.0, potential_collocation, expansion_coeff, 0.0,
                   potential_on_grid);
  // Compute (mu nu | A)
  Kokkos::parallel_for(
      "Scale basis by potential", policy,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        const double pot_g = potential_on_grid(g);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf), [=](const int i) {
              basis_collocation_scaled(i, g) = basis_collocation(i, g) * pot_g;
            });
      });
  KokkosBlas::gemm(space, "N", "T", 1.0, basis_collocation,
                   basis_collocation_scaled, 0.0, result);

  return result;
}

} // namespace Nukexc
