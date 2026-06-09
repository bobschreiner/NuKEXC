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

#include <KokkosLapack_svd.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace Nukexc {

DeviceView2DLeft compute_exact_exchange(const STOBasisSet basis,
                                        const STOBasisSet basis_aux,
                                        const FlatGrid grid,
                                        const DeviceView2DLeft mo_orbitals,
                                        const DeviceView1D mo_coeff) {

  ExecSpace space;
  const int N_bf = basis.nbf();
  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = grid.quad_points.extent(0);
  const int N_occ = mo_orbitals.extent(1);

  DeviceView2DLeft basis_collocation("Basis collocation", N_bf, N_quad);
  DeviceView2DLeft basis_aux_collocation("Auxillary basis collocation",
                                         N_bf_aux, N_quad);
  DeviceView2DLeft basis_collocation_scaled("Basis collocation Scaled", N_bf,
                                            N_quad);
  DeviceView1DLeft expansion_coeff("Expansion coeff", N_quad);
  DeviceView2DLeft result("Exchange matrix", N_bf, N_bf);
  DeviceView2DLeft aux_overlap("Aux overlap", N_bf_aux, N_bf_aux);
  DeviceView2DLeft three_center_integral("Three_center_integral", N_bf_aux,
                                         N_bf);


  fill_collocation(space, basis, grid.quad_points, basis_collocation);
  fill_collocation(space, basis_aux, grid.quad_points, basis_aux_collocation);

  DeviceView2DLeft potential_collocation =
      sto_potential_collocation(space, basis_aux, grid, basis_aux_collocation);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      "Scale potential ", policy,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        const double w_g = grid.weights(g);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf_aux),
            [=](const int alpha) { potential_collocation(alpha, g) *= w_g; });
      });

  // Compute (A|B)
  KokkosBlas::gemm(space, "N", "T", 1.0, basis_aux_collocation,
                   potential_collocation, 0.0, aux_overlap);
  // Invert (A|B)
  DeviceView2DLeft X = compute_half_invserse(aux_overlap, 1e-7);
  const int K = X.extent(1);
  DeviceView2DLeft three_center_integral_scaled("Three_center_integral_scaled",
                                                N_bf, K);

  // Loop over the occupied orbitals and compute contributions for each occupied
  // orbital
  for (unsigned int i = 0; i < N_occ; ++i) {

    auto mo_orbitals_subview = Kokkos::subview(mo_orbitals, Kokkos::ALL, i);

    KokkosBlas::gemv(space, "T", 1.0, basis_collocation, mo_orbitals_subview,
                     0.0, expansion_coeff);

    Kokkos::parallel_for(
        "Scale collocation ", policy,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int g = team_member.league_rank();
          const double expansion_coeff_g = expansion_coeff(g);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, N_bf), [=](const int mu) {
                basis_collocation_scaled(mu, g) =
                    basis_collocation(mu, g) * expansion_coeff_g;
              });
        });

    KokkosBlas::gemm(space, "N", "T", 1.0, potential_collocation,
                     basis_collocation_scaled, 0.0, three_center_integral);

    KokkosBlas::gemm(space, "T", "N", 1.0, three_center_integral, X, 0.0,
                     three_center_integral_scaled);

    KokkosBlas::gemm(space, "N", "T", mo_coeff(i), three_center_integral_scaled,
                     three_center_integral_scaled, 1.0, result);
  }
  return result;
}

} // namespace Nukexc
