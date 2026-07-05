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

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace Nukexc {

DeviceView2DLeft compute_exact_exchange(
    const ExecSpace space, const DeviceView2DLeft mo_orbitals,
    const DeviceView1D mo_coeff, const DeviceView2DLeft basis_collocation,
    const DeviceView2DLeft basis_aux_collocation,
    const DeviceView2DLeft potential_collocation_scaled,
    const DeviceView2DLeft half_inverse_X) {

  Kokkos::Profiling::pushRegion("Compute Exact Exchange Integral");
  const int N_bf = basis_collocation.extent(0);
  const int N_bf_aux = basis_aux_collocation.extent(0);
  const int N_quad = basis_collocation.extent(1);
  const int N_occ = mo_orbitals.extent(1);
  const int K = half_inverse_X.extent(1);

  DeviceView2DLeft basis_collocation_scaled("Basis collocation Scaled", N_bf,
                                            N_quad);
  DeviceView1DLeft expansion_coeff("Expansion coeff", N_quad);
  DeviceView2DLeft three_center_integral("Three_center_integral", N_bf_aux,
                                         N_bf);
  DeviceView2DLeft three_center_integral_scaled("Three_center_integral_scaled",
                                                N_bf, K);
  DeviceView2DLeft result("Exchange matrix", N_bf, N_bf);

  auto mo_coeff_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, mo_coeff);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
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

    KokkosBlas::gemm(space, "N", "T", 1.0, potential_collocation_scaled,
                     basis_collocation_scaled, 0.0, three_center_integral);

    KokkosBlas::gemm(space, "T", "N", 1.0, three_center_integral,
                     half_inverse_X, 0.0, three_center_integral_scaled);

    KokkosBlas::gemm(space, "N", "T", mo_coeff_h(i),
                     three_center_integral_scaled, three_center_integral_scaled,
                     1.0, result);
  }
  Kokkos::fence();
  Kokkos::Profiling::popRegion();
  return result;
}

} // namespace Nukexc
