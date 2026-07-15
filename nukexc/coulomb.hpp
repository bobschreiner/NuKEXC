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
#include "nukexc/octree.hpp"
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

DeviceView2DLeft
compute_coulomb(const ExecSpace space, const DeviceView2DLeft mo_orbitals,
                const DeviceView1D mo_coeff,
                const DeviceView2DLeft basis_collocation,
                const DeviceView2DLeft basis_aux_collocation,
                const DeviceView2DLeft potential_collocation_scaled,
                const DeviceView2DLeft half_inverse_X) {

  Kokkos::Profiling::pushRegion("Compute Coulomb Integral");
  const int N_bf = basis_collocation.extent(0);
  const int N_bf_aux = basis_aux_collocation.extent(0);
  const int N_quad = basis_collocation.extent(1);
  const int K = half_inverse_X.extent(1);

  DeviceView2DLeft basis_collocation_scaled("Basis collocation Scaled", N_bf,
                                            N_quad);
  DeviceView1DLeft expansion_coeff("Expansion coeff", N_bf_aux);
  DeviceView1DLeft potential_on_grid("Expansion coeff scaled", N_quad);
  DeviceView2DLeft result("Coulomb matrix", N_bf, N_bf);
  DeviceView1DLeft density =
      compute_density(basis_collocation, mo_orbitals, mo_coeff);

  DeviceView1DLeft scaling_factor("Scaling factor", K);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Compute expansion coefficients
  KokkosBlas::gemv(space, "N", 1.0, potential_collocation_scaled, density, 0.0,
                   expansion_coeff);

  // Apply (A|B)^{-1}
  KokkosBlas::gemv(space, "T", 1.0, half_inverse_X, expansion_coeff, 0.0,
                   scaling_factor);
  KokkosBlas::gemv(space, "N", 1.0, half_inverse_X, scaling_factor, 0.0,
                   expansion_coeff);

  // Compute potential on grid
  KokkosBlas::gemv(space, "T", 1.0, potential_collocation_scaled,
                   expansion_coeff, 0.0, potential_on_grid);

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

  Kokkos::fence();
  Kokkos::Profiling::popRegion();
  return result;
}

DeviceView2DLeft compute_coulomb_sparse(
    const ExecSpace space, const DeviceView2DLeft mo_orbitals,
    const DeviceView1D mo_coeff, const STOBasisSet basis,
    const STOBasisSet basis_aux, const FlatGrid grid, const NeighborList nl,
    const DeviceView2DLeft half_inverse_X) {

  Kokkos::Profiling::pushRegion("Compute Coulomb Integral Sparse");
  const int N_bf = basis.nbf();
  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = grid.quad_points.extent(0);
  const int N_elec = mo_coeff.extent(0);
  const int K = half_inverse_X.extent(1);

  const int max_points_per_box = nl.max_points_per_box;
  const int num_boxes = nl.offsets.extent(0) - 1;
  const int num_neighbors = nl.neighbors.extent(0);

  DeviceView1DLeft expansion_coeff("Expansion coeff", N_bf_aux);
  DeviceView2DLeft density_matrix("Density matrix", N_bf, N_bf);
  DeviceView2DLeft result("Coulomb matrix", N_bf, N_bf);

  Kokkos::parallel_for(
      "Fill Density matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
      KOKKOS_LAMBDA(const int mu, const int nu) {
        for (unsigned int k = 0; k < N_elec; ++k)
          density_matrix(mu, nu) +=
              mo_coeff(k) * mo_orbitals(mu, k) * mo_orbitals(nu, k);
      });

  DeviceView1DLeft scaling_factor("Scaling factor", K);

  Kokkos::TeamPolicy<ExecSpace> policy_boxes(space, num_boxes, Kokkos::AUTO());

  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
  typedef ExecSpace::scratch_memory_space ScratchSpace;
  typedef Kokkos::View<double *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_double;

  typedef Kokkos::View<Point *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_points;

  int scratch_size_team = shared_view_double::shmem_size(max_points_per_box) +
                          shared_view_points::shmem_size(max_points_per_box);

  policy_boxes.set_scratch_size(0, Kokkos::PerTeam(scratch_size_team));

  Kokkos::parallel_for(
      "Sparse Kernel: coeff(alpha) = sum_{ml} (alpha |l m) D_lm", policy_boxes,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int box_idx = team_member.league_rank();

        // Compute number of points per box
        const int start_points = box_idx * max_points_per_box;
        const int end_points =
            Kokkos::min(start_points + max_points_per_box, N_quad);
        const int num_points = end_points - start_points;

        // Compute number of neighbors per box
        const int start_neighbors = nl.offsets(box_idx);
        const int end_neighbors = nl.offsets(box_idx + 1);
        const int num_neighbors = end_neighbors - start_neighbors;

        shared_view_double weights_scratch(team_member.team_scratch(0),
                                           num_points);

        shared_view_points points_scratch(team_member.team_scratch(0),
                                          num_points);

        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                             [=](const int local_g) {
                               const int global_g = start_points + local_g;
                               weights_scratch(local_g) =
                                   grid.weights(global_g);
                               points_scratch(local_g) =
                                   grid.quad_points(global_g);
                             });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf_aux),
            [=](const int local_alpha) {
              const int n = basis_aux.n(local_alpha);
              const int l = basis_aux.l(local_alpha);
              const int m = basis_aux.m(local_alpha);
              const double zeta = basis_aux.zeta(local_alpha);

              double local_sum_alpha = 0;
              double local_sum_ij;

              double phi_i;
              double phi_j;

              int global_i;
              int global_j;

              for (int local_g = 0; local_g < num_points; ++local_g) {
                const int global_g = start_points + local_g;

                const double x =
                    points_scratch(local_g)[0] - basis_aux.O(local_alpha)[0];
                const double y =
                    points_scratch(local_g)[1] - basis_aux.O(local_alpha)[1];
                const double z =
                    points_scratch(local_g)[2] - basis_aux.O(local_alpha)[2];
                const double r =
                    dist(points_scratch(local_g), basis_aux.O(local_alpha));

                double potential_scaled =
                    sto_potential(n, l, m, x, y, z, r, zeta) *
                    weights_scratch(local_g);

                local_sum_ij = 0;
                for (unsigned int local_i = 0; local_i < num_neighbors;
                     ++local_i) {
                  global_i = nl.neighbors(start_neighbors + local_i);
                  basis_eval(basis, global_i, points_scratch(local_g)[0],
                             points_scratch(local_g)[1],
                             points_scratch(local_g)[2], phi_i);
                  for (unsigned int local_j = 0; local_j < num_neighbors;
                       ++local_j) {

                    global_j = nl.neighbors(start_neighbors + local_j);
                    basis_eval(basis, global_j, points_scratch(local_g)[0],
                               points_scratch(local_g)[1],
                               points_scratch(local_g)[2], phi_j);
                    local_sum_ij +=
                        phi_i * phi_j * density_matrix(global_i, global_j);
                  }
                }
                local_sum_alpha += local_sum_ij * potential_scaled;
              }
              Kokkos::atomic_add(&expansion_coeff(local_alpha),
                                 local_sum_alpha);
            });
      });

  space.fence();
  // Apply (A|B)^{-1}
  KokkosBlas::gemv(space, "T", 1.0, half_inverse_X, expansion_coeff, 0.0,
                   scaling_factor);
  KokkosBlas::gemv(space, "N", 1.0, half_inverse_X, scaling_factor, 0.0,
                   expansion_coeff);

  policy_boxes.set_scratch_size(0, Kokkos::PerTeam(scratch_size_team));
  Kokkos::parallel_for(
      "Sparse Kernel: J_{mu,nu} = sum_{alpha} (mu nu|alpha)", policy_boxes,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int box_idx = team_member.league_rank();

        // Compute number of points per box
        const int start_points = box_idx * max_points_per_box;
        const int end_points =
            Kokkos::min(start_points + max_points_per_box, N_quad);
        const int num_points = end_points - start_points;

        // Compute number of neighbors per box
        const int start_neighbors = nl.offsets(box_idx);
        const int end_neighbors = nl.offsets(box_idx + 1);
        const int num_neighbors = end_neighbors - start_neighbors;

        shared_view_double weights_scratch(team_member.team_scratch(0),
                                           num_points);

        shared_view_points points_scratch(team_member.team_scratch(0),
                                          num_points);

        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                             [=](const int local_g) {
                               const int global_g = start_points + local_g;
                               weights_scratch(local_g) =
                                   grid.weights(global_g);
                               points_scratch(local_g) =
                                   grid.quad_points(global_g);
                             });
        team_member.team_barrier();
        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                      num_neighbors),
            [=](const int local_i, const int local_j) {
              const int global_i = nl.neighbors(start_neighbors + local_i);
              const int global_j = nl.neighbors(start_neighbors + local_j);

              double phi_i;
              double phi_j;

              double local_result;
              for (int local_g = 0; local_g < num_points; ++local_g) {
                basis_eval(basis, global_i, points_scratch(local_g)[0],
                           points_scratch(local_g)[1],
                           points_scratch(local_g)[2], phi_i);
                basis_eval(basis, global_j, points_scratch(local_g)[0],
                           points_scratch(local_g)[1],
                           points_scratch(local_g)[2], phi_j);

                local_result = 0;
                for (int local_alpha = 0; local_alpha < N_bf_aux;
                     ++local_alpha) {
                  const int n = basis_aux.n(local_alpha);
                  const int l = basis_aux.l(local_alpha);
                  const int m = basis_aux.m(local_alpha);
                  const double zeta = basis_aux.zeta(local_alpha);
                  const double x =
                      points_scratch(local_g)[0] - basis_aux.O(local_alpha)[0];
                  const double y =
                      points_scratch(local_g)[1] - basis_aux.O(local_alpha)[1];
                  const double z =
                      points_scratch(local_g)[2] - basis_aux.O(local_alpha)[2];
                  const double r =
                      dist(points_scratch(local_g), basis_aux.O(local_alpha));

                  double potential_alpha =
                      sto_potential(n, l, m, x, y, z, r, zeta);

                  local_result += potential_alpha * phi_i * phi_j *
                                  weights_scratch(local_g) *
                                  expansion_coeff(local_alpha);
                }
                Kokkos::atomic_add(&result(global_i, global_j), local_result);
              }
            });
      });

  space.fence();
  Kokkos::Profiling::popRegion();
  return result;
}

} // namespace Nukexc
