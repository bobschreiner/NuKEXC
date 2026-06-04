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
#include "nukexc/grid.hpp"
#include "nukexc_config.hpp"
#include "octree.hpp"
#include "stobasis.hpp"

#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Copy_Impl.hpp>
#include <KokkosBatched_Dot.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Team_Impl.hpp>
#include <KokkosBatched_Util.hpp>

#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Pair.hpp>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <impl/Kokkos_Profiling.hpp>

namespace NuKEXC {

DeviceView1D
compute_density(const STOBasisSet basis, const FlatGrid grid,
                Kokkos::View<double **, ExecSpace> density_matrix) {

  ExecSpace space;
  const int N_bf = basis.nbf();
  const int N_quad = grid.quad_points.extent(0);

  DeviceView2DLeft collocation_values("Basis collocation", N_bf, N_quad);
  fill_collocation(space, basis, grid.quad_points, collocation_values);

  DeviceView1D density("density", N_quad);
  DeviceView2D intermediate_matrix("intermediate", N_bf, N_quad);

  KokkosBlas::gemm(space, "N", "N", 1.0, density_matrix, collocation_values,
                   0.0, intermediate_matrix);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Contract the basis in order to get the density
  Kokkos::parallel_for(
      "Contract Basis", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        double sum = 0;
        auto collocation_subview =
            Kokkos::subview(collocation_values, Kokkos::ALL(), g);
        auto intermediate_subview =
            Kokkos::subview(intermediate_matrix, Kokkos::ALL(), g);

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, N_bf),
            [=](const int i, double &update_sum) {
              update_sum += collocation_subview(i) * intermediate_subview(i);
            },
            sum);
        team_member.team_barrier();
        density(g) = sum;
      });

  return density;
};

DeviceView1D
compute_density(DeviceView2DLeft collocation_values,
                Kokkos::View<double **, ExecSpace> density_matrix) {

  ExecSpace space;
  const int N_bf = collocation_values.extent(0);
  const int N_quad = collocation_values.extent(1);

  DeviceView1D density("density", N_quad);
  DeviceView2D intermediate_matrix("intermediate", N_bf, N_quad);

  KokkosBlas::gemm(space, "N", "N", 1.0, density_matrix, collocation_values,
                   0.0, intermediate_matrix);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Contract the basis in order to get the density
  Kokkos::parallel_for(
      "Contract Basis", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        double sum = 0;
        auto collocation_subview =
            Kokkos::subview(collocation_values, Kokkos::ALL(), g);
        auto intermediate_subview =
            Kokkos::subview(intermediate_matrix, Kokkos::ALL(), g);

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, N_bf),
            [=](const int i, double &update_sum) {
              update_sum += collocation_subview(i) * intermediate_subview(i);
            },
            sum);
        team_member.team_barrier();
        density(g) = sum;
      });

  return density;
};

void compute_density_and_sigma(const DeviceView2DLeft collocation_values,
                               const DeviceView2DLeft collocation_gx,
                               const DeviceView2DLeft collocation_gy,
                               const DeviceView2DLeft collocation_gz,
                               const DeviceView2D density_matrix,
                               DeviceView1D rho, DeviceView1D gx_rho,
                               DeviceView1D gy_rho, DeviceView1D gz_rho,
                               DeviceView1D sigma) {

  ExecSpace space;
  const int N_bf = collocation_values.extent(0);
  const int N_quad = collocation_values.extent(1);

  DeviceView2D intermediate_matrix("intermediate", N_bf, N_quad);

  KokkosBlas::gemm(space, "N", "N", 1.0, density_matrix, collocation_values,
                   0.0, intermediate_matrix);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Contract the basis in order to get the density
  Kokkos::parallel_for(
      "Contract Basis", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        double sum_values = 0;
        double sum_gx = 0;
        double sum_gy = 0;
        double sum_gz = 0;
        auto collocation_subview =
            Kokkos::subview(collocation_values, Kokkos::ALL(), g);
        auto collocation_gx_subview =
            Kokkos::subview(collocation_gx, Kokkos::ALL(), g);
        auto collocation_gy_subview =
            Kokkos::subview(collocation_gy, Kokkos::ALL(), g);
        auto collocation_gz_subview =
            Kokkos::subview(collocation_gz, Kokkos::ALL(), g);

        auto intermediate_subview =
            Kokkos::subview(intermediate_matrix, Kokkos::ALL(), g);

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, N_bf),
            [=](const int i, double &update_sum, double &update_sum_gx,
                double &update_sum_gy, double &update_sum_gz) {
              update_sum += collocation_subview(i) * intermediate_subview(i);
              update_sum_gx +=
                  collocation_gx_subview(i) * intermediate_subview(i);
              update_sum_gy +=
                  collocation_gy_subview(i) * intermediate_subview(i);
              update_sum_gz +=
                  collocation_gz_subview(i) * intermediate_subview(i);
            },
            sum_values, sum_gx, sum_gy, sum_gz);
        team_member.team_barrier();
        rho(g) = sum_values;
        gx_rho(g) = 2.0 * sum_gx;
        gy_rho(g) = 2.0 * sum_gy;
        gz_rho(g) = 2.0 * sum_gz;
        sigma(g) = 4.0 * (sum_gx * sum_gx + sum_gy * sum_gy + sum_gz * sum_gz);
      });
};

void compute_density_and_sigma_and_tau(
    const DeviceView2DLeft collocation_values,
    const DeviceView2DLeft collocation_gx,
    const DeviceView2DLeft collocation_gy,
    const DeviceView2DLeft collocation_gz, const DeviceView2D density_matrix,
    DeviceView1D rho, DeviceView1D gx_rho, DeviceView1D gy_rho,
    DeviceView1D gz_rho, DeviceView1D sigma, DeviceView1D tau) {

  ExecSpace space;
  const int N_bf = collocation_values.extent(0);
  const int N_quad = collocation_values.extent(1);

  DeviceView2D intermediate_matrix("intermediate", N_bf, N_quad);
  DeviceView2D intermediate_matrix_gx("intermediate gx", N_bf, N_quad);
  DeviceView2D intermediate_matrix_gy("intermediate gy", N_bf, N_quad);
  DeviceView2D intermediate_matrix_gz("intermediate gz", N_bf, N_quad);

  KokkosBlas::gemm(space, "N", "N", 1.0, density_matrix, collocation_values,
                   0.0, intermediate_matrix);
  KokkosBlas::gemm(space, "N", "N", 1.0, density_matrix, collocation_gx, 0.0,
                   intermediate_matrix_gx);

  KokkosBlas::gemm(space, "N", "N", 1.0, density_matrix, collocation_gy, 0.0,
                   intermediate_matrix_gy);

  KokkosBlas::gemm(space, "N", "N", 1.0, density_matrix, collocation_gz, 0.0,
                   intermediate_matrix_gz);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Contract the basis in order to get the density
  Kokkos::parallel_for(
      "Contract Basis", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        double sum_values = 0;
        double sum_gx = 0;
        double sum_gy = 0;
        double sum_gz = 0;
        double sum_tau = 0;
        auto collocation_subview =
            Kokkos::subview(collocation_values, Kokkos::ALL(), g);
        auto collocation_gx_subview =
            Kokkos::subview(collocation_gx, Kokkos::ALL(), g);
        auto collocation_gy_subview =
            Kokkos::subview(collocation_gy, Kokkos::ALL(), g);
        auto collocation_gz_subview =
            Kokkos::subview(collocation_gz, Kokkos::ALL(), g);

        auto intermediate_subview =
            Kokkos::subview(intermediate_matrix, Kokkos::ALL(), g);
        auto intermediate_gx_subview =
            Kokkos::subview(intermediate_matrix_gx, Kokkos::ALL(), g);

        auto intermediate_gy_subview =
            Kokkos::subview(intermediate_matrix_gy, Kokkos::ALL(), g);

        auto intermediate_gz_subview =
            Kokkos::subview(intermediate_matrix_gz, Kokkos::ALL(), g);

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, N_bf),
            [=](const int i, double &update_sum, double &update_sum_gx,
                double &update_sum_gy, double &update_sum_gz,
                double &update_tau) {
              update_sum += collocation_subview(i) * intermediate_subview(i);
              update_sum_gx +=
                  collocation_gx_subview(i) * intermediate_subview(i);
              update_sum_gy +=
                  collocation_gy_subview(i) * intermediate_subview(i);
              update_sum_gz +=
                  collocation_gz_subview(i) * intermediate_subview(i);
              update_tau +=
                  collocation_gx_subview(i) * intermediate_gx_subview(i) +
                  collocation_gy_subview(i) * intermediate_gy_subview(i) +
                  collocation_gz_subview(i) * intermediate_gz_subview(i);
            },
            sum_values, sum_gx, sum_gy, sum_gz, sum_tau);
        team_member.team_barrier();
        rho(g) = sum_values;
        gx_rho(g) = 2.0 * sum_gx;
        gy_rho(g) = 2.0 * sum_gy;
        gz_rho(g) = 2.0 * sum_gz;
        sigma(g) = 4.0 * (sum_gx * sum_gx + sum_gy * sum_gy + sum_gz * sum_gz);
        tau(g) = 0.5 * sum_tau;
      });
};
}; // namespace NuKEXC
