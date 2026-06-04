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
#include "nukexc/partitioning.hpp"
#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include "stobasis.hpp"

#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas2_gemv_impl.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_gemm_impl.hpp>

#include <KokkosLapack_gesv.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <impl/Kokkos_CheckUsage.hpp>
#include <stdexcept>
#include <xc.h>

namespace NuKEXC {

std::pair<double, DeviceView2DLeft>
compute_lda(const STOBasisSet basis, const FlatGrid grid,
            const DeviceView2D density_matrix, const xc_func_type func) {

  ExecSpace space;
  // Make sure that the porovided xc functional is a LDA
  if (func.info->family != XC_FAMILY_LDA) {
    throw std::runtime_error(
        "Provided funtional is not a part of the LDA family");
  }

  // Get some constants
  const int N_quad = grid.quad_points.extent(0);
  const int N_bf = basis.nbf();

  // Compute densities rho and sigma
  DeviceView2DLeft collocation_values("Collocation values", N_bf, N_quad);
  DeviceView1D rho("Rho", N_quad);
  DeviceView1D vrho("VRho", N_quad);
  DeviceView1D exc("Exc", N_quad);

  fill_collocation(space, basis, grid.quad_points, collocation_values);

  rho = compute_density(collocation_values, density_matrix);

  // Copy densities rho and sigma to the host device
  auto rho_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho);

  auto vrho_h = Kokkos::create_mirror_view(vrho);
  auto exc_h = Kokkos::create_mirror_view(exc);

  Kokkos::deep_copy(vrho_h, 0.0);
  Kokkos::deep_copy(exc_h, 0.0);

  // Evaluate xc functionals on the host device with libxc
  xc_lda_exc_vxc(&func, N_quad, rho_h.data(), exc_h.data(), vrho_h.data());

  // Copy the values back to the execution device
  Kokkos::deep_copy(exc, exc_h);
  Kokkos::deep_copy(vrho, vrho_h);

  DeviceView2DLeft Z("Z LDA", N_bf, N_quad);
  DeviceView2DLeft V("V LDA", N_bf, N_bf);
  DeviceView2DLeft V_result("V LDA Result", N_bf, N_bf);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      "Compute Z_mu(r)", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        const double w_g = grid.weights(g);
        const double vrho_g = vrho(g);

        auto collocation_subview =
            Kokkos::subview(collocation_values, Kokkos::ALL(), g);
        auto Z_subview = Kokkos::subview(Z, Kokkos::ALL(), g);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf), [=](const int i) {
              double local_sum = 0.0;
              Z_subview(i) = w_g * (0.5 * collocation_subview(i) * vrho_g);
            });
      });

  // Compute gemm
  const double one(1.0);
  const double zero(0.0);
  KokkosBlas::gemm("N", "T", one, collocation_values, Z, zero, V);

  // Symmetrize result
  Kokkos::parallel_for(
      "Symmetrize Result",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
      KOKKOS_LAMBDA(const int i, const int j) {
        V_result(i, j) = V(i, j) + V(j, i);
      });

  double xc_energy = 0.0;

  // Integrate xc_energy density
  Kokkos::parallel_reduce(
      "Compute xc energy", N_quad,
      KOKKOS_LAMBDA(const int g, double &xc_update) {
        xc_update += grid.weights(g) * rho(g) * exc(g);
      },
      xc_energy);

  return std::make_pair(xc_energy, V_result);
}

std::pair<double, DeviceView2DLeft>
compute_gga(const STOBasisSet basis, const FlatGrid grid,
            const DeviceView2D density_matrix, const xc_func_type func) {

  ExecSpace space;
  // Make sure that the porovided xc functional is a GGA
  if (func.info->family != XC_FAMILY_GGA) {
    throw std::runtime_error(
        "Provided funtional is not a part of the GGA family");
  }

  // Get some constants
  const int N_quad = grid.quad_points.extent(0);
  const int N_bf = basis.nbf();

  // Compute densities rho and sigma
  DeviceView1D rho("Rho", N_quad);
  DeviceView1D gx_rho("Grad x Rho", N_quad);
  DeviceView1D gy_rho("Grad y Rho", N_quad);
  DeviceView1D gz_rho("Grad z Rho", N_quad);
  DeviceView1D sigma("Sigma", N_quad);
  DeviceView1D vrho("VRho", N_quad);
  DeviceView1D vsigma("VSigma", N_quad);
  DeviceView1D exc("Exc", N_quad);

  // Fill the collocation Views
  DeviceView2DLeft collocation_values("collocation values", N_bf, N_quad);
  DeviceView2DLeft collocation_gx("collocation gx", N_bf, N_quad);
  DeviceView2DLeft collocation_gy("collocation gy", N_bf, N_quad);
  DeviceView2DLeft collocation_gz("collocation gz", N_bf, N_quad);

  fill_collocation(space, basis, grid.quad_points, collocation_values);

  fill_grad_collocation(space, basis, grid.quad_points, collocation_gx,
                        collocation_gy, collocation_gz);

  compute_density_and_sigma(collocation_values, collocation_gx, collocation_gy,
                            collocation_gz, density_matrix, rho, gx_rho, gy_rho,
                            gz_rho, sigma);

  // Copy densities rho and sigma to the host device
  auto rho_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho);
  auto sigma_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma);

  auto vrho_h = Kokkos::create_mirror_view(vrho);
  auto vsigma_h = Kokkos::create_mirror_view(vsigma);
  auto exc_h = Kokkos::create_mirror_view(exc);

  Kokkos::deep_copy(vrho_h, 0.0);
  Kokkos::deep_copy(vsigma_h, 0.0);
  Kokkos::deep_copy(exc_h, 0.0);

  // Evaluate xc functionals on the host device with libxc
  xc_gga_exc_vxc(&func, N_quad, rho_h.data(), sigma_h.data(), exc_h.data(),
                 vrho_h.data(), vsigma_h.data());

  // Copy the values back to the execution device
  Kokkos::deep_copy(exc, exc_h);
  Kokkos::deep_copy(vrho, vrho_h);
  Kokkos::deep_copy(vsigma, vsigma_h);

  DeviceView2DLeft Z("Z GGA", N_bf, N_quad);
  DeviceView2DLeft V("V GGA", N_bf, N_bf);
  DeviceView2DLeft V_result("V GGA Result", N_bf, N_bf);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      "Compute Z_mu(r)", policy, KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        const double w_g = grid.weights(g);
        const double vrho_g = vrho(g);
        const double gx_rho_g = gx_rho(g);
        const double gy_rho_g = gy_rho(g);
        const double gz_rho_g = gz_rho(g);
        const double vsigma_g = vsigma(g);

        auto collocation_subview =
            Kokkos::subview(collocation_values, Kokkos::ALL(), g);
        auto collocation_gx_subview =
            Kokkos::subview(collocation_gx, Kokkos::ALL(), g);
        auto collocation_gy_subview =
            Kokkos::subview(collocation_gy, Kokkos::ALL(), g);
        auto collocation_gz_subview =
            Kokkos::subview(collocation_gz, Kokkos::ALL(), g);
        auto Z_subview = Kokkos::subview(Z, Kokkos::ALL(), g);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf), [=](const int i) {
              double local_sum = 0.0;
              local_sum += gx_rho_g * collocation_gx_subview(i);
              local_sum += gy_rho_g * collocation_gy_subview(i);
              local_sum += gz_rho_g * collocation_gz_subview(i);
              Z_subview(i) = w_g * (0.5 * collocation_subview(i) * vrho_g +
                                    2.0 * local_sum * vsigma_g);
            });
      });

  // Compute gemm
  const double one(1.0);
  const double zero(0.0);
  KokkosBlas::gemm("N", "T", one, collocation_values, Z, zero, V);

  // Symmetrize result
  Kokkos::parallel_for(
      "Symmetrize Result",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
      KOKKOS_LAMBDA(const int i, const int j) {
        V_result(i, j) = V(i, j) + V(j, i);
      });

  double xc_energy = 0.0;

  // Integrate xc_energy density
  Kokkos::parallel_reduce(
      "Compute xc energy", N_quad,
      KOKKOS_LAMBDA(const int g, double &xc_update) {
        xc_update += grid.weights(g) * rho(g) * exc(g);
      },
      xc_energy);

  return std::make_pair(xc_energy, V_result);
}

std::pair<double, DeviceView2DLeft>
compute_mgga(const STOBasisSet basis, const FlatGrid grid,
             const DeviceView2D density_matrix, const xc_func_type func) {

  // TODO:: Actually compute lapalcian and compute contribution to V
  // Right now we assue the the mgga funcitonal is indepdentant of the
  // laplacian

  ExecSpace space;
  // Make sure that the porovided xc functional is a GGA
  if (func.info->family != XC_FAMILY_MGGA) {
    throw std::runtime_error(
        "Provided funtional is not a part of the MGGA family");
  }

  // Get some constants
  const int N_quad = grid.quad_points.extent(0);
  const int N_bf = basis.nbf();

  // Compute densities rho and sigma
  DeviceView2DLeft collocation("Collocation", N_bf, N_quad);
  DeviceView1D rho("Rho", N_quad);
  DeviceView1D gx_rho("Grad x Rho", N_quad);
  DeviceView1D gy_rho("Grad y Rho", N_quad);
  DeviceView1D gz_rho("Grad z Rho", N_quad);
  DeviceView1D sigma("Sigma", N_quad);
  DeviceView1D tau("Tau", N_quad);
  DeviceView1D lapl("Laplacian", N_quad);

  DeviceView1D vrho("VRho", N_quad);
  DeviceView1D vsigma("VSigma", N_quad);
  DeviceView1D vtau("VTau", N_quad);
  DeviceView1D vlapl("VLaplacian", N_quad);
  DeviceView1D exc("Exc", N_quad);

  // Fill the collocation Views
  DeviceView2DLeft collocation_values("collocation values", N_bf, N_quad);
  DeviceView2DLeft collocation_gx("collocation gx", N_bf, N_quad);
  DeviceView2DLeft collocation_gy("collocation gy", N_bf, N_quad);
  DeviceView2DLeft collocation_gz("collocation gz", N_bf, N_quad);

  fill_collocation(space, basis, grid.quad_points, collocation_values);

  fill_grad_collocation(space, basis, grid.quad_points, collocation_gx,
                        collocation_gy, collocation_gz);

  compute_density_and_sigma_and_tau(
      collocation_values, collocation_gx, collocation_gy, collocation_gz,
      density_matrix, rho, gx_rho, gy_rho, gz_rho, sigma, tau);

  // Copy densities rho and sigma to the host device
  auto rho_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho);
  auto sigma_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma);
  auto tau_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, tau);
  auto lapl_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, lapl);

  auto vrho_h = Kokkos::create_mirror_view(vrho);
  auto vsigma_h = Kokkos::create_mirror_view(vsigma);
  auto vtau_h = Kokkos::create_mirror_view(vtau);
  auto vlapl_h = Kokkos::create_mirror_view(vlapl);
  auto exc_h = Kokkos::create_mirror_view(exc);

  Kokkos::deep_copy(exc_h, 0.0);
  Kokkos::deep_copy(vrho_h, 0.0);
  Kokkos::deep_copy(vsigma_h, 0.0);
  Kokkos::deep_copy(vtau_h, 0.0);
  Kokkos::deep_copy(vlapl_h, 0.0);

  // Evaluate xc functionals on the host device with libxc
  xc_mgga_exc_vxc(&func, N_quad, rho_h.data(), sigma_h.data(), lapl_h.data(),
                  tau_h.data(), exc_h.data(), vrho_h.data(), vsigma_h.data(),
                  vlapl_h.data(), vtau_h.data());

  // Copy the values back to the execution device
  Kokkos::deep_copy(exc, exc_h);
  Kokkos::deep_copy(vrho, vrho_h);
  Kokkos::deep_copy(vsigma, vsigma_h);
  Kokkos::deep_copy(vtau, vtau_h);
  Kokkos::deep_copy(vlapl, vlapl_h);

  DeviceView2DLeft Z("Z MGGA", N_bf, N_quad);
  DeviceView2DLeft V("V MGGA", N_bf, N_bf);
  DeviceView2DLeft W_x("W_x MGGA", N_bf, N_bf);
  DeviceView2DLeft W_y("W_y MGGA", N_bf, N_bf);
  DeviceView2DLeft W_z("W_x MGGA", N_bf, N_bf);
  DeviceView2DLeft V_result("V MGGA Result", N_bf, N_bf);

  Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      "Compute Z_mu(r) and W_mu(r)", policy,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int g = team_member.league_rank();
        const double w_g = grid.weights(g);
        const double vrho_g = vrho(g);
        const double gx_rho_g = gx_rho(g);
        const double gy_rho_g = gy_rho(g);
        const double gz_rho_g = gz_rho(g);
        const double vsigma_g = vsigma(g);
        const double vtau_g = vtau(g);

        auto collocation_subview =
            Kokkos::subview(collocation_values, Kokkos::ALL(), g);
        auto collocation_gx_subview =
            Kokkos::subview(collocation_gx, Kokkos::ALL(), g);
        auto collocation_gy_subview =
            Kokkos::subview(collocation_gy, Kokkos::ALL(), g);
        auto collocation_gz_subview =
            Kokkos::subview(collocation_gz, Kokkos::ALL(), g);
        auto Z_subview = Kokkos::subview(Z, Kokkos::ALL(), g);

        auto W_x_subview = Kokkos::subview(W_x, Kokkos::ALL(), g);
        auto W_y_subview = Kokkos::subview(W_y, Kokkos::ALL(), g);
        auto W_z_subview = Kokkos::subview(W_z, Kokkos::ALL(), g);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, N_bf), [=](const int i) {
              double local_sum = 0.0;
              local_sum += gx_rho_g * collocation_gx_subview(i);
              local_sum += gy_rho_g * collocation_gy_subview(i);
              local_sum += gz_rho_g * collocation_gz_subview(i);
              Z_subview(i) = w_g * (0.5 * collocation_subview(i) * vrho_g +
                                    2.0 * local_sum * vsigma_g);
              W_x_subview(i) =
                  w_g * (0.25 * vtau_g * collocation_gx_subview(i));
              W_y_subview(i) =
                  w_g * (0.25 * vtau_g * collocation_gy_subview(i));
              W_z_subview(i) =
                  w_g * (0.25 * vtau_g * collocation_gz_subview(i));
            });
      });

  // Compute gemm
  const double one(1.0);
  const double zero(0.0);
  KokkosBlas::gemm("N", "T", one, collocation, Z, zero, V);
  KokkosBlas::gemm("N", "T", one, collocation_gx, W_x, one, V);
  KokkosBlas::gemm("N", "T", one, collocation_gy, W_y, one, V);
  KokkosBlas::gemm("N", "T", one, collocation_gz, W_z, one, V);

  // Symmetrize result
  Kokkos::parallel_for(
      "Symmetrize Result",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
      KOKKOS_LAMBDA(const int i, const int j) {
        V_result(i, j) = V(i, j) + V(j, i);
      });

  double xc_energy = 0.0;

  // Integrate xc_energy density
  Kokkos::parallel_reduce(
      "Compute xc energy", N_quad,
      KOKKOS_LAMBDA(const int g, double &xc_update) {
        xc_update += grid.weights(g) * rho(g) * exc(g);
      },
      xc_energy);

  return std::make_pair(xc_energy, V_result);
}

} // namespace NuKEXC
