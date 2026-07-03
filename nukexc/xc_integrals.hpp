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

#include <stdexcept>
#include <xc.h>

namespace Nukexc {

struct XC_result {
  double energy;
  DeviceView2DLeft potential;
};

struct XC_result_polarized {
  double energy;
  DeviceView2DLeft potential_alpha;
  DeviceView2DLeft potential_beta;
};

XC_result compute_lda(const DeviceView2DLeft collocation_values,
                      const DeviceView1D weights,
                      const DeviceView2DLeft mo_orbitals,
                      const DeviceView1D mo_coeff, const xc_func_type func) {

  ExecSpace space;

  // Make sure that the porovided xc functional is a LDA
  if (func.info->family != XC_FAMILY_LDA) {
    throw std::runtime_error(
        "Provided funtional is not a part of the LDA family");
  }

  if (func.nspin != XC_UNPOLARIZED) {
    throw std::runtime_error(
        "compute_lda requires a functional initialized with XC_UNPOLARIZED; "
        "use compute_lsda for XC_POLARIZED");
  }

  // Get some constants
  const int N_quad = weights.extent(0);
  const int N_bf = collocation_values.extent(0);

  // Compute densities rho and sigma
  DeviceView1D rho("Rho", N_quad);
  DeviceView1D vrho("VRho", N_quad);
  DeviceView1D exc("Exc", N_quad);

  rho = compute_density(collocation_values, mo_orbitals, mo_coeff);

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
        const double w_g = weights(g);
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
        xc_update += weights(g) * rho(g) * exc(g);
      },
      xc_energy);

  return XC_result{xc_energy, V_result};
}

XC_result_polarized
compute_lsda(const DeviceView2DLeft collocation_values,
             const DeviceView1D weights, const DeviceView2DLeft mo_alpha,
             const DeviceView1D occ_alpha, const DeviceView2DLeft mo_beta,
             const DeviceView1D occ_beta, const xc_func_type func) {
  ExecSpace space;

  if (func.info->family != XC_FAMILY_LDA)
    throw std::runtime_error(
        "Provided functional is not part of the LDA family");
  if (func.nspin != XC_POLARIZED)
    throw std::runtime_error(
        "compute_lsda requires a functional initialized with XC_POLARIZED");

  const int N_quad = weights.extent(0);
  const int N_bf = collocation_values.extent(0);

  DeviceView1D rho_a = compute_density(collocation_values, mo_alpha, occ_alpha);
  DeviceView1D rho_b = compute_density(collocation_values, mo_beta, occ_beta);
  auto rho_a_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho_a);
  auto rho_b_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho_b);

  // libxc polarized layout: rho/vrho interleaved (up, down) per point.
  Kokkos::View<double *, HostSpace> rho_pol_h("rho_pol_h", 2 * N_quad);
  Kokkos::View<double *, HostSpace> vrho_pol_h("vrho_pol_h", 2 * N_quad);
  Kokkos::View<double *, HostSpace> exc_h("exc_h", N_quad);
  for (int g = 0; g < N_quad; ++g) {
    rho_pol_h(2 * g) = rho_a_h(g);
    rho_pol_h(2 * g + 1) = rho_b_h(g);
  }
  Kokkos::deep_copy(vrho_pol_h, 0.0);
  Kokkos::deep_copy(exc_h, 0.0);

  // ONE joint call: exc(g) and vrho_up/down(g) all come from the same
  // evaluation at the same (rho_up(g), rho_down(g)) pair.
  xc_lda_exc_vxc(&func, N_quad, rho_pol_h.data(), exc_h.data(),
                 vrho_pol_h.data());

  DeviceView1D exc("Exc", N_quad), vrho_up("VRho up", N_quad),
      vrho_down("VRho down", N_quad);
  auto exc_hv = Kokkos::create_mirror_view(exc);
  auto vu_hv = Kokkos::create_mirror_view(vrho_up);
  auto vd_hv = Kokkos::create_mirror_view(vrho_down);
  for (int g = 0; g < N_quad; ++g) {
    exc_hv(g) = exc_h(g);
    vu_hv(g) = vrho_pol_h(2 * g);
    vd_hv(g) = vrho_pol_h(2 * g + 1);
  }
  Kokkos::deep_copy(exc, exc_hv);
  Kokkos::deep_copy(vrho_up, vu_hv);
  Kokkos::deep_copy(vrho_down, vd_hv);

  auto build_potential = [&](const DeviceView1D &vrho_spin) {
    DeviceView2DLeft Z("Z LSDA", N_bf, N_quad);
    DeviceView2DLeft V("V LSDA", N_bf, N_bf);
    DeviceView2DLeft V_result("V LSDA Result", N_bf, N_bf);
    Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
    using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
    Kokkos::parallel_for(
        "Compute Z_mu(r)", policy,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int g = team_member.league_rank();
          const double w_g = weights(g);
          const double vrho_g = vrho_spin(g);
          auto colloc = Kokkos::subview(collocation_values, Kokkos::ALL(), g);
          auto Zsub = Kokkos::subview(Z, Kokkos::ALL(), g);
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, N_bf),
              [=](const int i) { Zsub(i) = w_g * (0.5 * colloc(i) * vrho_g); });
        });
    const double one(1.0), zero(0.0);
    KokkosBlas::gemm("N", "T", one, collocation_values, Z, zero, V);
    Kokkos::parallel_for(
        "Symmetrize Result",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
        KOKKOS_LAMBDA(const int i, const int j) {
          V_result(i, j) = V(i, j) + V(j, i);
        });
    return V_result;
  };

  DeviceView2DLeft V_alpha = build_potential(vrho_up);
  DeviceView2DLeft V_beta = build_potential(vrho_down);

  double xc_energy = 0.0;
  Kokkos::parallel_reduce(
      "Compute xc energy", N_quad,
      KOKKOS_LAMBDA(const int g, double &acc) {
        acc += weights(g) * (rho_a(g) + rho_b(g)) * exc(g);
      },
      xc_energy);

  return XC_result_polarized{xc_energy, V_alpha, V_beta};
}

XC_result compute_gga(const DeviceView2DLeft collocation_values,
                      const DeviceView2DLeft collocation_gx,
                      const DeviceView2DLeft collocation_gy,
                      const DeviceView2DLeft collocation_gz,
                      const DeviceView1D weights,
                      const DeviceView2DLeft mo_orbitals,
                      const DeviceView1D mo_coeff, const xc_func_type func) {

  ExecSpace space;
  // Make sure that the porovided xc functional is a GGA
  if (func.info->family != XC_FAMILY_GGA) {
    throw std::runtime_error(
        "Provided funtional is not a part of the GGA family");
  }

  // Get some constants
  const int N_quad = weights.extent(0);
  const int N_bf = collocation_values.extent(0);

  // Compute densities rho and sigma
  DeviceView1D rho("Rho", N_quad);
  DeviceView1D gx_rho("Grad x Rho", N_quad);
  DeviceView1D gy_rho("Grad y Rho", N_quad);
  DeviceView1D gz_rho("Grad z Rho", N_quad);
  DeviceView1D sigma("Sigma", N_quad);
  DeviceView1D vrho("VRho", N_quad);
  DeviceView1D vsigma("VSigma", N_quad);
  DeviceView1D exc("Exc", N_quad);

  compute_density_and_sigma(collocation_values, collocation_gx, collocation_gy,
                            collocation_gz, mo_orbitals, mo_coeff, rho, gx_rho,
                            gy_rho, gz_rho, sigma);

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
        const double w_g = weights(g);
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
        xc_update += weights(g) * rho(g) * exc(g);
      },
      xc_energy);

  return XC_result{xc_energy, V_result};
}

XC_result_polarized
compute_gga_lsda(const DeviceView2DLeft collocation_values,
                 const DeviceView2DLeft collocation_gx,
                 const DeviceView2DLeft collocation_gy,
                 const DeviceView2DLeft collocation_gz,
                 const DeviceView1D weights, const DeviceView2DLeft mo_alpha,
                 const DeviceView1D occ_alpha, const DeviceView2DLeft mo_beta,
                 const DeviceView1D occ_beta, const xc_func_type func) {

  ExecSpace space;

  if (func.info->family != XC_FAMILY_GGA)
    throw std::runtime_error(
        "Provided functional is not part of the GGA family");
  if (func.nspin != XC_POLARIZED)
    throw std::runtime_error(
        "compute_gga_lsda requires a functional initialized with XC_POLARIZED; "
        "use compute_gga for XC_UNPOLARIZED");

  const int N_quad = weights.extent(0);
  const int N_bf = collocation_values.extent(0);

  // ---- Per-spin densities, gradients, same-spin sigma (sigma_aa/sigma_bb) --
  DeviceView1D rho_a("rho_a", N_quad), gxa("gxa", N_quad), gya("gya", N_quad),
      gza("gza", N_quad), sigma_aa("sigma_aa", N_quad);
  DeviceView1D rho_b("rho_b", N_quad), gxb("gxb", N_quad), gyb("gyb", N_quad),
      gzb("gzb", N_quad), sigma_bb("sigma_bb", N_quad);

  compute_density_and_sigma(collocation_values, collocation_gx, collocation_gy,
                            collocation_gz, mo_alpha, occ_alpha, rho_a, gxa,
                            gya, gza, sigma_aa);
  compute_density_and_sigma(collocation_values, collocation_gx, collocation_gy,
                            collocation_gz, mo_beta, occ_beta, rho_b, gxb, gyb,
                            gzb, sigma_bb);

  // ---- Cross term sigma_ud = grad(rho_a) . grad(rho_b) ---------------------
  DeviceView1D sigma_ud("sigma_ud", N_quad);
  Kokkos::parallel_for(
      "Compute sigma_ud", N_quad, KOKKOS_LAMBDA(const int g) {
        sigma_ud(g) = gxa(g) * gxb(g) + gya(g) * gyb(g) + gza(g) * gzb(g);
      });

  // ---- Copy to host, interleave into libxc's polarized layout ---------------
  auto rho_a_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho_a);
  auto rho_b_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, rho_b);
  auto s_uu_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma_aa);
  auto s_ud_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma_ud);
  auto s_dd_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, sigma_bb);

  Kokkos::View<double *, HostSpace> rho_pol_h("rho_pol_h", 2 * N_quad);
  Kokkos::View<double *, HostSpace> sigma_pol_h("sigma_pol_h", 3 * N_quad);
  for (int g = 0; g < N_quad; ++g) {
    rho_pol_h(2 * g) = rho_a_h(g);
    rho_pol_h(2 * g + 1) = rho_b_h(g);
    sigma_pol_h(3 * g) = s_uu_h(g);
    sigma_pol_h(3 * g + 1) = s_ud_h(g);
    sigma_pol_h(3 * g + 2) = s_dd_h(g);
  }

  Kokkos::View<double *, HostSpace> vrho_pol_h("vrho_pol_h", 2 * N_quad);
  Kokkos::View<double *, HostSpace> vsigma_pol_h("vsigma_pol_h", 3 * N_quad);
  Kokkos::View<double *, HostSpace> exc_h("exc_h", N_quad);
  Kokkos::deep_copy(vrho_pol_h, 0.0);
  Kokkos::deep_copy(vsigma_pol_h, 0.0);
  Kokkos::deep_copy(exc_h, 0.0);

  // ---- One joint libxc call gives exc, vrho_up/down, vsigma_uu/ud/dd -------
  xc_gga_exc_vxc(&func, N_quad, rho_pol_h.data(), sigma_pol_h.data(),
                 exc_h.data(), vrho_pol_h.data(), vsigma_pol_h.data());

  DeviceView1D exc("exc", N_quad);
  DeviceView1D vrho_up("vrho_up", N_quad), vrho_down("vrho_down", N_quad);
  DeviceView1D vsigma_uu("vsigma_uu", N_quad), vsigma_ud("vsigma_ud", N_quad),
      vsigma_dd("vsigma_dd", N_quad);

  auto exc_hv = Kokkos::create_mirror_view(exc);
  auto vu_hv = Kokkos::create_mirror_view(vrho_up);
  auto vd_hv = Kokkos::create_mirror_view(vrho_down);
  auto vuu_hv = Kokkos::create_mirror_view(vsigma_uu);
  auto vud_hv = Kokkos::create_mirror_view(vsigma_ud);
  auto vdd_hv = Kokkos::create_mirror_view(vsigma_dd);
  for (int g = 0; g < N_quad; ++g) {
    exc_hv(g) = exc_h(g);
    vu_hv(g) = vrho_pol_h(2 * g);
    vd_hv(g) = vrho_pol_h(2 * g + 1);
    vuu_hv(g) = vsigma_pol_h(3 * g);
    vud_hv(g) = vsigma_pol_h(3 * g + 1);
    vdd_hv(g) = vsigma_pol_h(3 * g + 2);
  }
  Kokkos::deep_copy(exc, exc_hv);
  Kokkos::deep_copy(vrho_up, vu_hv);
  Kokkos::deep_copy(vrho_down, vd_hv);
  Kokkos::deep_copy(vsigma_uu, vuu_hv);
  Kokkos::deep_copy(vsigma_ud, vud_hv);
  Kokkos::deep_copy(vsigma_dd, vdd_hv);

  // ---- vec_up = 2*vsigma_uu*grad(rho_a) + vsigma_ud*grad(rho_b), and mirror -
  DeviceView1D vecx_up("vecx_up", N_quad), vecy_up("vecy_up", N_quad),
      vecz_up("vecz_up", N_quad);
  DeviceView1D vecx_dn("vecx_dn", N_quad), vecy_dn("vecy_dn", N_quad),
      vecz_dn("vecz_dn", N_quad);
  Kokkos::parallel_for(
      "Compute vec_up/vec_down", N_quad, KOKKOS_LAMBDA(const int g) {
        const double uu = vsigma_uu(g), ud = vsigma_ud(g), dd = vsigma_dd(g);
        vecx_up(g) = 2.0 * uu * gxa(g) + ud * gxb(g);
        vecy_up(g) = 2.0 * uu * gya(g) + ud * gyb(g);
        vecz_up(g) = 2.0 * uu * gza(g) + ud * gzb(g);
        vecx_dn(g) = 2.0 * dd * gxb(g) + ud * gxa(g);
        vecy_dn(g) = 2.0 * dd * gyb(g) + ud * gya(g);
        vecz_dn(g) = 2.0 * dd * gzb(g) + ud * gza(g);
      });

  // ---- Build Fock contribution for one spin channel -------------------------
  auto build_potential = [&](const DeviceView1D &vrho_spin,
                             const DeviceView1D &vecx, const DeviceView1D &vecy,
                             const DeviceView1D &vecz) {
    DeviceView2DLeft Z("Z GGA pol", N_bf, N_quad);
    DeviceView2DLeft V("V GGA pol", N_bf, N_bf);
    DeviceView2DLeft V_result("V GGA pol Result", N_bf, N_bf);

    Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
    using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
    Kokkos::parallel_for(
        "Compute Z_mu(r)", policy,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int g = team_member.league_rank();
          const double w_g = weights(g);
          const double vrho_g = vrho_spin(g);
          const double vx = vecx(g), vy = vecy(g), vz = vecz(g);

          auto colloc = Kokkos::subview(collocation_values, Kokkos::ALL(), g);
          auto colloc_gx = Kokkos::subview(collocation_gx, Kokkos::ALL(), g);
          auto colloc_gy = Kokkos::subview(collocation_gy, Kokkos::ALL(), g);
          auto colloc_gz = Kokkos::subview(collocation_gz, Kokkos::ALL(), g);
          auto Zsub = Kokkos::subview(Z, Kokkos::ALL(), g);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, N_bf), [=](const int i) {
                const double grad_dot =
                    vx * colloc_gx(i) + vy * colloc_gy(i) + vz * colloc_gz(i);
                Zsub(i) = w_g * (0.5 * colloc(i) * vrho_g + grad_dot);
              });
        });

    const double one(1.0), zero(0.0);
    KokkosBlas::gemm("N", "T", one, collocation_values, Z, zero, V);
    Kokkos::parallel_for(
        "Symmetrize Result",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_bf}),
        KOKKOS_LAMBDA(const int i, const int j) {
          V_result(i, j) = V(i, j) + V(j, i);
        });
    return V_result;
  };

  DeviceView2DLeft V_alpha =
      build_potential(vrho_up, vecx_up, vecy_up, vecz_up);
  DeviceView2DLeft V_beta =
      build_potential(vrho_down, vecx_dn, vecy_dn, vecz_dn);

  // ---- Energy: integrate against total density ------------------------------
  double xc_energy = 0.0;
  Kokkos::parallel_reduce(
      "Compute xc energy", N_quad,
      KOKKOS_LAMBDA(const int g, double &acc) {
        acc += weights(g) * (rho_a(g) + rho_b(g)) * exc(g);
      },
      xc_energy);

  return XC_result_polarized{xc_energy, V_alpha, V_beta};
}

XC_result compute_mgga(const STOBasisSet basis, const FlatGrid grid,
                       const DeviceView2DLeft mo_orbitals,
                       const DeviceView1D mo_coeff, const xc_func_type func) {

  // TODO:: Actually compute lapalcian and compute contribution to V
  // Right now we assume the the mgga funcitonal is indepdentant of the
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
      mo_orbitals, mo_coeff, rho, gx_rho, gy_rho, gz_rho, sigma, tau);

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

  return XC_result{xc_energy, V_result};
}

} // namespace Nukexc
