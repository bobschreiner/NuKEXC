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
#include "nukexc_utils.hpp"
#include "stobasis.hpp"

#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas2_gemv_impl.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_gemm_impl.hpp>

#include <KokkosLapack_gesv.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace NuKEXC {
KOKKOS_INLINE_FUNCTION
double I_tilde(const int n, const int l, const double r, const double zeta) {
  int a = n + l + 2;
  int b = n - l + 1;
  double zr = zeta * r;
  return lower_gamma(a, zr) /
             (Kokkos::pow(zeta, a) * Kokkos::pow(r, 2 * l + 1)) +
         upper_gamma(b, zr) / Kokkos::pow(zeta, b);
}

KOKKOS_INLINE_FUNCTION
double C_prefactor(const int n, const int l, const double zeta) {
  return 4 * M_PI * Kokkos::pow(2 * zeta, n + 0.5) /
         (Kokkos::sqrt(factorial(2 * n)) * (2 * l + 1));
}

// Potential — just three multiplications
KOKKOS_INLINE_FUNCTION
double sto_potential(const int n, const int l, const int m, const double x,
                     const double y, const double z, const double r,
                     const double zeta) {
  double val;
  real_solid_harmonic_cart_precomputed(l, m, x, y, z, val);
  return C_prefactor(n, l, zeta) * val * I_tilde(n, l, r, zeta);
}

DeviceView2DLeft sto_potential_collocation(const ExecSpace space,
                                           const STOBasisSet basis,
                                           const FlatGrid grid,
                                           const DeviceView2DLeft basis_vals) {

  const int N_bf = basis.nbf();
  const int N_quad = grid.quad_points.extent(0);

  DeviceView2DLeft potential_collocation("Potential collocation", N_bf, N_quad);

  Kokkos::parallel_for(
      "Compute potentials",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf, N_quad}),
      KOKKOS_LAMBDA(const int i, const int g) {
        const int n = basis.n(i);
        const int l = basis.l(i);
        const int m = basis.m(i);
        const double zeta = basis.zeta(i);
        const double x = grid.quad_points(g)[0] - basis.O(i)[0];
        const double y = grid.quad_points(g)[1] - basis.O(i)[1];
        const double z = grid.quad_points(g)[2] - basis.O(i)[2];
        const double r = dist(grid.quad_points(g), basis.O(i));
        potential_collocation(i, g) =
            sto_potential(n, l, m, x, y, z, r, zeta) + epsilon_shift;
      });
  space.fence();
  return potential_collocation;
}

DeviceView2DLeft compute_poisson(const STOBasisSet basis,
                                 const STOBasisSet basis_aux,
                                 const FlatGrid grid,
                                 const DeviceView2D density_matrix) {

  ExecSpace space;
  const int N_bf = basis.nbf();
  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = grid.quad_points.extent(0);

  DeviceView2DLeft basis_aux_collocation("Auxillary basis collocation",
                                         N_bf_aux, N_quad);

  DeviceView2DLeft basis_collocation("Basis collocation", N_bf, N_quad);
  DeviceView2DLeft basis_collocation_scaled("Basis collocation Scaled", N_bf,
                                            N_quad);

  DeviceView1DLeft expansion_coeff("Expansion coeff", N_bf_aux);

  DeviceView1DLeft potential_on_grid("Expansion coeff scaled", N_quad);

  DeviceView1DLeft density = compute_density(basis, grid, density_matrix);

  DeviceView2DLeft result("Poisson matrix", N_bf, N_bf);

  DeviceView2DLeft aux_overlap("Aux overlap", N_bf_aux, N_bf_aux);

  Kokkos::View<int *, Kokkos::LayoutLeft, ExecSpace> piv("pivot", N_bf_aux);

  fill_collocation(space, basis_aux, grid.quad_points, basis_aux_collocation);
  fill_collocation(space, basis, grid.quad_points, basis_collocation);

  DeviceView2DLeft potential_collocation =
      sto_potential_collocation(space, basis_aux, grid, basis_aux_collocation);

  Kokkos::parallel_for(
      "Scale potential",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(space, {0, 0}, {N_bf_aux, N_quad}),
      KOKKOS_LAMBDA(const int i, const int g) {
        potential_collocation(i, g) *= grid.weights(g);
      });
  space.fence();

  KokkosBlas::gemv(space, "N", 1.0, potential_collocation, density, 0.0,
                   expansion_coeff);

  KokkosBlas::gemm(space, "N", "T", 1.0, basis_aux_collocation,
                   potential_collocation, 0.0, aux_overlap);

  space.fence();

  KokkosLapack::gesv(space, aux_overlap, expansion_coeff, piv);

  KokkosBlas::gemv(space, "T", 1.0, potential_collocation, expansion_coeff, 0.0,
                   potential_on_grid);

  Kokkos::parallel_for(
      "Scale basis",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(space, {0, 0}, {N_bf, N_quad}),
      KOKKOS_LAMBDA(const int i, const int g) {
        basis_collocation_scaled(i, g) =
            basis_collocation(i, g) * potential_on_grid(g);
      });

  KokkosBlas::gemm(space, "N", "T", 1.0, basis_collocation,
                   basis_collocation_scaled, 0.0, result);
  space.fence();

  return result;
}

} // namespace NuKEXC
