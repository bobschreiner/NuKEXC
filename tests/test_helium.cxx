/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Bob Schreiner
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 *
 *    Redistribution and use in source and binary forms, with or without
 *    modification, are permitted provided that the following conditions are
 *    met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 *    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/coulomb.hpp>
#include <nukexc/diagonalizer.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp>
#include <nukexc/xc_integrals.hpp>

#include <cmath>
#include <vector>

#include <xc.h>
#include <xc_funcs.h>

using namespace Nukexc;
using bk_type = IntegratorXX::Becke<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

TEST_CASE("compute_gga -- Helium 1s gga", "[gga]") {
  const double ref_energy = -1.032549417787429;
  const double ref_potential = -0.657943275538615;

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}},
               std::vector<unsigned>{2u});
  auto grid = make_flat_grid<bk_type, ll_type>(mol, 2000, 100);

  // Primary basis: single 1s STO
  STOBasisSet basis = make_manual_basis({{1, 0, 0, 1.6875, 0., 0., 0.}});

  xc_func_type func;
  const int func_id = XC_GGA_X_PBE;
  if (xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0) {
    throw std::runtime_error("Failed to initialize Libxc functional");
  }

  // Density matrix: fully occupied single orbital, D_11 = 1
  DeviceView2DLeft mo_orbitals("Mo orbitals", 1, 1);
  DeviceView1D mo_coeff("MO coeff", 1);
  auto orbitals_h = Kokkos::create_mirror_view(mo_orbitals);
  auto coeff_h = Kokkos::create_mirror_view(mo_coeff);
  orbitals_h(0, 0) = 1.0;
  coeff_h(0) = 2.0;
  Kokkos::deep_copy(mo_orbitals, orbitals_h);
  Kokkos::deep_copy(mo_coeff, coeff_h);

  const int N_bf = basis.nbf();
  const int N_quad = grid.quad_points.extent(0);

  DeviceView2DLeft basis_collocation("Basis collocation", N_bf, N_quad);
  DeviceView2DLeft collocation_gx("Collocation gx", N_bf, N_quad);
  DeviceView2DLeft collocation_gy("Collocation gy", N_bf, N_quad);
  DeviceView2DLeft collocation_gz("Collocation gz", N_bf, N_quad);

  ExecSpace space;
  fill_collocation(space, basis, grid.quad_points, basis_collocation);
  fill_grad_collocation(space, basis, grid.quad_points, collocation_gx,
                        collocation_gy, collocation_gz);

  auto gga_result =
      compute_gga(basis_collocation, collocation_gx, collocation_gy,
                  collocation_gz, grid.weights, mo_orbitals, mo_coeff, func);

  // Clean up Libxc internal pointers
  xc_func_end(&func);

  auto gga_potential_h =
      Kokkos::create_mirror_view_and_copy(HostSpace{}, gga_result.potential);

  REQUIRE_THAT(gga_result.energy,
               Catch::Matchers::WithinRel(ref_energy, 1e-10));
  REQUIRE_THAT(gga_potential_h(0, 0),
               Catch::Matchers::WithinRel(ref_potential, 1e-10));
}

// ============================================================
int main() {
  Kokkos::initialize();
  int result = Catch::Session().run();
  Kokkos::finalize();
  return result;
}
