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
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
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

  auto gga_result = compute_gga(basis, grid, mo_orbitals, mo_coeff, func);

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
