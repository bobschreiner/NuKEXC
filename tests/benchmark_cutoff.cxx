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

#include <Kokkos_Core.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <impl/Kokkos_CheckUsage.hpp>
#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <iomanip>
#include <iostream>
#include <nukexc/grid.hpp>

#include <nukexc/molecule.hpp>
#include <nukexc/stobasis.hpp>

#include <catch2/catch_all.hpp>
#include <catch2/catch_assertion_info.hpp>
#include <vector>

using namespace NuKEXC;

TEST_CASE("Basis Cutoff", "[cutoff]") {

  using bk_type = IntegratorXX::Becke<double, double>;
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  for (double cutoff_tol = 1e-6; cutoff_tol > 1e-15; cutoff_tol /= 10) {
    Molecule mol;
    read_xyz("input/taxol.xyz", mol);
    STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", cutoff_tol);

    FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 40, 10);

    int N = basis.nbf();
    int G = grid.quad_points.extent(0);

    Kokkos::View<int> counter("counter");

    Kokkos::parallel_for(
        "Count points outside cutoff",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, G}),
        KOKKOS_LAMBDA(const int i, const int g) {
          double r = dist(basis.O(i), grid.quad_points(g));
          if (r > basis.cutoff_radii(i))
            Kokkos::atomic_fetch_add(&counter(), 1);
        });

    Kokkos::fence();
    auto count_h = Kokkos::create_mirror_view(counter);
    Kokkos::deep_copy(count_h, counter);
    double percent = (double)count_h() / (N * G) * 100;
    std::cout << "Quad notes outside of cutoff radius:  " << std::setw(4)
              << std::setprecision(4) << percent
              << "\%     tol = " << cutoff_tol << "\n";
  }
}

int main() {
  Kokkos::initialize();
  {
    int result = Catch::Session().run();
  }
  Kokkos::finalize();
  return 0;
};
